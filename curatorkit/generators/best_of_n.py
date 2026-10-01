"""
BestOfNTask — generate N completions per prompt, keep top-K by scorer.

Best-of-N (BoN) rejection sampling is the dominant inference-time
distillation strategy in 2024-2026 — OpenAI's o1 / o3 training stack,
DeepSeek's R1 distillation, and the Tulu 3 SFT pipeline all use BoN with
a learned reward model to select high-quality responses from candidate
pools.

Reference points:
  - Nakano et al., "WebGPT" (2021) — early BoN selection at inference.
  - Stiennon et al., "Learning to summarize from human feedback" (2020) —
    formalised the BoN curve.
  - Tulu 3 report (AI2, 2024) — current canonical SFT recipe using BoN
    with a reward model. https://arxiv.org/abs/2411.15124

CuratorKIT's BestOfNTask:

  For each input sample's `instruction`:
    1. Generate N candidate responses by sampling.
    2. Score each candidate with a `scorer` callable. The scorer signature
       is `(instruction, response, source_context) -> float`. This is
       deliberately abstract so any judge — Prometheus, DeepEval, a real
       reward model, or even a domain-specific heuristic — can plug in.
    3. Keep the top-K by score. K=1 produces SFT data; K=2 produces a
       chosen/rejected DPO pair from the same candidate pool (without the
       cost of a separate preference-generation pass).

This is the "real" GRPO-style data preparation pipeline — production
post-training stacks use this exact pattern (not the simpler
`GRPORolloutTask` which only emits a single rollout group per prompt).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from curatorkit.generators.base import STEP_VERSION, BaseGenerationTask
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

# Scorer signature: (instruction, response, source_context) -> float in [0, 1]
ScorerFn = Callable[[str, str, str], float]


# ───────────────────────────────────────────────────────────────────────────
# Built-in scorers
# ───────────────────────────────────────────────────────────────────────────


def make_prometheus_scorer(judge) -> ScorerFn:
    """Wrap a PrometheusJudge as a BoN scorer."""

    def score(instruction: str, response: str, source: str) -> float:
        r = judge.judge(instruction, response, reference=source or None)
        return r.score_normalized

    return score


def make_deepeval_scorer(judge) -> ScorerFn:
    """Wrap a DeepEvalJudge as a BoN scorer."""

    def score(instruction: str, response: str, source: str) -> float:
        r = judge.judge(instruction, response, context=source or None)
        return r.score_normalized

    return score


def make_factscore_scorer(judge) -> ScorerFn:
    """Wrap a FActScoreJudge as a BoN scorer (returns the factscore)."""

    def score(instruction: str, response: str, source: str) -> float:
        if not source:
            return 0.0
        return judge.judge(instruction, response, source).factscore

    return score


def make_length_scorer(target_tokens: int = 200) -> ScorerFn:
    """Heuristic scorer — useful as a baseline / sanity check."""

    def score(instruction: str, response: str, source: str) -> float:
        toks = response.split()
        if not toks:
            return 0.0
        return min(1.0, len(toks) / target_tokens)

    return score


# ───────────────────────────────────────────────────────────────────────────
# BestOfNTask
# ───────────────────────────────────────────────────────────────────────────


class BestOfNTask(BaseGenerationTask):
    """
    Generate N candidates per prompt, keep top-K by scorer.

    Parameters
    ----------
    llm : BaseLLM
        Generation backend.
    scorer : ScorerFn
        Callable (instruction, response, source) -> float in [0, 1].
        Build with `make_prometheus_scorer(judge)` or any of the helpers
        above, or pass your own callable.
    n_candidates : int
        Candidates per prompt. Production BoN uses N=8 to N=64. Default 8.
    top_k : int
        How many of the top candidates to keep per prompt.
          k=1  →  one SFT sample per prompt (the highest scorer).
          k=2  →  one DPO preference sample (chosen=top1, rejected=top2 OR
                  rejected=lowest, depending on `dpo_mode`).
          k>2  →  one GRPO sample carrying the top-k as `responses`
                  with `reward_scores` set to their scores.
    dpo_mode : "vs_lowest" | "consecutive"
        When top_k=2, choose how to pair. "vs_lowest" pairs top with the
        lowest-scoring candidate (sharper preference signal — Tulu 3
        default). "consecutive" pairs top with second-highest (softer).
    sampling_temperature : float
        Temperature for candidate generation. Default 0.9 — diversity matters.
    source_field : str
        Which field provides the source context the scorer sees. "input"
        (default) reads sample.input. "metadata.source_chunk" works too.
    """

    def __init__(
        self,
        llm: BaseLLM,
        scorer: ScorerFn,
        n_candidates: int = 8,
        top_k: int = 1,
        dpo_mode: str = "vs_lowest",
        sampling_temperature: float = 0.9,
        source_field: str = "input",
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=None, concurrency=concurrency)
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if top_k < 1 or top_k > n_candidates:
            raise ValueError(f"top_k must be in [1, n_candidates={n_candidates}], got {top_k}")
        if dpo_mode not in ("vs_lowest", "consecutive"):
            raise ValueError(f"dpo_mode must be 'vs_lowest' or 'consecutive', got {dpo_mode!r}")
        self.scorer = scorer
        self.n_candidates = int(n_candidates)
        self.top_k = int(top_k)
        self.dpo_mode = dpo_mode
        self.sampling_temperature = float(sampling_temperature)
        self.source_field = source_field

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        return [{"role": "user", "content": sample.instruction}]

    def _parse_response(self, sample, response):
        return []

    def _resolve_source(self, sample: DataSample) -> str:
        if self.source_field.startswith("metadata."):
            return str(sample.metadata.get(self.source_field.split(".", 1)[1], ""))
        return str(getattr(sample, self.source_field, "") or "")

    # ─────────────────────────────────────────────────────────────────────
    # Sampling + scoring
    # ─────────────────────────────────────────────────────────────────────

    def _sample_candidates(self, instruction: str) -> list[str]:
        out: list[str] = []
        for _ in range(self.n_candidates):
            try:
                r = self.llm.generate(
                    [{"role": "user", "content": instruction}],
                    temperature=self.sampling_temperature,
                )
                out.append((r.text or "").strip())
            except Exception:
                continue
        return [t for t in out if t]

    async def _sample_candidates_async(self, instruction: str) -> list[str]:
        async def one():
            try:
                r = await self.llm.agenerate(
                    [{"role": "user", "content": instruction}],
                    temperature=self.sampling_temperature,
                )
                return (r.text or "").strip()
            except Exception:
                return ""

        out = await asyncio.gather(*[one() for _ in range(self.n_candidates)])
        return [t for t in out if t]

    def _score_all(self, instruction: str, source: str, candidates: list[str]) -> list[float]:
        scores: list[float] = []
        for c in candidates:
            try:
                scores.append(float(self.scorer(instruction, c, source)))
            except Exception:
                scores.append(0.0)
        return scores

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline interface
    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        for sample in samples:
            r = self._process_one(sample, ts)
            if r is not None:
                results.append(r)
        return results

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        sem = asyncio.Semaphore(self.concurrency)

        async def one(s: DataSample):
            async with sem:
                return await self._process_one_async(s, ts)

        out = await asyncio.gather(*[one(s) for s in samples])
        return [x for x in out if x is not None]

    def _process_one(self, sample: DataSample, ts: datetime) -> DataSample | None:
        instruction = sample.instruction
        if not instruction:
            self._rejected.append(self._rej(sample, "best_of_n:no_instruction"))
            return None
        candidates = self._sample_candidates(instruction)
        if not candidates:
            self._rejected.append(self._rej(sample, "best_of_n:no_candidates_generated"))
            return None
        source = self._resolve_source(sample)
        scores = self._score_all(instruction, source, candidates)
        return self._build_output(sample, instruction, source, candidates, scores, ts)

    async def _process_one_async(self, sample: DataSample, ts: datetime):
        instruction = sample.instruction
        if not instruction:
            self._rejected.append(self._rej(sample, "best_of_n:no_instruction"))
            return None
        candidates = await self._sample_candidates_async(instruction)
        if not candidates:
            self._rejected.append(self._rej(sample, "best_of_n:no_candidates_generated"))
            return None
        source = self._resolve_source(sample)
        scores = self._score_all(instruction, source, candidates)
        return self._build_output(sample, instruction, source, candidates, scores, ts)

    # ─────────────────────────────────────────────────────────────────────
    # Output selection logic
    # ─────────────────────────────────────────────────────────────────────

    def _build_output(
        self,
        source_sample: DataSample,
        instruction: str,
        source_text: str,
        candidates: list[str],
        scores: list[float],
        ts: datetime,
    ) -> DataSample:
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top = ranked[: self.top_k]

        if self.top_k == 1:
            return self._emit_sft(source_sample, instruction, source_text, top[0], scores, ts)
        if self.top_k == 2:
            return self._emit_dpo(source_sample, instruction, source_text, ranked, scores, ts)
        return self._emit_grpo(source_sample, instruction, source_text, top, scores, ts)

    def _emit_sft(
        self,
        source_sample,
        instruction,
        source_text,
        winner,
        all_scores,
        ts,
    ) -> DataSample:
        text, score = winner
        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source_sample.source_uri or "best_of_n",
            instruction=instruction,
            input=source_text,
            output=text,
            label=score,
            task_type="instruction_following",
            metadata={
                "generator": "BestOfNTask",
                "selection_mode": "top1_sft",
                "n_candidates": self.n_candidates,
                "winner_score": score,
                "score_distribution": all_scores,
                "source_sample_id": source_sample.id,
            },
        )
        self._stamp(sample, ts, "top1_sft")
        return sample

    def _emit_dpo(
        self,
        source_sample,
        instruction,
        source_text,
        ranked,
        all_scores,
        ts,
    ) -> DataSample:
        chosen, chosen_score = ranked[0]
        if self.dpo_mode == "vs_lowest":
            rejected, rejected_score = ranked[-1]
        else:
            rejected, rejected_score = ranked[1]
        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source_sample.source_uri or "best_of_n",
            instruction=instruction,
            input=source_text,
            chosen=chosen,
            rejected=rejected,
            label=chosen_score,
            task_type="preference",
            metadata={
                "generator": "BestOfNTask",
                "selection_mode": f"dpo_{self.dpo_mode}",
                "n_candidates": self.n_candidates,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_distribution": all_scores,
                "source_sample_id": source_sample.id,
            },
        )
        self._stamp(sample, ts, f"dpo_{self.dpo_mode}")
        return sample

    def _emit_grpo(
        self,
        source_sample,
        instruction,
        source_text,
        top,
        all_scores,
        ts,
    ) -> DataSample:
        responses = [t for t, _ in top]
        rewards = [s for _, s in top]
        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source_sample.source_uri or "best_of_n",
            instruction=instruction,
            input=source_text,
            responses=responses,
            reward_scores=rewards,
            task_type="grpo",
            metadata={
                "generator": "BestOfNTask",
                "selection_mode": "top_k_grpo",
                "k": self.top_k,
                "n_candidates": self.n_candidates,
                "score_distribution": all_scores,
                "source_sample_id": source_sample.id,
            },
        )
        self._stamp(sample, ts, "top_k_grpo")
        return sample

    def _stamp(self, sample: DataSample, ts: datetime, mode: str) -> None:
        # Score-call count: 1 per candidate; gen-call count: n_candidates
        total_calls = self.n_candidates + self.n_candidates  # gen + score
        sample.append_provenance(
            ProvenanceRecord(
                step_name="BestOfNTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=self.llm.config_hash(),
                notes={
                    "selection_mode": mode,
                    "n_candidates": self.n_candidates,
                    "top_k": self.top_k,
                    "total_llm_calls_estimate": total_calls,
                },
            )
        )

    def _rej(self, source: DataSample, reason: str) -> RejectedSample:
        return RejectedSample(
            source_uri=source.source_uri or "best_of_n",
            instruction=source.instruction,
            rejection_reason=reason,
            rejecting_step="BestOfNTask",
        )
