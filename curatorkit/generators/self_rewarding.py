"""
SelfRewardingTask — model generates its own preference pairs (Yuan et al., ICML 2024).

Reference: "Self-Rewarding Language Models" — Yuan, Pang, et al. (ICML 2024)
  Paper: https://arxiv.org/abs/2401.10020

Canonical algorithm (per the paper's Figure 2 and Algorithm 1):

  For each input prompt:
    1. Generate N candidate responses by sampling (N=4 in the paper).
    2. Score each candidate with the model itself acting as judge, using
       an additive 5-point rubric. Average over K=3 sampled evaluations
       for stability.
    3. Pair the highest-scoring response with the lowest as
       (chosen, rejected). Discard if scores tie.

The result is a `preference` DataSample per input prompt — directly
consumable by DPOExporter. Across iterations the trained model improves
both its generation and its judgment, but that outer loop is the user's
responsibility (they retrain between Curator runs).

Cost: N + N*K calls per input prompt with default config = 4 + 12 = 16.
The cost gateway will surface this as `generation_self_rewarding` in the
per-stage attribution table.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import UTC, datetime

from curatorkit.generators.base import STEP_VERSION, BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

# ───────────────────────────────────────────────────────────────────────────
# Canonical 5-point additive rubric (paper Figure 2)
# ───────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {instruction}

<response>{response}</response>

After examining the user's instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Score: <total points>"

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria."""


_SCORE_RE = re.compile(r"Score\s*:\s*(\d+)", re.IGNORECASE)


def _parse_score(text: str) -> int:
    """Extract the integer 0..5 from a judge response. Defaults to 0 on parse failure."""
    m = _SCORE_RE.search(text or "")
    if not m:
        # Fallback: trailing integer
        m2 = re.search(r"(\d+)\s*$", (text or "").strip())
        if not m2:
            return 0
        try:
            v = int(m2.group(1))
        except ValueError:
            return 0
    else:
        try:
            v = int(m.group(1))
        except ValueError:
            return 0
    return max(0, min(5, v))


# ───────────────────────────────────────────────────────────────────────────
# SelfRewardingTask
# ───────────────────────────────────────────────────────────────────────────


class SelfRewardingTask(BaseGenerationTask):
    """
    One input prompt -> one DPO preference pair from the model's self-judgment.

    Parameters
    ----------
    llm : BaseLLM
        The model under self-reward. Both generation and judgment go through
        this backend; a separate `judge_llm` can be provided to keep them
        decoupled (useful for ablations).
    judge_llm : BaseLLM | None
        Separate judge backend. Defaults to `llm`.
    n_candidates : int
        Candidate responses per prompt. Paper uses 4.
    k_judge_samples : int
        Number of judge evaluations averaged per candidate. Paper uses 3.
    generation_temperature : float
        Sampling temperature for candidate generation. Paper uses 0.7+.
    judge_temperature : float
        Sampling temperature for the judge (kept low for stability).
    min_score_delta : int
        Minimum integer score gap between chosen and rejected to keep the
        pair. Default 1 — the paper discards ties.
    """

    def __init__(
        self,
        llm: BaseLLM,
        judge_llm: BaseLLM | None = None,
        n_candidates: int = 4,
        k_judge_samples: int = 3,
        generation_temperature: float = 0.7,
        judge_temperature: float = 0.3,
        min_score_delta: int = 1,
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=None, concurrency=concurrency)
        self.judge_llm = judge_llm or llm
        self.n_candidates = max(2, int(n_candidates))
        self.k_judge_samples = max(1, int(k_judge_samples))
        self.generation_temperature = float(generation_temperature)
        self.judge_temperature = float(judge_temperature)
        self.min_score_delta = max(1, int(min_score_delta))

    # _build_messages / _parse_response satisfy the ABC; we override run().
    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        return [{"role": "user", "content": sample.instruction}]

    def _parse_response(self, sample, response):
        return []

    # ─────────────────────────────────────────────────────────────────────
    # Single-sample workflow
    # ─────────────────────────────────────────────────────────────────────

    def _generate_candidates(self, instruction: str) -> list[tuple[str, LLMResponse | None]]:
        out: list[tuple[str, LLMResponse | None]] = []
        for _ in range(self.n_candidates):
            try:
                r = self.llm.generate(
                    [{"role": "user", "content": instruction}],
                    temperature=self.generation_temperature,
                )
                out.append((r.text.strip(), r))
            except Exception:
                continue
        return out

    async def _generate_candidates_async(self, instruction: str):
        async def one():
            try:
                r = await self.llm.agenerate(
                    [{"role": "user", "content": instruction}],
                    temperature=self.generation_temperature,
                )
                return (r.text.strip(), r)
            except Exception:
                return None

        results = await asyncio.gather(*[one() for _ in range(self.n_candidates)])
        return [r for r in results if r is not None]

    def _score_candidate(self, instruction: str, response: str) -> tuple[int, list[int]]:
        """Run k_judge_samples evaluations and return (avg_rounded, raw_scores)."""
        scores: list[int] = []
        prompt = _JUDGE_PROMPT.format(instruction=instruction, response=response)
        for _ in range(self.k_judge_samples):
            try:
                r = self.judge_llm.generate(
                    [{"role": "user", "content": prompt}],
                    temperature=self.judge_temperature,
                    max_tokens=512,
                )
                scores.append(_parse_score(r.text))
            except Exception:
                scores.append(0)
        avg = round(sum(scores) / max(1, len(scores)))
        return avg, scores

    async def _score_candidate_async(
        self,
        instruction: str,
        response: str,
    ) -> tuple[int, list[int]]:
        async def one():
            prompt = _JUDGE_PROMPT.format(instruction=instruction, response=response)
            try:
                r = await self.judge_llm.agenerate(
                    [{"role": "user", "content": prompt}],
                    temperature=self.judge_temperature,
                    max_tokens=512,
                )
                return _parse_score(r.text)
            except Exception:
                return 0

        scores = await asyncio.gather(*[one() for _ in range(self.k_judge_samples)])
        avg = round(sum(scores) / max(1, len(scores)))
        return avg, list(scores)

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline interface — overridden so we can run N+N*K calls per sample
    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)

        for sample in samples:
            result = self._process_one(sample, ts)
            if result is not None:
                results.append(result)
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

    # ─────────────────────────────────────────────────────────────────────

    def _process_one(self, sample: DataSample, ts: datetime) -> DataSample | None:
        instruction = sample.instruction
        if not instruction:
            self._rejected.append(self._rej(sample, "self_rewarding:no_instruction"))
            return None

        candidates = self._generate_candidates(instruction)
        if len(candidates) < 2:
            self._rejected.append(
                self._rej(
                    sample,
                    "self_rewarding:insufficient_candidates",
                )
            )
            return None

        scored: list[tuple[str, int, list[int]]] = []
        for text, _resp in candidates:
            avg, raw = self._score_candidate(instruction, text)
            scored.append((text, avg, raw))

        return self._build_preference_pair(sample, instruction, scored, ts)

    async def _process_one_async(
        self,
        sample: DataSample,
        ts: datetime,
    ) -> DataSample | None:
        instruction = sample.instruction
        if not instruction:
            self._rejected.append(self._rej(sample, "self_rewarding:no_instruction"))
            return None

        candidates = await self._generate_candidates_async(instruction)
        if len(candidates) < 2:
            self._rejected.append(
                self._rej(
                    sample,
                    "self_rewarding:insufficient_candidates",
                )
            )
            return None

        scored: list[tuple[str, int, list[int]]] = []
        for text, _resp in candidates:
            avg, raw = await self._score_candidate_async(instruction, text)
            scored.append((text, avg, raw))

        return self._build_preference_pair(sample, instruction, scored, ts)

    # ─────────────────────────────────────────────────────────────────────

    def _build_preference_pair(
        self,
        source: DataSample,
        instruction: str,
        scored: list[tuple[str, int, list[int]]],
        ts: datetime,
    ) -> DataSample | None:
        # Sort by score (asc) so [-1] is highest, [0] is lowest
        scored.sort(key=lambda x: x[1])
        lowest_text, lowest_score, lowest_raw = scored[0]
        highest_text, highest_score, highest_raw = scored[-1]

        if highest_score - lowest_score < self.min_score_delta:
            self._rejected.append(
                self._rej(
                    source,
                    f"self_rewarding:score_tie:{highest_score}={lowest_score}",
                )
            )
            return None

        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source.source_uri or "self_rewarding",
            instruction=instruction,
            chosen=highest_text,
            rejected=lowest_text,
            label=highest_score / 5.0,
            task_type="preference",
            metadata={
                "generator": "SelfRewardingTask",
                "source_sample_id": source.id,
                "chosen_score": highest_score,
                "rejected_score": lowest_score,
                "chosen_raw_scores": highest_raw,
                "rejected_raw_scores": lowest_raw,
                "n_candidates": self.n_candidates,
                "k_judge_samples": self.k_judge_samples,
            },
        )
        sample.append_provenance(
            ProvenanceRecord(
                step_name="SelfRewardingTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=self.llm.config_hash(),
                notes={
                    "chosen_score": highest_score,
                    "rejected_score": lowest_score,
                    "score_delta": highest_score - lowest_score,
                    "total_llm_calls": (
                        self.n_candidates + self.n_candidates * self.k_judge_samples
                    ),
                },
            )
        )
        return sample

    def _rej(self, source: DataSample, reason: str) -> RejectedSample:
        return RejectedSample(
            source_uri=source.source_uri or "self_rewarding",
            instruction=source.instruction,
            rejection_reason=reason,
            rejecting_step="SelfRewardingTask",
        )
