"""
RewardGate — LLM-as-judge quality filter for generated data.

Scores each sample using an LLM judge following the UltraFeedback rubric
(helpfulness, honesty, instruction-following, truthfulness). Samples
below the threshold are rejected with a structured reason.

The reward score is stored in DataSample.label for downstream use
(e.g., by the StratifiedSampler or for unpaired preference training).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from curatorkit.interfaces import BaseGate
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"

_PREFERENCE_TASK_TYPES = {"preference", "implicit_preference"}

_VALID_DIMENSIONS = {
    "helpfulness",
    "honesty",
    "instruction_following",
    "truthfulness",
    "depth",
    "creativity",
    "coherence",
}

_DEFAULT_REWARD_PROMPT = """You are an expert evaluator. Rate the quality of the following response on a scale from 0.0 to 1.0.

Instruction: {instruction}

Response: {response}

Evaluate on these dimensions:
{dimensions_text}

Respond in JSON format ONLY:
{{
  "overall_score": 0.XX,
  "dimension_scores": {{
    {dimensions_json}
  }},
  "strengths": "brief note",
  "weaknesses": "brief note"
}}

Scoring guide:
  0.9-1.0: Excellent — thorough, accurate, well-structured
  0.7-0.8: Good — mostly complete, minor issues
  0.5-0.6: Adequate — addresses the question but with gaps
  0.3-0.4: Poor — significant issues in accuracy or completeness
  0.0-0.2: Very poor — largely unhelpful or incorrect

/no_think"""

_DIMENSION_DESCRIPTIONS = {
    "helpfulness": "How useful is this response for the person asking?",
    "honesty": "Does the response acknowledge uncertainty when appropriate?",
    "instruction_following": "Does the response address all parts of the instruction?",
    "truthfulness": "Are all claims factually accurate (to the best of your knowledge)?",
    "depth": "Does the response provide sufficient detail and explanation?",
    "creativity": "Is the response original and insightful?",
    "coherence": "Is the response well-organized and easy to follow?",
}


class RewardGate(BaseGate):
    """
    Quality-score samples using an LLM judge and reject below threshold.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend for quality judgement.
    threshold : float
        Minimum quality score (0-1). Samples below this are rejected.
    dimensions : list[str]
        Quality dimensions to evaluate. Defaults to core UltraFeedback set.
    prompt_template : str | None
        Custom reward evaluation prompt.
    store_score_in_label : bool
        If True, store the overall score in DataSample.label.
    """

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        dimensions: list[str] | None = None,
        prompt_template: str | None = None,
        store_score_in_label: bool = True,
        concurrency: int = 16,
    ) -> None:
        self.llm = llm
        self.threshold = threshold
        self.dimensions = dimensions or ["helpfulness", "honesty", "instruction_following"]
        self.prompt_template = prompt_template
        self.store_score_in_label = store_score_in_label
        self.concurrency = concurrency

        # Validate dimensions
        for dim in self.dimensions:
            if dim not in _VALID_DIMENSIONS:
                raise ValueError(f"Unknown dimension '{dim}'. Valid: {sorted(_VALID_DIMENSIONS)}")

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "threshold": self.threshold,
                "dimensions": sorted(self.dimensions),
                "llm_model": self.llm.model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _is_preference_pair(self, sample: DataSample) -> bool:
        """True if this sample is a DPO preference pair with both sides populated."""
        return (
            sample.task_type in _PREFERENCE_TASK_TYPES
            and bool(sample.chosen)
            and bool(sample.rejected)
        )

    def _get_response_text(self, sample: DataSample) -> str:
        """Get the response text to evaluate for non-preference samples.

        Routing by task_type:
          grpo       → highest-reward response (or responses[0] if unscored)
          prompt_only → "" — no response generated yet, gate skips
          all others → output, then chosen, then ""
        """
        if sample.task_type == "grpo":
            if not sample.responses:
                return ""
            if sample.reward_scores and len(sample.reward_scores) == len(sample.responses):
                best = sample.reward_scores.index(max(sample.reward_scores))
                return sample.responses[best]
            return sample.responses[0]
        if sample.output:
            return sample.output
        if sample.chosen:
            return sample.chosen
        return ""

    def _build_reward_prompt(self, instruction: str, response: str) -> str:
        """Build the quality evaluation prompt."""
        if self.prompt_template:
            return self.prompt_template.format(
                instruction=instruction,
                response=response,
            )

        dimensions_text = "\n".join(
            f"- {dim}: {_DIMENSION_DESCRIPTIONS.get(dim, dim)}" for dim in self.dimensions
        )
        dimensions_json = ",\n    ".join(f'"{dim}": 0.XX' for dim in self.dimensions)

        return _DEFAULT_REWARD_PROMPT.format(
            instruction=instruction,
            response=response,
            dimensions_text=dimensions_text,
            dimensions_json=dimensions_json,
        )

    def _run_one(
        self, sample: DataSample, cfg_hash: str, ts
    ) -> tuple[DataSample | None, RejectedSample | None]:
        if self._is_preference_pair(sample):
            return self._run_one_preference(sample, cfg_hash, ts)

        response_text = self._get_response_text(sample)
        instruction = sample.instruction

        if not response_text:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={"skipped": True, "reason": "no_response_to_evaluate"},
                )
            )
            return sample, None

        try:
            score, dim_scores, details = self._evaluate_quality(instruction, response_text)
        except Exception as e:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={"error": str(e), "passed_on_error": True},
                )
            )
            return sample, None

        if score >= self.threshold:
            if self.store_score_in_label:
                sample.label = score
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "reward_score": score,
                        "dimension_scores": dim_scores,
                        "threshold": self.threshold,
                        "passed": True,
                    },
                )
            )
            return sample, None
        else:
            rej = RejectedSample(
                **sample.model_dump(),
                rejection_reason=f"below_reward_threshold:{score:.2f}",
                rejecting_step="RewardGate",
            )
            rej.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "reward_score": score,
                        "dimension_scores": dim_scores,
                        "threshold": self.threshold,
                        "passed": False,
                        **details,
                    },
                )
            )
            return None, rej

    def _run_one_preference(
        self, sample: DataSample, cfg_hash: str, ts
    ) -> tuple[DataSample | None, RejectedSample | None]:
        """
        Dual-score a DPO preference pair.

        A pair passes only when:
          chosen_score  >= threshold  (good answer meets quality bar)
          rejected_score <  threshold  (bad answer falls below quality bar)

        If rejected_score >= threshold, the pair is rejected because the
        adversarial/negative response is not sufficiently worse than chosen
        (insufficient quality contrast for DPO training).
        """
        instruction = sample.instruction
        try:
            chosen_score, chosen_dims, _ = self._evaluate_quality(instruction, sample.chosen)
            rejected_score, rejected_dims, _ = self._evaluate_quality(instruction, sample.rejected)
        except Exception as e:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={"error": str(e), "passed_on_error": True},
                )
            )
            return sample, None

        chosen_ok = chosen_score >= self.threshold
        rejected_ok = rejected_score < self.threshold

        if chosen_ok and rejected_ok:
            if self.store_score_in_label:
                sample.label = chosen_score
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "chosen_dims": chosen_dims,
                        "rejected_dims": rejected_dims,
                        "threshold": self.threshold,
                        "passed": True,
                    },
                )
            )
            return sample, None
        else:
            reason = (
                f"chosen_below_threshold:{chosen_score:.2f}"
                if not chosen_ok
                else f"rejected_above_threshold:{rejected_score:.2f}"
            )
            rej = RejectedSample(
                **sample.model_dump(),
                rejection_reason=f"dpo_pair_failed:{reason}",
                rejecting_step="RewardGate",
            )
            rej.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "chosen_dims": chosen_dims,
                        "rejected_dims": rejected_dims,
                        "threshold": self.threshold,
                        "passed": False,
                        "rejection_reason": reason,
                    },
                )
            )
            return None, rej

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)
        order = {s.id: i for i, s in enumerate(samples)}
        results: dict[int, tuple] = {}
        lock = threading.Lock()

        # Submit futures in a sliding window (concurrency*4 max in-flight) so
        # we never pre-allocate O(N) Future objects for very large batches.
        window = max(self.concurrency * 4, 128)
        chunk_size = min(window, len(samples))

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            with tqdm(
                total=len(samples), desc="RewardGate", unit="sample", disable=len(samples) <= 1
            ) as pbar:
                n_passed = n_rejected = 0
                for chunk_start in range(0, len(samples), chunk_size):
                    chunk = samples[chunk_start : chunk_start + chunk_size]
                    futures = {
                        pool.submit(self._run_one, s, cfg_hash, ts): order[s.id] for s in chunk
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        p, r = future.result()
                        with lock:
                            results[idx] = (p, r)
                        if p is not None:
                            n_passed += 1
                        else:
                            n_rejected += 1
                        pbar.set_postfix(passed=n_passed, rejected=n_rejected)
                        pbar.update(1)

        for i in range(len(samples)):
            p, r = results[i]
            if p is not None:
                passed.append(p)
            if r is not None:
                rejected.append(r)

        return passed, rejected

    def _evaluate_quality(
        self, instruction: str, response: str
    ) -> tuple[float, dict[str, float], dict]:
        """Call the LLM judge and parse the quality evaluation."""
        prompt = self._build_reward_prompt(instruction, response)

        resp = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )

        text = resp.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        try:
            parsed = json.loads(text)
            overall = float(parsed.get("overall_score", 0.0))
            overall = max(0.0, min(1.0, overall))

            dim_scores = {}
            raw_dims = parsed.get("dimension_scores", {})
            for dim in self.dimensions:
                if dim in raw_dims:
                    dim_scores[dim] = max(0.0, min(1.0, float(raw_dims[dim])))

            details = {
                "strengths": parsed.get("strengths", ""),
                "weaknesses": parsed.get("weaknesses", ""),
            }

            return overall, dim_scores, details

        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: extract a number
            score_match = re.search(r"(\d+\.?\d*)", text)
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 10.0
                return max(0.0, min(1.0, score)), {}, {"parse_error": True}

            return 0.5, {}, {"parse_error": True, "raw_response": text[:200]}

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def _evaluate_quality_async(
        self, instruction: str, response: str
    ) -> tuple[float, dict[str, float], dict]:
        prompt = self._build_reward_prompt(instruction, response)
        resp = await self.llm.agenerate(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        text = resp.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        try:
            parsed = json.loads(text)
            overall = max(0.0, min(1.0, float(parsed.get("overall_score", 0.0))))
            dim_scores = {
                dim: max(0.0, min(1.0, float(parsed.get("dimension_scores", {}).get(dim, 0.0))))
                for dim in self.dimensions
                if dim in parsed.get("dimension_scores", {})
            }
            details = {
                "strengths": parsed.get("strengths", ""),
                "weaknesses": parsed.get("weaknesses", ""),
            }
            return overall, dim_scores, details
        except (json.JSONDecodeError, ValueError, TypeError):
            m = re.search(r"(\d+\.?\d*)", text)
            if m:
                s = float(m.group(1))
                return max(0.0, min(1.0, s / 10.0 if s > 1.0 else s)), {}, {"parse_error": True}
            return 0.5, {}, {"parse_error": True, "raw_response": text[:200]}

    async def _run_one_async(
        self, sample: DataSample, cfg_hash: str, ts, semaphore: asyncio.Semaphore
    ) -> tuple[DataSample | None, RejectedSample | None]:
        async with semaphore:
            if self._is_preference_pair(sample):
                return await self._run_one_preference_async(sample, cfg_hash, ts)

            response_text = self._get_response_text(sample)
            instruction = sample.instruction

            if not response_text:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="RewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"skipped": True, "reason": "no_response_to_evaluate"},
                    )
                )
                return sample, None

            try:
                score, dim_scores, details = await self._evaluate_quality_async(
                    instruction, response_text
                )
            except Exception as e:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="RewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"error": str(e), "passed_on_error": True},
                    )
                )
                return sample, None

            if score >= self.threshold:
                if self.store_score_in_label:
                    sample.label = score
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="RewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "reward_score": score,
                            "dimension_scores": dim_scores,
                            "threshold": self.threshold,
                            "passed": True,
                        },
                    )
                )
                return sample, None
            else:
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"below_reward_threshold:{score:.2f}",
                    rejecting_step="RewardGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="RewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "reward_score": score,
                            "dimension_scores": dim_scores,
                            "threshold": self.threshold,
                            "passed": False,
                            **details,
                        },
                    )
                )
                return None, rej

    async def _run_one_preference_async(
        self, sample: DataSample, cfg_hash: str, ts
    ) -> tuple[DataSample | None, RejectedSample | None]:
        """Async dual-scoring for DPO preference pairs (runs both evals concurrently)."""
        instruction = sample.instruction
        try:
            (
                (chosen_score, chosen_dims, _),
                (rejected_score, rejected_dims, _),
            ) = await asyncio.gather(
                self._evaluate_quality_async(instruction, sample.chosen),
                self._evaluate_quality_async(instruction, sample.rejected),
            )
        except Exception as e:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={"error": str(e), "passed_on_error": True},
                )
            )
            return sample, None

        chosen_ok = chosen_score >= self.threshold
        rejected_ok = rejected_score < self.threshold

        if chosen_ok and rejected_ok:
            if self.store_score_in_label:
                sample.label = chosen_score
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "chosen_dims": chosen_dims,
                        "rejected_dims": rejected_dims,
                        "threshold": self.threshold,
                        "passed": True,
                    },
                )
            )
            return sample, None
        else:
            reason = (
                f"chosen_below_threshold:{chosen_score:.2f}"
                if not chosen_ok
                else f"rejected_above_threshold:{rejected_score:.2f}"
            )
            rej = RejectedSample(
                **sample.model_dump(),
                rejection_reason=f"dpo_pair_failed:{reason}",
                rejecting_step="RewardGate",
            )
            rej.append_provenance(
                ProvenanceRecord(
                    step_name="RewardGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "chosen_dims": chosen_dims,
                        "rejected_dims": rejected_dims,
                        "threshold": self.threshold,
                        "passed": False,
                        "rejection_reason": reason,
                    },
                )
            )
            return None, rej

    async def run_async(
        self, samples: list[DataSample]
    ) -> tuple[list[DataSample], list[RejectedSample]]:
        """Async execution — uses agenerate() with semaphore-bounded concurrency."""
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self.concurrency)

        results = await atqdm.gather(
            *[self._run_one_async(s, cfg_hash, ts, semaphore) for s in samples],
            desc="RewardGate",
            unit="sample",
            disable=len(samples) <= 1,
        )

        passed = [p for p, r in results if p is not None]
        rejected = [r for p, r in results if r is not None]
        return passed, rejected
