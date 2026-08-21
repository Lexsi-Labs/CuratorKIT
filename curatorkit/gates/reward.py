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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from curatorkit.gates._score_parsing import extract_score, template_mentions_key
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
        self._fallback_count = 0
        self._scored_count = 0

        # Validate dimensions
        for dim in self.dimensions:
            if dim not in _VALID_DIMENSIONS:
                raise ValueError(f"Unknown dimension '{dim}'. Valid: {sorted(_VALID_DIMENSIONS)}")

        # Static check, before any LLM calls: does the custom template even
        # ask for the key this gate parses out of the judge's response? If
        # not, every sample will fail to produce a real score regardless of
        # how good the judge's actual answers are — warn now instead of only
        # discovering it after the whole pipeline has run.
        if not template_mentions_key(prompt_template, "overall_score"):
            warnings.warn(
                "reward_prompt_template does not mention 'overall_score' — RewardGate parses "
                "that exact key from the judge's JSON response to decide pass/fail. Without it, "
                "scores will be recovered by averaging your requested dimensions or extracting a "
                "number from the raw response, which may reject far more samples than expected. "
                'Add `"overall_score": 0.XX` to your template\'s expected output JSON.',
                UserWarning,
                stacklevel=2,
            )

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

        self._warn_if_fallback_heavy()
        return passed, rejected

    def _warn_if_fallback_heavy(self) -> None:
        """One aggregated warning if scores had to be recovered via fallback
        parsing instead of the documented `overall_score` key — mirrors the
        SFT exporters' empty-row warning rather than warning per-sample."""
        if self._fallback_count == 0 or self._scored_count == 0:
            return
        warnings.warn(
            f"RewardGate: {self._fallback_count}/{self._scored_count} judge responses had no "
            "'overall_score' and were scored via dimension-averaging or raw-text extraction "
            "instead — check that your reward_prompt_template's expected output JSON includes "
            '`"overall_score": 0.XX`.',
            UserWarning,
            stacklevel=2,
        )

    def _evaluate_quality(
        self, instruction: str, response: str
    ) -> tuple[float, dict[str, float], dict]:
        """Call the LLM judge and parse the quality evaluation."""
        prompt = self._build_reward_prompt(instruction, response)

        resp = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )

        text = resp.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        overall, used_fallback = extract_score(
            parsed, text, "overall_score", dimension_keys=tuple(self.dimensions)
        )
        self._scored_count += 1
        if used_fallback:
            self._fallback_count += 1

        dim_scores = {}
        if isinstance(parsed, dict):
            raw_dims = parsed.get("dimension_scores", parsed)
            for dim in self.dimensions:
                if dim in raw_dims:
                    try:
                        dim_scores[dim] = max(0.0, min(1.0, float(raw_dims[dim])))
                    except (TypeError, ValueError):
                        continue

        details = {
            "strengths": parsed.get("strengths", "") if isinstance(parsed, dict) else "",
            "weaknesses": parsed.get("weaknesses", "") if isinstance(parsed, dict) else "",
        }
        if used_fallback:
            details["parse_error"] = True

        return overall, dim_scores, details

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def _evaluate_quality_async(
        self, instruction: str, response: str
    ) -> tuple[float, dict[str, float], dict]:
        prompt = self._build_reward_prompt(instruction, response)
        resp = await self.llm.agenerate(
            [{"role": "user", "content": prompt}],
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )
        text = resp.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        overall, used_fallback = extract_score(
            parsed, text, "overall_score", dimension_keys=tuple(self.dimensions)
        )
        self._scored_count += 1
        if used_fallback:
            self._fallback_count += 1

        dim_scores = {}
        if isinstance(parsed, dict):
            raw_dims = parsed.get("dimension_scores", parsed)
            for dim in self.dimensions:
                if dim in raw_dims:
                    try:
                        dim_scores[dim] = max(0.0, min(1.0, float(raw_dims[dim])))
                    except (TypeError, ValueError):
                        continue

        details = {
            "strengths": parsed.get("strengths", "") if isinstance(parsed, dict) else "",
            "weaknesses": parsed.get("weaknesses", "") if isinstance(parsed, dict) else "",
        }
        if used_fallback:
            details["parse_error"] = True

        return overall, dim_scores, details

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
        self._warn_if_fallback_heavy()
        return passed, rejected
