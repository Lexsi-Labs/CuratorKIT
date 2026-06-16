"""
HallucinationGate — verify generated content is grounded in source text.

Uses an LLM judge to compare generated answers against their source chunks.
The source chunk is retrieved via provenance metadata (page, parent_heading,
source_file) that PDFReader attaches during ingestion.

Samples that fail grounding become RejectedSample with reason
'hallucination_contract_failed:{score}'.

This gate only runs on samples that have source context available
(typically PDF-derived QA pairs). Samples without source context pass through.
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

_DEFAULT_GROUNDING_PROMPT = """You are an expert fact-checker. Evaluate whether the given answer is fully supported by the source text.

Source text:
---
{source_text}
---

Question: {question}
Answer: {answer}

Evaluate the answer on these dimensions:
1. Factual accuracy — Are all specific claims (dates, names, numbers, outcomes) directly supported by the source text?
2. Source grounding — Does every factual claim trace back to the source? Claims from general world knowledge that are NOT in the source must be penalised even if they sound plausible.
3. No contradiction — Does the answer avoid contradicting any fact stated in the source?
4. No drift — The answer must not substitute source-specific facts with different facts from outside the source.

IMPORTANT: If the answer contains facts, numbers, names, or claims that cannot be verified from the source text — even if they are generally plausible — treat those as unsupported and lower the score accordingly. An answer that ignores the source and relies on external knowledge should score below 0.5.

Respond in JSON format ONLY:
{{
  "grounding_score": 0.XX,
  "supported_claims": ["..."],
  "unsupported_claims": ["..."],
  "verdict": "grounded | partially_grounded | hallucinated"
}}

Score guidelines:
  1.0 = all claims directly supported by the source text
  0.7-0.9 = mostly grounded, only minor phrasing not verbatim in source
  0.4-0.6 = some claims supported, others from external knowledge
  0.0-0.3 = answer relies on external knowledge or contradicts the source

/no_think"""


class HallucinationGate(BaseGate):
    """
    Verify generated answers are grounded in their source text.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend for grounding judgement.
    threshold : float
        Minimum grounding score (0-1). Samples below this are rejected.
    prompt_template : str | None
        Custom grounding evaluation prompt.
    skip_if_no_context : bool
        If True, samples without source context pass through.
        If False, samples without source context are rejected.
    """

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        prompt_template: str | None = None,
        skip_if_no_context: bool = True,
        concurrency: int = 16,
    ) -> None:
        self.llm = llm
        self.threshold = threshold
        self.prompt_template = prompt_template or _DEFAULT_GROUNDING_PROMPT
        self.skip_if_no_context = skip_if_no_context
        self.concurrency = concurrency

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "threshold": self.threshold,
                "llm_model": self.llm.model,
                "skip_if_no_context": self.skip_if_no_context,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_source_context(self, sample: DataSample) -> str | None:
        """Extract source text from sample's input or metadata."""
        # QA samples typically store context in input
        if sample.input and len(sample.input) > 50:
            return sample.input

        # Check metadata for source chunk
        if "source_chunk" in sample.metadata:
            return sample.metadata["source_chunk"]

        return None

    def _get_question(self, sample: DataSample) -> str:
        """Get the question/instruction."""
        return sample.instruction

    def _get_answer(self, sample: DataSample) -> str:
        """Get the generated answer to evaluate.

        Routing by task_type:
          preference / implicit_preference → chosen (grounded good answer only;
              rejected is intentionally adversarial and is not grounding-checked)
          grpo  → highest-reward response (best-effort grounding check on rollouts)
          prompt_only → "" (no response generated yet — gate skips)
          all others → output
        """
        if sample.task_type in {"preference", "implicit_preference"} and sample.chosen:
            return sample.chosen
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

    def _run_one(
        self, sample: DataSample, cfg_hash: str, ts
    ) -> tuple[DataSample | None, RejectedSample | None]:
        """Evaluate a single sample. Returns (passed, None) or (None, rejected)."""
        source_context = self._get_source_context(sample)
        if source_context is None:
            if self.skip_if_no_context:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="HallucinationGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"skipped": True, "reason": "no_source_context"},
                    )
                )
                return sample, None
            return None, RejectedSample(
                **sample.model_dump(),
                rejection_reason="hallucination_gate:no_source_context",
                rejecting_step="HallucinationGate",
            )

        question = self._get_question(sample)
        answer = self._get_answer(sample)
        if not answer:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="HallucinationGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={"skipped": True, "reason": "no_answer_to_evaluate"},
                )
            )
            return sample, None

        try:
            score, verdict, details = self._evaluate_grounding(source_context, question, answer)
        except Exception as e:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="HallucinationGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={"error": str(e), "passed_on_error": True},
                )
            )
            return sample, None

        if score >= self.threshold:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="HallucinationGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "grounding_score": score,
                        "verdict": verdict,
                        "threshold": self.threshold,
                        "passed": True,
                    },
                )
            )
            if sample.label is None:
                sample.label = score
            return sample, None
        else:
            rej = RejectedSample(
                **sample.model_dump(),
                rejection_reason=f"hallucination_contract_failed:{score:.2f}",
                rejecting_step="HallucinationGate",
            )
            rej.append_provenance(
                ProvenanceRecord(
                    step_name="HallucinationGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "grounding_score": score,
                        "verdict": verdict,
                        "threshold": self.threshold,
                        "passed": False,
                        **details,
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
                total=len(samples),
                desc="HallucinationGate",
                unit="sample",
                disable=len(samples) <= 1,
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

    def _evaluate_grounding(
        self, source_text: str, question: str, answer: str
    ) -> tuple[float, str, dict]:
        """Call the LLM judge and parse the grounding evaluation."""
        prompt = self.prompt_template.format(
            source_text=source_text,
            question=question,
            answer=answer,
        )

        response = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )

        text = response.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        try:
            parsed = json.loads(text)
            score = float(parsed.get("grounding_score", 0.0))
            score = max(0.0, min(1.0, score))
            verdict = parsed.get("verdict", "unknown")
            details = {
                "supported_claims": parsed.get("supported_claims", []),
                "unsupported_claims": parsed.get("unsupported_claims", []),
            }
            return score, verdict, details
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: try to extract a score from the text
            score_match = re.search(r"(\d+\.?\d*)", text)
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 10.0  # Handle 0-10 scale
                return max(0.0, min(1.0, score)), "parse_fallback", {}

            return 0.5, "parse_error", {"raw_response": text[:200]}

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def _evaluate_grounding_async(
        self, source_text: str, question: str, answer: str
    ) -> tuple[float, str, dict]:
        prompt = self.prompt_template.format(
            source_text=source_text,
            question=question,
            answer=answer,
        )
        response = await self.llm.agenerate(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        text = response.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        try:
            parsed = json.loads(text)
            score = max(0.0, min(1.0, float(parsed.get("grounding_score", 0.0))))
            verdict = parsed.get("verdict", "unknown")
            details = {
                "supported_claims": parsed.get("supported_claims", []),
                "unsupported_claims": parsed.get("unsupported_claims", []),
            }
            return score, verdict, details
        except (json.JSONDecodeError, ValueError, TypeError):
            m = re.search(r"(\d+\.?\d*)", text)
            if m:
                s = float(m.group(1))
                return max(0.0, min(1.0, s / 10.0 if s > 1.0 else s)), "parse_fallback", {}
            return 0.5, "parse_error", {"raw_response": text[:200]}

    async def _run_one_async(
        self, sample: DataSample, cfg_hash: str, ts, semaphore: asyncio.Semaphore
    ) -> tuple[DataSample | None, RejectedSample | None]:
        async with semaphore:
            source_context = self._get_source_context(sample)
            if source_context is None:
                if self.skip_if_no_context:
                    sample.append_provenance(
                        ProvenanceRecord(
                            step_name="HallucinationGate",
                            step_version=STEP_VERSION,
                            timestamp=ts,
                            config_hash=cfg_hash,
                            notes={"skipped": True, "reason": "no_source_context"},
                        )
                    )
                    return sample, None
                return None, RejectedSample(
                    **sample.model_dump(),
                    rejection_reason="hallucination_gate:no_source_context",
                    rejecting_step="HallucinationGate",
                )

            question = self._get_question(sample)
            answer = self._get_answer(sample)
            if not answer:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="HallucinationGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"skipped": True, "reason": "no_answer_to_evaluate"},
                    )
                )
                return sample, None

            try:
                score, verdict, details = await self._evaluate_grounding_async(
                    source_context, question, answer
                )
            except Exception as e:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="HallucinationGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"error": str(e), "passed_on_error": True},
                    )
                )
                return sample, None

            if score >= self.threshold:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="HallucinationGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "grounding_score": score,
                            "verdict": verdict,
                            "threshold": self.threshold,
                            "passed": True,
                        },
                    )
                )
                if sample.label is None:
                    sample.label = score
                return sample, None
            else:
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"hallucination_contract_failed:{score:.2f}",
                    rejecting_step="HallucinationGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="HallucinationGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "grounding_score": score,
                            "verdict": verdict,
                            "threshold": self.threshold,
                            "passed": False,
                            **details,
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
            desc="HallucinationGate",
            unit="sample",
            disable=len(samples) <= 1,
        )

        passed = [p for p, r in results if p is not None]
        rejected = [r for p, r in results if r is not None]
        return passed, rejected
