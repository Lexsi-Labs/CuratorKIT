"""
BadSampleInjector — controlled hallucination injection for gate evaluation.

Takes a fraction of generated QA samples and replaces their answers with
LLM-generated adversarial variants. Each variant is tagged with:
  metadata["injected_failure"]  = True
  metadata["injection_type"]    = <type string>

The hallucination gate has no access to these tags — it evaluates purely on
(source_text, question, answer). This gives ground-truth labels for gate
precision/recall without human annotation.

Injection types
---------------
contradicts_source
    Answer sounds fluent but directly contradicts a claim in the source.
    Expected gate diagnosis: GENERATOR_PARAMETRIC

parametric_drift
    Answer uses plausible world knowledge that ignores the specific source text.
    Expected gate diagnosis: GENERATOR_PARAMETRIC

cross_chunk_drift
    Answer references information from a *different* passage than the one
    provided (simulated by asking model to answer from general knowledge
    about the domain topic rather than the passage).
    Expected gate diagnosis: SOURCE_CROSS_CHUNK

high_temperature_drift
    Original answer is re-generated at temperature=1.4 to simulate the
    model drifting from the source at high sampling temperature.
    Expected gate diagnosis: GENERATOR_TEMPERATURE
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Literal

from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample

logger = logging.getLogger(__name__)

InjectionType = Literal[
    "contradicts_source",
    "parametric_drift",
    "cross_chunk_drift",
    "high_temperature_drift",
    "domain_mismatch",
    "instruction_quality",
]

ALL_INJECTION_TYPES: list[InjectionType] = [
    "contradicts_source",
    "parametric_drift",
    "cross_chunk_drift",
    "high_temperature_drift",
    "domain_mismatch",
    "instruction_quality",
]

_CONTRADICTS_PROMPT = """\
You are given a source passage and a question. Write a plausible-sounding answer \
that SUBTLY CONTRADICTS or MISREPRESENTS specific facts in the source passage.

Rules:
- The answer must sound fluent and confident
- It must contain at least one factual claim that directly contradicts the source
- It should NOT be obviously nonsensical — it should look like a genuine answer at first glance
- Do not mention that you are contradicting anything

Source passage:
---
{source}
---

Question: {question}

Contradicting answer (fluent but factually wrong relative to the source):"""

_PARAMETRIC_DRIFT_PROMPT = """\
You are given a source passage and a question. Write an answer using your general \
world knowledge about this topic, IGNORING the specific details stated in the source passage.

Rules:
- Answer should sound plausible and factually reasonable in general
- Do NOT ground your answer in the specific content of the passage above
- Use general knowledge about the topic, not the passage text

Source passage (ignore the specific details):
---
{source}
---

Question: {question}

Answer (based on general knowledge, not the passage):"""

_CROSS_CHUNK_DRIFT_PROMPT = """\
You are given a source passage and a question. Write an answer that introduces \
plausible-sounding information from OUTSIDE the given passage — as if you had \
access to a related but different document on the same topic.

Rules:
- The answer should be coherent and topically related
- It must reference specific details that are NOT present in the passage above
- Do not copy phrases directly from the passage

Source passage:
---
{source}
---

Question: {question}

Answer (referencing information from beyond the given passage):"""


class BadSampleInjector:
    """
    Inject controlled failure samples into a generated QA corpus.

    Parameters
    ----------
    llm : BaseLLM
        LLM used to generate adversarial answers.
    injection_rate : float
        Fraction of samples to replace with injected failures (0–1).
    injection_types : list[InjectionType]
        Which failure types to inject. Types are sampled uniformly.
    seed : int | None
        Random seed for reproducible injection selection.
    """

    def __init__(
        self,
        llm: BaseLLM,
        injection_rate: float = 0.20,
        injection_types: list[InjectionType] | None = None,
        seed: int | None = 42,
    ) -> None:
        self.llm = llm
        self.injection_rate = max(0.0, min(1.0, injection_rate))
        self.injection_types = injection_types or ALL_INJECTION_TYPES
        self._rng = random.Random(seed)

    def inject(self, samples: list[DataSample]) -> list[DataSample]:
        """
        Replace injection_rate fraction of samples with adversarial variants.

        Returns a new list (original list is not mutated). Injected samples
        are shuffled back into the list at their original positions so the
        gate cannot exploit ordering.

        The returned list has metadata["injected_failure"]=True on injected
        samples. All other samples are unchanged.
        """
        if not samples or self.injection_rate == 0.0:
            return list(samples)

        n_inject = max(1, round(len(samples) * self.injection_rate))
        inject_indices = set(self._rng.sample(range(len(samples)), min(n_inject, len(samples))))

        result: list[DataSample] = []
        for i, sample in enumerate(samples):
            if i not in inject_indices:
                result.append(sample)
                continue

            inj_type: InjectionType = self._rng.choice(self.injection_types)
            injected = self._inject_one(sample, inj_type)
            result.append(injected if injected is not None else sample)

        return result

    def inject_by_type(
        self,
        samples: list[DataSample],
        counts: dict[InjectionType, int],
    ) -> list[DataSample]:
        """
        Inject exact counts of each failure type (useful for controlled ablations).

        samples are sampled without replacement per type.
        """
        available = list(range(len(samples)))
        self._rng.shuffle(available)
        used: set[int] = set()
        result = list(samples)

        for inj_type, count in counts.items():
            candidates = [i for i in available if i not in used]
            chosen = candidates[:count]
            for i in chosen:
                injected = self._inject_one(samples[i], inj_type)
                if injected is not None:
                    result[i] = injected
                used.add(i)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Per-sample injection
    # ─────────────────────────────────────────────────────────────────────────

    def _inject_one(self, sample: DataSample, inj_type: InjectionType) -> DataSample | None:
        """Return a new DataSample with an adversarial answer, or None on failure."""
        try:
            if inj_type == "high_temperature_drift":
                return self._inject_high_temperature(sample)
            return self._inject_llm(sample, inj_type)
        except Exception as exc:
            logger.warning("Injection failed for %s (%s): %s", sample.id, inj_type, exc)
            return None

    def _inject_llm(self, sample: DataSample, inj_type: InjectionType) -> DataSample | None:
        source = sample.input.strip() or sample.output.strip()
        question = sample.instruction.strip()

        if not source or not question:
            return None

        if inj_type == "contradicts_source":
            prompt = _CONTRADICTS_PROMPT.format(source=source, question=question)
        elif inj_type == "parametric_drift":
            prompt = _PARAMETRIC_DRIFT_PROMPT.format(source=source, question=question)
        elif inj_type == "cross_chunk_drift":
            prompt = _CROSS_CHUNK_DRIFT_PROMPT.format(source=source, question=question)
        else:
            return None

        response = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        bad_answer = response.text.strip()
        if not bad_answer:
            return None

        return self._make_injected(sample, bad_answer, inj_type)

    def _inject_high_temperature(self, sample: DataSample) -> DataSample | None:
        """Re-generate the answer at high temperature to induce drift."""
        source = sample.input.strip() or sample.output.strip()
        question = sample.instruction.strip()
        if not source or not question:
            return None

        from curatorkit.diagnostic.failure_modes import PROMPT_TEMPLATES

        prompt = PROMPT_TEMPLATES["default"].format(source=source, question=question)
        response = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=1.4,
            max_tokens=512,
        )
        bad_answer = response.text.strip()
        if not bad_answer:
            return None

        return self._make_injected(sample, bad_answer, "high_temperature_drift")

    @staticmethod
    def _make_injected(
        original: DataSample, bad_answer: str, inj_type: InjectionType
    ) -> DataSample:
        """Clone sample, swap in the adversarial answer, tag metadata."""
        d = original.model_dump()
        d["id"] = str(uuid.uuid4())
        d["output"] = bad_answer
        d["metadata"] = {
            **d.get("metadata", {}),
            "injected_failure": True,
            "injection_type": inj_type,
            "original_sample_id": original.id,
        }
        return DataSample(**d)


def injection_evaluation_report(
    rejected_ids: set[str],
    all_samples: list[DataSample],
) -> dict:
    """
    Compute gate precision/recall on the injected subset.

    Parameters
    ----------
    rejected_ids : set of sample IDs that the gate rejected
    all_samples  : full list of samples passed to the gate (before filtering)

    Returns
    -------
    dict with keys:
      n_injected          — total injected bad samples
      n_clean             — total clean (non-injected) samples
      injected_caught     — injected samples the gate correctly rejected
      injected_missed     — injected samples the gate wrongly passed
      clean_rejected      — clean samples the gate wrongly rejected (false positives)
      gate_recall         — injected_caught / n_injected
      uninj_rejection_rate            — clean_rejected / n_clean
      by_type             — per injection_type breakdown
    """
    injected = [s for s in all_samples if s.metadata.get("injected_failure")]
    clean = [s for s in all_samples if not s.metadata.get("injected_failure")]

    injected_caught = sum(1 for s in injected if s.id in rejected_ids)
    injected_missed = len(injected) - injected_caught
    clean_rejected = sum(1 for s in clean if s.id in rejected_ids)

    n_inj = len(injected)
    n_clean = len(clean)

    by_type: dict[str, dict] = {}
    for inj_type in ALL_INJECTION_TYPES:
        subset = [s for s in injected if s.metadata.get("injection_type") == inj_type]
        caught = sum(1 for s in subset if s.id in rejected_ids)
        by_type[inj_type] = {
            "total": len(subset),
            "caught": caught,
            "recall": round(caught / len(subset), 4) if subset else None,
        }

    return {
        "n_injected": n_inj,
        "n_clean": n_clean,
        "injected_caught": injected_caught,
        "injected_missed": injected_missed,
        "clean_rejected": clean_rejected,
        "gate_recall": round(injected_caught / n_inj, 4) if n_inj else None,
        "uninj_rejection_rate": round(clean_rejected / n_clean, 4) if n_clean else None,
        "by_type": by_type,
    }
