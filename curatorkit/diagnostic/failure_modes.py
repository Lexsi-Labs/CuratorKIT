"""
Failure mode taxonomy for the CuratorKIT diagnostic loop.

FailureMode classifies WHY a sample was rejected by a quality gate
(hallucination, reward, or diversity), so rejections become actionable:
each mode maps to a concrete fix (lower the temperature, tighten the
grounding prompt, regenerate the question, ...) instead of an opaque drop.

Recovery is INLINE: DiagnosticProbe attempts each probe path and stores the
passing sample in FailureDiagnosis.recovered_sample if any path succeeds.
There is no separate pass 2 — the pipeline routes recovered samples forward
immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from curatorkit.schema import DataSample


class FailureMode(str, Enum):
    # ── HallucinationGate failure causes ─────────────────────────────────────
    SOURCE_AMBIGUOUS = "source_ambiguous"  # source/response relationship unclear
    GENERATOR_TEMPERATURE = "generator_temperature"  # high temperature causes source drift
    GENERATOR_PARAMETRIC = "generator_parametric"  # model ignores source, uses prior knowledge
    THRESHOLD_MARGINAL = "threshold_marginal"  # score just below threshold, unstable

    # ── RewardGate failure causes ─────────────────────────────────────────────
    INSTRUCTION_QUALITY = "instruction_quality"  # generated question is poor quality
    RESPONSE_QUALITY = "response_quality"  # generated answer is poor quality
    DOMAIN_MISMATCH = "domain_mismatch"  # generation prompt wrong for this domain

    # ── DiversityGate failure cause ───────────────────────────────────────────
    NEAR_DUPLICATE = "near_duplicate"  # too similar to an already-accepted sample

    # ── Fallback ──────────────────────────────────────────────────────────────
    UNKNOWN = "unknown"  # probe inconclusive


# Prompt templates used by DiagnosticProbe for inline regeneration.
# "strict_grounding" → parametric-drift probe (force model to use only source text)
# "domain_specific"  → domain-mismatch probe
# "generate_question"→ instruction-quality probe (regenerate the question)
# "default"          → fallback regeneration template
PROMPT_TEMPLATES: dict[str, str] = {
    "strict_grounding": (
        "You are answering a question. You MUST answer using ONLY information "
        "explicitly stated in the source text below. Do not use any external "
        "knowledge or make inferences beyond what the text directly states.\n\n"
        "Source text:\n---\n{source}\n---\n\n"
        "Question: {question}\n\n"
        "Answer strictly from the source text:"
    ),
    "domain_specific": (
        "Source text:\n---\n{source}\n---\n\n"
        "Based on the source text above, answer the following question "
        "with precise factual detail.\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "generate_question": (
        "Read the source text below and write ONE clear, specific question "
        "that can be answered using ONLY the information in the text. "
        "The question must be self-contained and end with a question mark.\n\n"
        "Source text:\n---\n{source}\n---\n\n"
        "Question:"
    ),
    "default": ("Source text:\n---\n{source}\n---\n\nQuestion: {question}\n\nAnswer:"),
}


@dataclass
class FailureDiagnosis:
    """
    Result of DiagnosticProbe.diagnose(). Attached to RejectedSample.diagnosis.

    Fields
    ------
    mode             : FailureMode — the diagnosed cause
    evidence         : list[bool] — Probe 1 pass/fail pattern [T=0.3, T=0.5]
    probe_calls      : int — total LLM calls consumed by all probes
    notes            : dict — extra info (e.g. which prompt variant succeeded)
    recovered_sample : DataSample | None — the passing re-generation from the probe,
                       if any probe path succeeded. Pipeline routes this back into
                       the accepted pool inline. None means all probes were exhausted.

    Typical uses
    ------------
    mode + evidence aggregate into per-mode rejection breakdowns
    (see PipelineDiagnostics and diagnostic_summary.json); probe_calls
    tracks the LLM budget the probe consumed, so recovery yield can be
    cost-normalised; recovered_sample is not None marks an actual inline
    recovery.
    """

    mode: FailureMode
    evidence: list[bool] = field(default_factory=list)
    probe_calls: int = 0
    notes: dict[str, Any] = field(default_factory=dict)
    recovered_sample: DataSample | None = field(default=None, repr=False)

    @property
    def was_recovered(self) -> bool:
        """True when the probe produced an inline passing re-generation."""
        return self.recovered_sample is not None

    @classmethod
    def from_mode(
        cls,
        mode: FailureMode,
        evidence: list[bool],
        probe_calls: int,
        notes: dict[str, Any] | None = None,
        recovered_sample: DataSample | None = None,
    ) -> FailureDiagnosis:
        return cls(
            mode=mode,
            evidence=evidence,
            probe_calls=probe_calls,
            notes=notes or {},
            recovered_sample=recovered_sample,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict for rejected.jsonl output."""
        return {
            "mode": self.mode.value,
            "was_recovered": self.was_recovered,
            "evidence": self.evidence,
            "probe_calls": self.probe_calls,
            "notes": self.notes,
        }
