"""
ToxicityGate — two-stage toxicity filter.

Stage 1 (local classifier — fast, deterministic):
  Detoxify classifier scores each text field 0.0–1.0 across dimensions:
    toxicity, severe_toxicity, obscene, identity_attack, insult, threat,
    sexual_explicit (model=unbiased, the default).
  Decision by max score across all relevant fields:
    max_score < classifier_pass_threshold   → pass  (no LLM call)
    max_score >= classifier_reject_threshold → reject (no LLM call)
    in between                               → escalate to Stage 2

Stage 2 (LLM judge, borderline only):
  Fires only for samples in the borderline band. The LLM provides a
  calibrated score plus a reasoning string written to the provenance chain.
  If llm=None, borderline samples are rejected at classifier_pass_threshold.

Design notes:
  - DPO/preference data: chosen and rejected fields are scored independently;
    max score across fields drives the decision. The rejected completion may be
    deliberately edgy — we still want to catch outright toxic content.
  - GRPO data: each response is scored independently.
  - Use detoxify_model="unbiased" (default) over "original" for legal/medical
    corpora — the unbiased model suppresses false positives on legitimate
    discussion of sensitive topics (crime, medication, disease).
  - Classifier thresholds [0.1, 0.5] may need tuning per corpus: academic
    and legal text scores 0.1–0.25 on toxicity even when clean. Raise
    classifier_pass_threshold to ~0.2 if you see excessive LLM escalations.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

if TYPE_CHECKING:
    from curatorkit.llm.base import BaseLLM

STEP_VERSION = "1.0.0"

_LLM_JUDGE_PROMPT = """\
You are a content safety evaluator for LLM fine-tuning datasets.

Assess whether the text below contains harmful, toxic, or unsafe content
that should be excluded from a training corpus.

Rate toxicity 0.0–1.0:
  0.0–0.15  Safe — no harmful content, including frank academic/legal/medical discussion
  0.15–0.35 Marginal — edgy or controversial but not harmful; likely fine for training
  0.35–0.60 Moderate — content some would find harmful; review recommended
  0.60–0.80 High — clearly harmful, offensive, or dangerous
  0.80–1.00 Severe — extremely harmful, illegal, or actively dangerous

Text:
---
{text}
---

Respond with JSON only (no markdown fences):
{{"toxicity_score": <float>, "categories": [<strings>], "reasoning": "<one sentence>"}}

Valid categories: toxicity, severe_toxicity, obscene, identity_attack, insult, threat, sexual_explicit, safe
/no_think"""


def _ensure_detoxify() -> Any:
    try:
        from detoxify import Detoxify

        return Detoxify
    except ImportError as e:
        raise ImportError(
            "detoxify is not installed. Install with: pip install 'curatorkit[hygiene]'"
        ) from e


class ToxicityGate(BaseGate):
    """
    Two-stage toxicity filter.

    Stage 1: Detoxify classifier (fast, deterministic, no API cost).
    Stage 2: LLM judge for borderline samples only (optional).

    Parameters
    ----------
    classifier_pass_threshold : float
        Max toxicity score below which a sample passes without LLM escalation.
        Default 0.1; raise to ~0.2 for academic/legal/medical corpora.
    classifier_reject_threshold : float
        Max toxicity score at or above which a sample is rejected without LLM.
        Default 0.5. If llm=None, the band [pass_threshold, reject_threshold)
        is also rejected (no escalation path).
    llm : BaseLLM | None
        LLM backend for borderline samples. If None, samples in the borderline
        band are rejected at classifier_pass_threshold.
    llm_reject_threshold : float
        LLM toxicity score at or above which a borderline sample is rejected.
    detoxify_model : str
        "unbiased" (default, recommended), "original", or "multilingual".
    text_field : str
        "auto" scores the fields most relevant to the task_type; or pass
        a specific DataSample field name.
    """

    def __init__(
        self,
        classifier_pass_threshold: float = 0.1,
        classifier_reject_threshold: float = 0.5,
        llm: BaseLLM | None = None,
        llm_reject_threshold: float = 0.5,
        detoxify_model: str = "unbiased",
        text_field: str = "auto",
    ) -> None:
        if classifier_pass_threshold >= classifier_reject_threshold:
            raise ValueError(
                "classifier_pass_threshold must be strictly less than classifier_reject_threshold"
            )
        self.classifier_pass_threshold = classifier_pass_threshold
        self.classifier_reject_threshold = classifier_reject_threshold
        self.llm = llm
        self.llm_reject_threshold = llm_reject_threshold
        self.detoxify_model = detoxify_model
        self.text_field = text_field
        self._classifier: Any = None

    def _load_classifier(self) -> Any:
        if self._classifier is None:
            Detoxify = _ensure_detoxify()
            self._classifier = Detoxify(self.detoxify_model)
        return self._classifier

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "classifier_pass_threshold": self.classifier_pass_threshold,
                "classifier_reject_threshold": self.classifier_reject_threshold,
                "llm_reject_threshold": self.llm_reject_threshold,
                "detoxify_model": self.detoxify_model,
                "text_field": self.text_field,
                "has_llm": self.llm is not None,
                "llm_hash": self.llm.config_hash() if self.llm is not None else None,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _fields_to_score(self, sample: DataSample) -> list[tuple[str, str]]:
        """
        Return (field_name, text) pairs to score for this sample.
        For multi-field task types each field is scored independently
        so one toxic completion in a preference pair triggers rejection.
        """
        if self.text_field != "auto":
            text = getattr(sample, self.text_field, "") or ""
            return [(self.text_field, text)] if text.strip() else []

        task = sample.task_type
        if task == "language_modeling":
            return [("output", sample.output)] if sample.output.strip() else []

        if task in ("preference", "implicit_preference"):
            pairs = []
            if sample.chosen.strip():
                pairs.append(("chosen", sample.chosen))
            if sample.rejected.strip():
                pairs.append(("rejected", sample.rejected))
            return pairs

        if task == "grpo":
            return [(f"response_{i}", r) for i, r in enumerate(sample.responses) if r.strip()]

        # instruction_following, conversational, prompt_only, source_chunk, etc.
        pairs = []
        if sample.output.strip():
            pairs.append(("output", sample.output))
        if sample.instruction.strip():
            pairs.append(("instruction", sample.instruction))
        return pairs

    def _score_field(self, text: str) -> dict[str, float]:
        """Run Detoxify on a single text field. Truncates to 2 048 chars."""
        classifier = self._load_classifier()
        raw = classifier.predict(text[:2048])
        return {k: round(float(v), 4) for k, v in raw.items()}

    def _llm_verdict(self, text: str) -> tuple[float, str, list[str]]:
        """Call LLM judge. Returns (score, reasoning, categories)."""
        assert self.llm is not None
        prompt = _LLM_JUDGE_PROMPT.format(text=text[:1500])
        try:
            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            m = re.search(r"\{[^{}]+\}", response.text, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return (
                    float(data.get("toxicity_score", 1.0)),
                    str(data.get("reasoning", "")),
                    list(data.get("categories", [])),
                )
        except Exception:
            pass
        return 1.0, "LLM judge parse failed; defaulting to reject", []

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        if not samples:
            return [], []

        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []

        from tqdm import tqdm

        for sample in tqdm(samples, desc="ToxicityGate", unit="sample"):
            field_pairs = self._fields_to_score(sample)

            if not field_pairs:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="ToxicityGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"passed": True, "skipped": True, "reason": "no_text"},
                    )
                )
                passed.append(sample)
                continue

            # Score each field independently; track worst-case field
            field_scores: dict[str, dict[str, float]] = {}
            worst_field = ""
            worst_max = 0.0

            for field_name, text in field_pairs:
                scores = self._score_field(text)
                field_max = max(scores.values())
                field_scores[field_name] = scores
                if field_max > worst_max:
                    worst_max = field_max
                    worst_field = field_name

            # Stage 1: clearly safe
            if worst_max < self.classifier_pass_threshold:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="ToxicityGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": True,
                            "phase": "classifier",
                            "max_toxicity_score": worst_max,
                            "worst_field": worst_field,
                            "field_scores": field_scores,
                        },
                    )
                )
                passed.append(sample)
                continue

            # Stage 1: clearly toxic or no LLM available
            if worst_max >= self.classifier_reject_threshold or self.llm is None:
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"toxic_content:classifier:{worst_max:.3f}",
                    rejecting_step="ToxicityGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="ToxicityGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": False,
                            "phase": "classifier",
                            "max_toxicity_score": worst_max,
                            "worst_field": worst_field,
                            "field_scores": field_scores,
                        },
                    )
                )
                rejected.append(rej)
                continue

            # Stage 2: borderline — escalate to LLM judge on the worst field
            worst_text = next(text for fname, text in field_pairs if fname == worst_field)
            llm_score, reasoning, categories = self._llm_verdict(worst_text)

            if llm_score >= self.llm_reject_threshold:
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"toxic_content:llm_judge:{llm_score:.3f}",
                    rejecting_step="ToxicityGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="ToxicityGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": False,
                            "phase": "llm_judge",
                            "classifier_max_score": worst_max,
                            "worst_field": worst_field,
                            "llm_score": llm_score,
                            "categories": categories,
                            "reasoning": reasoning,
                        },
                    )
                )
                rejected.append(rej)
            else:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="ToxicityGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": True,
                            "phase": "llm_judge",
                            "classifier_max_score": worst_max,
                            "worst_field": worst_field,
                            "llm_score": llm_score,
                            "categories": categories,
                            "reasoning": reasoning,
                        },
                    )
                )
                passed.append(sample)

        return passed, rejected
