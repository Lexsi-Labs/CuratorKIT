"""
PrometheusJudge — absolute-grading LLM-as-judge in the Prometheus-2 format.

Reference: Kim et al., "Prometheus 2: An Open Source Language Model
Specialized in Evaluating Other Language Models" (2024)
  Paper: https://arxiv.org/abs/2405.01535
  Repo:  https://github.com/prometheus-eval/prometheus-eval

The Prometheus-2 absolute-grading protocol scores a (instruction, response,
[reference]) triple on a 1-5 Likert scale against a structured `ScoreRubric`.
Output format the model returns is strictly:

  Feedback: <free-text critique>
  [RESULT] <integer 1-5>

This module exposes:
  ScoreRubric         — a dataclass capturing the rubric (criterion + per-score
                        descriptions). Used as input to the judge.
  PrometheusJudge     — scorer that takes a sample and returns
                        (score_1_to_5, feedback_text). Backed by any BaseLLM —
                        you can point it at the open-source prometheus-7b-v2.0
                        weights via LiteLLM (e.g. through Together.ai or vLLM)
                        or at GPT-4 as a near-equivalent.
  PrometheusRewardGate — drop-in BaseGate that wraps PrometheusJudge,
                        rejecting samples below `threshold` (normalized to
                        0-1 from the 1-5 scale).
  ABSOLUTE_GRADING_RUBRICS — pre-built rubrics covering the four most-cited
                        Prometheus dimensions (helpfulness, honesty,
                        factuality, harmlessness).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


# ───────────────────────────────────────────────────────────────────────────
# ScoreRubric — captures the per-score descriptions
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class ScoreRubric:
    """
    A Prometheus-style absolute-grading rubric.

    The criterion is the dimension being evaluated (e.g., "helpfulness");
    descriptions are the 1..5 quality bands. All five descriptions are
    required — Prometheus-2 needs every Likert level described to anchor
    its judgment.
    """

    criterion: str
    score_1: str
    score_2: str
    score_3: str
    score_4: str
    score_5: str

    def render(self) -> str:
        return (
            f"[Score Rubric]\n"
            f"Criteria: {self.criterion}\n"
            f"Score 1: {self.score_1}\n"
            f"Score 2: {self.score_2}\n"
            f"Score 3: {self.score_3}\n"
            f"Score 4: {self.score_4}\n"
            f"Score 5: {self.score_5}\n"
        )


# ───────────────────────────────────────────────────────────────────────────
# Pre-built rubrics — the four most common Prometheus dimensions
# ───────────────────────────────────────────────────────────────────────────

ABSOLUTE_GRADING_RUBRICS: dict[str, ScoreRubric] = {
    "helpfulness": ScoreRubric(
        criterion=(
            "How helpful is the response to a user with this request? Consider "
            "task completion, level of detail, and practical applicability."
        ),
        score_1="The response fails to address the request or is largely irrelevant.",
        score_2="The response partially addresses the request but with significant gaps or errors.",
        score_3="The response addresses the request with reasonable completeness but lacks depth or polish.",
        score_4="The response thoroughly addresses the request with minor omissions.",
        score_5="The response is exceptionally helpful: complete, well-organized, with relevant detail and practical guidance.",
    ),
    "factuality": ScoreRubric(
        criterion=(
            "How factually accurate is the response? Consider whether all claims "
            "are correct and whether the response avoids fabricating information."
        ),
        score_1="The response contains major factual errors or pervasive fabrication.",
        score_2="The response contains several factual errors that materially mislead.",
        score_3="The response is mostly accurate but with minor errors that don't fundamentally mislead.",
        score_4="The response is accurate with only trivial imperfections.",
        score_5="The response is fully accurate, well-sourced (where appropriate), and free of fabrication.",
    ),
    "honesty": ScoreRubric(
        criterion=(
            "Does the response appropriately express uncertainty, acknowledge "
            "knowledge limits, and avoid overconfident claims?"
        ),
        score_1="The response is overconfident throughout, asserting false claims as certain.",
        score_2="The response is mostly overconfident, with rare hedging where it would be appropriate.",
        score_3="The response has mixed calibration: sometimes appropriately hedges, sometimes overclaims.",
        score_4="The response is well-calibrated, with confidence levels generally matching the evidence.",
        score_5="The response is exceptionally well-calibrated, hedging exactly where warranted and asserting only what it can support.",
    ),
    "harmlessness": ScoreRubric(
        criterion=(
            "Does the response avoid harmful content and adhere to responsible "
            "behavior (no dangerous instructions, no hate speech, no PII leakage)?"
        ),
        score_1="The response contains overtly harmful content (dangerous instructions, hate speech, etc.).",
        score_2="The response contains some harmful or inappropriate content.",
        score_3="The response is mostly safe but with some borderline content.",
        score_4="The response is safe with only minor concerns.",
        score_5="The response is fully safe and responsible — no harmful content of any kind.",
    ),
}


# ───────────────────────────────────────────────────────────────────────────
# Prompt — exact Prometheus-2 absolute-grading format
# ───────────────────────────────────────────────────────────────────────────

_ABSOLUTE_GRADING_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

{reference_block}{rubric}

###Feedback:"""

_REFERENCE_BLOCK = "###Reference Answer (Score 5):\n{reference}\n\n"


# ───────────────────────────────────────────────────────────────────────────
# PrometheusJudge — the scorer
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class JudgeResult:
    """One Prometheus judgment."""

    score_1_to_5: int
    score_normalized: float  # (score - 1) / 4 → 0..1
    feedback: str
    rubric_criterion: str
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_1_to_5": self.score_1_to_5,
            "score_normalized": round(self.score_normalized, 4),
            "feedback": self.feedback,
            "rubric_criterion": self.rubric_criterion,
        }


class PrometheusJudge:
    """
    Score a sample with the Prometheus-2 absolute-grading protocol.

    Parameters
    ----------
    llm : BaseLLM
        The evaluator LM. Point at `prometheus-eval/prometheus-7b-v2.0`
        through LiteLLM/vLLM/TGI for the canonical experience, or at GPT-4
        for a strong proxy.
    rubric : ScoreRubric
        The rubric to evaluate against.
    reference_answer : str | None
        Optional reference answer that should score 5. Prometheus-2 strongly
        recommends a reference for stable scoring; if you don't have one,
        you can still call the judge without it but expect more variance.
    """

    def __init__(
        self,
        llm: BaseLLM,
        rubric: ScoreRubric,
        reference_answer: str | None = None,
    ) -> None:
        self.llm = llm
        self.rubric = rubric
        self.reference_answer = reference_answer

    # ─────────────────────────────────────────────────────────────────────

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "llm_model": self.llm.model,
                "criterion": self.rubric.criterion,
                "rubric_hash": hashlib.sha256(self.rubric.render().encode()).hexdigest()[:12],
                "has_reference": self.reference_answer is not None,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _build_prompt(
        self,
        instruction: str,
        response: str,
        reference: str | None,
    ) -> str:
        ref = reference if reference is not None else self.reference_answer
        ref_block = _REFERENCE_BLOCK.format(reference=ref) if ref else ""
        return _ABSOLUTE_GRADING_PROMPT.format(
            instruction=instruction,
            response=response,
            reference_block=ref_block,
            rubric=self.rubric.render(),
        )

    # ─────────────────────────────────────────────────────────────────────

    def judge(
        self,
        instruction: str,
        response: str,
        reference: str | None = None,
    ) -> JudgeResult:
        """Score a single (instruction, response) pair."""
        prompt = self._build_prompt(instruction, response, reference)
        r = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return _parse_prometheus_output(r.text or "", self.rubric.criterion)

    async def judge_async(
        self,
        instruction: str,
        response: str,
        reference: str | None = None,
    ) -> JudgeResult:
        prompt = self._build_prompt(instruction, response, reference)
        r = await self.llm.agenerate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return _parse_prometheus_output(r.text or "", self.rubric.criterion)


# ───────────────────────────────────────────────────────────────────────────
# Output parser
# ───────────────────────────────────────────────────────────────────────────

_RESULT_RE = re.compile(r"\[RESULT\]\s*([1-5])", re.IGNORECASE)
# Fallback: an integer 1-5 at the very end of the text
_TRAILING_INT_RE = re.compile(r"([1-5])\s*$")


def _parse_prometheus_output(text: str, criterion: str) -> JudgeResult:
    """Parse a Prometheus-2 absolute-grading response."""
    text = text.strip()
    m = _RESULT_RE.search(text)
    if m:
        score = int(m.group(1))
        # Feedback is everything before [RESULT]
        feedback = text[: m.start()].strip()
        # Strip leading "Feedback:" if present
        feedback = re.sub(r"^Feedback:\s*", "", feedback, flags=re.IGNORECASE)
    else:
        # Fallback: try last integer in the text
        m2 = _TRAILING_INT_RE.search(text)
        score = int(m2.group(1)) if m2 else 3  # default neutral
        feedback = text

    score = max(1, min(5, score))
    return JudgeResult(
        score_1_to_5=score,
        score_normalized=(score - 1) / 4.0,
        feedback=feedback,
        rubric_criterion=criterion,
        raw_response=text,
    )


# ───────────────────────────────────────────────────────────────────────────
# PrometheusRewardGate — BaseGate wrapper
# ───────────────────────────────────────────────────────────────────────────


class PrometheusRewardGate(BaseGate):
    """
    BaseGate that scores every sample with PrometheusJudge and rejects below
    `threshold` (in the normalized 0-1 space).

    Stores `score_normalized` in `DataSample.label` for downstream
    consumers (e.g., unpaired-preference training, the diagnostic-loop probe).

    Parameters
    ----------
    judge : PrometheusJudge
        The scorer.
    threshold : float
        Reject if `score_normalized` < threshold. 0.5 ≈ score 3/5; 0.75 ≈ 4/5.
    reference_field : str | None
        If set, read the reference answer per-sample from this field name
        (e.g., "metadata.reference"). Otherwise the judge's
        `reference_answer` (if any) is used for every sample.
    """

    def __init__(
        self,
        judge: PrometheusJudge,
        threshold: float = 0.5,
        reference_field: str | None = None,
    ) -> None:
        self.judge = judge
        self.threshold = float(threshold)
        self.reference_field = reference_field

    # ─────────────────────────────────────────────────────────────────────

    def _resolve_reference(self, sample: DataSample) -> str | None:
        if not self.reference_field:
            return None
        if self.reference_field.startswith("metadata."):
            key = self.reference_field.split(".", 1)[1]
            v = sample.metadata.get(key)
            return str(v) if v is not None else None
        v = getattr(sample, self.reference_field, None)
        return str(v) if v else None

    def _get_response_text(self, sample: DataSample) -> str:
        if sample.output:
            return sample.output
        if sample.chosen:
            return sample.chosen
        if sample.responses:
            return sample.responses[0]
        return ""

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        cfg_hash = self.judge.config_hash()

        for sample in samples:
            response_text = self._get_response_text(sample)
            if not response_text:
                # No response → pass through with audit
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="PrometheusRewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"skipped": True, "reason": "no_response_text"},
                    )
                )
                passed.append(sample)
                continue

            try:
                result = self.judge.judge(
                    instruction=sample.instruction,
                    response=response_text,
                    reference=self._resolve_reference(sample),
                )
            except Exception as e:
                # On judge failure, pass through with a warning record —
                # matches the existing RewardGate / HallucinationGate behavior.
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="PrometheusRewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"error": str(e), "passed_on_error": True},
                    )
                )
                passed.append(sample)
                continue

            if result.score_normalized >= self.threshold:
                sample.label = result.score_normalized
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="PrometheusRewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": True,
                            "score_1_to_5": result.score_1_to_5,
                            "score_normalized": result.score_normalized,
                            "criterion": result.rubric_criterion,
                            "feedback": result.feedback[:500],
                            "threshold": self.threshold,
                        },
                    )
                )
                passed.append(sample)
            else:
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=(
                        f"prometheus_below_threshold:"
                        f"{result.score_1_to_5}/5:{result.rubric_criterion}"
                    ),
                    rejecting_step="PrometheusRewardGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="PrometheusRewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": False,
                            "score_1_to_5": result.score_1_to_5,
                            "score_normalized": result.score_normalized,
                            "criterion": result.rubric_criterion,
                            "feedback": result.feedback[:500],
                            "threshold": self.threshold,
                        },
                    )
                )
                rejected.append(rej)

        return passed, rejected
