"""
DeepEvalJudge — G-Eval-style LLM-as-judge adapter.

Reference: Confident AI's DeepEval framework — https://docs.confident-ai.com/
Original G-Eval: Liu et al., "G-Eval: NLG Evaluation using GPT-4 with
Better Human Alignment" (NAACL 2023) — https://arxiv.org/abs/2303.16634

Differences from Prometheus-2:
  Prometheus → fixed 1-5 Likert, requires a structured per-score rubric
              and (recommended) a reference answer.
  G-Eval     → continuous 0-10 score derived from chain-of-thought
              evaluation steps the model generates on the fly. No
              per-score rubric required; the criterion + evaluation
              steps drive scoring. Output is "Score: N" or "N/10".

The CuratorKIT DeepEvalJudge:
  - Asks the model to generate evaluation_steps from `criteria`
    (if `evaluation_steps` not provided)
  - Then scores the (instruction, response) using those steps
  - Normalizes the 0-10 score to 0-1 for use with the gate machinery

This module is a self-contained re-implementation of the G-Eval scoring
loop — it does NOT import the `deepeval` package (the package is mostly a
pytest-style runner, not a callable judge). The result is the same
G-Eval algorithm, callable from anywhere a BaseLLM is.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


# ───────────────────────────────────────────────────────────────────────────
# Prompts
# ───────────────────────────────────────────────────────────────────────────

_EVALUATION_STEPS_PROMPT = """Given the evaluation criterion below, write a numbered list of 3-5 specific evaluation steps a judge should follow to score a response on this criterion.

Criterion: {criterion}

Respond with ONLY the numbered list, one step per line. No preamble."""


_SCORING_PROMPT = """You are evaluating an LLM response against a specific criterion.

Criterion: {criterion}

Evaluation steps (follow these in order):
{evaluation_steps}

Instruction:
{instruction}

Response to evaluate:
{response}
{context_block}
Score the response on a scale from 0 to 10 where 10 is best. Give a brief
one-line rationale, then output the score.

Format your reply EXACTLY as:
Rationale: <one-line rationale>
Score: <integer 0-10>"""


_CONTEXT_BLOCK = """
Source context (the response should be grounded in this):
{context}
"""


# Regex for the score line — accepts "Score: 7", "Score: 7/10", "7/10".
_SCORE_RE = re.compile(
    r"(?:score\s*:\s*)?(\d+(?:\.\d+)?)\s*(?:/\s*10)?",
    re.IGNORECASE,
)


# ───────────────────────────────────────────────────────────────────────────
# DeepEvalJudge
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class GEvalResult:
    """One G-Eval-style judgment."""

    score_0_to_10: float
    score_normalized: float  # / 10.0
    rationale: str
    criterion: str
    evaluation_steps: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_0_to_10": round(self.score_0_to_10, 3),
            "score_normalized": round(self.score_normalized, 4),
            "rationale": self.rationale,
            "criterion": self.criterion,
            "evaluation_steps": self.evaluation_steps,
        }


class DeepEvalJudge:
    """
    G-Eval style continuous scorer.

    Parameters
    ----------
    llm : BaseLLM
        Evaluator LM. Stronger models give better-calibrated G-Eval scores
        (the paper uses GPT-4). Weaker models work but with more variance.
    criterion : str
        The evaluation criterion description. e.g., "factual correctness
        with respect to the source context".
    evaluation_steps : list[str] | None
        Pre-canned evaluation steps. If None, the judge generates them
        from `criterion` on first use and caches them — this is the
        G-Eval paper's auto-CoT approach.
    use_context : bool
        If True, the judge expects each sample to carry a source context
        (read from `sample.input` or `sample.metadata["source_chunk"]`)
        and includes it in the scoring prompt. Default False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        criterion: str,
        evaluation_steps: list[str] | None = None,
        use_context: bool = False,
    ) -> None:
        self.llm = llm
        self.criterion = criterion
        self._evaluation_steps: list[str] = list(evaluation_steps or [])
        self.use_context = bool(use_context)

    # ─────────────────────────────────────────────────────────────────────

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "llm_model": self.llm.model,
                "criterion": self.criterion,
                "evaluation_steps": self._evaluation_steps,
                "use_context": self.use_context,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def evaluation_steps(self) -> list[str]:
        """Return the cached evaluation steps. Generates them on first call."""
        if not self._evaluation_steps:
            try:
                r = self.llm.generate(
                    [
                        {
                            "role": "user",
                            "content": _EVALUATION_STEPS_PROMPT.format(criterion=self.criterion),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                self._evaluation_steps = self._parse_steps(r.text or "")
            except Exception:
                # Hard fallback so the judge still scores something
                self._evaluation_steps = [
                    f"Check whether the response addresses '{self.criterion}'.",
                    "Identify omissions or errors with respect to the criterion.",
                    "Weigh the response's strengths and weaknesses.",
                    "Assign an integer score from 0 to 10.",
                ]
        return list(self._evaluation_steps)

    async def evaluation_steps_async(self) -> list[str]:
        if not self._evaluation_steps:
            try:
                r = await self.llm.agenerate(
                    [
                        {
                            "role": "user",
                            "content": _EVALUATION_STEPS_PROMPT.format(criterion=self.criterion),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                self._evaluation_steps = self._parse_steps(r.text or "")
            except Exception:
                self._evaluation_steps = [
                    f"Check whether the response addresses '{self.criterion}'.",
                    "Identify omissions or errors with respect to the criterion.",
                    "Weigh the response's strengths and weaknesses.",
                    "Assign an integer score from 0 to 10.",
                ]
        return list(self._evaluation_steps)

    # ─────────────────────────────────────────────────────────────────────

    def _parse_steps(self, text: str) -> list[str]:
        """Parse numbered evaluation steps from the LM's reply."""
        steps: list[str] = []
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            # Strip leading "1.", "1)", "-" markers
            cleaned = re.sub(r"^(?:\d+[\.\)]|\-|\*)\s*", "", raw)
            if cleaned:
                steps.append(cleaned)
        if not steps:
            steps = [text.strip()]
        return steps[:8]  # cap so the scoring prompt stays compact

    def _build_scoring_prompt(
        self,
        instruction: str,
        response: str,
        context: str | None,
    ) -> str:
        steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.evaluation_steps()))
        ctx_block = _CONTEXT_BLOCK.format(context=context) if (self.use_context and context) else ""
        return _SCORING_PROMPT.format(
            criterion=self.criterion,
            evaluation_steps=steps,
            instruction=instruction,
            response=response,
            context_block=ctx_block,
        )

    # ─────────────────────────────────────────────────────────────────────

    def judge(
        self,
        instruction: str,
        response: str,
        context: str | None = None,
    ) -> GEvalResult:
        prompt = self._build_scoring_prompt(instruction, response, context)
        r = self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return _parse_geval_output(
            r.text or "",
            self.criterion,
            self._evaluation_steps,
        )

    async def judge_async(
        self,
        instruction: str,
        response: str,
        context: str | None = None,
    ) -> GEvalResult:
        # Ensure steps are populated (may need an LLM call)
        await self.evaluation_steps_async()
        prompt = self._build_scoring_prompt(instruction, response, context)
        r = await self.llm.agenerate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return _parse_geval_output(
            r.text or "",
            self.criterion,
            self._evaluation_steps,
        )


# ───────────────────────────────────────────────────────────────────────────
# Parser
# ───────────────────────────────────────────────────────────────────────────


def _parse_geval_output(
    text: str,
    criterion: str,
    steps: list[str],
) -> GEvalResult:
    text = text.strip()

    rationale = ""
    m_rat = re.search(r"Rationale:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
    if m_rat:
        rationale = m_rat.group(1).strip()

    # Pick the LAST score-like number in the text — G-Eval models often
    # discuss low/high before stating the final score.
    score = 5.0  # neutral default
    matches = list(_SCORE_RE.finditer(text))
    if matches:
        try:
            score = float(matches[-1].group(1))
        except ValueError:
            score = 5.0
    score = max(0.0, min(10.0, score))

    return GEvalResult(
        score_0_to_10=score,
        score_normalized=score / 10.0,
        rationale=rationale,
        criterion=criterion,
        evaluation_steps=list(steps),
        raw_response=text,
    )


# ───────────────────────────────────────────────────────────────────────────
# DeepEvalRewardGate
# ───────────────────────────────────────────────────────────────────────────


class DeepEvalRewardGate(BaseGate):
    """
    BaseGate that scores every sample with DeepEvalJudge and rejects below
    `threshold` (normalized 0-1).

    Parameters
    ----------
    judge : DeepEvalJudge
        The G-Eval scorer.
    threshold : float
        Reject if `score_normalized` < threshold. Default 0.6 ≈ score 6/10.
    context_field : str | None
        Where to find per-sample context. None → no context (default).
        "input" reads sample.input. "metadata.source_chunk" reads
        sample.metadata["source_chunk"]. Only used when the judge's
        `use_context` is True.
    """

    def __init__(
        self,
        judge: DeepEvalJudge,
        threshold: float = 0.6,
        context_field: str | None = None,
    ) -> None:
        self.judge = judge
        self.threshold = float(threshold)
        self.context_field = context_field

    # ─────────────────────────────────────────────────────────────────────

    def _resolve_context(self, sample: DataSample) -> str | None:
        if not self.context_field:
            return None
        if self.context_field.startswith("metadata."):
            key = self.context_field.split(".", 1)[1]
            v = sample.metadata.get(key)
            return str(v) if v is not None else None
        v = getattr(sample, self.context_field, None)
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
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="DeepEvalRewardGate",
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
                    context=self._resolve_context(sample),
                )
            except Exception as e:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="DeepEvalRewardGate",
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
                        step_name="DeepEvalRewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": True,
                            "score_0_to_10": result.score_0_to_10,
                            "score_normalized": result.score_normalized,
                            "criterion": result.criterion,
                            "rationale": result.rationale[:500],
                            "threshold": self.threshold,
                        },
                    )
                )
                passed.append(sample)
            else:
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=(f"deepeval_below_threshold:{result.score_0_to_10:.1f}/10"),
                    rejecting_step="DeepEvalRewardGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="DeepEvalRewardGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": False,
                            "score_0_to_10": result.score_0_to_10,
                            "score_normalized": result.score_normalized,
                            "criterion": result.criterion,
                            "rationale": result.rationale[:500],
                            "threshold": self.threshold,
                        },
                    )
                )
                rejected.append(rej)

        return passed, rejected
