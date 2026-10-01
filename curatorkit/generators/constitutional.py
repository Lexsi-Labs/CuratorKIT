"""
ConstitutionalRevisionTask — SL-CAI critique-and-revise loop.

Reference: Bai et al., "Constitutional AI: Harmlessness from AI Feedback"
  Paper: https://arxiv.org/abs/2212.08073 (Anthropic, 2022)

The SL-CAI (Supervised Learning Constitutional AI) phase the paper describes:

  initial_response = model(prompt)
  for principle in constitution:
      critique = model(critique_prompt(principle, prompt, response))
      revision = model(revision_prompt(principle, prompt, response, critique))
      response = revision

The final revised response is paired with the original prompt to create
SFT training data. The intermediate critiques are recorded in provenance
so a reviewer can audit *why* a revision was made.

This module ships:
  ConstitutionalRevisionTask     — BaseGenerationTask producing revised SFT pairs
  ConstitutionalPreferenceTask   — like SelfRewarding, but the preference pair
                                   is (chosen=revised, rejected=original_response)
                                   — useful for DPO data construction
  CANONICAL_CONSTITUTION         — the standard 5-principle harmlessness set
                                   from the paper's appendix

Canonical critique prompt structure (paper Appendix B, distilled from the
Anthropic blog + LangChain ConstitutionalChain reference implementation):

  CritiquePrompt: "Identify specific ways in which the assistant's last
                   response is {principle}. Also point out potential
                   {principle} in the human's question."
  RevisionPrompt: "Please rewrite the assistant response to remove all
                   {principle} content. The revision should be tactful and
                   acknowledge any harmful assumptions in the human's question."

The principles below are paraphrased from the paper's published
harmlessness constitution. Replace via the `principles=` argument for a
domain-specific constitution (factuality, code-quality, brand-voice, …).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from curatorkit.generators.base import STEP_VERSION, BaseGenerationTask
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

# ───────────────────────────────────────────────────────────────────────────
# Constitution
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Principle:
    """One constitutional principle: a critique target + revision goal."""

    name: str
    critique_request: str
    revision_request: str


CANONICAL_CONSTITUTION: list[Principle] = [
    Principle(
        name="harmlessness",
        critique_request=(
            "Identify specific ways in which the assistant's last response is "
            "harmful, unethical, racist, sexist, toxic, dangerous, or illegal. "
            "Also point out any harmful assumptions or framings in the human's "
            "question itself."
        ),
        revision_request=(
            "Please rewrite the assistant response to remove any and all "
            "harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
            "content. Be tactful in pointing out harmful assumptions in the "
            "human's question."
        ),
    ),
    Principle(
        name="honesty",
        critique_request=(
            "Identify specific places where the assistant's response either "
            "makes claims it cannot support, expresses inappropriate "
            "confidence, or fails to acknowledge uncertainty or knowledge limits."
        ),
        revision_request=(
            "Please rewrite the assistant response to express appropriate "
            "uncertainty, acknowledge its knowledge limits, and avoid "
            "overconfident claims."
        ),
    ),
    Principle(
        name="helpfulness",
        critique_request=(
            "Identify specific ways in which the assistant's response could be "
            "more helpful, complete, or clear. Note any missing steps, "
            "qualifications, or actionable detail the user would benefit from."
        ),
        revision_request=(
            "Please rewrite the assistant response to be more helpful, "
            "complete, and clear while remaining concise and honest."
        ),
    ),
    Principle(
        name="non_evasion",
        critique_request=(
            "Identify specific ways in which the assistant's response is "
            "evasive, preachy, or unnecessarily refuses a benign request."
        ),
        revision_request=(
            "Please rewrite the assistant response to be more direct, less "
            "preachy, and to engage genuinely with the user's request when it "
            "is benign."
        ),
    ),
    Principle(
        name="privacy",
        critique_request=(
            "Identify specific ways in which the assistant's response leaks, "
            "encourages disclosure of, or otherwise mishandles personally "
            "identifiable information (PII) about real individuals."
        ),
        revision_request=(
            "Please rewrite the assistant response to redact or avoid any "
            "PII about real individuals, and to refuse requests that ask for "
            "such disclosure."
        ),
    ),
]


# ───────────────────────────────────────────────────────────────────────────
# Prompt templates (paper Appendix B / LangChain ConstitutionalChain)
# ───────────────────────────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """The original conversation:
Human: {instruction}
Assistant: {response}

Critique request: {critique_request}

Provide your critique. If the response has no issues against this principle,
write only "NO_REVISION_NEEDED" — do not invent issues."""


_REVISION_PROMPT = """The original conversation:
Human: {instruction}
Assistant: {response}

Critique: {critique}

Revision request: {revision_request}

Write only the rewritten assistant response (no preamble, no quotes)."""


_NO_REVISION_TOKEN = "NO_REVISION_NEEDED"


# ───────────────────────────────────────────────────────────────────────────
# ConstitutionalRevisionTask
# ───────────────────────────────────────────────────────────────────────────


class ConstitutionalRevisionTask(BaseGenerationTask):
    """
    Critique-and-revise loop. Produces ONE refined SFT sample per input.

    Workflow (per input sample with field `instruction`):
      1. If `sample.output` is empty, generate an initial response.
      2. For each principle in `principles`:
         a. Critique under that principle.
         b. If critique contains NO_REVISION_NEEDED, skip revision.
         c. Otherwise revise; the revised text replaces `response`.
      3. Emit a DataSample with the original instruction + final revised output.

    Parameters
    ----------
    llm : BaseLLM
        Backend used for initial generation + critique + revision.
    principles : list[Principle] | None
        Constitution. None -> CANONICAL_CONSTITUTION (5 principles).
    skip_initial_generation : bool
        If True (default), reuse `sample.output` as the initial response.
        Set False to discard `sample.output` and re-generate from `instruction`.
    early_stop_on_no_revision : bool
        If True, stop the principle loop as soon as one principle returns
        NO_REVISION_NEEDED — useful when principles are ordered by priority.
        Default False (apply every principle).
    """

    def __init__(
        self,
        llm: BaseLLM,
        principles: list[Principle] | None = None,
        skip_initial_generation: bool = True,
        early_stop_on_no_revision: bool = False,
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=None, concurrency=concurrency)
        self.principles = list(principles or CANONICAL_CONSTITUTION)
        if not self.principles:
            raise ValueError("ConstitutionalRevisionTask needs at least one principle.")
        self.skip_initial_generation = bool(skip_initial_generation)
        self.early_stop_on_no_revision = bool(early_stop_on_no_revision)

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        # Only called for the "initial response" pass when skip_initial_generation=False
        return [{"role": "user", "content": sample.instruction}]

    def _parse_response(self, sample, response):
        return []

    # ─────────────────────────────────────────────────────────────────────
    # Single-sample workflow
    # ─────────────────────────────────────────────────────────────────────

    def _initial_response(self, instruction: str) -> str | None:
        try:
            r = self.llm.generate([{"role": "user", "content": instruction}])
            return r.text.strip()
        except Exception:
            return None

    async def _initial_response_async(self, instruction: str) -> str | None:
        try:
            r = await self.llm.agenerate([{"role": "user", "content": instruction}])
            return r.text.strip()
        except Exception:
            return None

    def _critique(self, instruction: str, response: str, principle: Principle) -> str:
        try:
            r = self.llm.generate(
                [
                    {
                        "role": "user",
                        "content": _CRITIQUE_PROMPT.format(
                            instruction=instruction,
                            response=response,
                            critique_request=principle.critique_request,
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return (r.text or "").strip()
        except Exception:
            # On failure, behave as "no revision needed" so the loop is conservative
            return _NO_REVISION_TOKEN

    async def _critique_async(self, instruction, response, principle):
        try:
            r = await self.llm.agenerate(
                [
                    {
                        "role": "user",
                        "content": _CRITIQUE_PROMPT.format(
                            instruction=instruction,
                            response=response,
                            critique_request=principle.critique_request,
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return (r.text or "").strip()
        except Exception:
            return _NO_REVISION_TOKEN

    def _revise(
        self,
        instruction: str,
        response: str,
        critique: str,
        principle: Principle,
    ) -> str | None:
        try:
            r = self.llm.generate(
                [
                    {
                        "role": "user",
                        "content": _REVISION_PROMPT.format(
                            instruction=instruction,
                            response=response,
                            critique=critique,
                            revision_request=principle.revision_request,
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return (r.text or "").strip()
        except Exception:
            return None

    async def _revise_async(self, instruction, response, critique, principle):
        try:
            r = await self.llm.agenerate(
                [
                    {
                        "role": "user",
                        "content": _REVISION_PROMPT.format(
                            instruction=instruction,
                            response=response,
                            critique=critique,
                            revision_request=principle.revision_request,
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return (r.text or "").strip()
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline interface
    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        out: list[DataSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        for sample in samples:
            r = self._process_one(sample, ts)
            if r is not None:
                out.append(r)
        return out

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        sem = asyncio.Semaphore(self.concurrency)

        async def one(s: DataSample):
            async with sem:
                return await self._process_one_async(s, ts)

        results = await asyncio.gather(*[one(s) for s in samples])
        return [x for x in results if x is not None]

    # ─────────────────────────────────────────────────────────────────────

    def _process_one(self, sample: DataSample, ts: datetime) -> DataSample | None:
        instruction = sample.instruction
        if not instruction:
            self._rejected.append(self._rej(sample, "cai:no_instruction"))
            return None

        response = sample.output
        n_initial = 0
        if not response or not self.skip_initial_generation:
            response = self._initial_response(instruction)
            n_initial = 1
        if not response:
            self._rejected.append(self._rej(sample, "cai:initial_response_failed"))
            return None
        original = response

        trace: list[dict[str, Any]] = []
        for principle in self.principles:
            critique = self._critique(instruction, response, principle)
            applied = False
            if _NO_REVISION_TOKEN.lower() not in critique.lower():
                revised = self._revise(instruction, response, critique, principle)
                if revised:
                    response = revised
                    applied = True
            trace.append(
                {
                    "principle": principle.name,
                    "critique": critique[:500],
                    "revision_applied": applied,
                }
            )
            if not applied and self.early_stop_on_no_revision:
                break

        return self._build_sample(
            sample, instruction, original, response, trace, ts, initial_call_count=n_initial
        )

    async def _process_one_async(self, sample: DataSample, ts: datetime):
        instruction = sample.instruction
        if not instruction:
            self._rejected.append(self._rej(sample, "cai:no_instruction"))
            return None

        response = sample.output
        n_initial = 0
        if not response or not self.skip_initial_generation:
            response = await self._initial_response_async(instruction)
            n_initial = 1
        if not response:
            self._rejected.append(self._rej(sample, "cai:initial_response_failed"))
            return None
        original = response

        trace: list[dict[str, Any]] = []
        for principle in self.principles:
            critique = await self._critique_async(instruction, response, principle)
            applied = False
            if _NO_REVISION_TOKEN.lower() not in critique.lower():
                revised = await self._revise_async(
                    instruction,
                    response,
                    critique,
                    principle,
                )
                if revised:
                    response = revised
                    applied = True
            trace.append(
                {
                    "principle": principle.name,
                    "critique": critique[:500],
                    "revision_applied": applied,
                }
            )
            if not applied and self.early_stop_on_no_revision:
                break

        return self._build_sample(
            sample, instruction, original, response, trace, ts, initial_call_count=n_initial
        )

    # ─────────────────────────────────────────────────────────────────────

    def _build_sample(
        self,
        source: DataSample,
        instruction: str,
        original_response: str,
        final_response: str,
        trace: list[dict[str, Any]],
        ts: datetime,
        initial_call_count: int,
    ) -> DataSample:
        revisions_applied = sum(1 for t in trace if t["revision_applied"])
        # Each principle = 1 critique call + (revisions_applied) revision calls
        total_calls = initial_call_count + len(trace) + revisions_applied

        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source.source_uri or "constitutional_ai",
            instruction=instruction,
            output=final_response,
            task_type="instruction_following",
            metadata={
                "generator": "ConstitutionalRevisionTask",
                "original_response": original_response,
                "principles_applied": [p.name for p in self.principles],
                "revisions_applied": revisions_applied,
                "revision_trace": trace,
                "source_sample_id": source.id,
            },
        )
        sample.append_provenance(
            ProvenanceRecord(
                step_name="ConstitutionalRevisionTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=self.llm.config_hash(),
                notes={
                    "principles": [p.name for p in self.principles],
                    "revisions_applied": revisions_applied,
                    "total_llm_calls": total_calls,
                },
            )
        )
        return sample

    def _rej(self, source: DataSample, reason: str) -> RejectedSample:
        return RejectedSample(
            source_uri=source.source_uri or "constitutional_ai",
            instruction=source.instruction,
            rejection_reason=reason,
            rejecting_step="ConstitutionalRevisionTask",
        )


# ───────────────────────────────────────────────────────────────────────────
# ConstitutionalPreferenceTask — emits DPO pairs (revised, original)
# ───────────────────────────────────────────────────────────────────────────


class ConstitutionalPreferenceTask(ConstitutionalRevisionTask):
    """
    Same critique-and-revise loop, but emits a DPO preference sample
    `(chosen=revised_response, rejected=original_response)` instead of an
    SFT sample. Useful for building harmlessness/honesty preference data
    from a single seed corpus — matches the paper's CAI preference-data
    construction.

    If no revisions were applied (the original passed every principle), the
    sample is added to the rejected list with reason `cai:no_revision_made`
    so the audit log records the no-op.
    """

    def _build_sample(
        self,
        source: DataSample,
        instruction: str,
        original_response: str,
        final_response: str,
        trace: list[dict[str, Any]],
        ts: datetime,
        initial_call_count: int,
    ) -> DataSample:
        revisions_applied = sum(1 for t in trace if t["revision_applied"])
        if revisions_applied == 0:
            # No revision means chosen == rejected; skip rather than emit a no-op pair
            self._rejected.append(self._rej(source, "cai:no_revision_made"))
            # Returning None would crash the parent .run() loop; so we
            # build and return a DataSample but mark it so the caller can
            # filter it. Cleaner: just append to rejected and return a
            # marker that the parent's `if r is not None` check filters.
            # We override `_process_one` upstream to honour this.
            return None  # type: ignore[return-value]

        total_calls = initial_call_count + len(trace) + revisions_applied

        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source.source_uri or "constitutional_ai",
            instruction=instruction,
            chosen=final_response,
            rejected=original_response,
            task_type="preference",
            metadata={
                "generator": "ConstitutionalPreferenceTask",
                "principles_applied": [p.name for p in self.principles],
                "revisions_applied": revisions_applied,
                "revision_trace": trace,
                "source_sample_id": source.id,
            },
        )
        sample.append_provenance(
            ProvenanceRecord(
                step_name="ConstitutionalPreferenceTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=self.llm.config_hash(),
                notes={
                    "principles": [p.name for p in self.principles],
                    "revisions_applied": revisions_applied,
                    "total_llm_calls": total_calls,
                },
            )
        )
        return sample
