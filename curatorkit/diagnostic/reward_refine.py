"""
RewardRefiner — targeted single-retry recovery for RewardGate failures.

For each reward-rejected sample, reads the lowest-scoring dimension and the
judge's weakness note from provenance, then prompts the generator to rewrite
the answer specifically to improve that dimension. The refined answer is
re-evaluated by the same RewardGate — one LLM call per sample.

Supported by:
  Self-Refine (Madaan et al., 2023) — critique-conditioned iterative refinement
  CRITIC (Gou et al., 2023) — targeted LLM feedback loops
  RAIN (Liu et al., 2023) — reward-signal-guided rewriting
"""

from __future__ import annotations

import copy
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from curatorkit.interfaces import BaseGate, BaseNormalizer
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

logger = logging.getLogger(__name__)

_REWARD_REFINE_PROMPT = """\
The following answer was evaluated and scored low on {axis}.

Judge feedback: "{weakness}"

Rewrite the answer to specifically improve {axis}. Keep the answer grounded \
in the source passage — do not introduce facts not present in the source.

Question: {instruction}

Source passage:
---
{source}
---

Original answer: {answer}

Improved answer: /no_think"""

_INSTRUCTION_REFINE_PROMPT = """\
The following question is poorly formed, vague, or ambiguous.

Judge feedback: "{weakness}"

Rewrite the question to be clear, specific, and directly answerable from \
the source passage below.

Source passage:
---
{source}
---

Original question: {question}

Improved question: /no_think"""


class RewardRefiner(BaseNormalizer):
    """
    Single-retry recovery for RewardGate-rejected samples.

    Parameters
    ----------
    generator_llm : BaseLLM
        LLM used for answer rewriting.
    reward_gate : BaseGate
        The RewardGate instance to re-evaluate refined samples.
    refine_instruction : bool
        If True, also attempt question refinement when the failure axis is
        instruction_following or when no axis is identified.
        Default False (answer rewrite only).
    """

    def __init__(
        self,
        generator_llm: BaseLLM,
        reward_gate: BaseGate,
        refine_instruction: bool = False,
        concurrency: int = 32,
        refine_prompt_template: str | None = None,
        instruction_refine_template: str | None = None,
    ) -> None:
        self.generator_llm = generator_llm
        self.reward_gate = reward_gate
        self.refine_instruction = refine_instruction
        self.concurrency = concurrency
        self._refine_prompt = refine_prompt_template or _REWARD_REFINE_PROMPT
        self._instruction_prompt = instruction_refine_template or _INSTRUCTION_REFINE_PROMPT

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        """BaseNormalizer pipeline interface.

        When used as a pipeline step after RewardGate, all samples in the list
        have already passed the gate — nothing to refine. This method is a no-op
        pass-through for direct pipeline use. Call refine() directly for
        post-pipeline targeted recovery on RewardGate rejects.
        """
        return samples

    def _config_hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {
                "refine_instruction": self.refine_instruction,
                "concurrency": self.concurrency,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def refine(
        self, rejected: list[RejectedSample]
    ) -> tuple[list[DataSample], list[RejectedSample]]:
        """
        Targeted refinement on reward-rejected samples.

        Generation is concurrent (ThreadPoolExecutor); all candidates are then
        evaluated in a single bulk RewardGate call.

        Returns
        -------
        recovered : list[DataSample]  — samples that passed RewardGate after refinement
        still_rejected : list[RejectedSample]  — samples that still fail
        """
        if not rejected:
            return [], []

        # Step 1: generate all refined candidates concurrently
        def _generate(sample: RejectedSample) -> DataSample | None:
            # Rejected-too-good pairs cannot be fixed by refining chosen — skip.
            if "rejected_above_threshold" in (sample.rejection_reason or ""):
                return None
            try:
                axis, weakness = self._get_failure_axis(sample)
                if self.refine_instruction and axis in ("instruction_following", ""):
                    return self._refine_instruction(sample, weakness)
                return self._refine_answer(sample, axis, weakness)
            except Exception as exc:
                logger.warning("RewardRefiner generation failed for %s: %s", sample.id, exc)
                return None

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            raw_candidates = list(pool.map(_generate, rejected))

        # Split into valid candidates vs generation failures
        pairs = [(orig, cand) for orig, cand in zip(rejected, raw_candidates) if cand is not None]
        gen_failed = [orig for orig, cand in zip(rejected, raw_candidates) if cand is None]

        if not pairs:
            return [], rejected

        orig_list, cand_list = zip(*pairs)

        # Step 2: evaluate all candidates in one bulk gate call
        passed, _ = self.reward_gate.run(list(cand_list))
        passed_ids = {s.id for s in passed}

        gate_failed = [
            orig for orig, cand in zip(orig_list, cand_list) if cand.id not in passed_ids
        ]

        return list(passed), gen_failed + gate_failed

    # ─────────────────────────────────────────────────────────────────────────

    def _refine_one(self, sample: RejectedSample) -> DataSample | None:
        axis, weakness = self._get_failure_axis(sample)

        if self.refine_instruction and axis in ("instruction_following", ""):
            refined = self._refine_instruction(sample, weakness)
        else:
            refined = self._refine_answer(sample, axis, weakness)

        if refined is None:
            return None

        passed, _ = self.reward_gate.run([refined])
        return passed[0] if passed else None

    def _refine_answer(self, sample: RejectedSample, axis: str, weakness: str) -> DataSample | None:
        prompt = self._refine_prompt.format(
            axis=axis or "overall quality",
            weakness=weakness or "the response needs improvement",
            instruction=sample.instruction,
            source=sample.input or "",
            answer=sample.output,
        )
        try:
            resp = self.generator_llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
        except Exception as exc:
            logger.debug("RewardRefiner answer rewrite failed: %s", exc)
            return None

        new_output = resp.text.strip()
        if not new_output:
            return None

        return self._build_candidate(sample, new_output, axis, "answer_rewrite")

    def _refine_instruction(self, sample: RejectedSample, weakness: str) -> DataSample | None:
        prompt = self._instruction_prompt.format(
            weakness=weakness or "the question is unclear",
            source=sample.input or "",
            question=sample.instruction,
        )
        try:
            resp = self.generator_llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=128,
            )
        except Exception as exc:
            logger.debug("RewardRefiner instruction rewrite failed: %s", exc)
            return None

        new_instruction = resp.text.strip()
        if not new_instruction:
            return None

        candidate = self._build_candidate(
            sample, sample.output, "instruction_following", "instruction_rewrite"
        )
        candidate.instruction = new_instruction
        return candidate

    def _build_candidate(
        self,
        original: RejectedSample,
        new_output: str,
        axis: str,
        refine_type: str,
    ) -> DataSample:
        metadata = copy.deepcopy(original.metadata)
        metadata["reward_refined"] = True
        metadata["refinement_axis"] = axis
        metadata["refinement_type"] = refine_type

        s = DataSample(
            id=str(uuid.uuid4()),
            source_uri=original.source_uri,
            instruction=original.instruction,
            input=original.input,
            output=new_output,
            # Preserve DPO pair structure so RewardGate dual-scoring applies on re-eval.
            # chosen gets the refined answer; rejected (adversarial) is unchanged.
            chosen=new_output if original.chosen else "",
            rejected=original.rejected,
            task_type=original.task_type,
            metadata=metadata,
            provenance_chain=list(original.provenance_chain),
        )
        s.append_provenance(
            ProvenanceRecord(
                step_name="RewardRefiner",
                step_version="1.0.0",
                timestamp=datetime.now(UTC),
                config_hash="",
                notes={
                    "original_id": original.id,
                    "refinement": refine_type,
                    "axis": axis,
                },
            )
        )
        return s

    # ─────────────────────────────────────────────────────────────────────────

    def _get_failure_axis(self, sample: RejectedSample) -> tuple[str, str]:
        """
        Extract the weakest dimension and judge weakness note from provenance.
        Returns (axis, weakness_note).
        """
        for rec in reversed(sample.provenance_chain):
            if rec.step_name == "RewardGate" and not rec.notes.get("passed", True):
                dim_scores = rec.notes.get("dimension_scores", {})
                weakness = rec.notes.get("weaknesses", "")
                if dim_scores:
                    axis = min(dim_scores, key=lambda d: dim_scores[d])
                    return axis, weakness
                return "", weakness
        return "", ""
