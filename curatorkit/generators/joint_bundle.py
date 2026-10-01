"""
JointBundleTask — single-pass SFT + DPO + GRPO from one source chunk.

The SFT pair, the DPO chosen/rejected pair, and the GRPO
rollout group all derive from the *same* source chunk in a *single* LLM
pass, with structurally linked provenance via a shared `bundle_id`.

Why this matters: existing frameworks generate SFT, DPO, and GRPO in
separate passes from separate prompts, which means the preference pair's
"rejected" is not actually a worse version of the SFT answer — it's just a
different generation. A reviewer can't audit *why* one response is
preferred. With a joint pass:

  1. One LLM call emits a structured bundle:
     - sft_response       — the high-quality target answer
     - chosen / rejected  — paired responses to the SAME prompt
     - rollouts[]         — N independent responses for GRPO with scores
  2. All emitted DataSamples carry `bundle_id` in metadata and share
     `source_uri` + `source_chunk_hash` in provenance.
  3. A downstream consumer can JOIN sft.jsonl × dpo.jsonl × grpo.jsonl on
     bundle_id to reconstruct which artifacts came from the same source.

This is two LLM calls per input chunk (one for SFT + DPO + rollouts; one
optional reward-scoring pass per rollout), so cost is bounded.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from curatorkit.generators.base import STEP_VERSION, BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

_JOINT_PROMPT = """You are generating training data for a language model from a source text.
For the given source, produce a coherent BUNDLE of artifacts:

  1. ONE high-quality instruction the source can answer.
  2. THE preferred (chosen) answer — accurate, complete, grounded in the source.
  3. ONE rejected answer — plausible but with a specific flaw (omission,
     overclaim, or partially-correct reasoning). It MUST still address the
     instruction (no off-topic refusals).
  4. {num_rollouts} alternative responses ("rollouts") of varying quality, to
     enable GRPO training. The rollouts should differ from "chosen" and
     "rejected" — they explore the response space.

ALL artifacts must derive from the SAME source. The "rejected" must be a
plausible-but-worse answer to the SAME instruction as "chosen" — not a
different question.

Source text:
---
{source}
---

Respond in this exact JSON format (no other text):
{{
  "instruction": "...",
  "chosen": "...",
  "rejected": "...",
  "rollouts": ["...", "...", "...", "..."]
}}"""


_REWARD_PROMPT = """Score the following response on a 0.0-1.0 scale for how
well it answers the instruction using ONLY the source text. Reply with just
the number.

Source:
---
{source}
---
Instruction: {instruction}
Response: {response}

Score (0.0-1.0):"""


class JointBundleTask(BaseGenerationTask):
    """
    One source → SFT + DPO + GRPO bundle.

    Parameters
    ----------
    llm : BaseLLM
        Generation backend.
    num_rollouts : int
        How many rollouts to request in the GRPO bundle. Default 4.
    score_rollouts : bool
        If True (default), make num_rollouts extra LLM calls to score each
        rollout for the GRPO `reward_scores` field. Disable to halve cost
        if your downstream pipeline scores rollouts itself.
    source_field : str
        Which field of the input sample holds the source chunk.
        "auto" picks: output (pretrain), input (context), instruction (last).
    """

    def __init__(
        self,
        llm: BaseLLM,
        num_rollouts: int = 4,
        score_rollouts: bool = True,
        source_field: str = "auto",
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=None, concurrency=concurrency)
        self.num_rollouts = max(2, int(num_rollouts))
        self.score_rollouts = bool(score_rollouts)
        self.source_field = source_field

    # ─────────────────────────────────────────────────────────────────────

    def _get_source(self, sample: DataSample) -> str:
        if self.source_field != "auto":
            return getattr(sample, self.source_field, "") or ""
        if sample.task_type == "language_modeling" and sample.output:
            return sample.output
        if sample.input:
            return sample.input
        return sample.instruction or sample.output

    # _build_messages / _parse_response satisfy the ABC but we override run().
    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": _JOINT_PROMPT.format(
                    source=self._get_source(sample),
                    num_rollouts=self.num_rollouts,
                ),
            }
        ]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        return []

    # ─────────────────────────────────────────────────────────────────────
    # Synchronous run override — handles bundle generation + optional scoring
    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)

        for sample in samples:
            bundle_samples = self._process_one(sample, ts)
            results.extend(bundle_samples)

        return results

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(s: DataSample) -> list[DataSample]:
            async with sem:
                return await self._process_one_async(s, ts)

        outcomes = await asyncio.gather(*[_one(s) for s in samples])
        return [item for batch in outcomes for item in batch]

    # ─────────────────────────────────────────────────────────────────────

    def _process_one(self, sample: DataSample, ts: datetime) -> list[DataSample]:
        source = self._get_source(sample)
        if not source:
            self._rejected.append(self._rej(sample, "joint_bundle:no_source"))
            return []

        messages = self._build_messages(sample)
        try:
            response = self.llm.generate(messages)
        except Exception as e:
            self._rejected.append(
                self._rej(
                    sample,
                    f"joint_bundle:llm_error:{type(e).__name__}",
                )
            )
            return []

        bundle = self._parse_bundle(response.text)
        if bundle is None:
            self._rejected.append(
                self._rej(
                    sample,
                    "joint_bundle:parse_error",
                )
            )
            return []

        # Score rollouts (optional)
        reward_scores: list[float] = []
        if self.score_rollouts:
            for r in bundle["rollouts"]:
                score = self._score_rollout(source, bundle["instruction"], r)
                reward_scores.append(score)

        return self._materialize_bundle(
            sample,
            source,
            bundle,
            response,
            reward_scores,
            ts,
        )

    async def _process_one_async(
        self,
        sample: DataSample,
        ts: datetime,
    ) -> list[DataSample]:
        source = self._get_source(sample)
        if not source:
            self._rejected.append(self._rej(sample, "joint_bundle:no_source"))
            return []

        try:
            response = await self.llm.agenerate(self._build_messages(sample))
        except Exception as e:
            self._rejected.append(
                self._rej(
                    sample,
                    f"joint_bundle:llm_error:{type(e).__name__}",
                )
            )
            return []

        bundle = self._parse_bundle(response.text)
        if bundle is None:
            self._rejected.append(
                self._rej(
                    sample,
                    "joint_bundle:parse_error",
                )
            )
            return []

        reward_scores: list[float] = []
        if self.score_rollouts:
            for r in bundle["rollouts"]:
                reward_scores.append(
                    await self._score_rollout_async(
                        source,
                        bundle["instruction"],
                        r,
                    )
                )

        return self._materialize_bundle(
            sample,
            source,
            bundle,
            response,
            reward_scores,
            ts,
        )

    # ─────────────────────────────────────────────────────────────────────

    def _parse_bundle(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

        required = {"instruction", "chosen", "rejected", "rollouts"}
        if not required.issubset(parsed.keys()):
            return None
        if not isinstance(parsed["rollouts"], list) or not parsed["rollouts"]:
            return None
        if not all(isinstance(r, str) for r in parsed["rollouts"]):
            return None
        return parsed

    def _score_rollout(self, source: str, instruction: str, response: str) -> float:
        try:
            r = self.llm.generate(
                [
                    {
                        "role": "user",
                        "content": _REWARD_PROMPT.format(
                            source=source,
                            instruction=instruction,
                            response=response,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=8,
            )
        except Exception:
            return 0.5
        return _parse_score(r.text)

    async def _score_rollout_async(
        self,
        source: str,
        instruction: str,
        response: str,
    ) -> float:
        try:
            r = await self.llm.agenerate(
                [
                    {
                        "role": "user",
                        "content": _REWARD_PROMPT.format(
                            source=source,
                            instruction=instruction,
                            response=response,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=8,
            )
        except Exception:
            return 0.5
        return _parse_score(r.text)

    # ─────────────────────────────────────────────────────────────────────
    # Bundle → linked DataSamples
    # ─────────────────────────────────────────────────────────────────────

    def _materialize_bundle(
        self,
        source_sample: DataSample,
        source: str,
        bundle: dict[str, Any],
        response: LLMResponse,
        reward_scores: list[float],
        ts: datetime,
    ) -> list[DataSample]:
        bundle_id = str(uuid.uuid4())
        chunk_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        prov_notes_base = {
            "generator": "JointBundleTask",
            "bundle_id": bundle_id,
            "source_chunk_hash": chunk_hash,
            "source_sample_id": source_sample.id,
            "total_llm_calls": 1 + (len(bundle["rollouts"]) if self.score_rollouts else 0),
        }
        cfg_hash = self.llm.config_hash()

        results: list[DataSample] = []

        # ── SFT (instruction → chosen) ─────────────────────────────────
        sft = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source_sample.source_uri,
            instruction=bundle["instruction"],
            input=source,
            output=bundle["chosen"],
            task_type="instruction_following",
            metadata={
                **prov_notes_base,
                "role_in_bundle": "sft",
            },
        )
        sft.append_provenance(
            ProvenanceRecord(
                step_name="JointBundleTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=cfg_hash,
                notes={**prov_notes_base, "role_in_bundle": "sft", **response.to_provenance_dict()},
            )
        )
        results.append(sft)

        # ── DPO (chosen / rejected on same instruction) ───────────────
        dpo = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source_sample.source_uri,
            instruction=bundle["instruction"],
            input=source,
            chosen=bundle["chosen"],
            rejected=bundle["rejected"],
            task_type="preference",
            metadata={
                **prov_notes_base,
                "role_in_bundle": "dpo",
            },
        )
        dpo.append_provenance(
            ProvenanceRecord(
                step_name="JointBundleTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=cfg_hash,
                notes={**prov_notes_base, "role_in_bundle": "dpo", **response.to_provenance_dict()},
            )
        )
        results.append(dpo)

        # ── GRPO (N rollouts + reward_scores) ─────────────────────────
        grpo = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source_sample.source_uri,
            instruction=bundle["instruction"],
            input=source,
            responses=list(bundle["rollouts"]),
            reward_scores=reward_scores or [0.5] * len(bundle["rollouts"]),
            task_type="grpo",
            metadata={
                **prov_notes_base,
                "role_in_bundle": "grpo",
                "scored": self.score_rollouts,
            },
        )
        grpo.append_provenance(
            ProvenanceRecord(
                step_name="JointBundleTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=cfg_hash,
                notes={
                    **prov_notes_base,
                    "role_in_bundle": "grpo",
                    "num_rollouts": len(bundle["rollouts"]),
                    "scored": self.score_rollouts,
                    **response.to_provenance_dict(),
                },
            )
        )
        results.append(grpo)

        return results

    def _rej(self, sample: DataSample, reason: str) -> RejectedSample:
        return RejectedSample(
            **sample.model_dump(),
            rejection_reason=reason,
            rejecting_step="JointBundleTask",
        )


def _parse_score(text: str) -> float:
    """Extract a 0.0-1.0 number from the LLM's free-text scoring response."""
    m = re.search(r"(\d+(?:\.\d+)?)", text or "")
    if not m:
        return 0.5
    s = float(m.group(1))
    if s > 1.0:
        s = s / 10.0
    return max(0.0, min(1.0, s))
