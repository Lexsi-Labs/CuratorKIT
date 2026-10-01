"""
MagpieTask — cold-start instruction synthesis (Xu et al., ICLR 2025).

Reference: "Magpie: Alignment Data Synthesis from Scratch by Prompting
Aligned LLMs with Nothing" — https://arxiv.org/abs/2406.08464

The canonical Magpie algorithm:
  Step 1: Send the aligned model ONLY its pre-query template
          (e.g., `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n`).
          Auto-regressive generation completes the user turn — this is the
          synthesised instruction. No seed data required.
  Step 2: Wrap the generated instruction with the assistant pre-query template
          and prompt the model again. Output is the response.

Two implementation modes:
  raw_prefix=True   — Backend supports raw-completion calls. Send the exact
                      template prefix. This matches the paper.
  raw_prefix=False  — Backend is a chat API (OpenAI/Anthropic/etc.) which
                      forbids partial-prefix completion. We approximate by
                      using a "produce one user-style question" system prompt
                      and a second call for the response. Lower fidelity but
                      works through any LiteLLM provider.

Cold-start usage: pass an empty list (or a list of placeholder samples) to
the task. It generates `n_samples` instructions either way.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime

from curatorkit.generators.base import STEP_VERSION, BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

# ───────────────────────────────────────────────────────────────────────────
# Canonical Magpie template prefixes (a few common chat templates).
# Backends that support raw-completion can use any of these directly.
# ───────────────────────────────────────────────────────────────────────────

CANONICAL_PRE_QUERY_TEMPLATES: dict[str, str] = {
    "llama3": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"),
    "llama2": "<s>[INST] ",
    "qwen2": "<|im_start|>user\n",
    "mistral": "<s>[INST] ",
    # The default is the Llama-3 family because it's the one the paper
    # validates most extensively.
    "default": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"),
}


# Chat-API approximation prompt — used when raw_prefix=False.
# Crafted to be model-agnostic: every aligned chat model can comply.
_CHAT_PRIMING_INSTRUCTION = """You are roleplaying as a user of a conversational AI assistant.
Generate ONE realistic user request that a person might send to such an assistant.

Constraints:
- The request must stand alone (a reader should understand it without context).
- It should be a real task: a question, an instruction, or a problem to solve.
- Aim for diversity — vary topics (coding, writing, science, daily-life advice, math, etc.).
- Do NOT generate the assistant's response. Output ONLY the user request, as plain text, with no preamble or quotation marks."""


class MagpieTask(BaseGenerationTask):
    """
    Cold-start instruction synthesis. Produces (instruction, response) pairs
    without any seed data.

    Parameters
    ----------
    llm : BaseLLM
        The aligned model to harvest instructions from.
    n_samples : int
        Total cold-start samples to produce. Required for the run-without-input
        usage; ignored when called with a non-empty input list.
    raw_prefix : bool
        True  → use canonical Magpie (assumes backend supports prefix prompting,
                e.g., vLLM raw completions).
        False → chat-API approximation. Default False for portability.
    template_id : str
        Key into CANONICAL_PRE_QUERY_TEMPLATES when raw_prefix=True.
    pre_query_template : str | None
        Override the template entirely. Wins over template_id.
    response_temperature : float
        Temperature for the response generation pass (Step 2). The
        instruction pass (Step 1) always uses self.llm's default temperature
        so the instruction distribution matches the paper.
    skip_short_instructions : int
        Drop instructions shorter than this many characters (paper recommends
        ≥10 to filter out one-token "noise" outputs).
    """

    def __init__(
        self,
        llm: BaseLLM,
        n_samples: int = 100,
        raw_prefix: bool = False,
        template_id: str = "default",
        pre_query_template: str | None = None,
        response_temperature: float | None = None,
        skip_short_instructions: int = 10,
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=None, concurrency=concurrency)
        self.n_samples = max(1, int(n_samples))
        self.raw_prefix = bool(raw_prefix)
        self.template_id = template_id
        self.pre_query_template = (
            pre_query_template
            or CANONICAL_PRE_QUERY_TEMPLATES.get(template_id)
            or CANONICAL_PRE_QUERY_TEMPLATES["default"]
        )
        self.response_temperature = response_temperature
        self.skip_short_instructions = max(0, int(skip_short_instructions))

    # ─────────────────────────────────────────────────────────────────────
    # BaseGenerationTask abstracts — Magpie does two LLM calls per sample,
    # so we override `run` / `run_async` instead of using the single-call
    # base implementation. _build_messages / _parse_response are required
    # by the ABC but the work happens in run().
    # ─────────────────────────────────────────────────────────────────────

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        return [{"role": "user", "content": _CHAT_PRIMING_INSTRUCTION}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        # Unused for Magpie — kept to satisfy the ABC.
        return []

    # ─────────────────────────────────────────────────────────────────────
    # Single-sample two-pass call
    # ─────────────────────────────────────────────────────────────────────

    def _step1_instruction(self) -> tuple[str | None, LLMResponse | None]:
        """Step 1 of Magpie: harvest one user-turn instruction from the model."""
        if self.raw_prefix:
            # Send the literal pre-query template as a raw-completion prompt.
            # The backend must support this; LiteLLMBackend / OllamaBackend
            # will route via a "text completion" call when the message list
            # is a single empty-system + prefix arrangement.
            messages = [{"role": "system", "content": self.pre_query_template}]
        else:
            messages = [{"role": "user", "content": _CHAT_PRIMING_INSTRUCTION}]

        try:
            r = self.llm.generate(messages, temperature=self.llm.temperature)
        except Exception:
            return None, None
        text = r.text.strip()
        # Strip leading/trailing quotes that some chat models add anyway
        if text.startswith(('"', "'")) and text.endswith(('"', "'")):
            text = text[1:-1].strip()
        if len(text) < self.skip_short_instructions:
            return None, r
        return text, r

    def _step2_response(self, instruction: str) -> tuple[str | None, LLMResponse | None]:
        """Step 2: prompt the model with the harvested instruction → response."""
        messages = [{"role": "user", "content": instruction}]
        temp = (
            self.response_temperature
            if self.response_temperature is not None
            else self.llm.temperature
        )
        try:
            r = self.llm.generate(messages, temperature=temp)
        except Exception:
            return None, None
        return r.text.strip(), r

    async def _step1_instruction_async(self) -> tuple[str | None, LLMResponse | None]:
        if self.raw_prefix:
            messages = [{"role": "system", "content": self.pre_query_template}]
        else:
            messages = [{"role": "user", "content": _CHAT_PRIMING_INSTRUCTION}]
        try:
            r = await self.llm.agenerate(messages, temperature=self.llm.temperature)
        except Exception:
            return None, None
        text = r.text.strip()
        if text.startswith(('"', "'")) and text.endswith(('"', "'")):
            text = text[1:-1].strip()
        if len(text) < self.skip_short_instructions:
            return None, r
        return text, r

    async def _step2_response_async(
        self, instruction: str
    ) -> tuple[str | None, LLMResponse | None]:
        messages = [{"role": "user", "content": instruction}]
        temp = (
            self.response_temperature
            if self.response_temperature is not None
            else self.llm.temperature
        )
        try:
            r = await self.llm.agenerate(messages, temperature=temp)
        except Exception:
            return None, None
        return r.text.strip(), r

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline interface — override to drive both passes
    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        """
        Cold-start mode: if `samples` is empty, generate self.n_samples from
        scratch. Otherwise produce one Magpie pair per input sample (the
        input is used only for source-attribution metadata; its content is
        discarded — this is the design).
        """
        self._rejected = []
        results: list[DataSample] = []
        target_n = self.n_samples if not samples else len(samples)
        ts = datetime.now(UTC).replace(tzinfo=None)

        for i in range(target_n):
            attribution = samples[i] if (samples and i < len(samples)) else None
            sample = self._magpie_one(attribution, ts)
            if sample is not None:
                results.append(sample)

        return results

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        target_n = self.n_samples if not samples else len(samples)
        ts = datetime.now(UTC).replace(tzinfo=None)
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(i: int):
            attribution = samples[i] if (samples and i < len(samples)) else None
            async with sem:
                return await self._magpie_one_async(attribution, ts)

        outcomes = await asyncio.gather(*[_one(i) for i in range(target_n)])
        return [s for s in outcomes if s is not None]

    # ─────────────────────────────────────────────────────────────────────

    def _magpie_one(
        self,
        attribution: DataSample | None,
        ts: datetime,
    ) -> DataSample | None:
        instr, r1 = self._step1_instruction()
        if not instr:
            self._rejected.append(
                self._rejected_record(
                    attribution,
                    "magpie_step1_empty_or_short",
                )
            )
            return None
        response, r2 = self._step2_response(instr)
        if not response:
            self._rejected.append(
                self._rejected_record(
                    attribution,
                    "magpie_step2_empty",
                )
            )
            return None

        return self._build_sample(instr, response, r1, r2, attribution, ts)

    async def _magpie_one_async(
        self,
        attribution: DataSample | None,
        ts: datetime,
    ) -> DataSample | None:
        instr, r1 = await self._step1_instruction_async()
        if not instr:
            self._rejected.append(
                self._rejected_record(
                    attribution,
                    "magpie_step1_empty_or_short",
                )
            )
            return None
        response, r2 = await self._step2_response_async(instr)
        if not response:
            self._rejected.append(
                self._rejected_record(
                    attribution,
                    "magpie_step2_empty",
                )
            )
            return None
        return self._build_sample(instr, response, r1, r2, attribution, ts)

    def _build_sample(
        self,
        instruction: str,
        response: str,
        r1: LLMResponse | None,
        r2: LLMResponse | None,
        attribution: DataSample | None,
        ts: datetime,
    ) -> DataSample:
        prompt_hash = hashlib.sha256(
            json.dumps(
                {"prefix": self.pre_query_template[:40], "instruction": instruction}
            ).encode()
        ).hexdigest()[:12]
        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=(attribution.source_uri if attribution else "magpie_cold_start"),
            instruction=instruction,
            output=response,
            task_type="instruction_following",
            metadata={
                "generator": "MagpieTask",
                "raw_prefix": self.raw_prefix,
                "template_id": self.template_id,
                "step1_tokens": r1.total_tokens if r1 else 0,
                "step2_tokens": r2.total_tokens if r2 else 0,
            },
        )
        sample.append_provenance(
            ProvenanceRecord(
                step_name="MagpieTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=self.llm.config_hash(),
                notes={
                    "prompt_hash": prompt_hash,
                    "raw_prefix": self.raw_prefix,
                    "template_id": self.template_id,
                    "step1_latency": r1.latency_seconds if r1 else 0,
                    "step2_latency": r2.latency_seconds if r2 else 0,
                    "total_llm_calls": 2,
                },
            )
        )
        return sample

    def _rejected_record(
        self,
        attribution: DataSample | None,
        reason: str,
    ) -> RejectedSample:
        return RejectedSample(
            source_uri=(attribution.source_uri if attribution else "magpie_cold_start"),
            instruction="",
            rejection_reason=reason,
            rejecting_step="MagpieTask",
        )
