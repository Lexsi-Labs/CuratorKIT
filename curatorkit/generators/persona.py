"""
PersonaDrivenTask — persona-conditioned instruction synthesis.

Reference: "Scaling Synthetic Data Creation with 1,000,000,000 Personas"
(Tencent AI Lab Seattle, 2024) — https://arxiv.org/abs/2406.20094
Project: https://github.com/tencent-ailab/persona-hub
Canonical templates: persona-hub/code/prompt_templates.py

The paper's canonical workflow is TWO LLM calls per persona:
  Pass 1 — "Guess a prompt that the following persona may ask you to do: ..."
           model returns ONLY the prompt (e.g., starting with "User prompt:")
  Pass 2 — Send that prompt back to the model normally → assistant response

This file implements the paper's canonical templates verbatim where they
were public (instruction, math) and the same template *structure* for the
extras (logic, knowledge, tool). Two-pass is the default. Set
`generate_answer=False` to skip pass 2 — useful when you want to harvest
prompts only and pair them with a different answer model downstream.

Supported task styles (set via `style=`):
  "instruction" → "Guess a prompt that the following persona may ask you to do..."
  "math"        → "Create a math problem related to the following persona..."
  "logic"       → "Create a logical reasoning problem related to the following persona..."
  "knowledge"   → "Write a knowledge-rich passage authored by the following persona..."
  "tool"        → "Define a tool/function the following persona would invoke..."
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import UTC, datetime

from curatorkit.generators.base import STEP_VERSION, BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

# ───────────────────────────────────────────────────────────────────────────
# Canonical templates (Tencent persona-hub/code/prompt_templates.py)
# ───────────────────────────────────────────────────────────────────────────

_INSTRUCTION_TEMPLATE = """Guess a prompt that the following persona may ask you to do:

{persona}

Note:

1. The prompt should be informative and specific.

2. Your output should start with "User prompt:" """


_MATH_TEMPLATE = """Create a math problem related to the following persona:

{persona}

Note:

1. The math problem should be challenging and involve advanced mathematical skills and knowledge. Only top talents can solve it correctly.

2. You should make full use of the persona description to create the math problem to ensure that the math problem is unique and specific to the persona.

3. Your response should always start with "Math problem:". Your response should not include a solution to the created math problem.

4. Your created math problem should include no more than 2 sub-problems."""


# Same structure, extended to the cases the paper lists as use-cases.
_LOGIC_TEMPLATE = """Create a logical reasoning problem related to the following persona:

{persona}

Note:

1. The problem should be non-trivial and require multi-step reasoning.

2. You should make full use of the persona description so the problem is specific to the persona's domain.

3. Your response should always start with "Logic problem:". Your response should not include the solution."""


_KNOWLEDGE_TEMPLATE = """Write a knowledge-rich short passage authored by the following persona:

{persona}

Note:

1. The passage should be 3-6 sentences, written from this persona's expert perspective.

2. Pick a specific topic this persona would know in depth.

3. Your response should always start with "Passage:" """


_TOOL_TEMPLATE = """Define one tool (function) that the following persona would invoke:

{persona}

Note:

1. Pick a tool this persona realistically needs in their domain.

2. Output ONLY the tool definition.

3. Your response should always start with "Tool:" """


# Style → (template, output-prefix to strip when parsing)
_STYLE_SPEC: dict[str, tuple[str, str]] = {
    "instruction": (_INSTRUCTION_TEMPLATE, "User prompt:"),
    "math": (_MATH_TEMPLATE, "Math problem:"),
    "logic": (_LOGIC_TEMPLATE, "Logic problem:"),
    "knowledge": (_KNOWLEDGE_TEMPLATE, "Passage:"),
    "tool": (_TOOL_TEMPLATE, "Tool:"),
}


_STYLE_TASK_TYPES: dict[str, str] = {
    "instruction": "instruction_following",
    "math": "instruction_following",
    "logic": "instruction_following",
    "knowledge": "language_modeling",
    "tool": "instruction_following",
}


# ───────────────────────────────────────────────────────────────────────────
# PersonaDrivenTask
# ───────────────────────────────────────────────────────────────────────────


class PersonaDrivenTask(BaseGenerationTask):
    """
    Generate persona-conditioned data using the canonical Persona-Hub
    two-pass workflow.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend.
    style : str
        One of "instruction" | "math" | "logic" | "knowledge" | "tool".
    persona_field : str
        Where the persona description lives on the input sample.
        "instruction" (default) reads sample.instruction. Pass
        "metadata.persona" to read sample.metadata["persona"].
    generate_answer : bool
        If True (default), make a second LLM call to produce the assistant
        response. If False, only Pass 1 runs and the prompt is stored in
        `instruction` with an empty `output` — useful when you want to
        harvest prompts only and pair them with a different model later.
    prompt_template : str | None
        Override the style's default template. Must contain `{persona}`.
    output_prefix : str | None
        Override the expected output prefix used to strip the leading
        "User prompt:" / "Math problem:" / etc. tag.
    """

    def __init__(
        self,
        llm: BaseLLM,
        style: str = "instruction",
        persona_field: str = "instruction",
        generate_answer: bool = True,
        prompt_template: str | None = None,
        output_prefix: str | None = None,
        concurrency: int = 10,
    ) -> None:
        if style not in _STYLE_SPEC:
            raise ValueError(f"style must be one of {sorted(_STYLE_SPEC)}, got {style!r}")
        default_template, default_prefix = _STYLE_SPEC[style]
        super().__init__(
            llm=llm,
            prompt_template=prompt_template or default_template,
            concurrency=concurrency,
        )
        self.style = style
        self.persona_field = persona_field
        self.generate_answer = bool(generate_answer)
        self.output_prefix = output_prefix or default_prefix

    # ─────────────────────────────────────────────────────────────────────

    def _get_persona(self, sample: DataSample) -> str:
        if self.persona_field == "instruction":
            return sample.instruction or ""
        if self.persona_field.startswith("metadata."):
            key = self.persona_field.split(".", 1)[1]
            return str(sample.metadata.get(key, ""))
        return str(getattr(sample, self.persona_field, "") or "")

    def _strip_prefix(self, text: str) -> str:
        """Strip the canonical 'User prompt:' / 'Math problem:' etc. prefix."""
        text = text.strip()
        # Match the prefix case-insensitively at the start
        pattern = re.escape(self.output_prefix)
        text = re.sub(rf"^\s*{pattern}\s*", "", text, count=1, flags=re.IGNORECASE)
        return text.strip()

    # _build_messages / _parse_response satisfy the ABC; we override run().
    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": self.prompt_template.format(persona=self._get_persona(sample)),
            }
        ]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        return []

    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        for sample in samples:
            r = self._process_one(sample, ts)
            if r is not None:
                results.append(r)
        return results

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(s: DataSample) -> DataSample | None:
            async with sem:
                return await self._process_one_async(s, ts)

        outcomes = await asyncio.gather(*[_one(s) for s in samples])
        return [r for r in outcomes if r is not None]

    # ─────────────────────────────────────────────────────────────────────

    def _process_one(
        self,
        sample: DataSample,
        ts: datetime,
    ) -> DataSample | None:
        persona = self._get_persona(sample)
        if not persona:
            self._rejected.append(self._rej(sample, "persona_missing"))
            return None

        # Pass 1 — harvest the prompt
        prompt_msg = [
            {
                "role": "user",
                "content": self.prompt_template.format(persona=persona),
            }
        ]
        try:
            r1 = self.llm.generate(prompt_msg)
        except Exception as e:
            self._rejected.append(self._rej(sample, f"persona_pass1_error:{type(e).__name__}"))
            return None
        harvested = self._strip_prefix(r1.text or "")
        if not harvested:
            self._rejected.append(self._rej(sample, "persona_pass1_empty"))
            return None

        # Pass 2 — generate the answer (optional)
        answer = ""
        r2: LLMResponse | None = None
        if self.generate_answer:
            try:
                r2 = self.llm.generate(
                    [{"role": "user", "content": harvested}],
                )
                answer = (r2.text or "").strip()
            except Exception as e:
                self._rejected.append(
                    self._rej(
                        sample,
                        f"persona_pass2_error:{type(e).__name__}",
                    )
                )
                return None

        return self._build_sample(persona, harvested, answer, r1, r2, sample, ts)

    async def _process_one_async(
        self,
        sample: DataSample,
        ts: datetime,
    ) -> DataSample | None:
        persona = self._get_persona(sample)
        if not persona:
            self._rejected.append(self._rej(sample, "persona_missing"))
            return None

        prompt_msg = [
            {
                "role": "user",
                "content": self.prompt_template.format(persona=persona),
            }
        ]
        try:
            r1 = await self.llm.agenerate(prompt_msg)
        except Exception as e:
            self._rejected.append(self._rej(sample, f"persona_pass1_error:{type(e).__name__}"))
            return None
        harvested = self._strip_prefix(r1.text or "")
        if not harvested:
            self._rejected.append(self._rej(sample, "persona_pass1_empty"))
            return None

        answer = ""
        r2: LLMResponse | None = None
        if self.generate_answer:
            try:
                r2 = await self.llm.agenerate(
                    [{"role": "user", "content": harvested}],
                )
                answer = (r2.text or "").strip()
            except Exception as e:
                self._rejected.append(
                    self._rej(
                        sample,
                        f"persona_pass2_error:{type(e).__name__}",
                    )
                )
                return None

        return self._build_sample(persona, harvested, answer, r1, r2, sample, ts)

    # ─────────────────────────────────────────────────────────────────────

    def _build_sample(
        self,
        persona: str,
        harvested: str,
        answer: str,
        r1: LLMResponse,
        r2: LLMResponse | None,
        source: DataSample,
        ts: datetime,
    ) -> DataSample:
        task_type = _STYLE_TASK_TYPES[self.style]

        # Knowledge passages flow into output (the passage IS the data).
        if self.style == "knowledge":
            instruction = ""
            output = harvested
        else:
            instruction = harvested
            output = answer

        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source.source_uri or "persona_hub",
            instruction=instruction,
            output=output,
            task_type=task_type,
            metadata={
                "generator": "PersonaDrivenTask",
                "persona": persona,
                "style": self.style,
                "two_pass": self.generate_answer,
            },
        )
        sample.append_provenance(
            ProvenanceRecord(
                step_name="PersonaDrivenTask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=self.llm.config_hash(),
                notes={
                    "style": self.style,
                    "two_pass": self.generate_answer,
                    "pass1_tokens": r1.total_tokens,
                    "pass2_tokens": r2.total_tokens if r2 else 0,
                    "total_llm_calls": 2 if self.generate_answer else 1,
                },
            )
        )
        return sample

    def _rej(self, sample: DataSample, reason: str) -> RejectedSample:
        return RejectedSample(
            source_uri=sample.source_uri or "persona_hub",
            instruction=sample.instruction,
            rejection_reason=reason,
            rejecting_step="PersonaDrivenTask",
        )
