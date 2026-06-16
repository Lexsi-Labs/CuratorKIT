"""
ChainOfThoughtTask — generate step-by-step reasoning before the final answer.

Corpus-aware: when the input sample carries raw source text (task_type=
'language_modeling'), the generator includes the source in the CoT prompt so
reasoning is grounded in the passage.  The output DataSample always carries
input=source_context so HallucinationGate can verify grounding.

Produces instruction-following samples where the output includes explicit
chain-of-thought reasoning. This is critical for training models that need
to show their work (math, logic, code reasoning, complex analysis).

Two modes:
  - "wrap": Takes an existing instruction + output and wraps the output
    in chain-of-thought reasoning
  - "generate": Takes an instruction and generates both the CoT and answer
"""

from __future__ import annotations

import json
import re
import uuid

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample

_DEFAULT_COT_GENERATE_PROMPT = """Solve the following step by step. Show your reasoning clearly before giving the final answer.

Instruction:
{instruction}

Format your response as:
## Reasoning
(step-by-step thinking)

## Answer
(final answer)"""

_COT_GENERATE_PROMPT_WITH_CONTEXT = """Based on the source passage, solve the following step by step. Ground your reasoning strictly in the passage.

Source passage:
---
{context}
---

Instruction:
{instruction}

Format your response as:
## Reasoning
(step-by-step thinking grounded in the passage)

## Answer
(final answer)"""

_CORPUS_COT_PROMPT = """Based on the following source passage, provide a detailed step-by-step analysis.

Source passage:
---
{context}
---

Show your reasoning process clearly:

## Reasoning
(step-by-step analysis of the key ideas in the passage)

## Key Points
(main insights and conclusions)"""

_DEFAULT_COT_WRAP_PROMPT = """Given the following instruction and its correct answer, generate detailed step-by-step reasoning that leads to this answer.

Instruction: {instruction}

Correct answer: {answer}

Generate the chain-of-thought reasoning that naturally leads to this answer.

Respond in JSON format ONLY:
{{
  "reasoning": "step-by-step reasoning here...",
  "answer": "final answer (can be the same or a refined version of the original)"
}}"""


class ChainOfThoughtTask(BaseGenerationTask):
    """
    Generate chain-of-thought reasoning for instructions.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend for generation.
    mode : str
        "generate": Generate both CoT and answer from instruction only.
        "wrap": Given instruction + existing answer, generate CoT reasoning.
    prompt_template : str | None
        Custom prompt template.
    cot_marker : str
        String used to separate reasoning from the final answer in the output.
    """

    def __init__(
        self,
        llm: BaseLLM,
        mode: str = "generate",
        prompt_template: str | None = None,
        cot_marker: str = "\n\n## Answer\n",
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=prompt_template, concurrency=concurrency)
        self.mode = mode
        self.cot_marker = cot_marker
        if prompt_template:
            required = ["instruction", "answer"] if mode == "wrap" else ["instruction"]
            self._validate_template(prompt_template, required)

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        source_context = self._get_source_context(sample)
        instruction = sample.instruction

        if self.mode == "wrap" and sample.output:
            template = self.prompt_template or _DEFAULT_COT_WRAP_PROMPT
            prompt = template.format(
                instruction=instruction,
                answer=sample.output,
            )
        elif not instruction and source_context:
            # Corpus mode: analyze source text
            prompt = _CORPUS_COT_PROMPT.format(context=source_context)
        elif source_context:
            prompt = _COT_GENERATE_PROMPT_WITH_CONTEXT.format(
                instruction=instruction,
                context=source_context,
            )
        else:
            template = self.prompt_template or _DEFAULT_COT_GENERATE_PROMPT
            prompt = template.format(instruction=instruction)

        return [{"role": "user", "content": prompt}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        text = response.text.strip()

        if self.mode == "wrap":
            return self._parse_wrap_response(sample, text)
        else:
            return self._parse_generate_response(sample, text)

    def _parse_wrap_response(self, sample: DataSample, text: str) -> list[DataSample]:
        """Parse JSON response from wrap mode."""
        clean = re.sub(r"```(?:json)?\s*", "", text)
        clean = re.sub(r"```\s*$", "", clean).strip()

        reasoning = ""
        answer = sample.output

        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                reasoning = parsed.get("reasoning", "")
                answer = parsed.get("answer", sample.output)
        except json.JSONDecodeError:
            reasoning = text

        if not reasoning:
            return []

        source_context = self._get_source_context(sample)
        full_output = f"## Reasoning\n{reasoning}{self.cot_marker}{answer}"

        return [
            DataSample(
                id=str(uuid.uuid4()),
                source_uri=sample.source_uri,
                instruction=sample.instruction,
                input=source_context,
                output=full_output,
                task_type="instruction_following",
                metadata={
                    "generation_source": "cot_generator",
                    "cot_mode": "wrap",
                    "has_chain_of_thought": True,
                    "source_sample_id": sample.id,
                },
                provenance_chain=list(sample.provenance_chain),
            )
        ]

    def _parse_generate_response(self, sample: DataSample, text: str) -> list[DataSample]:
        """Parse freeform CoT + answer from generate mode."""
        if not text:
            return []

        source_context = self._get_source_context(sample)
        instruction = sample.instruction or "Analyze and discuss the source passage."

        return [
            DataSample(
                id=str(uuid.uuid4()),
                source_uri=sample.source_uri,
                instruction=instruction,
                input=source_context,
                output=text,
                task_type="instruction_following",
                metadata={
                    "generation_source": "cot_generator",
                    "cot_mode": "generate",
                    "has_chain_of_thought": True,
                    "corpus_mode": not sample.instruction and bool(source_context),
                    "source_sample_id": sample.id,
                },
                provenance_chain=list(sample.provenance_chain),
            )
        ]
