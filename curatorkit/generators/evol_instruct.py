"""
EvolInstructTask — evolve simple instructions into harder variants.

Corpus-aware: when the input sample carries raw source text (task_type=
'language_modeling'), the generator creates a grounded instruction from the
source text instead of evolving an empty instruction.  The output DataSample
always carries input=source_context so HallucinationGate can verify grounding.

Implements the core idea from WizardLM's Evol-Instruct: given a simple
instruction, produce a more complex version that requires deeper reasoning.

Evolution strategies:
  - add_constraints: Add specific requirements or constraints
  - deepen: Require deeper domain knowledge
  - concretize: Replace generic references with specific examples
  - increase_reasoning: Require multi-step reasoning
  - broaden: Increase scope to cover related topics

Each evolution preserves the original instruction in provenance.
"""

from __future__ import annotations

import json
import re
import uuid

from tqdm import tqdm

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample

_EVOLUTION_STRATEGIES = [
    "add_constraints",
    "deepen",
    "concretize",
    "increase_reasoning",
    "broaden",
]

_DEFAULT_EVOL_PROMPT = """You are an instruction evolution expert. Your task is to make the given instruction more complex and challenging using the specified evolution strategy.

Original instruction:
---
{instruction}
---

Evolution strategy: {strategy}

Strategy descriptions:
- add_constraints: Add 2-3 specific requirements, edge cases, or constraints that make the task harder
- deepen: Require deeper domain expertise or more advanced knowledge
- concretize: Replace generic references with specific, realistic examples or scenarios
- increase_reasoning: Require multi-step reasoning, comparison, or analysis
- broaden: Expand scope to cover related sub-topics or cross-domain connections

Requirements:
- The evolved instruction MUST be solvable (don't create impossible tasks)
- The evolved instruction should be self-contained (no references to "the original")
- Keep the same general topic but increase complexity
- The evolved instruction should be 1-3 sentences

Respond in JSON format ONLY:
{{"evolved_instruction": "...", "strategy_applied": "{strategy}", "complexity_notes": "..."}}"""

_EVOL_PROMPT_WITH_CONTEXT = """You are an instruction evolution expert. Your task is to make the given instruction more complex using the specified evolution strategy. The instruction must remain answerable from the source passage.

Source passage (evolved instruction must be answerable from this):
---
{context}
---

Original instruction:
---
{instruction}
---

Evolution strategy: {strategy}

Strategy descriptions:
- add_constraints: Add 2-3 specific requirements or constraints grounded in the passage
- deepen: Require deeper analysis of concepts present in the passage
- concretize: Reference specific details, examples, or scenarios from the passage
- increase_reasoning: Require multi-step reasoning about the passage content
- broaden: Expand scope to cover related sub-topics also present in the passage

Requirements:
- The evolved instruction MUST be answerable from the passage above
- The evolved instruction should be self-contained
- Keep the same general topic but increase complexity
- The evolved instruction should be 1-3 sentences

Respond in JSON format ONLY:
{{"evolved_instruction": "...", "strategy_applied": "{strategy}", "complexity_notes": "..."}}"""

_CORPUS_EVOL_PROMPT = """You are an instruction creation expert. Your task is to create a specific, complex instruction that:
1. Is directly answerable from the provided source passage
2. Uses the "{strategy}" complexity strategy

Source passage:
---
{context}
---

Strategy descriptions:
- add_constraints: Create an instruction with 2-3 specific requirements or constraints
- deepen: Create an instruction requiring deep analysis of passage concepts
- concretize: Create an instruction referencing specific details from the passage
- increase_reasoning: Create an instruction requiring multi-step reasoning
- broaden: Create an instruction covering multiple related topics from the passage

Requirements:
- The instruction MUST be answerable from the passage above (no outside knowledge needed)
- The instruction should be specific and unambiguous
- The instruction should be 1-3 sentences

Respond in JSON format ONLY:
{{"evolved_instruction": "...", "strategy_applied": "{strategy}", "complexity_notes": "..."}}"""

_DEFAULT_ANSWER_PROMPT = """Answer the following instruction thoroughly and accurately.

Instruction:
{instruction}

Provide a detailed, well-structured response."""

_ANSWER_PROMPT_WITH_CONTEXT = """Answer the following instruction accurately and thoroughly using the provided source passage.

Source passage:
---
{context}
---

Instruction:
{instruction}

Provide a detailed, well-structured response grounded in the passage above."""


class EvolInstructTask(BaseGenerationTask):
    """
    Evolve instructions into harder variants via LLM.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend for generation.
    prompt_template : str | None
        Custom prompt template. Must contain {instruction} and {strategy}.
    num_evolutions : int
        Number of evolved variants per instruction (each uses a different strategy).
    strategies : list[str] | None
        Which evolution strategies to use. Defaults to all five.
    generate_answers : bool
        If True, also generate answers for evolved instructions.
    answer_prompt_template : str | None
        Custom template for answer generation.
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str | None = None,
        num_evolutions: int = 1,
        strategies: list[str] | None = None,
        generate_answers: bool = True,
        answer_prompt_template: str | None = None,
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=prompt_template, concurrency=concurrency)
        self.num_evolutions = max(1, num_evolutions)
        self.strategies = strategies or _EVOLUTION_STRATEGIES
        self.generate_answers = generate_answers
        self.answer_prompt_template = answer_prompt_template or _DEFAULT_ANSWER_PROMPT
        if prompt_template:
            self._validate_template(prompt_template, ["instruction", "strategy", "context"])
        if answer_prompt_template:
            self._validate_template(
                answer_prompt_template, ["instruction"], "answer_prompt_template"
            )

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        """Build messages for a single evolution. Strategy is set via metadata."""
        strategy = sample.metadata.get("_evol_strategy", self.strategies[0])
        source_context = self._get_source_context(sample)
        instruction = sample.instruction

        if not instruction and source_context:
            # Corpus mode: create a grounded instruction from scratch
            prompt = _CORPUS_EVOL_PROMPT.format(context=source_context, strategy=strategy)
        elif source_context:
            template = self.prompt_template or _EVOL_PROMPT_WITH_CONTEXT
            prompt = template.format(
                instruction=instruction,
                strategy=strategy,
                context=source_context,
            )
        else:
            template = self.prompt_template or _DEFAULT_EVOL_PROMPT
            prompt = template.format(instruction=instruction, strategy=strategy)

        return [{"role": "user", "content": prompt}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        text = response.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        evolved_instruction = ""
        strategy_used = sample.metadata.get("_evol_strategy", "unknown")
        complexity_notes = ""

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                evolved_instruction = parsed.get("evolved_instruction", "")
                strategy_used = parsed.get("strategy_applied", strategy_used)
                complexity_notes = parsed.get("complexity_notes", "")
        except json.JSONDecodeError:
            evolved_instruction = text

        if not evolved_instruction:
            return []

        source_context = self._get_source_context(sample)

        return [
            DataSample(
                id=str(uuid.uuid4()),
                source_uri=sample.source_uri,
                instruction=evolved_instruction,
                input=source_context,
                output="",  # Answer generated in second pass if generate_answers=True
                task_type="instruction_following",
                metadata={
                    "generation_source": "evol_instruct",
                    "evolution_strategy": strategy_used,
                    "complexity_notes": complexity_notes,
                    "original_instruction": sample.instruction,
                    "corpus_mode": not sample.instruction and bool(source_context),
                    "source_sample_id": sample.id,
                },
                provenance_chain=list(sample.provenance_chain),
            )
        ]

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        """
        Run evolution with strategy cycling.

        Each sample gets num_evolutions variants, cycling through strategies.
        If generate_answers is True, a second LLM pass generates answers.
        """
        expanded: list[DataSample] = []
        for sample in samples:
            for i in range(self.num_evolutions):
                strategy = self.strategies[i % len(self.strategies)]
                tagged = sample.model_copy(deep=True)
                tagged.metadata["_evol_strategy"] = strategy
                expanded.append(tagged)

        evolved = super().run(expanded)

        if not self.generate_answers:
            return evolved

        from concurrent.futures import ThreadPoolExecutor, as_completed

        answered: list[DataSample] = [None] * len(evolved)  # type: ignore[list-item]

        def _answer_one(idx_sample):
            idx, sample = idx_sample
            try:
                source_context = sample.input  # already set by _parse_response
                if source_context:
                    prompt = _ANSWER_PROMPT_WITH_CONTEXT.format(
                        instruction=sample.instruction,
                        context=source_context,
                    )
                else:
                    prompt = self.answer_prompt_template.format(
                        instruction=sample.instruction,
                    )
                response = self.llm.generate([{"role": "user", "content": prompt}])
                sample.output = response.text.strip()
            except Exception:
                pass
            return idx, sample

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(_answer_one, (i, s)): i for i, s in enumerate(evolved)}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[EvolInstruct] answering",
                unit="sample",
            ):
                idx, sample = future.result()
                answered[idx] = sample

        return [s for s in answered if s is not None]
