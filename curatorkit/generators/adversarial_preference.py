"""
AdversarialPreferenceTask — generate DPO preference pairs with adversarially
hallucinated rejected responses.

Unlike PreferenceGenerationTask (two_pass), which generates a quality-degraded
response as the rejected answer, this task generates:
  - chosen  = faithful answer grounded in the source chunk
  - rejected = adversarially injected answer (specific hallucination failure mode)

The result is a stronger DPO training signal: the model learns to reject
specific failure modes (parametric drift, source contradiction, domain mismatch)
rather than just "lower quality" responses.

Output task_type: "preference"

Usage:
    task = AdversarialPreferenceTask(
        llm=generator_llm,
        num_questions=1,
        injection_rate=0.5,          # fraction of pairs that get adversarial rejected
        injection_types=["contradicts_source", "parametric_drift"],
    )
    samples = task.run(source_chunks)
"""

from __future__ import annotations

import random
import uuid

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord

# Faithful QA generation prompt — same structure as QAGenerationTask default
_DEFAULT_FAITHFUL_PROMPT = """You are an expert question-answer generator. Generate {num_questions} question(s) and answer(s) based ONLY on the provided source text.

Source text:
---
{context}
---

Requirements:
- Questions must be answerable from the source text alone
- Answers must be fully supported by the source text
- Do not use external knowledge
- Difficulty: {difficulty}

Respond in JSON format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

/no_think"""

# Adversarial rejected answer prompts by injection type
_ADVERSARIAL_PROMPTS: dict[str, str] = {
    "contradicts_source": """You are generating a deliberately incorrect answer for dataset quality testing.

Source text:
---
{context}
---

Question: {question}

Write a fluent, confident answer that directly contradicts a specific fact in the source text.
The answer should sound plausible but contain a clear factual error relative to the source.
Return ONLY the answer text, no explanation.

/no_think""",
    "parametric_drift": """You are generating a deliberately incorrect answer for dataset quality testing.

Source text:
---
{context}
---

Question: {question}

Write a fluent answer that ignores the source text and instead relies on general world knowledge.
The answer should be plausible from general knowledge but not grounded in the source.
Return ONLY the answer text, no explanation.

/no_think""",
    "domain_mismatch": """You are generating a deliberately incorrect answer for dataset quality testing.

Source text:
---
{context}
---

Question: {question}

Write a fluent answer that uses terminology and framing from a completely different domain
(e.g., use legal terms for a medical question, or scientific jargon for a general question).
The answer should be technically wrong for this context.
Return ONLY the answer text, no explanation.

/no_think""",
    "instruction_quality": """You are generating a deliberately incorrect answer for dataset quality testing.

Source text:
---
{context}
---

Question: {question}

Write a vague, ambiguous answer that avoids directly addressing the question.
Use hedging language and generic statements that could apply to many different questions.
Return ONLY the answer text, no explanation.

/no_think""",
}

_DEFAULT_INJECTION_TYPES = list(_ADVERSARIAL_PROMPTS.keys())


class AdversarialPreferenceTask(BaseGenerationTask):
    """
    Generate DPO preference pairs with adversarially hallucinated rejected responses.

    For each source chunk, generates `num_questions` faithful QA pairs (chosen),
    then for `injection_rate` fraction of pairs generates an adversarial variant
    of the answer as the rejected response. The remaining pairs use a quality-
    degraded variant (lower temperature re-generation) as the rejected response.

    Parameters
    ----------
    llm : BaseLLM
        Generator LLM (used for both faithful and adversarial generation).
    num_questions : int
        Preference pairs per source chunk.
    injection_rate : float
        Fraction of pairs to inject with adversarial rejected responses (0–1).
        Remaining pairs get a naive re-generation at higher temperature as rejected.
    injection_types : list[str] | None
        Which adversarial types to sample from. None = all four.
        Options: contradicts_source, parametric_drift, domain_mismatch, instruction_quality
    seed : int | None
        RNG seed for reproducible injection assignment.
    faithful_prompt_template : str | None
        Custom prompt for faithful answer generation. Must include
        {context}, {num_questions}, {difficulty} placeholders.
    adversarial_prompt_template : str | None
        Custom prompt for adversarial rejected generation. Must include
        {context}, {question} placeholders.
    difficulty : str
        "easy" | "medium" | "hard" — passed to faithful generation prompt.
    concurrency : int
    """

    def __init__(
        self,
        llm: BaseLLM,
        num_questions: int = 1,
        injection_rate: float = 0.5,
        injection_types: list[str] | None = None,
        seed: int | None = 42,
        faithful_prompt_template: str | None = None,
        adversarial_prompt_template: str | None = None,
        difficulty: str = "medium",
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, concurrency=concurrency)
        self.num_questions = num_questions
        self.injection_rate = injection_rate
        self.injection_types = injection_types or _DEFAULT_INJECTION_TYPES
        self.rng = random.Random(seed)
        self.faithful_prompt = faithful_prompt_template or _DEFAULT_FAITHFUL_PROMPT
        self.adversarial_prompt = adversarial_prompt_template
        self.difficulty = difficulty

    @property
    def task_name(self) -> str:
        return "AdversarialPreferenceTask"

    def _get_context(self, sample: DataSample) -> str:
        if sample.task_type == "language_modeling":
            return sample.output
        return sample.input or sample.output

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        context = self._get_context(sample)
        prompt = self.faithful_prompt.format(
            context=context,
            num_questions=self.num_questions,
            difficulty=self.difficulty,
        )
        return [{"role": "user", "content": prompt}]

    def _build_adversarial_messages(
        self, context: str, question: str, injection_type: str
    ) -> list[dict[str, str]]:
        if self.adversarial_prompt:
            prompt = self.adversarial_prompt.format(
                context=context,
                question=question,
                injection_type=injection_type,
            )
        else:
            template = _ADVERSARIAL_PROMPTS.get(
                injection_type, _ADVERSARIAL_PROMPTS["parametric_drift"]
            )
            prompt = template.format(context=context, question=question)
        return [{"role": "user", "content": prompt}]

    def _build_naive_rejected_messages(self, context: str, question: str) -> list[dict[str, str]]:
        """Naive rejected: re-generate at higher temperature (no adversarial intent)."""
        prompt = self.faithful_prompt.format(
            context=context,
            num_questions=1,
            difficulty="hard",
        )
        return [{"role": "user", "content": prompt}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        import json

        text = response.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[: text.rfind("```")]

        try:
            pairs = json.loads(text)
            if not isinstance(pairs, list):
                pairs = [pairs]
        except json.JSONDecodeError:
            return []

        context = self._get_context(sample)
        results = []

        for pair in pairs:
            question = pair.get("question") or pair.get("q", "")
            answer = pair.get("answer") or pair.get("a", "")
            if not question or not answer:
                continue

            # Decide injection type for this pair
            use_adversarial = self.rng.random() < self.injection_rate
            injection_type = self.rng.choice(self.injection_types) if use_adversarial else None

            # Generate rejected response
            if use_adversarial:
                adv_msgs = self._build_adversarial_messages(context, question, injection_type)
                try:
                    adv_resp = self.llm.generate(adv_msgs, temperature=0.8)
                    rejected_text = adv_resp.text.strip()
                except Exception:
                    rejected_text = ""
            else:
                # Naive rejected: higher temperature regeneration
                naive_msgs = self._build_naive_rejected_messages(context, question)
                try:
                    naive_resp = self.llm.generate(naive_msgs, temperature=1.1)
                    # Parse first answer from response
                    naive_text = naive_resp.text.strip()
                    try:
                        naive_pairs = json.loads(naive_text)
                        if isinstance(naive_pairs, list) and naive_pairs:
                            rejected_text = naive_pairs[0].get("answer", naive_text)
                        else:
                            rejected_text = naive_text
                    except json.JSONDecodeError:
                        rejected_text = naive_text
                except Exception:
                    rejected_text = ""

            if not rejected_text:
                continue

            ds = DataSample(
                id=str(uuid.uuid4()),
                source_uri=sample.source_uri,
                instruction=question,
                input=context,
                output=answer,  # output mirrors chosen for downstream compatibility
                chosen=answer,
                rejected=rejected_text,
                task_type="preference",
                metadata={
                    "generation_source": "adversarial_preference",
                    "source_sample_id": sample.id,
                    "injection_type": injection_type or "naive_high_temp",
                    "adversarial_rejected": use_adversarial,
                    "difficulty": self.difficulty,
                },
                provenance_chain=list(sample.provenance_chain),
            )
            ds.append_provenance(
                ProvenanceRecord(
                    step_name="AdversarialPreferenceTask",
                    step_version="1.0.0",
                    config_hash=self.llm.config_hash(),
                    notes={
                        **response.to_provenance_dict(),
                        "injection_type": injection_type or "naive_high_temp",
                        "adversarial_rejected": use_adversarial,
                        "source_sample_id": sample.id,
                    },
                )
            )
            results.append(ds)

        return results
