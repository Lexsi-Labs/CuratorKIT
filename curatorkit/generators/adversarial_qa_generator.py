"""
AdversarialQAGenerationTask — adversarial injection in the first generation pass.

Subclasses QAGenerationTask. For each seed, probabilistically selects either
the faithful grounded prompt (normal generation) or an adversarial prompt that
produces an ungrounded answer. The hallucination gate has no access to the
injection metadata — it evaluates (source, question, answer) blindly.

All injection types are single-pass: one LLM call per seed, returning a JSON
array of {question, answer} pairs — identical format to the faithful path.

Injection types
---------------
contradicts_source
    Answers directly contradict specific facts from the source.
    → Expected diagnosis: GENERATOR_PARAMETRIC

parametric_drift
    Answers use general world knowledge, ignoring the source entirely.
    → Expected diagnosis: GENERATOR_PARAMETRIC

high_temperature_drift
    Faithful prompt generated at T=1.4 instead of T=0.7.
    → Expected diagnosis: GENERATOR_TEMPERATURE

domain_mismatch
    Answers use terminology and framing from a different domain.
    → Expected diagnosis: DOMAIN_MISMATCH

instruction_quality
    Questions are deliberately vague/ambiguous; answers respond to the
    vague question rather than grounding in source facts.
    → Expected diagnosis: INSTRUCTION_QUALITY

Future work: ContrastivePairBuilder will use this to produce
(faithful, adversarial) pairs for DPO dataset construction.
See ~/CONTRASTIVE_DATASET_PLAN.md for the full roadmap.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from datetime import UTC, datetime
from typing import Literal

from curatorkit.generators.qa_generator import _DEFAULT_QA_PROMPT, QAGenerationTask
from curatorkit.llm.base import LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

InjectionType = Literal[
    "contradicts_source",
    "parametric_drift",
    "high_temperature_drift",
    "domain_mismatch",
    "instruction_quality",
]

ALL_INJECTION_TYPES: list[InjectionType] = [
    "contradicts_source",
    "parametric_drift",
    "high_temperature_drift",
    "domain_mismatch",
    "instruction_quality",
]

# ── Single-pass adversarial JSON prompts ──────────────────────────────────────
# All prompts return the same [{question, answer}] JSON format as the faithful
# prompt so _extract_qa_pairs() handles them without special-casing.

_CONTRADICTS_JSON_PROMPT = """\
You are given a source passage. Generate {num_questions} question-answer pairs \
where each answer DIRECTLY AND CLEARLY CONTRADICTS a specific verifiable fact \
in the source — a wrong date, name, number, location, or causal relationship.

Rules:
- Questions must be factual questions answerable from the passage
- Each answer MUST change at least one concrete fact: swap a number, \
flip a cause/effect, replace a named entity, or invert a stated outcome
- The contradiction must be obvious to anyone who reads the source carefully
- Answers must still read as confident and complete sentences
- Do NOT mention that answers are incorrect

Source passage:
---
{context}
---

Respond ONLY with a JSON array, no other text:
[{{"question": "...", "answer": "..."}}] /no_think"""

_PARAMETRIC_DRIFT_JSON_PROMPT = """\
You are given a source passage for question inspiration only. Generate \
{num_questions} question-answer pairs where each answer relies entirely on \
general world knowledge and contains ZERO information from the passage.

Rules:
- Questions should be factual questions about the passage topic
- Answers must come from your training knowledge, not from the passage — \
treat the passage as if it does not exist
- The answer must be on a DIFFERENT specific fact than what the passage states \
(different date, different number, different person, different outcome)
- Do not quote, paraphrase, or echo any phrase from the passage

Source passage (do NOT use any content from this in your answers):
---
{context}
---

Respond ONLY with a JSON array, no other text:
[{{"question": "...", "answer": "..."}}] /no_think"""

_DOMAIN_MISMATCH_JSON_PROMPT = """\
You are given a source passage. Generate {num_questions} question-answer pairs \
where each answer uses vocabulary and framing from a DIFFERENT domain than the \
passage (e.g., answer a medical question with legal framing, or a technical \
question with culinary framing).

Rules:
- Questions should be reasonable questions about the passage content
- Answers must use the wrong domain's vocabulary and framing
- Do not acknowledge the domain mismatch

Source passage:
---
{context}
---

Respond ONLY with a JSON array, no other text:
[{{"question": "...", "answer": "..."}}] /no_think"""

_INSTRUCTION_QUALITY_JSON_PROMPT = """\
You are given a source passage. Generate {num_questions} question-answer pairs \
where each question is deliberately VAGUE, AMBIGUOUS, or POORLY SPECIFIED.

Rules:
- Questions must be too vague to answer precisely: remove specific entities, \
dates, or key terms so the question is unclear
- Questions must still be topically related to the passage
- Answers should respond to the vague question without grounding in specific \
source facts
- Do NOT write clear or specific questions

Source passage:
---
{context}
---

Respond ONLY with a JSON array, no other text:
[{{"question": "...", "answer": "..."}}] /no_think"""


class AdversarialQAGenerationTask(QAGenerationTask):
    """
    Generate QA pairs with a controlled fraction of adversarial samples
    injected directly in the first pass.

    Parameters
    ----------
    llm : BaseLLM
        LLM for both faithful and adversarial generation.
    injection_rate : float
        Fraction of seeds to generate adversarially (0–1). Default 0.20.
    injection_types : list[InjectionType] | None
        Which adversarial types to use. Sampled uniformly. Default: all five.
    num_questions : int
        QA pairs per seed (faithful or adversarial). Default 3.
    seed : int | None
        Random seed for reproducible injection selection.
    high_temp : float
        Temperature used for high_temperature_drift injection. Default 1.4.
    """

    def __init__(
        self,
        llm,
        injection_rate: float = 0.20,
        injection_types: list[InjectionType] | None = None,
        num_questions: int = 3,
        seed: int | None = 42,
        high_temp: float = 1.4,
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, num_questions=num_questions, **kwargs)
        self.injection_rate = max(0.0, min(1.0, injection_rate))
        self.injection_types = injection_types or ALL_INJECTION_TYPES
        self.high_temp = high_temp
        self._rng = random.Random(seed)
        self._injection_plan: dict[str, InjectionType | None] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, seeds: list[DataSample]) -> list[DataSample]:
        self._injection_plan = self._make_plan([s.id for s in seeds])
        return self._run_with_plan(seeds)

    def run_multi_passage(
        self, seed_pairs: list[tuple[DataSample, DataSample]]
    ) -> list[DataSample]:
        pair_ids = [a.id for a, _ in seed_pairs]
        self._injection_plan = self._make_plan(pair_ids)
        return self._run_multi_passage_with_plan(seed_pairs)

    # ─────────────────────────────────────────────────────────────────────────
    # Injection plan
    # ─────────────────────────────────────────────────────────────────────────

    def _make_plan(self, ids: list[str]) -> dict[str, InjectionType | None]:
        n_inject = max(0, round(len(ids) * self.injection_rate))
        inject_ids = set(self._rng.sample(ids, min(n_inject, len(ids))))
        plan: dict[str, InjectionType | None] = {}
        for sid in ids:
            if sid in inject_ids:
                plan[sid] = self._rng.choice(self.injection_types)
            else:
                plan[sid] = None
        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # Single-chunk run loop
    # ─────────────────────────────────────────────────────────────────────────

    def _run_with_plan(self, seeds: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC)

        for seed in seeds:
            inj_type = self._injection_plan.get(seed.id)
            try:
                results.extend(self._generate_one(seed, inj_type, ts))
            except Exception as exc:
                self._rejected.append(
                    RejectedSample(
                        **seed.model_dump(),
                        rejection_reason=f"generation_failed:{exc}",
                        rejecting_step=self.task_name,
                    )
                )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-passage run loop
    # ─────────────────────────────────────────────────────────────────────────

    def _run_multi_passage_with_plan(
        self, seed_pairs: list[tuple[DataSample, DataSample]]
    ) -> list[DataSample]:
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC)

        for chunk_a, chunk_b in seed_pairs:
            inj_type = self._injection_plan.get(chunk_a.id)
            ctx_a = self._get_context(chunk_a)
            ctx_b = self._get_context(chunk_b)
            if not ctx_a or not ctx_b:
                continue
            combined_ctx = ctx_a + "\n\n" + ctx_b
            try:
                results.extend(
                    self._generate_multi_one(chunk_a, chunk_b, combined_ctx, inj_type, ts)
                )
            except Exception as exc:
                self._rejected.append(
                    RejectedSample(
                        **chunk_a.model_dump(),
                        rejection_reason=f"multi_passage_generation_failed:{exc}",
                        rejecting_step=self.task_name,
                    )
                )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Per-seed generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_one(
        self,
        seed: DataSample,
        inj_type: InjectionType | None,
        ts: datetime,
    ) -> list[DataSample]:
        source_ctx = self._get_context(seed)
        if inj_type is None:
            messages = self._build_messages(seed)
            temperature = None
        else:
            messages, temperature = self._adversarial_messages(source_ctx, inj_type)

        response = self.llm.generate(
            messages,
            **({"temperature": temperature} if temperature is not None else {}),
        )
        qa_pairs = self._extract_qa_pairs(response.text.strip()) or self._fallback_extract(
            response.text.strip()
        )

        return self._build_samples(seed, qa_pairs, source_ctx, inj_type, messages, response, ts)

    def _generate_multi_one(
        self,
        chunk_a: DataSample,
        chunk_b: DataSample,
        combined_ctx: str,
        inj_type: InjectionType | None,
        ts: datetime,
    ) -> list[DataSample]:
        from curatorkit.generators.qa_generator import _MULTI_PASSAGE_QA_PROMPT

        if inj_type is None:
            prompt = _MULTI_PASSAGE_QA_PROMPT.format(
                num_passages=2,
                num_questions=self.num_questions,
                passage_1=self._get_context(chunk_a),
                passage_2=self._get_context(chunk_b),
            )
            messages = [{"role": "user", "content": prompt}]
            temperature = None
        else:
            messages, temperature = self._adversarial_messages(combined_ctx, inj_type)

        response = self.llm.generate(
            messages,
            **({"temperature": temperature} if temperature is not None else {}),
        )
        qa_pairs = self._extract_qa_pairs(response.text.strip()) or self._fallback_extract(
            response.text.strip()
        )

        base_metadata = {
            "generation_source": "adversarial_qa_generator_multi_passage"
            if inj_type
            else "qa_generator_multi_passage",
            "source_sample_id_a": chunk_a.id,
            "source_sample_id_b": chunk_b.id,
            "chunk_index_a": chunk_a.metadata.get("chunk_index"),
            "chunk_index_b": chunk_b.metadata.get("chunk_index"),
            "domain": chunk_a.metadata.get("domain", ""),
            "multi_passage": True,
        }
        if inj_type:
            base_metadata["injected_failure"] = True
            base_metadata["injection_type"] = inj_type

        prompt_hash = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:12]
        results = []
        for qa in qa_pairs:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if not q or not a:
                continue
            s = DataSample(
                id=str(uuid.uuid4()),
                source_uri=chunk_a.source_uri,
                instruction=q,
                input=combined_ctx,
                output=a,
                task_type="instruction_following",
                metadata={**base_metadata},
                provenance_chain=list(chunk_a.provenance_chain),
            )
            s.append_provenance(
                ProvenanceRecord(
                    step_name=self.task_name,
                    step_version="1.0.0",
                    timestamp=ts,
                    config_hash=self.llm.config_hash(),
                    notes={
                        **response.to_provenance_dict(),
                        "prompt_hash": prompt_hash,
                        "injection": inj_type or "none",
                        "mode": "multi_passage",
                    },
                )
            )
            results.append(s)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Adversarial prompt builder — single LLM call per injection type
    # ─────────────────────────────────────────────────────────────────────────

    def _adversarial_messages(
        self,
        source_ctx: str,
        inj_type: InjectionType,
    ) -> tuple[list[dict], float | None]:
        """Return (messages, temperature_override) for a single adversarial call."""
        temperature = None

        if inj_type == "contradicts_source":
            prompt = _CONTRADICTS_JSON_PROMPT.format(
                context=source_ctx, num_questions=self.num_questions
            )
        elif inj_type == "parametric_drift":
            prompt = _PARAMETRIC_DRIFT_JSON_PROMPT.format(
                context=source_ctx, num_questions=self.num_questions
            )
        elif inj_type == "high_temperature_drift":
            prompt = _DEFAULT_QA_PROMPT.format(context=source_ctx, num_questions=self.num_questions)
            temperature = self.high_temp
        elif inj_type == "domain_mismatch":
            prompt = _DOMAIN_MISMATCH_JSON_PROMPT.format(
                context=source_ctx, num_questions=self.num_questions
            )
        elif inj_type == "instruction_quality":
            prompt = _INSTRUCTION_QUALITY_JSON_PROMPT.format(
                context=source_ctx, num_questions=self.num_questions
            )
        else:
            assert False, f"Unknown injection type: {inj_type}"

        return [{"role": "user", "content": prompt}], temperature

    # ─────────────────────────────────────────────────────────────────────────
    # Sample assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _build_samples(
        self,
        seed: DataSample,
        qa_pairs: list[dict],
        source_ctx: str,
        inj_type: InjectionType | None,
        messages: list[dict],
        response: LLMResponse,
        ts: datetime,
    ) -> list[DataSample]:
        prompt_hash = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:12]
        results = []

        for qa in qa_pairs:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if not q or not a:
                continue

            metadata: dict = {
                "generation_source": ("adversarial_qa_generator" if inj_type else "qa_generator"),
                "source_sample_id": seed.id,
                "difficulty": self.difficulty,
                **{
                    k: v
                    for k, v in seed.metadata.items()
                    if k in ("chunk_index", "domain", "source_file")
                },
            }
            if inj_type:
                metadata["injected_failure"] = True
                metadata["injection_type"] = inj_type

            s = DataSample(
                id=str(uuid.uuid4()),
                source_uri=seed.source_uri,
                instruction=q,
                input=source_ctx,
                output=a,
                task_type="instruction_following",
                metadata=metadata,
                provenance_chain=list(seed.provenance_chain),
            )
            s.append_provenance(
                ProvenanceRecord(
                    step_name=self.task_name,
                    step_version="1.0.0",
                    timestamp=ts,
                    config_hash=self.llm.config_hash(),
                    notes={
                        **response.to_provenance_dict(),
                        "prompt_hash": prompt_hash,
                        "injection": inj_type or "none",
                    },
                )
            )
            results.append(s)

        return results
