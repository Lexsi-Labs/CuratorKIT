"""
QAGenerationTask — generate grounded question-answer pairs from text chunks.

Supports two generation modes:

Single-chunk (default):
  Given one text chunk, generate N grounded QA pairs.

Multi-passage (run_multi_passage):
  Given pairs of adjacent chunks, generate N synthesis questions that require
  information from BOTH passages to answer. This produces harder samples where
  the generator must bridge across passages — increasing natural hallucination
  rate at the gate and making failure modes more diverse.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC

from tqdm import tqdm

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample

_DEFAULT_QA_PROMPT = """You are an expert at creating high-quality question-answer pairs from source text.

Given the following text, generate exactly {num_questions} question-answer pair(s).

Requirements:
- Questions should be diverse (factual, conceptual, analytical)
- Answers MUST be fully grounded in the provided text — do not add external knowledge
- Each answer should be detailed (2-5 sentences)
- Questions should be standalone (a reader should understand them without seeing the source)

Source text:
---
{context}
---

Respond in the following JSON format ONLY (no other text):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
] /no_think"""

_DEFAULT_TABLE_QA_PROMPT = """You are an expert at creating question-answer pairs from tabular data.

Given the following table content, generate exactly {num_questions} question-answer pair(s).

Requirements:
- Questions should require reading and interpreting the table
- Answers MUST be grounded in the table data — do not add external knowledge
- Include questions about specific values, comparisons, and trends where applicable

Table content:
---
{context}
---

Respond in the following JSON format ONLY (no other text):
[
  {{"question": "...", "answer": "..."}}
]"""

_MULTI_PASSAGE_QA_PROMPT = """You are an expert at creating challenging question-answer pairs that require synthesizing information across multiple passages.

Below are {num_passages} related passages from the same source:

Passage 1:
---
{passage_1}
---

Passage 2:
---
{passage_2}
---

Generate exactly {num_questions} question-answer pairs following these rules:
- Questions MUST require facts from MORE THAN ONE passage to answer completely
- Use comparative, causal, or connective question types ("How does X relate to Y?", "What explains the difference between...")
- Answers must be fully grounded in the passages above — do not add external knowledge
- Each answer should be 2-4 sentences and draw explicitly from both passages
- Questions must be self-contained (understandable without seeing the passages)

Respond in the following JSON format ONLY (no other text):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


class QAGenerationTask(BaseGenerationTask):
    """
    Generate question-answer pairs from text chunks.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend for generation.
    prompt_template : str | None
        Custom prompt template. Must contain {context} and {num_questions}.
    num_questions : int
        Number of QA pairs to generate per chunk.
    table_prompt_template : str | None
        Separate prompt for table-derived chunks.
    difficulty : str
        Difficulty level hint: "easy", "medium", "hard".
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str | None = None,
        num_questions: int = 3,
        table_prompt_template: str | None = None,
        difficulty: str = "medium",
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=prompt_template, concurrency=concurrency)
        self.num_questions = max(1, num_questions)
        self.table_prompt_template = table_prompt_template or _DEFAULT_TABLE_QA_PROMPT
        self.difficulty = difficulty

    def _get_context(self, sample: DataSample) -> str:
        """Extract the text chunk from the sample."""
        # Language modeling samples store text in output
        if sample.task_type == "language_modeling" and sample.output:
            return sample.output
        # Context-bearing samples
        if sample.input:
            return sample.input
        # Fallback: instruction might hold the chunk
        if sample.instruction:
            return sample.instruction
        return sample.output

    def _is_table_sample(self, sample: DataSample) -> bool:
        """Check if this sample is table-derived."""
        return (
            sample.metadata.get("_table_skipped", False)
            or sample.task_type == "table_qa"
            or sample.metadata.get("content_type") == "table"
        )

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        context = self._get_context(sample)

        if self._is_table_sample(sample):
            template = self.table_prompt_template
        else:
            template = self.prompt_template or _DEFAULT_QA_PROMPT

        # Add difficulty hint if present in template or append it
        prompt = template.format(
            context=context,
            num_questions=self.num_questions,
            difficulty=self.difficulty,
        )

        if self.difficulty != "medium" and "{difficulty}" not in template:
            prompt += f"\n\nDifficulty level: {self.difficulty}"

        return [{"role": "user", "content": prompt}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        """Parse JSON array of QA pairs from LLM response."""
        text = response.text.strip()
        qa_pairs = self._extract_qa_pairs(text)

        if not qa_pairs:
            # Fallback: try to parse as a single QA
            qa_pairs = self._fallback_extract(text)

        context = self._get_context(sample)
        results: list[DataSample] = []

        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()

            if not question or not answer:
                continue

            task_type = "table_qa" if self._is_table_sample(sample) else "instruction_following"

            results.append(
                DataSample(
                    id=str(uuid.uuid4()),
                    source_uri=sample.source_uri,
                    instruction=question,
                    input=context,
                    output=answer,
                    task_type=task_type,
                    metadata={
                        "generation_source": "qa_generator",
                        "source_sample_id": sample.id,
                        "difficulty": self.difficulty,
                        # Carry forward PDF provenance for hallucination gate
                        **{
                            k: v
                            for k, v in sample.metadata.items()
                            if k
                            in (
                                "page",
                                "parent_heading",
                                "chunk_index",
                                "content_type",
                                "source_file",
                                "domain",
                            )
                        },
                    },
                    provenance_chain=list(sample.provenance_chain),
                )
            )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-passage generation
    # ─────────────────────────────────────────────────────────────────────────

    def run_multi_passage(
        self,
        seed_pairs: list[tuple[DataSample, DataSample]],
    ) -> list[DataSample]:
        """
        Generate cross-passage synthesis QA from pairs of adjacent chunks.

        Each pair produces num_questions QA samples whose answers require
        facts from both passages. The combined context (passage_1 + passage_2)
        is stored in DataSample.input so the hallucination gate judges against
        the full source.

        Parameters
        ----------
        seed_pairs : list of (chunk_a, chunk_b) DataSample tuples
        """
        import hashlib
        import json as _json
        from datetime import datetime

        from curatorkit.schema import ProvenanceRecord, RejectedSample

        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC)

        for chunk_a, chunk_b in tqdm(seed_pairs, desc="[QAGenerator] multi-passage", unit="pair"):
            ctx_a = self._get_context(chunk_a)
            ctx_b = self._get_context(chunk_b)

            if not ctx_a or not ctx_b:
                continue

            prompt = _MULTI_PASSAGE_QA_PROMPT.format(
                num_passages=2,
                num_questions=self.num_questions,
                passage_1=ctx_a,
                passage_2=ctx_b,
            )
            messages = [{"role": "user", "content": prompt}]

            try:
                response = self.llm.generate(messages)
            except Exception as exc:
                self._rejected.append(
                    RejectedSample(
                        **chunk_a.model_dump(),
                        rejection_reason=f"multi_passage_generation_failed:{exc}",
                        rejecting_step=self.task_name,
                    )
                )
                continue

            text = response.text.strip()
            qa_pairs = self._extract_qa_pairs(text) or self._fallback_extract(text)

            if not qa_pairs:
                self._rejected.append(
                    RejectedSample(
                        **chunk_a.model_dump(),
                        rejection_reason="multi_passage_parse_failed",
                        rejecting_step=self.task_name,
                    )
                )
                continue

            # Combined context for hallucination gate
            combined_context = ctx_a + "\n\n" + ctx_b
            prompt_hash = hashlib.sha256(_json.dumps(messages).encode()).hexdigest()[:12]

            for qa in qa_pairs:
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                if not question or not answer:
                    continue

                sample = DataSample(
                    id=str(uuid.uuid4()),
                    source_uri=chunk_a.source_uri,
                    instruction=question,
                    input=combined_context,
                    output=answer,
                    task_type="instruction_following",
                    metadata={
                        "generation_source": "qa_generator_multi_passage",
                        "source_sample_id_a": chunk_a.id,
                        "source_sample_id_b": chunk_b.id,
                        "chunk_index_a": chunk_a.metadata.get("chunk_index"),
                        "chunk_index_b": chunk_b.metadata.get("chunk_index"),
                        "domain": chunk_a.metadata.get("domain", ""),
                        "difficulty": self.difficulty,
                        "multi_passage": True,
                    },
                    provenance_chain=list(chunk_a.provenance_chain),
                )
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name=self.task_name,
                        step_version="1.0.0",
                        timestamp=ts,
                        config_hash=self.llm.config_hash(),
                        notes={
                            **response.to_provenance_dict(),
                            "prompt_hash": prompt_hash,
                            "source_sample_id_a": chunk_a.id,
                            "source_sample_id_b": chunk_b.id,
                            "mode": "multi_passage",
                        },
                    )
                )
                results.append(sample)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Parsing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_qa_pairs(self, text: str) -> list[dict[str, str]]:
        """Try to parse JSON array from the response."""
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [p for p in parsed if isinstance(p, dict)]
            except json.JSONDecodeError:
                pass

        return []

    def _fallback_extract(self, text: str) -> list[dict[str, str]]:
        """Regex fallback for non-JSON responses."""
        pairs = []
        # Pattern: Q: ... A: ...
        q_matches = re.findall(
            r"(?:Q(?:uestion)?[:\d.)\s]*)(.*?)(?:A(?:nswer)?[:\d.)\s]*)(.*?)(?=(?:Q(?:uestion)?[:\d.)\s])|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        for q, a in q_matches:
            q = q.strip()
            a = a.strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
        return pairs
