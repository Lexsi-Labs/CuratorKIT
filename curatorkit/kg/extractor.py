"""
KGExtractor — LLM-based entity + relation extraction from source chunks.

Takes a corpus of DataSamples (typically PDF-chunked or pretrain-style),
calls the LLM to extract (subject, predicate, object) triples from each
chunk, and writes them into a KnowledgeGraph with provenance back to the
originating chunk.

This is the GraphGen "Synthesizer Model" step (Yang et al., 2025). The
prompt asks for a strict JSON list of triples. Parse failures land in
rejected (via the standard BaseGenerationTask machinery would, but this
class is intentionally NOT a generation task — it produces a KG, not new
DataSamples — so it owns its own loop).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import warnings

from curatorkit.kg.store import KnowledgeGraph, Triple
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample

_EXTRACT_PROMPT = """You are a knowledge-graph builder. Extract factual
(subject, predicate, object) triples from the source text below.

Rules:
- Each triple must be DIRECTLY supported by the source. Do not infer.
- Subjects and objects must be specific named entities or concrete concepts
  (proper nouns, technical terms, dates, numbers — not pronouns).
- Predicates must be short verb phrases ("is_a", "founded_by", "treats",
  "located_in", "discovered_in", "causes"). Use snake_case.
- Aim for 3-12 triples for a typical paragraph. If the source has no
  extractable factual content, return an empty list.

Source text:
---
{source}
---

Respond in this exact JSON format (no other text):
[
  {{"subject": "...", "predicate": "...", "object": "..."}},
  ...
]"""


class KGExtractor:
    """
    LLM-driven (subject, predicate, object) extractor.

    Parameters
    ----------
    llm : BaseLLM
        The "Synthesizer" model. GraphGen uses a strong base (e.g., GPT-4
        class) — weaker models leak too much noise into the graph.
    source_field : str
        Which field of the input sample holds the text to extract from.
        "auto" picks: output (pretrain), input (context), instruction (last).
    min_confidence : float
        Drop extracted triples whose triple-level confidence (in the JSON
        response, or a default of 1.0) falls below this threshold.
    concurrency : int
        Async worker pool size.
    """

    def __init__(
        self,
        llm: BaseLLM,
        source_field: str = "auto",
        min_confidence: float = 0.5,
        concurrency: int = 10,
    ) -> None:
        self.llm = llm
        self.source_field = source_field
        self.min_confidence = float(min_confidence)
        self.concurrency = int(concurrency)

    # ─────────────────────────────────────────────────────────────────────

    def _get_source(self, sample: DataSample) -> str:
        if self.source_field != "auto":
            return getattr(sample, self.source_field, "") or ""
        if sample.task_type == "language_modeling" and sample.output:
            return sample.output
        if sample.input:
            return sample.input
        return sample.instruction or sample.output

    def _chunk_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ─────────────────────────────────────────────────────────────────────

    def extract(self, samples: list[DataSample]) -> KnowledgeGraph:
        """Synchronously extract triples from every sample into one graph."""
        kg = KnowledgeGraph()
        for sample in samples:
            triples = self._extract_one(sample)
            kg.add_triples(triples)
        return kg

    async def extract_async(self, samples: list[DataSample]) -> KnowledgeGraph:
        """Async extraction with concurrency control."""
        kg = KnowledgeGraph()
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(s: DataSample) -> list[Triple]:
            async with sem:
                return await self._extract_one_async(s)

        results = await asyncio.gather(*[_one(s) for s in samples])
        for triples in results:
            kg.add_triples(triples)
        return kg

    # ─────────────────────────────────────────────────────────────────────

    def _extract_one(self, sample: DataSample) -> list[Triple]:
        source = self._get_source(sample)
        if not source:
            return []
        prompt = _EXTRACT_PROMPT.format(source=source)
        try:
            r = self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as e:
            warnings.warn(
                f"[KGExtractor] LLM failed for sample {sample.id}: {e}",
                stacklevel=2,
            )
            return []
        return self._parse(r.text or "", sample, source)

    async def _extract_one_async(self, sample: DataSample) -> list[Triple]:
        source = self._get_source(sample)
        if not source:
            return []
        prompt = _EXTRACT_PROMPT.format(source=source)
        try:
            r = await self.llm.agenerate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception:
            return []
        return self._parse(r.text or "", sample, source)

    def _parse(
        self,
        text: str,
        sample: DataSample,
        source: str,
    ) -> list[Triple]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []

        if not isinstance(parsed, list):
            return []

        chunk_hash = self._chunk_hash(source)
        out: list[Triple] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            s = str(entry.get("subject", "")).strip()
            p = str(entry.get("predicate", "")).strip()
            o = str(entry.get("object", "")).strip()
            conf = float(entry.get("confidence", 1.0))
            if not s or not p or not o:
                continue
            if conf < self.min_confidence:
                continue
            out.append(
                Triple(
                    subject=s,
                    predicate=p,
                    object=o,
                    source_uri=sample.source_uri or "kg",
                    source_chunk_hash=chunk_hash,
                    confidence=conf,
                )
            )
        return out
