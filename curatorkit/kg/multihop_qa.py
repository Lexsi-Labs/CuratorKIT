"""
MultiHopQATask — generate atomic / aggregated / multi-hop QA from a graph.

Reference: GraphGen (Yang et al., 2025) — three QA scenarios:
  atomic     — single (s, p, o) triple → "Q: <s p> ?  A: <o>"
  aggregated — multiple triples sharing a subject → richer compound answer
  multi_hop  — k-hop path → multi-step reasoning question

The generator walks the KnowledgeGraph using `sample_paths()` to drive
generation density without enumerating an exponential path space. Each
generated DataSample carries:
  - `source_uri`           — the originating chunk's URI
  - `metadata["kg_path"]`  — the (s,p,o) sequence the question was built from
  - `metadata["hops"]`     — path length
  - `metadata["mode"]`     — atomic / aggregated / multi_hop

Multi-hop questions are auditable because the path itself is the citation — a reviewer can audit which
triples support which inference, satisfying the "structural traceability"
claim of the provenance-grounded hallucination contract.
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
from curatorkit.kg.store import KnowledgeGraph
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

# ───────────────────────────────────────────────────────────────────────────
# Prompts
# ───────────────────────────────────────────────────────────────────────────

_ATOMIC_PROMPT = """Generate ONE atomic question-answer pair from this fact:

Fact: ({subject}) -[{predicate}]-> ({object})

The question should test recall of this specific fact. The answer must be
the object value exactly.

Respond in this exact JSON format (no other text):
{{"question": "...", "answer": "..."}}"""

_AGGREGATED_PROMPT = """Generate ONE aggregated question-answer pair from these
facts about "{subject}":

{facts}

The question should require integrating information from multiple facts.
The answer should synthesize them into a coherent response.

Respond in this exact JSON format (no other text):
{{"question": "...", "answer": "..."}}"""

_MULTIHOP_PROMPT = """Generate ONE multi-hop reasoning question that requires
following this {n_hops}-hop chain of facts:

{path}

The question MUST require all {n_hops} steps to answer. The answer is the
final entity in the chain.

Respond in this exact JSON format (no other text):
{{"question": "...", "answer": "...", "reasoning_steps": ["...", "..."]}}"""


# ───────────────────────────────────────────────────────────────────────────
# MultiHopQATask
# ───────────────────────────────────────────────────────────────────────────


class MultiHopQATask(BaseGenerationTask):
    """
    Generate QA samples from a pre-built KnowledgeGraph.

    Unlike most BaseGenerationTask subclasses, this one ignores its input
    `samples` list and instead walks the graph to produce QA pairs. The
    input samples are kept as a count signal (one input → one output) so
    the runner produces a predictable volume.

    Parameters
    ----------
    llm : BaseLLM
        Generator.
    kg : KnowledgeGraph
        The graph to walk.
    mode : str
        "atomic" | "aggregated" | "multi_hop" | "mix" (rotate through all).
    max_hops : int
        Path-length cap for multi-hop mode. Default 3.
    n_samples : int
        Total QA pairs to produce when called without input. Required.
    """

    def __init__(
        self,
        llm: BaseLLM,
        kg: KnowledgeGraph,
        mode: str = "multi_hop",
        max_hops: int = 3,
        n_samples: int = 50,
        concurrency: int = 10,
    ) -> None:
        if mode not in ("atomic", "aggregated", "multi_hop", "mix"):
            raise ValueError(f"mode must be one of atomic/aggregated/multi_hop/mix, got {mode!r}")
        super().__init__(llm=llm, prompt_template=None, concurrency=concurrency)
        self.kg = kg
        self.mode = mode
        self.max_hops = max(1, int(max_hops))
        self.n_samples = max(1, int(n_samples))

    # ─────────────────────────────────────────────────────────────────────
    # BaseGenerationTask abstracts — task overrides run/run_async so these
    # are only used by the failed-path fall-back machinery.
    # ─────────────────────────────────────────────────────────────────────

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        return [{"role": "user", "content": ""}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        return []

    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        target_n = self.n_samples if not samples else len(samples)
        ts = datetime.now(UTC).replace(tzinfo=None)
        plan = self._plan_generations(target_n)
        results: list[DataSample] = []
        for entry in plan:
            sample = self._generate_one(entry, ts)
            if sample is not None:
                results.append(sample)
        return results

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        self._rejected = []
        target_n = self.n_samples if not samples else len(samples)
        ts = datetime.now(UTC).replace(tzinfo=None)
        plan = self._plan_generations(target_n)
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(entry: dict[str, Any]):
            async with sem:
                return await self._generate_one_async(entry, ts)

        outcomes = await asyncio.gather(*[_one(e) for e in plan])
        return [s for s in outcomes if s is not None]

    # ─────────────────────────────────────────────────────────────────────
    # Generation planning
    # ─────────────────────────────────────────────────────────────────────

    def _plan_generations(self, target_n: int) -> list[dict[str, Any]]:
        """
        Build a list of generation entries, each carrying the mode + the
        graph data needed (a triple, a subject's fact-bundle, or a path).
        """
        plan: list[dict[str, Any]] = []
        if not self.kg.triples():
            return plan

        modes = [self.mode] if self.mode != "mix" else ["atomic", "aggregated", "multi_hop"]

        for i in range(target_n):
            mode = modes[i % len(modes)]
            entry = self._build_entry(mode, i)
            if entry is not None:
                plan.append(entry)
        return plan

    def _build_entry(self, mode: str, idx: int) -> dict[str, Any] | None:
        if mode == "atomic":
            triples = self.kg.triples()
            if not triples:
                return None
            t = triples[idx % len(triples)]
            return {"mode": "atomic", "triple": t}

        if mode == "aggregated":
            # Find a subject with ≥2 outgoing predicates
            for entity in self.kg.entities():
                succ = self.kg.successors(entity)
                if len(succ) >= 2:
                    return {"mode": "aggregated", "subject": entity, "facts": succ}
            return None

        if mode == "multi_hop":
            paths = self.kg.sample_paths(
                max_hops=self.max_hops,
                sample_size=1,
                seed=idx,
            )
            paths = [p for p in paths if len(p) >= 2]
            if not paths:
                return None
            return {"mode": "multi_hop", "path": paths[0]}

        return None

    # ─────────────────────────────────────────────────────────────────────
    # Per-entry generation
    # ─────────────────────────────────────────────────────────────────────

    def _generate_one(
        self,
        entry: dict[str, Any],
        ts: datetime,
    ) -> DataSample | None:
        prompt, source_attr = self._build_prompt(entry)
        if prompt is None:
            return None
        try:
            r = self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
        except Exception as e:
            self._rejected.append(self._rej(entry, f"multihop_llm_error:{type(e).__name__}"))
            return None
        parsed = self._parse(r.text)
        if parsed is None:
            self._rejected.append(self._rej(entry, "multihop_parse_error"))
            return None
        return self._materialize(entry, parsed, r, source_attr, ts)

    async def _generate_one_async(
        self,
        entry: dict[str, Any],
        ts: datetime,
    ) -> DataSample | None:
        prompt, source_attr = self._build_prompt(entry)
        if prompt is None:
            return None
        try:
            r = await self.llm.agenerate(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
        except Exception as e:
            self._rejected.append(self._rej(entry, f"multihop_llm_error:{type(e).__name__}"))
            return None
        parsed = self._parse(r.text)
        if parsed is None:
            self._rejected.append(self._rej(entry, "multihop_parse_error"))
            return None
        return self._materialize(entry, parsed, r, source_attr, ts)

    # ─────────────────────────────────────────────────────────────────────

    def _build_prompt(self, entry: dict[str, Any]) -> tuple[str | None, str]:
        mode = entry["mode"]
        if mode == "atomic":
            t = entry["triple"]
            return _ATOMIC_PROMPT.format(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object,
            ), t.source_uri

        if mode == "aggregated":
            facts = "\n".join(f"- {entry['subject']} -[{p}]-> {o}" for p, o in entry["facts"])
            return _AGGREGATED_PROMPT.format(
                subject=entry["subject"],
                facts=facts,
            ), ""

        if mode == "multi_hop":
            path = entry["path"]
            path_text = "\n".join(
                f"  Step {i + 1}: ({s}) -[{p}]-> ({o})" for i, (s, p, o) in enumerate(path)
            )
            return _MULTIHOP_PROMPT.format(
                n_hops=len(path),
                path=path_text,
            ), ""

        return None, ""

    def _parse(self, text: str) -> dict[str, Any] | None:
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
        if not isinstance(parsed, dict):
            return None
        if not parsed.get("question") or not parsed.get("answer"):
            return None
        return parsed

    def _materialize(
        self,
        entry: dict[str, Any],
        parsed: dict[str, Any],
        response: LLMResponse,
        source_attr: str,
        ts: datetime,
    ) -> DataSample:
        mode = entry["mode"]
        hops = len(entry["path"]) if mode == "multi_hop" else 1
        kg_path: list[tuple[str, str, str]]
        if mode == "atomic":
            t = entry["triple"]
            kg_path = [(t.subject, t.predicate, t.object)]
        elif mode == "aggregated":
            kg_path = [(entry["subject"], p, o) for p, o in entry["facts"]]
        else:
            kg_path = list(entry["path"])

        sample = DataSample(
            id=str(uuid.uuid4()),
            source_uri=source_attr or "kg_synthetic",
            instruction=str(parsed["question"]).strip(),
            output=str(parsed["answer"]).strip(),
            task_type="instruction_following",
            metadata={
                "generator": "MultiHopQATask",
                "mode": mode,
                "hops": hops,
                "kg_path": kg_path,
                "reasoning_steps": parsed.get("reasoning_steps", []),
            },
        )
        sample.append_provenance(
            ProvenanceRecord(
                step_name="MultiHopQATask",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=self.llm.config_hash(),
                notes={
                    "mode": mode,
                    "hops": hops,
                    "kg_path_hash": hashlib.sha256(json.dumps(kg_path).encode()).hexdigest()[:12],
                    **response.to_provenance_dict(),
                },
            )
        )
        return sample

    def _rej(self, entry: dict[str, Any], reason: str) -> RejectedSample:
        return RejectedSample(
            source_uri="kg_synthetic",
            instruction="",
            rejection_reason=reason,
            rejecting_step="MultiHopQATask",
            metadata={"mode": entry.get("mode", "")},
        )
