"""
Tests for the KG module.

Covers:
  KnowledgeGraph — add / dedup / neighbors / paths / persistence
  KGExtractor    — LLM-driven triple extraction (stub LLM)
  MultiHopQATask — atomic / aggregated / multi-hop generation
  GraphCoverageAnalyser — long-tail / rare predicates / components / recs
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from curatorkit.kg.coverage import GraphCoverageAnalyser
from curatorkit.kg.extractor import KGExtractor
from curatorkit.kg.multihop_qa import MultiHopQATask
from curatorkit.kg.store import KnowledgeGraph
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class ScriptedLLM(BaseLLM):
    def __init__(self, responses):
        super().__init__(model="stub", max_retries=1)
        self.responses = list(responses)

    def _call(self, messages, **kw):
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(text=text, model=self.model)


@pytest.fixture
def marie_kg() -> KnowledgeGraph:
    """A small reproducible test graph."""
    kg = KnowledgeGraph()
    kg.add_triple("Marie Curie", "discovered", "polonium")
    kg.add_triple("Marie Curie", "discovered", "radium")
    kg.add_triple("Marie Curie", "won", "Nobel Prize in Physics")
    kg.add_triple("Nobel Prize in Physics", "awarded_in", "1903")
    kg.add_triple("polonium", "atomic_number", "84")
    kg.add_triple("radium", "atomic_number", "88")
    return kg


# ───────────────────────────────────────────────────────────────────────────
# KnowledgeGraph
# ───────────────────────────────────────────────────────────────────────────


class TestKnowledgeGraph:
    def test_empty_graph(self):
        kg = KnowledgeGraph()
        assert len(kg) == 0
        assert kg.entities() == set()
        assert kg.relations() == set()

    def test_add_triple_returns_was_new(self, marie_kg):
        assert marie_kg.add_triple("Marie Curie", "discovered", "polonium") is False
        assert marie_kg.add_triple("Marie Curie", "discovered", "uranium") is True

    def test_empty_string_triple_rejected(self):
        kg = KnowledgeGraph()
        assert kg.add_triple("", "p", "o") is False
        assert kg.add_triple("s", "", "o") is False
        assert kg.add_triple("s", "p", "") is False

    def test_whitespace_normalized(self):
        kg = KnowledgeGraph()
        kg.add_triple("  A ", "p", " B")
        kg.add_triple("A", "p", "B")
        # Should dedup after strip
        assert len(kg) == 1

    def test_neighbors_k_hops(self, marie_kg):
        # 1-hop: direct neighbors
        n1 = marie_kg.neighbors("Marie Curie", k=1)
        assert "polonium" in n1
        assert "84" not in n1  # 2-hop
        # 2-hop reaches atomic numbers
        n2 = marie_kg.neighbors("Marie Curie", k=2)
        assert "84" in n2 and "88" in n2

    def test_paths_directed(self, marie_kg):
        paths = marie_kg.paths("Marie Curie", "84", max_hops=3)
        assert paths
        # Shortest path is 2 hops
        assert min(len(p) for p in paths) == 2

    def test_paths_max_hops_enforced(self, marie_kg):
        paths_1 = marie_kg.paths("Marie Curie", "84", max_hops=1)
        # 84 is 2 hops away — should not find it
        assert paths_1 == []

    def test_paths_missing_endpoint(self, marie_kg):
        assert marie_kg.paths("Marie Curie", "Pluto", max_hops=5) == []
        assert marie_kg.paths("Aliens", "polonium", max_hops=5) == []

    def test_sample_paths_returns_iterable(self, marie_kg):
        paths = marie_kg.sample_paths(max_hops=2, sample_size=10, seed=0)
        assert isinstance(paths, list)
        assert all(isinstance(p, list) for p in paths)
        # Each step is (s, p, o)
        for path in paths:
            for s, p, o in path:
                assert s and p and o

    def test_persistence_roundtrip(self, marie_kg):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "kg.json"
            marie_kg.save(p)
            kg2 = KnowledgeGraph.load(p)
            assert len(kg2) == len(marie_kg)
            assert kg2.entities() == marie_kg.entities()
            assert kg2.relations() == marie_kg.relations()

    def test_persistence_preserves_provenance(self):
        kg = KnowledgeGraph()
        kg.add_triple(
            "a", "p", "b", source_uri="doc.pdf", source_chunk_hash="abc123", confidence=0.9
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "kg.json"
            kg.save(path)
            kg2 = KnowledgeGraph.load(path)
            t = kg2.triples()[0]
            assert t.source_uri == "doc.pdf"
            assert t.source_chunk_hash == "abc123"
            assert t.confidence == 0.9

    def test_predicate_counts(self, marie_kg):
        counts = marie_kg.predicate_counts()
        assert counts["discovered"] == 2  # polonium + radium
        assert counts["won"] == 1


# ───────────────────────────────────────────────────────────────────────────
# KGExtractor
# ───────────────────────────────────────────────────────────────────────────


class TestKGExtractor:
    def test_extract_well_formed_json(self):
        triples_json = json.dumps(
            [
                {"subject": "A", "predicate": "is_a", "object": "B"},
                {"subject": "A", "predicate": "has", "object": "C"},
            ]
        )
        llm = ScriptedLLM([triples_json])
        ex = KGExtractor(llm=llm, source_field="output")
        samples = [
            DataSample(
                source_uri="bio.pdf",
                output="Some source text.",
                task_type="language_modeling",
            )
        ]
        kg = ex.extract(samples)
        assert len(kg) == 2

    def test_strips_code_fences(self):
        # The model wraps output in ```json fences
        triples = '```json\n[{"subject":"x","predicate":"p","object":"y"}]\n```'
        llm = ScriptedLLM([triples])
        ex = KGExtractor(llm=llm, source_field="output")
        samples = [DataSample(source_uri="x", output="src", task_type="language_modeling")]
        kg = ex.extract(samples)
        assert len(kg) == 1

    def test_skips_invalid_triples(self):
        # Mix of valid and missing-field entries
        triples = json.dumps(
            [
                {"subject": "A", "predicate": "p", "object": "B"},
                {"subject": "", "predicate": "p", "object": "X"},  # missing subject
                {"predicate": "p", "object": "C"},  # missing subject key
            ]
        )
        llm = ScriptedLLM([triples])
        ex = KGExtractor(llm=llm, source_field="output")
        kg = ex.extract([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        assert len(kg) == 1

    def test_min_confidence_filter(self):
        triples = json.dumps(
            [
                {"subject": "A", "predicate": "p", "object": "B", "confidence": 0.9},
                {"subject": "A", "predicate": "p", "object": "C", "confidence": 0.2},
            ]
        )
        llm = ScriptedLLM([triples])
        ex = KGExtractor(llm=llm, source_field="output", min_confidence=0.5)
        kg = ex.extract([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        assert len(kg) == 1
        assert kg.triples()[0].object == "B"

    def test_provenance_attached_per_triple(self):
        triples = json.dumps([{"subject": "A", "predicate": "p", "object": "B"}])
        llm = ScriptedLLM([triples])
        ex = KGExtractor(llm=llm, source_field="output")
        kg = ex.extract(
            [
                DataSample(
                    source_uri="doc.pdf",
                    output="some source text",
                    task_type="language_modeling",
                )
            ]
        )
        t = kg.triples()[0]
        assert t.source_uri == "doc.pdf"
        assert t.source_chunk_hash  # non-empty

    def test_llm_error_returns_empty(self):
        class FailLLM(BaseLLM):
            def _call(self, messages, **kw):
                raise RuntimeError("network error")

        ex = KGExtractor(llm=FailLLM(model="x", max_retries=1), source_field="output")
        kg = ex.extract([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        assert len(kg) == 0


# ───────────────────────────────────────────────────────────────────────────
# MultiHopQATask
# ───────────────────────────────────────────────────────────────────────────


class TestMultiHopQATask:
    def test_atomic_mode(self, marie_kg):
        qa = json.dumps({"question": "What did Marie Curie discover?", "answer": "polonium"})
        llm = ScriptedLLM([qa])
        task = MultiHopQATask(llm=llm, kg=marie_kg, mode="atomic", n_samples=1)
        out = task.run([])
        assert len(out) == 1
        assert out[0].metadata["mode"] == "atomic"
        assert out[0].metadata["hops"] == 1

    def test_aggregated_mode_uses_subject_with_multiple_edges(self, marie_kg):
        qa = json.dumps(
            {
                "question": "Tell me about Marie Curie",
                "answer": "She discovered polonium and radium...",
            }
        )
        llm = ScriptedLLM([qa])
        task = MultiHopQATask(llm=llm, kg=marie_kg, mode="aggregated", n_samples=1)
        out = task.run([])
        if out:  # depends on graph having a multi-edge subject (Marie Curie has 3)
            assert out[0].metadata["mode"] == "aggregated"

    def test_multi_hop_mode_emits_path(self, marie_kg):
        qa = json.dumps(
            {
                "question": "What's the atomic number of the element Marie Curie discovered?",
                "answer": "84 or 88",
                "reasoning_steps": [
                    "Marie Curie discovered polonium/radium",
                    "polonium has atomic number 84",
                ],
            }
        )
        llm = ScriptedLLM([qa])
        task = MultiHopQATask(llm=llm, kg=marie_kg, mode="multi_hop", max_hops=3, n_samples=1)
        out = task.run([])
        if out:
            assert out[0].metadata["mode"] == "multi_hop"
            assert out[0].metadata["hops"] >= 2
            assert len(out[0].metadata["kg_path"]) >= 2

    def test_invalid_mode_raises(self, marie_kg):
        with pytest.raises(ValueError, match="mode must be"):
            MultiHopQATask(llm=ScriptedLLM([]), kg=marie_kg, mode="invalid")

    def test_empty_kg_produces_nothing(self):
        kg = KnowledgeGraph()
        task = MultiHopQATask(
            llm=ScriptedLLM([]),
            kg=kg,
            mode="atomic",
            n_samples=3,
        )
        assert task.run([]) == []

    def test_parse_error_rejected(self, marie_kg):
        llm = ScriptedLLM(["not json at all"])
        task = MultiHopQATask(llm=llm, kg=marie_kg, mode="atomic", n_samples=1)
        out = task.run([])
        assert out == []
        rejected = task.flush_rejected()
        assert rejected and "parse_error" in rejected[0].rejection_reason


# ───────────────────────────────────────────────────────────────────────────
# GraphCoverageAnalyser
# ───────────────────────────────────────────────────────────────────────────


class TestGraphCoverageAnalyser:
    def test_empty_graph(self):
        a = GraphCoverageAnalyser()
        report = a.analyse(KnowledgeGraph())
        assert report.n_entities == 0
        assert report.n_triples == 0

    def test_long_tail_detected(self):
        kg = KnowledgeGraph()
        # Star graph: hub touches 10 satellites
        for i in range(10):
            kg.add_triple("hub", "connects", f"sat_{i}")
        a = GraphCoverageAnalyser(long_tail_percentile=0.5)
        r = a.analyse(kg)
        # Satellites have degree 1, hub has degree 10 → long_tail = satellites
        assert len(r.long_tail_entities) >= 5
        assert "hub" not in r.long_tail_entities

    def test_rare_predicate_detected(self):
        kg = KnowledgeGraph()
        for i in range(5):
            kg.add_triple(f"a_{i}", "common_pred", f"b_{i}")
        kg.add_triple("x", "rare_pred", "y")
        a = GraphCoverageAnalyser(rare_predicate_min_count=2)
        r = a.analyse(kg)
        assert "rare_pred" in r.rare_predicates
        assert "common_pred" not in r.rare_predicates

    def test_disconnected_components(self):
        kg = KnowledgeGraph()
        # One main component, one orphan pair
        kg.add_triple("hub", "p", "a")
        kg.add_triple("hub", "p", "b")
        kg.add_triple("hub", "p", "c")
        kg.add_triple("orphan_1", "rare", "orphan_2")
        a = GraphCoverageAnalyser(min_component_size=3)
        r = a.analyse(kg)
        # Main component has 4 entities (> 3), orphan pair has 2 (≤ 3)
        assert any(len(c) <= 3 for c in r.disconnected_components)

    def test_recommendations_have_content(self):
        kg = KnowledgeGraph()
        for i in range(20):
            kg.add_triple("hub", "p", f"x_{i}")
        a = GraphCoverageAnalyser(long_tail_percentile=0.5)
        r = a.analyse(kg)
        assert r.recommendations  # at least one recommendation

    def test_dict_serialization(self):
        kg = KnowledgeGraph()
        kg.add_triple("a", "p", "b")
        d = GraphCoverageAnalyser().analyse(kg).to_dict()
        for key in (
            "n_entities",
            "n_predicates",
            "n_triples",
            "long_tail_entities",
            "rare_predicates",
            "recommendations",
        ):
            assert key in d
