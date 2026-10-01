"""
Tests for coverage-directed generation, the DeepEval judge and PDF PII scrub:

  - CoverageDirectedGenerator     (curatorkit/coverage/directed.py)
  - DeepEvalJudge / DeepEvalRewardGate  (curatorkit/judges/deepeval.py)
  - PDFChunker pii_scrub          (curatorkit/connectors/pdf.py)
"""

from __future__ import annotations

import json
import warnings

import pytest

from curatorkit.connectors.pdf import PDFChunker
from curatorkit.coverage.analyser import CoverageReport
from curatorkit.coverage.directed import (
    CoverageDirectedGenerator,
    _load_diversity_stats_from_manifest,
    _thin_clusters_from_stats,
)
from curatorkit.generators.qa_generator import QAGenerationTask
from curatorkit.judges import DeepEvalJudge, DeepEvalRewardGate
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class ScriptedLLM(BaseLLM):
    def __init__(self, responses):
        super().__init__(model="stub", max_retries=1)
        self.responses = list(responses)

    def _call(self, messages, **kw):
        text = self.responses.pop(0) if self.responses else "ok"
        return LLMResponse(text=text, model=self.model)


# ───────────────────────────────────────────────────────────────────────────
# CoverageDirectedGenerator
# ───────────────────────────────────────────────────────────────────────────


class TestCoverageDirectedGenerator:
    @pytest.fixture(autouse=True)
    def _silence(self):
        warnings.simplefilter("ignore")

    def test_passthrough_with_factor_1(self):
        """oversample_factor=1.0 means no duplication regardless of cluster_id."""
        report = CoverageReport(n_samples=10, k_clusters=8, thin_clusters=[2])
        inner = QAGenerationTask(
            llm=ScriptedLLM([json.dumps([{"question": "Q", "answer": "A"}])] * 3),
            num_questions=1,
        )
        wrapper = CoverageDirectedGenerator(
            inner=inner,
            coverage_report=report,
            oversample_factor=1.0,
        )
        samples = [
            DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 2}),
        ]
        out = wrapper.run(samples)
        assert len(out) == 1  # no oversample

    def test_oversamples_thin_cluster_inputs(self):
        report = CoverageReport(n_samples=10, k_clusters=8, thin_clusters=[2, 5])
        # ScriptedLLM provides enough JSON for up to 10 expansions
        inner = QAGenerationTask(
            llm=ScriptedLLM([json.dumps([{"question": "Q", "answer": "A"}])] * 20),
            num_questions=1,
        )
        wrapper = CoverageDirectedGenerator(
            inner=inner,
            coverage_report=report,
            oversample_factor=3.0,
        )
        samples = [
            DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 2}),
            DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 5}),
            DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 9}),
        ]
        out = wrapper.run(samples)
        # 2 thin-cluster samples × (3.0 oversample = 2 extras each) + 3 originals = 7
        assert len(out) == 7

    def test_non_thin_samples_left_alone(self):
        report = CoverageReport(n_samples=10, k_clusters=8, thin_clusters=[2])
        inner = QAGenerationTask(
            llm=ScriptedLLM([json.dumps([{"question": "Q", "answer": "A"}])] * 5),
            num_questions=1,
        )
        wrapper = CoverageDirectedGenerator(
            inner=inner,
            coverage_report=report,
            oversample_factor=2.0,
        )
        samples = [
            DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 9}),
            DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 9}),
        ]
        out = wrapper.run(samples)
        assert len(out) == 2  # no oversample, neither is thin

    def test_prompt_enrichment_injects_hint(self):
        report = CoverageReport(n_samples=10, k_clusters=8, thin_clusters=[2])
        report.recommendations = ["Add more samples for cluster 2"]
        inner = QAGenerationTask(
            llm=ScriptedLLM([json.dumps([{"question": "Q", "answer": "A"}])]),
            num_questions=1,
        )
        wrapper = CoverageDirectedGenerator(
            inner=inner,
            coverage_report=report,
            oversample_factor=1.0,
            enrich_prompts=True,
        )
        samples = [DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 2})]
        wrapper.run(samples)
        # coverage_hint was injected into the input sample's metadata
        assert "coverage_hint" in samples[0].metadata
        assert "cluster" in samples[0].metadata["coverage_hint"].lower()

    def test_provenance_stamped(self):
        report = CoverageReport(n_samples=10, k_clusters=8, thin_clusters=[2])
        inner = QAGenerationTask(
            llm=ScriptedLLM([json.dumps([{"question": "Q", "answer": "A"}])]),
            num_questions=1,
        )
        wrapper = CoverageDirectedGenerator(
            inner=inner,
            coverage_report=report,
            oversample_factor=1.0,
        )
        out = wrapper.run(
            [
                DataSample(source_uri="x", input="ctx", metadata={"cluster_id": 2}),
            ]
        )
        steps = [r.step_name for r in out[0].provenance_chain]
        assert "CoverageDirectedGenerator" in steps

    def test_load_from_manifest(self, tmp_path):
        manifest = {
            "diversity_stats": {
                "thin_clusters": [1, 4],
                "recommendations": ["targeted hint"],
            }
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest))
        loaded = _load_diversity_stats_from_manifest(p)
        assert loaded is not None
        assert _thin_clusters_from_stats(loaded) == [1, 4]

    def test_missing_manifest_path_returns_none(self, tmp_path):
        result = _load_diversity_stats_from_manifest(tmp_path / "nope.json")
        assert result is None

    def test_cannot_pass_both_report_and_manifest(self):
        with pytest.raises(ValueError, match="not both"):
            CoverageDirectedGenerator(
                inner=QAGenerationTask(llm=ScriptedLLM([]), num_questions=1),
                coverage_report=CoverageReport(n_samples=0, k_clusters=0),
                manifest_path="anywhere",
            )

    def test_rebalance_task_types(self):
        # Build a report claiming "code" is over-represented; oversample
        # "qa" task_type samples (they are under-represented)
        report = CoverageReport(n_samples=0, k_clusters=0)
        report.task_type_distribution = {"code": 80, "qa": 10, "creative": 10}
        inner = QAGenerationTask(
            llm=ScriptedLLM([json.dumps([{"question": "Q", "answer": "A"}])] * 10),
            num_questions=1,
        )
        wrapper = CoverageDirectedGenerator(
            inner=inner,
            coverage_report=report,
            oversample_factor=2.0,
            rebalance_task_types=True,
        )
        samples = [
            DataSample(source_uri="x", input="ctx", task_type="code"),
            DataSample(source_uri="x", input="ctx", task_type="qa"),
            DataSample(source_uri="x", input="ctx", task_type="creative"),
        ]
        out = wrapper.run(samples)
        # "qa" and "creative" are below the mean (≈33); both get oversampled
        # by ×2 → 2 extras → 5 total
        assert len(out) == 5


# ───────────────────────────────────────────────────────────────────────────
# DeepEval judge
# ───────────────────────────────────────────────────────────────────────────


class TestDeepEvalJudge:
    def test_canonical_geval_format_parsed(self):
        llm = ScriptedLLM(["Rationale: Good answer.\nScore: 8"])
        judge = DeepEvalJudge(
            llm=llm,
            criterion="helpfulness",
            evaluation_steps=["s1", "s2", "s3"],
        )
        r = judge.judge("Q?", "A")
        assert r.score_0_to_10 == 8.0
        assert r.score_normalized == 0.8
        assert "Good answer" in r.rationale

    def test_picks_last_score(self):
        """G-Eval models often discuss low/high before stating final score."""
        llm = ScriptedLLM(
            [
                "I considered scores between 3 and 9 but Score: 7",
            ]
        )
        judge = DeepEvalJudge(llm=llm, criterion="x", evaluation_steps=["s"])
        r = judge.judge("Q", "A")
        assert r.score_0_to_10 == 7.0

    def test_clamp_to_0_10(self):
        llm = ScriptedLLM(["Score: 15"])
        judge = DeepEvalJudge(llm=llm, criterion="x", evaluation_steps=["s"])
        r = judge.judge("Q", "A")
        assert 0 <= r.score_0_to_10 <= 10

    def test_auto_generates_evaluation_steps(self):
        """When evaluation_steps not provided, judge generates them on first use."""
        llm = ScriptedLLM(
            [
                "1. Read the response\n2. Check facts\n3. Score it",  # steps generation
                "Rationale: meh\nScore: 5",  # actual scoring
            ]
        )
        judge = DeepEvalJudge(llm=llm, criterion="helpfulness")
        result = judge.judge("Q", "A")
        steps = judge.evaluation_steps()
        assert len(steps) >= 3
        assert result.score_0_to_10 == 5.0

    def test_default_score_for_garbage_output(self):
        llm = ScriptedLLM(["nothing parseable"])
        judge = DeepEvalJudge(llm=llm, criterion="x", evaluation_steps=["s"])
        r = judge.judge("Q", "A")
        assert 0 <= r.score_0_to_10 <= 10  # default neutral

    def test_context_included_when_use_context_true(self):
        llm = ScriptedLLM(["Rationale: ok\nScore: 6"])
        judge = DeepEvalJudge(
            llm=llm,
            criterion="groundedness",
            evaluation_steps=["s"],
            use_context=True,
        )
        # We can't easily inspect the prompt the LLM saw, but the judge
        # should not error when context is passed.
        r = judge.judge("Q?", "A", context="Some source text.")
        assert r.score_0_to_10 == 6.0


class TestDeepEvalRewardGate:
    @pytest.fixture(autouse=True)
    def _silence(self):
        warnings.simplefilter("ignore")

    def test_pass_and_reject(self):
        llm = ScriptedLLM(
            [
                "Rationale: Great.\nScore: 9",
                "Rationale: Poor.\nScore: 3",
            ]
        )
        judge = DeepEvalJudge(llm=llm, criterion="helpfulness", evaluation_steps=["s"])
        gate = DeepEvalRewardGate(judge=judge, threshold=0.6)
        passed, rejected = gate.run(
            [
                DataSample(source_uri="x", instruction="q1", output="excellent"),
                DataSample(source_uri="x", instruction="q2", output="bad"),
            ]
        )
        assert len(passed) == 1 and len(rejected) == 1
        assert passed[0].label == 0.9
        assert "deepeval_below_threshold" in rejected[0].rejection_reason

    def test_passes_through_no_response(self):
        judge = DeepEvalJudge(llm=ScriptedLLM([]), criterion="x", evaluation_steps=["s"])
        gate = DeepEvalRewardGate(judge=judge)
        s = DataSample(source_uri="x", instruction="q", output="")
        passed, rejected = gate.run([s])
        assert len(passed) == 1 and not rejected


# ───────────────────────────────────────────────────────────────────────────
# PDFChunker PII scrub
# ───────────────────────────────────────────────────────────────────────────


class TestPDFChunkerPIIScrub:
    @pytest.fixture(autouse=True)
    def _silence(self):
        warnings.simplefilter("ignore")

    def test_off_passthrough(self):
        c = PDFChunker(pii_scrub="off")
        text, counts, drop = c._scrub_chunk_text("email me at a@b.com")
        assert text == "email me at a@b.com"
        assert counts == {}
        assert drop is False

    def test_redact_replaces_inline(self):
        c = PDFChunker(pii_scrub="redact")
        text, counts, drop = c._scrub_chunk_text("Phone: 555-123-4567")
        assert "<PHONE_NUMBER>" in text
        assert counts["PHONE_NUMBER"] == 1
        assert drop is False

    def test_drop_signals_drop_flag(self):
        c = PDFChunker(pii_scrub="drop")
        _, counts, drop = c._scrub_chunk_text("SSN: 123-45-6789")
        assert drop is True
        assert "US_SSN" in counts

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="pii_scrub"):
            PDFChunker(pii_scrub="bogus")

    def test_entities_filter(self):
        # Restrict to EMAIL only — phone should pass through unscrubbed
        c = PDFChunker(pii_scrub="redact", pii_entities=["EMAIL_ADDRESS"])
        text, counts, _ = c._scrub_chunk_text("Phone: 555-123-4567")
        assert text == "Phone: 555-123-4567"
        assert counts == {}
