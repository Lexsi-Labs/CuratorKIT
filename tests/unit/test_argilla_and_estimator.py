"""
Tests for ArgillaExporter (fallback path) and DryRunCostEstimator.

ArgillaExporter is tested via its local-JSONL fallback so the test suite
doesn't require a running Argilla server. The push path is exercised in
the integration suite if Argilla is available.
"""

from __future__ import annotations

import json
import warnings

import pytest

from curatorkit.cost.estimator import (
    DryRunCostEstimator,
    _approx_tokens,
)
from curatorkit.diagnostic.failure_modes import FailureDiagnosis, FailureMode
from curatorkit.exporters.argilla import (
    ArgillaExporter,
    _sample_to_record_dict,
)
from curatorkit.gates.hallucination import HallucinationGate
from curatorkit.gates.schema import SchemaGate
from curatorkit.generators.joint_bundle import JointBundleTask
from curatorkit.generators.magpie import MagpieTask
from curatorkit.generators.qa_generator import QAGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, RejectedSample


class StubLLM(BaseLLM):
    def _call(self, messages, **kw):
        return LLMResponse(text="ok", model=self.model)


# ───────────────────────────────────────────────────────────────────────────
# ArgillaExporter — fallback path
# ───────────────────────────────────────────────────────────────────────────


class TestArgillaExporter:
    @pytest.fixture(autouse=True)
    def _silence(self):
        warnings.simplefilter("ignore")

    def test_writes_local_jsonl_when_no_api_url(self, tmp_path):
        exp = ArgillaExporter(api_url="", api_key="")
        samples = [DataSample(source_uri="x", instruction="q", output="a")]
        exp.export(samples, tmp_path)
        path = tmp_path / "argilla_records.jsonl"
        assert path.exists()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["fields"]["instruction"] == "q"

    def test_includes_rejected_samples(self, tmp_path):
        rej = RejectedSample(
            source_uri="x",
            instruction="q",
            output="bad",
            rejection_reason="hallucination_contract_failed:0.30",
            rejecting_step="HallucinationGate",
        )
        # Assigned, not passed to the constructor — the same way the pipeline
        # attaches a probe verdict to a rejection.
        rej.diagnosis = FailureDiagnosis(
            mode=FailureMode.GENERATOR_TEMPERATURE, evidence=[False, True], probe_calls=2
        )
        exp = ArgillaExporter(
            api_url="",
            api_key="",
            rejected_samples=[rej],
        )
        exp.export([], tmp_path)
        records = [
            json.loads(line)
            for line in (tmp_path / "argilla_records.jsonl").read_text().splitlines()
        ]
        assert len(records) == 1
        r = records[0]
        assert r["metadata"]["record_type"] == "rejected"
        assert r["metadata"]["failure_mode"] == "generator_temperature"
        assert json.loads(r["metadata"]["diagnosis"])["probe_calls"] == 2

    def test_record_dict_carries_bundle_id(self):
        s = DataSample(
            source_uri="x",
            instruction="q",
            chosen="good",
            rejected="bad",
            task_type="preference",
            metadata={"bundle_id": "bundle-abc", "role_in_bundle": "dpo"},
        )
        r = _sample_to_record_dict(s, record_type="passed")
        assert r["metadata"]["bundle_id"] == "bundle-abc"
        assert r["metadata"]["role_in_bundle"] == "dpo"
        assert r["fields"]["chosen"] == "good"
        assert r["fields"]["rejected"] == "bad"

    def test_empty_field_fallback(self):
        # Argilla requires at least one field; if every text field is empty,
        # the record dict must still be valid.
        s = DataSample(source_uri="x")
        r = _sample_to_record_dict(s, record_type="passed")
        assert r["fields"]  # non-empty
        assert "instruction" in r["fields"]

    def test_include_rejected_false(self, tmp_path):
        rej = RejectedSample(
            source_uri="x",
            instruction="q",
            rejection_reason="test",
            rejecting_step="X",
        )
        exp = ArgillaExporter(
            api_url="",
            api_key="",
            include_rejected=False,
            rejected_samples=[rej],
        )
        exp.export([], tmp_path)
        records = [
            json.loads(line)
            for line in (tmp_path / "argilla_records.jsonl").read_text().splitlines()
        ]
        assert len(records) == 0


# ───────────────────────────────────────────────────────────────────────────
# DryRunCostEstimator
# ───────────────────────────────────────────────────────────────────────────


class TestDryRunCostEstimator:
    def test_approx_tokens_handles_empty(self):
        assert _approx_tokens("") == 0
        assert _approx_tokens("hello world") >= 1

    def test_no_llm_steps_yields_empty_estimate(self):
        est = DryRunCostEstimator(n_input_samples=100).estimate([SchemaGate()])
        assert est.total_tokens == 0
        assert est.per_stage == []

    def test_single_generator(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        steps = [QAGenerationTask(llm=llm, num_questions=3)]
        e = DryRunCostEstimator(n_input_samples=100).estimate(steps)
        assert len(e.per_stage) == 1
        s = e.per_stage[0]
        assert s.stage == "QAGenerationTask"
        assert s.calls == 100  # one per sample
        assert s.total_tokens > 0

    def test_magpie_counts_two_calls_per_sample(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        e = DryRunCostEstimator(n_input_samples=50).estimate(
            [
                MagpieTask(llm=llm, n_samples=50),
            ]
        )
        assert e.per_stage[0].calls == 100  # 50 × 2

    def test_joint_bundle_with_rollout_scoring(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        e = DryRunCostEstimator(n_input_samples=20).estimate(
            [
                JointBundleTask(llm=llm, num_rollouts=4, score_rollouts=True),
            ]
        )
        # 1 bundle call + 4 score calls per sample = 5 × 20 = 100
        assert e.per_stage[0].calls == 100

    def test_joint_bundle_without_rollout_scoring(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        e = DryRunCostEstimator(n_input_samples=20).estimate(
            [
                JointBundleTask(llm=llm, num_rollouts=4, score_rollouts=False),
            ]
        )
        assert e.per_stage[0].calls == 20

    def test_gate_attenuation(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        steps = [
            QAGenerationTask(llm=llm, num_questions=3),
            HallucinationGate(llm=llm),
        ]
        e = DryRunCostEstimator(
            n_input_samples=100,
            gate_pass_rate=1.0,
        ).estimate(steps)
        # Without attenuation, both stages see 100 input samples
        assert e.per_stage[0].calls == 100
        assert e.per_stage[1].calls == 100

        # With 0.5 pass rate after the gate, second LLM step would be at 50
        e2 = DryRunCostEstimator(
            n_input_samples=100,
            gate_pass_rate=0.5,
        ).estimate(steps + [HallucinationGate(llm=llm)])
        # First gate runs at full 100, second gate at attenuated 50
        gate_calls = [s.calls for s in e2.per_stage if s.stage == "HallucinationGate"]
        assert gate_calls == [100, 50]

    def test_adaptive_loop_adds_probe_stage(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        e = DryRunCostEstimator(n_input_samples=100).estimate(
            [HallucinationGate(llm=llm)],
            adaptive_loop=True,
        )
        stages = [s.stage for s in e.per_stage]
        assert any("DiagnosticProbe" in s for s in stages)

    def test_adaptive_loop_off_no_probe_stage(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        e = DryRunCostEstimator(n_input_samples=100).estimate(
            [HallucinationGate(llm=llm)],
            adaptive_loop=False,
        )
        assert not any("DiagnosticProbe" in s.stage for s in e.per_stage)

    def test_markdown_rendering(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        e = DryRunCostEstimator(n_input_samples=10).estimate(
            [
                QAGenerationTask(llm=llm, num_questions=3),
            ]
        )
        md = e.render_markdown()
        assert "# Pre-flight Cost Estimate" in md
        assert "Per-stage breakdown" in md
        assert "QAGenerationTask" in md

    def test_to_dict_serializable(self):
        llm = StubLLM(model="openai/gpt-4o-mini", max_retries=1)
        e = DryRunCostEstimator(n_input_samples=10).estimate(
            [
                QAGenerationTask(llm=llm, num_questions=3),
            ]
        )
        d = e.to_dict()
        # Round-trip through json.dumps to ensure it's serializable
        json.dumps(d)
        assert d["total_tokens"] > 0
        assert "per_stage" in d
        assert "warnings" in d
