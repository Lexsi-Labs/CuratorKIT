"""
Cross-module integration tests for the v0.4.0 generators, judges and KG.

Verifies that the new generators / judges / KG components compose correctly
with the existing pipeline machinery (Pipeline runner, Curator wiring,
exporters).
"""

from __future__ import annotations

import json
import warnings

import pytest

from curatorkit.diagnostic.failure_modes import FailureDiagnosis, FailureMode
from curatorkit.exporters.argilla import ArgillaExporter
from curatorkit.generators.joint_bundle import JointBundleTask
from curatorkit.generators.magpie import MagpieTask
from curatorkit.generators.persona import PersonaDrivenTask
from curatorkit.judges import (
    ABSOLUTE_GRADING_RUBRICS,
    PrometheusJudge,
    PrometheusRewardGate,
)
from curatorkit.kg.extractor import KGExtractor
from curatorkit.kg.multihop_qa import MultiHopQATask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.pipeline import Pipeline
from curatorkit.schema import DataSample


class ScriptedLLM(BaseLLM):
    def __init__(self, responses, name="stub"):
        super().__init__(model=name, max_retries=1)
        self.responses = list(responses)

    def _call(self, messages, **kw):
        text = self.responses.pop(0) if self.responses else "ok"
        return LLMResponse(text=text, model=self.model)


class TestIntegrationNew:
    @pytest.fixture(autouse=True)
    def _silence(self):
        warnings.simplefilter("ignore")

    def test_magpie_then_prometheus_in_pipeline(self):
        """Magpie cold-start feeds into Prometheus reward filtering."""
        # Magpie: 1 instruction + 1 response per sample, 2 samples
        # Prometheus: 1 judgment per sample
        magpie_llm = ScriptedLLM(
            [
                "Write a fibonacci function.",
                "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
                "Reverse a string.",
                "return s[::-1]",
            ],
            name="generator",
        )
        prom_llm = ScriptedLLM(
            [
                "Feedback: Excellent code. [RESULT] 5",
                "Feedback: Adequate. [RESULT] 3",
            ],
            name="prometheus-7b-v2.0",
        )

        magpie = MagpieTask(llm=magpie_llm, n_samples=2)
        judge = PrometheusJudge(
            llm=prom_llm,
            rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"],
        )
        gate = PrometheusRewardGate(judge=judge, threshold=0.75)

        pipeline = Pipeline(steps=[magpie, gate])
        result = pipeline.run()
        # 2 generated. Only the score-5 (label=1.0) clears 0.75 threshold;
        # the score-3 (label=0.5) is rejected.
        assert len(result.passed) == 1
        assert len(result.rejected) == 1
        assert "prometheus_below_threshold" in result.rejected[0].rejection_reason

    def test_joint_bundle_emits_correct_distribution(self):
        """One JointBundle pass yields exactly 1 SFT + 1 DPO + 1 GRPO."""
        bundle = json.dumps(
            {
                "instruction": "What is photosynthesis?",
                "chosen": "Plants convert light to glucose.",
                "rejected": "Plants eat sunlight.",
                "rollouts": ["A", "B", "C", "D"],
            }
        )
        llm = ScriptedLLM(
            [bundle] + ["0.9", "0.7", "0.5", "0.3"],
            name="bundle",
        )
        task = JointBundleTask(llm=llm, num_rollouts=4)
        src = DataSample(
            source_uri="bio.pdf", output="Plants use light.", task_type="language_modeling"
        )
        result = task.run([src])
        by_role = {s.metadata["role_in_bundle"]: s for s in result}
        assert set(by_role.keys()) == {"sft", "dpo", "grpo"}

        # All three share one bundle_id
        bundle_ids = {s.metadata["bundle_id"] for s in result}
        assert len(bundle_ids) == 1

        # GRPO rewards reflect the scoring pass
        assert by_role["grpo"].reward_scores == [0.9, 0.7, 0.5, 0.3]

    def test_persona_metadata_field_routes_correctly(self):
        """Persona-driven generator reads from metadata.persona."""
        llm = ScriptedLLM(
            [
                "User prompt: Solve fizzbuzz",
                "for i in range(1, 16): ...",
            ]
        )
        task = PersonaDrivenTask(
            llm=llm,
            style="instruction",
            persona_field="metadata.persona",
        )
        s = DataSample(
            source_uri="x",
            instruction="ignored",
            metadata={"persona": "a junior Python developer"},
        )
        result = task.run([s])
        assert len(result) == 1
        assert "fizzbuzz" in result[0].instruction.lower()

    def test_kg_workflow_extract_to_qa(self):
        """KG extractor builds graph; MultiHopQATask generates QA from it."""
        triples_json = json.dumps(
            [
                {"subject": "Marie Curie", "predicate": "discovered", "object": "polonium"},
                {"subject": "polonium", "predicate": "atomic_number", "object": "84"},
            ]
        )
        extract_llm = ScriptedLLM([triples_json])
        extractor = KGExtractor(llm=extract_llm, source_field="output")
        samples = [
            DataSample(
                source_uri="curie.pdf",
                output="Marie Curie discovered polonium.",
                task_type="language_modeling",
            )
        ]
        kg = extractor.extract(samples)
        assert len(kg) == 2
        # Provenance carried through
        assert all(t.source_uri == "curie.pdf" for t in kg.triples())

        qa_json = json.dumps(
            {
                "question": "What did Marie Curie discover?",
                "answer": "polonium",
            }
        )
        qa_llm = ScriptedLLM([qa_json])
        task = MultiHopQATask(llm=qa_llm, kg=kg, mode="atomic", n_samples=1)
        qa = task.run([])
        assert len(qa) == 1
        # kg_path stamped into metadata for citation audit
        assert qa[0].metadata["kg_path"]
        assert qa[0].metadata["mode"] == "atomic"

    def test_argilla_exporter_with_diagnostic_rejected(self, tmp_path):
        """ArgillaExporter carries the probe's FailureDiagnosis for review."""
        from curatorkit.schema import RejectedSample

        rej = RejectedSample(
            source_uri="x",
            instruction="q",
            rejection_reason="hallucination_contract_failed:0.30",
            rejecting_step="HallucinationGate",
        )
        rej.diagnosis = FailureDiagnosis(mode=FailureMode.GENERATOR_TEMPERATURE)
        exp = ArgillaExporter(api_url="", api_key="", rejected_samples=[rej])
        exp.export([], tmp_path)
        records = [
            json.loads(line)
            for line in (tmp_path / "argilla_records.jsonl").read_text().splitlines()
        ]
        assert records[0]["metadata"]["failure_mode"] == "generator_temperature"
        assert "diagnosis" in records[0]["metadata"]
