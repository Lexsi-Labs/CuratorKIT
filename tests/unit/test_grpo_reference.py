"""The gold answer on an input row must survive GRPORolloutTask and the GRPO export,
so a verifiable-reward trainer can score the exported rollouts."""

from __future__ import annotations

import asyncio
import json

from curatorkit.exporters.grpo import GRPOExporter
from curatorkit.generators.grpo_rollout import GRPORolloutTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class _FakeLLM(BaseLLM):
    def _call(self, messages, **kw):
        return LLMResponse(text="a response", model="fake")


def _task() -> GRPORolloutTask:
    llm = _FakeLLM(model="fake", temperature=0.7, max_tokens=100)
    return GRPORolloutTask(llm=llm, num_responses=2, score_responses=False)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestRolloutCarriesReference:
    def test_sync_run_keeps_gold_answer(self):
        seed = DataSample(source_uri="t://", instruction="2+2?", output="4")
        (rollout,) = _task().run([seed])
        assert rollout.output == "4"

    def test_async_run_keeps_gold_answer(self):
        seed = DataSample(source_uri="t://", instruction="2+2?", output="4")
        (rollout,) = asyncio.run(_task().run_async([seed]))
        assert rollout.output == "4"

    def test_corpus_chunk_text_is_not_a_reference(self):
        chunk = DataSample(source_uri="t://", output="long passage", task_type="language_modeling")
        (rollout,) = _task().run([chunk])
        assert rollout.output == ""


class TestExporterWritesReference:
    def test_reference_column_written_when_present(self, tmp_path):
        samples = [
            DataSample(source_uri="t://", instruction="2+2?", output="4",
                       responses=["4", "5"], reward_scores=[1.0, 0.0], task_type="grpo"),
            DataSample(source_uri="t://", instruction="3*5?", output="15",
                       responses=["15", "8"], reward_scores=[1.0, 0.0], task_type="grpo"),
        ]
        GRPOExporter().export(samples, tmp_path)
        assert [r["reference"] for r in _rows(tmp_path / "grpo.jsonl")] == ["4", "15"]

    def test_every_row_has_the_column_if_any_row_has_it(self, tmp_path):
        samples = [
            DataSample(source_uri="t://", instruction="2+2?", output="4", task_type="grpo"),
            DataSample(source_uri="t://", instruction="open question", task_type="grpo"),
        ]
        GRPOExporter().export(samples, tmp_path)
        assert [r["reference"] for r in _rows(tmp_path / "grpo.jsonl")] == ["4", ""]

    def test_no_reference_column_when_no_row_has_one(self, tmp_path):
        samples = [DataSample(source_uri="t://", instruction="open question", task_type="grpo")]
        GRPOExporter().export(samples, tmp_path)
        assert "reference" not in _rows(tmp_path / "grpo.jsonl")[0]
