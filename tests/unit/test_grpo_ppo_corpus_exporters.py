"""
Tests for GRPOExporter / PPOExporter / CorpusExporter's skip+warn behavior.

Before this, all three wrote a record for every sample regardless of
whether the field(s) they need were populated (GRPOExporter, PPOExporter),
or skipped silently with no visibility into how many or why
(CorpusExporter) — unlike DPOExporter, which already warned. Now all four
follow the same convention: skip a sample only when the exporter's one
truly indispensable field is empty, and warn once with a count.
"""

from __future__ import annotations

import json

import pytest

from curatorkit.exporters.corpus import CorpusExporter
from curatorkit.exporters.grpo import GRPOExporter
from curatorkit.exporters.ppo import PPOExporter
from curatorkit.schema import DataSample


class TestGRPOExporter:
    def test_exports_valid_rollout(self, tmp_path):
        samples = [
            DataSample(
                source_uri="t://",
                instruction="Explain X",
                responses=["a", "b"],
                reward_scores=[0.9, 0.4],
                task_type="grpo",
            )
        ]
        GRPOExporter().export(samples, tmp_path)
        lines = (tmp_path / "grpo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["prompt"] == "Explain X"
        assert record["responses"] == ["a", "b"]

    def test_empty_responses_and_rewards_do_not_warn(self, tmp_path, recwarn):
        """Documented, intentional case: task_type="grpo" before
        GRPORolloutTask has actually run any rollouts yet."""
        samples = [
            DataSample(source_uri="t://", instruction="Explain X", task_type="grpo")
        ]
        GRPOExporter().export(samples, tmp_path)
        assert len(recwarn) == 0
        lines = (tmp_path / "grpo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["responses"] == []

    def test_wrong_task_type_is_skipped_and_warns(self, tmp_path):
        """instruction_following has nothing to do with GRPO rollouts —
        a non-empty instruction alone must not make it "compatible"."""
        samples = [
            DataSample(
                source_uri="t://", instruction="What is 2+2?", output="4",
                task_type="instruction_following",
            )
        ]
        with pytest.warns(UserWarning, match=r"GRPOExporter skipped 1/1"):
            GRPOExporter().export(samples, tmp_path)
        lines = (tmp_path / "grpo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0

    def test_empty_prompt_is_skipped_and_warns(self, tmp_path):
        samples = [
            DataSample(source_uri="t://", output="pretrain text", task_type="language_modeling")
        ]
        with pytest.warns(UserWarning, match=r"GRPOExporter skipped 1/1"):
            GRPOExporter().export(samples, tmp_path)
        lines = (tmp_path / "grpo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0


class TestPPOExporter:
    def test_exports_valid_prompt(self, tmp_path):
        samples = [
            DataSample(source_uri="t://", instruction="Explain X", task_type="prompt_only")
        ]
        PPOExporter().export(samples, tmp_path)
        lines = (tmp_path / "ppo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"prompt": "Explain X"}

    def test_empty_instruction_is_skipped_and_warns(self, tmp_path):
        samples = [
            DataSample(source_uri="t://", output="pretrain text", task_type="language_modeling")
        ]
        with pytest.warns(UserWarning, match=r"PPOExporter skipped 1/1"):
            PPOExporter().export(samples, tmp_path)
        lines = (tmp_path / "ppo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0

    def test_wrong_task_type_is_skipped_and_warns(self, tmp_path):
        """A non-empty instruction alone doesn't make instruction_following
        data a PPO seed prompt — only prompt_only is."""
        samples = [
            DataSample(
                source_uri="t://", instruction="What is 2+2?", output="4",
                task_type="instruction_following",
            )
        ]
        with pytest.warns(UserWarning, match=r"PPOExporter skipped 1/1"):
            PPOExporter().export(samples, tmp_path)
        lines = (tmp_path / "ppo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0

    def test_no_warning_when_all_exported(self, tmp_path, recwarn):
        samples = [
            DataSample(source_uri="t://", instruction="Explain X", task_type="prompt_only")
        ]
        PPOExporter().export(samples, tmp_path)
        assert len(recwarn) == 0


class TestCorpusExporter:
    def test_exports_valid_chunk(self, tmp_path):
        samples = [
            DataSample(source_uri="t://", output="chunk text", task_type="language_modeling")
        ]
        CorpusExporter().export(samples, tmp_path)
        lines = (tmp_path / "corpus.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["text"] == "chunk text"

    def test_empty_output_and_input_is_skipped_and_warns(self, tmp_path):
        samples = [
            DataSample(
                source_uri="t://",
                instruction="q",
                chosen="good",
                rejected="bad",
                task_type="preference",
            )
        ]
        with pytest.warns(UserWarning, match=r"CorpusExporter skipped 1/1"):
            CorpusExporter().export(samples, tmp_path)
        lines = (tmp_path / "corpus.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0

    def test_wrong_task_type_is_skipped_and_warns_even_with_text(self, tmp_path):
        """Having non-empty output alone doesn't make instruction_following
        data a corpus chunk — only language_modeling/source_chunk are."""
        samples = [
            DataSample(
                source_uri="t://", instruction="q", output="a",
                task_type="instruction_following",
            )
        ]
        with pytest.warns(UserWarning, match=r"CorpusExporter skipped 1/1"):
            CorpusExporter().export(samples, tmp_path)
        lines = (tmp_path / "corpus.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0

    def test_source_chunk_task_type_is_accepted(self, tmp_path):
        samples = [
            DataSample(source_uri="t://", input="chunk from input field", task_type="source_chunk")
        ]
        CorpusExporter().export(samples, tmp_path)
        lines = (tmp_path / "corpus.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["text"] == "chunk from input field"

    def test_no_warning_when_all_exported(self, tmp_path, recwarn):
        samples = [
            DataSample(source_uri="t://", output="chunk text", task_type="language_modeling")
        ]
        CorpusExporter().export(samples, tmp_path)
        assert len(recwarn) == 0
