"""Guards against silent failure modes at the pipeline edges.

1. SFT exporters skip samples whose task_type isn't alpaca/sharegpt-compatible
   (e.g. preference data has no place in an Alpaca/ShareGPT file — it's
   skipped and warned about, not written with an empty instruction/output).
2. Within a compatible task_type, a sample that still ends up with an empty
   instruction/output (e.g. a malformed row) is written but warned about —
   the shape is right, the content isn't.
3. LiteLLMBackend fails fast at construction when litellm is missing,
   instead of rejecting every sample mid-run.
"""

from __future__ import annotations

import importlib.util
import json
import warnings

import pytest

from curatorkit.exporters.alpaca import AlpacaExporter
from curatorkit.exporters.sharegpt import ShareGPTExporter
from curatorkit.schema import DataSample


def _sft_sample(i: int) -> DataSample:
    return DataSample(
        instruction=f"Question {i} about optimisers and learning-rate schedules?",
        output=f"Answer {i}: warmup then cosine decay is a common default.",
        task_type="instruction_following",
        source_uri="test://export-guards",
    )


def _preference_sample(i: int) -> DataSample:
    return DataSample(
        instruction="",
        output="",
        chosen=f"Chosen answer {i}",
        rejected=f"Rejected answer {i}",
        task_type="preference",
        source_uri="test://export-guards",
    )


def _malformed_sft_sample(i: int) -> DataSample:
    """Correct task_type, but a genuinely empty output — e.g. a QA
    generation task that failed to produce an answer for this row."""
    return DataSample(
        instruction=f"Question {i}?",
        output="",
        task_type="instruction_following",
        source_uri="test://export-guards",
    )


class TestExporterTaskTypeMismatch:
    """preference data has no place in an SFT-shaped export — skipped and
    warned about, not silently written with empty fields."""

    def test_alpaca_skips_preference_data(self, tmp_path):
        samples = [_preference_sample(i) for i in range(3)]
        with pytest.warns(UserWarning, match="skipped 3/3"):
            AlpacaExporter().export(samples, tmp_path)
        lines = (tmp_path / "sft_alpaca.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0

    def test_alpaca_silent_on_matching_data(self, tmp_path):
        samples = [_sft_sample(i) for i in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AlpacaExporter().export(samples, tmp_path)
        lines = [json.loads(line) for line in open(tmp_path / "sft_alpaca.jsonl")]
        assert all(line["instruction"] and line["output"] for line in lines)

    def test_sharegpt_skips_preference_data(self, tmp_path):
        samples = [_preference_sample(i) for i in range(2)]
        with pytest.warns(UserWarning, match="skipped 2/2"):
            ShareGPTExporter().export(samples, tmp_path)
        lines = (tmp_path / "sft_sharegpt.jsonl").read_text().strip().splitlines()
        assert len(lines) == 0

    def test_sharegpt_silent_on_matching_data(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ShareGPTExporter().export([_sft_sample(0)], tmp_path)


class TestExporterEmptyFieldWithinCompatibleType:
    """task_type is right, but the content is still missing — the row is
    still written (it's a real, if broken, SFT-shaped sample), just warned
    about, since discarding it silently could hide a real generation bug."""

    def test_alpaca_warns_on_empty_output(self, tmp_path):
        samples = [_malformed_sft_sample(i) for i in range(3)]
        with pytest.warns(UserWarning, match="empty"):
            AlpacaExporter().export(samples, tmp_path)
        lines = (tmp_path / "sft_alpaca.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_sharegpt_warns_on_empty_output(self, tmp_path):
        samples = [_malformed_sft_sample(i) for i in range(2)]
        with pytest.warns(UserWarning, match="empty"):
            ShareGPTExporter().export(samples, tmp_path)
        lines = (tmp_path / "sft_sharegpt.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2


class TestLiteLLMFailFast:
    @pytest.mark.skipif(
        importlib.util.find_spec("litellm") is not None,
        reason="litellm installed — the fail-fast path doesn't apply",
    )
    def test_constructor_raises_helpful_importerror(self):
        from curatorkit.llm.litellm import LiteLLMBackend

        with pytest.raises(ImportError, match="curatorkit\\[generation\\]"):
            LiteLLMBackend(model="openai/gpt-4o-mini")
