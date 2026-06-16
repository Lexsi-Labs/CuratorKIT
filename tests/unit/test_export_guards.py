"""Guards against silent failure modes at the pipeline edges.

1. SFT exporters warn when the samples don't carry SFT-shaped content
   (e.g. preference data exported as Alpaca produces empty rows).
2. LiteLLMBackend fails fast at construction when litellm is missing,
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


class TestExporterMismatchWarning:
    def test_alpaca_warns_on_empty_sft_fields(self, tmp_path):
        samples = [_preference_sample(i) for i in range(3)]
        with pytest.warns(UserWarning, match="doesn't match the Alpaca SFT format"):
            AlpacaExporter().export(samples, tmp_path)
        # rows are still written — the warning is loud, not destructive
        lines = (tmp_path / "sft_alpaca.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_alpaca_silent_on_matching_data(self, tmp_path):
        samples = [_sft_sample(i) for i in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AlpacaExporter().export(samples, tmp_path)
        lines = [json.loads(line) for line in open(tmp_path / "sft_alpaca.jsonl")]
        assert all(line["instruction"] and line["output"] for line in lines)

    def test_sharegpt_warns_on_empty_sft_fields(self, tmp_path):
        samples = [_preference_sample(i) for i in range(2)]
        with pytest.warns(UserWarning, match="empty"):
            ShareGPTExporter().export(samples, tmp_path)

    def test_sharegpt_silent_on_matching_data(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ShareGPTExporter().export([_sft_sample(0)], tmp_path)


class TestLiteLLMFailFast:
    @pytest.mark.skipif(
        importlib.util.find_spec("litellm") is not None,
        reason="litellm installed — the fail-fast path doesn't apply",
    )
    def test_constructor_raises_helpful_importerror(self):
        from curatorkit.llm.litellm import LiteLLMBackend

        with pytest.raises(ImportError, match="curatorkit\\[generation\\]"):
            LiteLLMBackend(model="openai/gpt-4o-mini")
