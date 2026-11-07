"""PreferenceGenerationTask reads a pair only from JSON or from labels that start a line."""
from types import SimpleNamespace

import pytest

from curatorkit.generators.preference_gen import PreferenceGenerationTask
from curatorkit.schema import DataSample


def _parse(text):
    task = PreferenceGenerationTask.__new__(PreferenceGenerationTask)
    task.mode = "quality_degradation"
    task._is_corpus_mode = lambda s: False
    task._get_source_context = lambda s: ""
    sample = DataSample(source_uri="x", instruction="What is 2+2?", output="4", task_type="sft")
    return [(o.chosen, o.rejected) for o in task._parse_response(sample, SimpleNamespace(text=text))]


@pytest.mark.parametrize(
    "text",
    [
        '{"chosen": "4", "rejected": "5"}',
        '```json\n{"chosen": "4", "rejected": "5"}\n```',
        'Sure! Here you go: {"chosen": "4", "rejected": "5"} Hope it helps.',
        "Chosen: 4\nRejected: 5",
        "**Chosen:** 4\n**Rejected:** 5",
    ],
)
def test_a_pair_is_read(text):
    assert _parse(text) == [("4", "5")]


@pytest.mark.parametrize(
    "text",
    [
        '{"chosen": "4", "rejected": "4"}',
        '{"chosen": "Four", "rejected": "  four "}',
        '{"chosen": "", "rejected": ""}',
        "I cannot write a worse answer on purpose; a better use is to ask something else.",
        '[{"chosen": "4", "rejected": "5"}]',
    ],
)
def test_text_that_is_not_a_distinct_pair_is_rejected(text):
    assert _parse(text) == []
