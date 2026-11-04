"""QA, chain-of-thought and adversarial-preference generators do not turn prose into data."""
from types import SimpleNamespace

import pytest

from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask
from curatorkit.generators.cot_generator import ChainOfThoughtTask
from curatorkit.generators.qa_generator import QAGenerationTask
from curatorkit.schema import DataSample

SAMPLE = DataSample(
    source_uri="x", instruction="Q?", input="Paris is the capital of France.",
    output="Paris", task_type="instruction_following",
)


def _qa(text):
    task = QAGenerationTask.__new__(QAGenerationTask)
    task._get_context = lambda s: "ctx"
    task._is_table_sample = lambda s: False
    task.difficulty = "easy"
    return [(o.instruction, o.output) for o in task._parse_response(SAMPLE, SimpleNamespace(text=text))]


@pytest.mark.parametrize(
    "text",
    [
        '[{"question": "Capital of France?", "answer": "Paris"}]',
        'Here: [{"question": "Capital of France?", "answer": "Paris"}] Enjoy!',
        'Sure: {"question": "Capital of France?", "answer": "Paris"}',
        "Q: Capital of France?\nA: Paris",
        "**Question 1:** Capital of France?\n**Answer 1:** Paris",
    ],
)
def test_qa_pair_is_read(text):
    assert _qa(text) == [("Capital of France?", "Paris")]


def test_qa_two_labelled_pairs_are_both_read():
    assert _qa("Q1: One?\nA1: a\nQ2: Two?\nA2: b") == [("One?", "a"), ("Two?", "b")]


def test_qa_refusal_prose_is_not_a_pair():
    assert _qa("I cannot generate questions about this topic as it contains sensitive material.") == []


def _cot(text):
    task = ChainOfThoughtTask.__new__(ChainOfThoughtTask)
    task.mode, task.cot_marker = "wrap", "\n\n## Answer\n"
    task._get_source_context = lambda s: ""
    return [o.output for o in task._parse_response(SAMPLE, SimpleNamespace(text=text))]


def test_cot_wrap_reads_json_inside_prose():
    out = _cot('Here you go: {"reasoning": "Paris is in France.", "answer": "Paris"}')
    assert out and "Paris is in France." in out[0] and "Here you go" not in out[0]


@pytest.mark.parametrize("text", ["I'm sorry, I can't help with that.", "Step 1: think. Step 2: Paris."])
def test_cot_wrap_prose_is_not_a_chain_of_thought(text):
    assert _cot(text) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ('[{"question": "Q?", "answer": "A"}]', 1),
        ('Here: [{"question": "Q?", "answer": "A"}]', 1),
        ('{"question": "Q?", "answer": "A"}', 1),
        ('["just", "strings"]', 0),
        ('"a string"', 0),
        ("no json", 0),
    ],
)
def test_adversarial_pairs_are_dicts_only(text, expected):
    assert len(AdversarialPreferenceTask._extract_pairs(text)) == expected
