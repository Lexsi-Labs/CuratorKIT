"""A multi-turn reply must be a user-first, strictly alternating conversation that ends on the assistant."""
import json
from types import SimpleNamespace

import pytest

from curatorkit.generators.multiturn_gen import MultiTurnTask
from curatorkit.schema import DataSample


def _task():
    task = MultiTurnTask.__new__(MultiTurnTask)
    task.mode, task.num_turns, task.include_context = "single_call", 3, True
    task._get_source_context = lambda s: ""
    return task


def _turn(role, content):
    return {"role": role, "content": content}


def _parse(turns):
    sample = DataSample(source_uri="x", instruction="Q?", task_type="prompt_only")
    return _task()._parse_response(sample, SimpleNamespace(text=json.dumps({"turns": turns})))


def test_a_valid_conversation_is_kept():
    out = _parse([_turn("user", "Q1"), _turn("assistant", "A1"), _turn("user", "Q2"), _turn("assistant", "A2")])
    assert len(out) == 1 and out[0].instruction == "Q1" and out[0].output == "A1"


def test_a_dangling_user_question_is_dropped_not_the_conversation():
    out = _parse([_turn("user", "Q1"), _turn("assistant", "A1"), _turn("user", "Q2")])
    assert len(out) == 1 and out[0].metadata["total_turns"] == 2


@pytest.mark.parametrize(
    "turns",
    [
        [_turn("assistant", "Hi"), _turn("user", "Q1"), _turn("assistant", "A1")],
        [_turn("system", "be nice"), _turn("user", "Q1"), _turn("assistant", "A1")],
        [_turn("user", "Q1"), _turn("user", "Q2"), _turn("assistant", "A1")],
        [_turn("user", "Q1"), _turn("assistant", "")],
        [_turn("user", "Q1"), _turn("bot", "A1")],
        [_turn("user", "Q1"), {"role": "assistant", "content": {"x": 1}}],
        [_turn("user", "Q1"), _turn("assistant", "A1"), _turn("assistant", "A1b")],
    ],
)
def test_a_malformed_conversation_is_rejected(turns):
    assert _parse(turns) == []
