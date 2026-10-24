"""A scorer that is down, or replies with no score, must not silently give every rollout
a neutral 0.5 reward (all-equal rewards mean zero GRPO advantage): the group is rejected."""

from __future__ import annotations

import asyncio

import pytest

from curatorkit.generators.grpo_rollout import GRPORolloutTask, ScoringError
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class _FakeLLM(BaseLLM):
    def __init__(self, text: str = "a response", fail: bool = False, **kw):
        super().__init__(model="fake", temperature=0.7, max_tokens=100, max_retries=0, **kw)
        self._text, self._fail = text, fail

    def _call(self, messages, **kw):
        if self._fail:
            raise ConnectionError("scorer unreachable")
        return LLMResponse(text=self._text, model="fake")

    async def _acall(self, messages, **kw):
        return self._call(messages, **kw)


def _task(scorer: BaseLLM) -> GRPORolloutTask:
    return GRPORolloutTask(llm=_FakeLLM(), scoring_llm=scorer, num_responses=2)


SEED = [DataSample(source_uri="t://", instruction="2+2?")]


@pytest.mark.parametrize("mode", ["sync", "async"])
class TestScoringFailsClosed:
    def _run(self, task, mode):
        return task.run(SEED) if mode == "sync" else asyncio.run(task.run_async(SEED))

    def test_unreachable_scorer_rejects_instead_of_scoring_half(self, mode):
        task = _task(_FakeLLM(fail=True))
        assert self._run(task, mode) == []
        (rej,) = task.rejected
        assert rej.rejection_reason.startswith("scoring_failed:RuntimeError:")

    def test_unparseable_scorer_reply_rejects(self, mode):
        task = _task(_FakeLLM(text="I cannot rate this"))
        assert self._run(task, mode) == []
        (rej,) = task.rejected
        assert rej.rejection_reason.startswith("scoring_failed:JudgeParseError:")

    def test_valid_scores_still_pass_through(self, mode):
        task = _task(_FakeLLM(text='{"score": 0.8, "reasoning": "ok"}'))
        (out,) = self._run(task, mode)
        assert out.reward_scores == [0.8, 0.8]


def test_score_with_trailing_text_is_still_parsed():
    # the model wrapped valid JSON in extra text; recoverable, not a failure
    assert GRPORolloutTask._parse_score('{"score": 0.7} thanks') == 0.7


def test_parse_score_raises_scoring_error_when_nothing_numeric():
    with pytest.raises(ScoringError):
        GRPORolloutTask._parse_score("no number here")
