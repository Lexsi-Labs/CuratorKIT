"""
Regression tests for BaseLLM.generate()/agenerate() retry-count handling.

Before this fix, `max_retries` was used directly as the total attempt count
via `range(1, self.max_retries + 1)` — with `max_retries=0` this range is
empty, so the LLM was never called at all, yet the code still raised
"LLM call failed after 0 retries: None" as if a real failure had occurred.
`max_retries` now means "retries after the first attempt": total calls made
is `max_retries + 1`, so `0` means one attempt (no retries) and `1` means
one initial attempt plus one retry (two calls total).
"""

from __future__ import annotations

import asyncio

import pytest

from curatorkit.llm.base import BaseLLM, LLMResponse


class _FailNTimesLLM(BaseLLM):
    """Fails its first `fail_count` calls, then succeeds."""

    def __init__(self, fail_count: int = 0, **kwargs):
        super().__init__(model="fake/retry-test", **kwargs)
        self.fail_count = fail_count
        self.calls = 0

    def _call(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError(f"transient failure #{self.calls}")
        return LLMResponse(text="ok", model=self.model)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Backoff sleeps are real time.sleep/asyncio.sleep calls — skip them so
    multi-attempt tests don't actually wait."""
    monkeypatch.setattr("curatorkit.llm.base.time.sleep", lambda *_a, **_k: None)

    async def _fast_sleep(*_a, **_k):
        return None

    monkeypatch.setattr("curatorkit.llm.base.asyncio.sleep", _fast_sleep)


class TestSyncGenerateMaxRetriesFloor:
    def test_max_retries_zero_still_calls_once_and_succeeds(self):
        llm = _FailNTimesLLM(fail_count=0, max_retries=0)
        response = llm.generate([{"role": "user", "content": "hi"}])
        assert response.text == "ok"
        assert llm.calls == 1

    def test_max_retries_zero_with_failure_reports_one_attempt(self):
        llm = _FailNTimesLLM(fail_count=99, max_retries=0)
        with pytest.raises(RuntimeError, match=r"failed after 1 attempt\(s\)"):
            llm.generate([{"role": "user", "content": "hi"}])
        assert llm.calls == 1

    def test_max_retries_one_makes_two_calls(self):
        """max_retries=1 means one retry after the first attempt: 2 calls total."""
        llm = _FailNTimesLLM(fail_count=99, max_retries=1)
        with pytest.raises(RuntimeError, match=r"failed after 2 attempt\(s\)"):
            llm.generate([{"role": "user", "content": "hi"}])
        assert llm.calls == 2

    def test_max_retries_three_recovers_on_second_attempt(self):
        llm = _FailNTimesLLM(fail_count=1, max_retries=3)
        response = llm.generate([{"role": "user", "content": "hi"}])
        assert response.text == "ok"
        assert llm.calls == 2

    def test_max_retries_three_exhausts_after_four_calls(self):
        """3 retries + the first attempt = 4 total calls."""
        llm = _FailNTimesLLM(fail_count=99, max_retries=3)
        with pytest.raises(RuntimeError, match=r"failed after 4 attempt\(s\)"):
            llm.generate([{"role": "user", "content": "hi"}])
        assert llm.calls == 4


class TestAsyncGenerateMaxRetriesFloor:
    def test_max_retries_zero_still_calls_once_and_succeeds(self):
        llm = _FailNTimesLLM(fail_count=0, max_retries=0)
        response = asyncio.run(llm.agenerate([{"role": "user", "content": "hi"}]))
        assert response.text == "ok"
        assert llm.calls == 1

    def test_max_retries_zero_with_failure_reports_one_attempt(self):
        llm = _FailNTimesLLM(fail_count=99, max_retries=0)
        with pytest.raises(RuntimeError, match=r"failed after 1 attempt\(s\)"):
            asyncio.run(llm.agenerate([{"role": "user", "content": "hi"}]))
        assert llm.calls == 1

    def test_max_retries_three_recovers_on_second_attempt(self):
        llm = _FailNTimesLLM(fail_count=1, max_retries=3)
        response = asyncio.run(llm.agenerate([{"role": "user", "content": "hi"}]))
        assert response.text == "ok"
        assert llm.calls == 2
