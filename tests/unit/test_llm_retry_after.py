"""
BaseLLM waits as long as the provider asks (Retry-After) before a retry, up to `retry_after_max`.

Before, every retry waited 1 s, 2 s, 4 s (with jitter) whatever the provider said, so with the
default three retries all attempts fell inside one per-minute rate-limit window and could not
succeed. With no header the wait is unchanged.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from curatorkit.llm.base import BaseLLM, LLMResponse, retry_after_seconds


class _Resp:
    def __init__(self, headers, status_code=429):
        self.headers = headers
        self.status_code = status_code


class _ProviderError(Exception):
    def __init__(self, msg="rate limited", retry_after=None, headers=None, status_code=429):
        super().__init__(msg)
        if retry_after is not None:
            self.retry_after = retry_after
        self.status_code = status_code
        if headers is not None:
            self.response = _Resp(headers, status_code)


class _FailingLLM(BaseLLM):
    """Raises `error` for the first `fails` calls, then answers."""

    def __init__(self, error, fails=1, **kw):
        super().__init__(model="fake/ra", **kw)
        self.error, self.fails, self.calls = error, fails, 0

    def _call(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.error
        return LLMResponse(text="ok", model=self.model)


@pytest.fixture
def sleeps(monkeypatch):
    waited: list[float] = []
    monkeypatch.setattr("curatorkit.llm.base.time.sleep", waited.append)

    async def _sleep(seconds):
        waited.append(seconds)

    monkeypatch.setattr("curatorkit.llm.base.asyncio.sleep", _sleep)
    return waited


MSG = [{"role": "user", "content": "hi"}]


def test_no_header_keeps_the_exponential_backoff_with_jitter(sleeps):
    llm = _FailingLLM(_ProviderError(status_code=500), fails=3, max_retries=3)
    assert llm.generate(MSG).text == "ok"
    assert len(sleeps) == 3
    for got, base in zip(sleeps, (1, 2, 4)):
        assert 0.5 * base <= got <= 1.5 * base


def test_retry_after_attribute_is_honoured(sleeps):
    llm = _FailingLLM(_ProviderError(retry_after=7), max_retries=3)
    assert llm.generate(MSG).text == "ok"
    assert sleeps == [7.0]


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"Retry-After": "12"}, 12.0),
        ({"retry-after": "2.5"}, 2.5),
        ({"retry-after-ms": "1500", "retry-after": "30"}, 1.5),
        ({"x-ratelimit-reset-requests": "6m0s"}, 360.0),
        ({"x-ratelimit-reset-requests": "250ms"}, 0.25),
    ],
)
def test_response_headers_are_honoured(headers, expected, sleeps):
    llm = _FailingLLM(_ProviderError(headers=headers), max_retries=3, retry_after_max=1000)
    llm.generate(MSG)
    assert sleeps == [pytest.approx(expected)]


def test_retry_after_http_date():
    when = datetime.now(UTC) + timedelta(seconds=30)
    got = retry_after_seconds(_ProviderError(headers={"retry-after": format_datetime(when)}))
    assert 25 <= got <= 30


def test_reset_header_is_ignored_unless_the_status_is_429(sleeps):
    error = _ProviderError(headers={"x-ratelimit-reset-requests": "30s"}, status_code=500)
    assert retry_after_seconds(error) is None


def test_wait_is_capped_by_retry_after_max(sleeps):
    _FailingLLM(_ProviderError(retry_after=500), max_retries=3).generate(MSG)
    _FailingLLM(_ProviderError(retry_after=500), max_retries=3, retry_after_max=5).generate(MSG)
    assert sleeps == [60.0, 5.0]


def test_provider_error_behind_a_wrapper_is_found_through_cause():
    try:
        try:
            raise _ProviderError(retry_after=4)
        except _ProviderError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert retry_after_seconds(outer) == 4.0


def test_unreadable_values_fall_back_to_backoff(sleeps):
    llm = _FailingLLM(_ProviderError(headers={"retry-after": "soon"}), max_retries=3)
    llm.generate(MSG)
    assert 0.5 <= sleeps[0] <= 1.5


def test_async_generate_honours_retry_after(sleeps):
    llm = _FailingLLM(_ProviderError(retry_after=9), max_retries=3)
    assert asyncio.run(llm.agenerate(MSG)).text == "ok"
    assert sleeps == [9.0]


def test_config_reaches_the_backend():
    from curatorkit.curator import Curator, CuratorConfig, LLMOverride
    from curatorkit.llm.litellm import LiteLLMBackend
    from curatorkit.llm.ollama import OllamaBackend

    assert CuratorConfig().llm_retry_after_max == 60.0
    curator = Curator(CuratorConfig(llm_retry_after_max=12.5))
    ollama = curator._make_backend(LLMOverride(model="ollama/llama3"), 0.7, 100, 10.0, 3, True, {})
    assert isinstance(ollama, OllamaBackend) and ollama.retry_after_max == 12.5
    pytest.importorskip("litellm")
    hosted = curator._make_backend(LLMOverride(model="openai/gpt-4o-mini"), 0.7, 100, 10.0, 3, True, {})
    assert isinstance(hosted, LiteLLMBackend) and hosted.retry_after_max == 12.5


def test_yaml_llm_block_reaches_the_backend():
    from curatorkit.cli import _build_llm
    from curatorkit.config import LLMConfig, PipelineConfig

    cfg = PipelineConfig(llm=LLMConfig(model="ollama/llama3", retry_after_max=7.0))
    assert _build_llm(None, cfg, role="generation").retry_after_max == 7.0
