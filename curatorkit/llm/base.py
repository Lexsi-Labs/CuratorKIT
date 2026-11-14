"""
BaseLLM — abstract base class for all LLM backends.

Every generation task calls BaseLLM.generate() — not a provider-specific SDK.
Swap model by changing a single string. Subclasses implement _call() and
optionally _acall() for async support.

Design decisions:
  - generate() returns LLMResponse (text + metadata), not raw strings,
    so provenance can record token usage and model info.
  - Retry logic lives in the base class via tenacity, not in subclasses.
  - Temperature, max_tokens, and stop sequences are per-call overridable
    but default to the values set at construction time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _seconds(value: Any) -> float | None:
    """A wait in seconds from a number, "7", "1.5", an HTTP date, or "6m0s" / "250ms" (OpenAI)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    text = str(value).strip()
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text)
    if parts and "".join(n + u for n, u in parts) == text:
        return sum(float(n) * _DURATION_UNITS[u] for n, u in parts)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


def retry_after_seconds(error: BaseException) -> float | None:
    """How long the provider asked us to wait, or None when it did not say.

    Reads, in order: a `retry_after` attribute on the error (set by some SDKs), then the response
    headers the SDKs expose on `error.response` / `error.headers`: `retry-after-ms`,
    `retry-after` (seconds or an HTTP date) and, on a 429, OpenAI's `x-ratelimit-reset-requests`.
    The error's `__cause__` is checked too, since a wrapper may hold the provider error.
    """
    for err in (error, getattr(error, "__cause__", None)):
        if err is None:
            continue
        found = _seconds(getattr(err, "retry_after", None))
        if found is not None:
            return found
        headers = getattr(getattr(err, "response", None), "headers", None) or getattr(
            err, "headers", None
        )
        if not headers:
            continue
        try:
            lowered = {str(k).lower(): v for k, v in headers.items()}
        except AttributeError:
            continue
        if "retry-after-ms" in lowered and (ms := _seconds(lowered["retry-after-ms"])) is not None:
            return ms / 1000
        if (found := _seconds(lowered.get("retry-after"))) is not None:
            return found
        status = getattr(err, "status_code", None) or getattr(
            getattr(err, "response", None), "status_code", None
        )
        if status == 429 and (found := _seconds(lowered.get("x-ratelimit-reset-requests"))) is not None:
            return found
    return None


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_provenance_dict(self) -> dict[str, Any]:
        """Extract fields suitable for a ProvenanceRecord.notes entry."""
        return {
            "llm_model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": round(self.latency_seconds, 3),
        }


class BaseLLM(ABC):
    """
    Abstract base for LLM backends.

    Subclasses must implement:
      _call(messages, **kwargs) -> LLMResponse

    Subclasses may optionally implement:
      _acall(messages, **kwargs) -> LLMResponse   (for async generation)

    Parameters
    ----------
    model : str
        Model identifier string (format depends on backend).
    temperature : float
        Default temperature for generation.
    max_tokens : int
        Default maximum tokens for generation.
    api_key : str | None
        API key override. Falls back to environment variable if None.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Number of retries after the first attempt on transient failures.
        Total calls made is `max_retries + 1` — `0` means one attempt, no
        retries; the first attempt is always made regardless of this value.
    retry_after_max : float
        Longest wait, in seconds, taken from a provider's `Retry-After` (or `retry_after`
        / rate-limit reset header) before a retry. Without such a header the wait is the
        exponential backoff of 1 s, 2 s, 4 s ... (capped at 30 s) with jitter.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_after_max: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_after_max = retry_after_max

    @abstractmethod
    def _call(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous LLM call. Must be implemented by subclasses."""
        ...

    async def _acall(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Async LLM call. Default implementation wraps _call() in a thread pool.
        Override in subclasses that have native async support.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._call(messages, **kwargs))

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        """Seconds to wait before the retry that follows failed attempt number `attempt`.

        The provider's own request (`retry_after_seconds`, capped at `retry_after_max`) wins;
        without one, exponential backoff of 1 s, 2 s, 4 s ... capped at 30 s, with jitter.
        """
        asked = retry_after_seconds(error)
        if asked is not None:
            return min(asked, self.retry_after_max)
        return min(2 ** (attempt - 1), 30) * random.uniform(0.5, 1.5)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Synchronous generation with retry logic.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style message list: [{"role": "user", "content": "..."}]
        temperature : float | None
            Override default temperature for this call.
        max_tokens : int | None
            Override default max_tokens for this call.
        stop : list[str] | None
            Stop sequences.

        Returns
        -------
        LLMResponse
        """
        merged = {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs,
        }
        if stop is not None:
            merged["stop"] = stop

        # max_retries counts retries *after* the first attempt, so total calls
        # is max_retries + 1 — max_retries=0 still means "call it once."
        total_attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                t0 = time.monotonic()
                response = self._call(messages, **merged)
                response.latency_seconds = time.monotonic() - t0
                return response
            except Exception as e:
                last_error = e
                if attempt < total_attempts:
                    time.sleep(self._retry_delay(e, attempt))
                continue

        raise RuntimeError(
            f"LLM call failed after {total_attempts} attempt(s) (max_retries={self.max_retries}): {last_error}"
        ) from last_error

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Async generation with retry logic.

        Same interface as generate() but returns a coroutine.
        """
        merged = {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs,
        }
        if stop is not None:
            merged["stop"] = stop

        total_attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                t0 = time.monotonic()
                response = await self._acall(messages, **merged)
                response.latency_seconds = time.monotonic() - t0
                return response
            except Exception as e:
                last_error = e
                if attempt < total_attempts:
                    await asyncio.sleep(self._retry_delay(e, attempt))
                continue

        raise RuntimeError(
            f"Async LLM call failed after {total_attempts} attempt(s) (max_retries={self.max_retries}): {last_error}"
        ) from last_error

    def config_hash(self) -> str:
        """Hash the LLM configuration for provenance tracking."""
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, temperature={self.temperature})"
