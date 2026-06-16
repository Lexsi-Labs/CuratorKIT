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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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
        Number of retries on transient failures.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

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

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                response = self._call(messages, **merged)
                response.latency_seconds = time.monotonic() - t0
                return response
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s...
                    time.sleep(min(2 ** (attempt - 1), 30) * random.uniform(0.5, 1.5))
                continue

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_error}"
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

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                response = await self._acall(messages, **merged)
                response.latency_seconds = time.monotonic() - t0
                return response
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 30) * random.uniform(0.5, 1.5))
                continue

        raise RuntimeError(
            f"Async LLM call failed after {self.max_retries} retries: {last_error}"
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
