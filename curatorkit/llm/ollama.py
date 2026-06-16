"""
OllamaBackend — direct HTTP client for local Ollama models.

No API key required. Ollama must be running on the host.
Uses the /api/chat endpoint for OpenAI-compatible message format.

Usage:
    from curatorkit.llm.ollama import OllamaBackend

    llm = OllamaBackend(model="llama3")
    response = llm.generate([{"role": "user", "content": "Hello"}])
    print(response.text)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from curatorkit.llm.base import BaseLLM, LLMResponse


class OllamaBackend(BaseLLM):
    """
    LLM backend for local Ollama models via HTTP.

    Parameters
    ----------
    model : str
        Ollama model name, e.g. "llama3", "mistral", "phi3".
    base_url : str
        Ollama server URL. Defaults to http://localhost:11434.
    temperature : float
        Default temperature.
    max_tokens : int
        Default max tokens (maps to num_predict in Ollama).
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Retries on transient failures.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=None,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.base_url = base_url.rstrip("/")

    def _call(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if "stop" in kwargs:
            payload["options"]["stop"] = kwargs.pop("stop")

        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}"
            ) from e

        message = body.get("message", {})
        text = message.get("content", "")

        # Ollama response includes eval_count (completion tokens) and
        # prompt_eval_count (prompt tokens) when available
        prompt_tokens = body.get("prompt_eval_count", 0)
        completion_tokens = body.get("eval_count", 0)

        return LLMResponse(
            text=text,
            model=body.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            metadata={
                "done": body.get("done", True),
                "total_duration_ns": body.get("total_duration", 0),
            },
        )
