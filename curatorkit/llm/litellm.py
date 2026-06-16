"""
LiteLLMBackend — unified LLM interface via LiteLLM.

Wraps litellm.completion() to support OpenAI, Anthropic, Gemini, Ollama,
vLLM, AWS Bedrock, Azure, Cohere, Mistral, and 100+ other providers
through a single interface.

Install: pip install curatorkit[generation]

Usage:
    from curatorkit.llm.litellm import LiteLLMBackend

    llm = LiteLLMBackend(model="openai/gpt-4o-mini")
    response = llm.generate([{"role": "user", "content": "Hello"}])
    print(response.text)
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from curatorkit.llm.base import BaseLLM, LLMResponse

# Captures the content inside <think>...</think> so we can fall back to it
# when the model puts its answer entirely inside the thinking block.
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _ensure_litellm() -> Any:
    """Import litellm with a helpful error if not installed."""
    try:
        import litellm

        return litellm
    except ImportError as e:
        raise ImportError(
            "litellm is not installed. Install it with: pip install curatorkit[generation]"
        ) from e


class LiteLLMBackend(BaseLLM):
    """
    LLM backend using LiteLLM for provider-agnostic access.

    Parameters
    ----------
    model : str
        LiteLLM model string, e.g.:
          "openai/gpt-4o-mini"
          "anthropic/claude-sonnet-4-20250514"
          "gemini/gemini-1.5-flash"
          "ollama/llama3"
          "bedrock/anthropic.claude-3-sonnet"
    temperature : float
        Default temperature.
    max_tokens : int
        Default max tokens.
    api_key : str | None
        API key override. If None, litellm reads from the standard
        environment variable for each provider (OPENAI_API_KEY,
        ANTHROPIC_API_KEY, etc.).
    api_base : str | None
        Custom API base URL (useful for vLLM, Ollama, etc.).
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Retries on transient failures.
    drop_params : bool
        If True, litellm silently drops unsupported params for each provider.
    extra_body : dict | None
        Extra fields merged into every request body. Use this to pass
        provider-specific parameters that litellm doesn't expose natively.
        Common use-case: per-request thinking control on vLLM/SGLang —
          {"chat_template_kwargs": {"enable_thinking": False}}
        This takes precedence over the server's --default-chat-template-kwargs,
        so generator and judge can have different thinking modes even when
        they share the same endpoint.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        drop_params: bool = True,
        extra_body: dict | None = None,
    ) -> None:
        # Fail fast at construction: without this, a missing litellm install
        # surfaces only as per-sample generation_failed rejections mid-run.
        _ensure_litellm()
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.api_base = api_base
        self.drop_params = drop_params
        self.extra_body = extra_body or {}
        self._thinking_warned = False  # emit the stripping warning at most once

    def _call(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        litellm = _ensure_litellm()

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "timeout": self.timeout,
            "drop_params": self.drop_params,
        }

        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        elif self.api_base:
            # Local endpoints (Ollama, vLLM) don't need a real key, but the
            # OpenAI client errors if api_key is completely absent.
            call_kwargs["api_key"] = "nokey"
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.extra_body:
            call_kwargs["extra_body"] = self.extra_body

        # Pass through any extra kwargs (stop, response_format, etc.)
        call_kwargs.update(kwargs)

        response = litellm.completion(**call_kwargs)

        choice = response.choices[0]
        text = self._postprocess(choice.message.content or "")
        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=text,
            model=getattr(response, "model", self.model) or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            metadata={
                "finish_reason": getattr(choice, "finish_reason", None),
                "provider": self.model.split("/")[0] if "/" in self.model else "unknown",
            },
        )

    async def _acall(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Native async using litellm.acompletion()."""
        litellm = _ensure_litellm()

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "timeout": self.timeout,
            "drop_params": self.drop_params,
        }

        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        elif self.api_base:
            call_kwargs["api_key"] = "nokey"
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.extra_body:
            call_kwargs["extra_body"] = self.extra_body

        call_kwargs.update(kwargs)

        response = await litellm.acompletion(**call_kwargs)

        choice = response.choices[0]
        text = self._postprocess(choice.message.content or "")
        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=text,
            model=getattr(response, "model", self.model) or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            metadata={
                "finish_reason": getattr(choice, "finish_reason", None),
                "provider": self.model.split("/")[0] if "/" in self.model else "unknown",
            },
        )

    def _postprocess(self, text: str) -> str:
        """Strip thinking tokens; fall back to thinking content when response is empty.

        Some reasoning models put their structured answer (e.g. JSON) entirely
        inside the <think> block and emit nothing after it.  Stripping the block
        would leave an empty string and cause downstream JSON parsers to return
        fallback scores.  When that happens, we use the thinking content itself
        as the response — it is model-agnostic and requires no prompt changes.
        """
        m = _THINK_RE.search(text)
        clean = _THINK_RE.sub("", text, count=1).lstrip()

        if m and not self._thinking_warned:
            warnings.warn(
                f"[CuratorKIT] Thinking tokens detected in response from '{self.model}' "
                f"and stripped automatically. To suppress thinking at the source, either "
                f"serve with --default-chat-template-kwargs '{{\"enable_thinking\": false}}' "
                f'or set llm_extra_body={{"chat_template_kwargs": {{"enable_thinking": False}}}} '
                f"in CuratorConfig.",
                stacklevel=3,
            )
            self._thinking_warned = True

        if not clean.strip() and m:
            # Answer was entirely inside the thinking block — use it as fallback.
            return m.group(1).strip()

        return clean
