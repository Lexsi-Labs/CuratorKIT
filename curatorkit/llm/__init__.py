"""
LLM abstraction layer for CuratorKIT.

Provides a unified interface to 100+ LLM providers via LiteLLM,
plus a dedicated Ollama backend for local models.
"""

from curatorkit.llm.base import BaseLLM, LLMResponse

__all__ = ["BaseLLM", "LLMResponse"]
