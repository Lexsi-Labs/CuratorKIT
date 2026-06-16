"""
Token counting utilities.

Whitespace tokenizer is the default — fast, dependency-free, and predictable.
tiktoken is opt-in via use_tiktoken=True. Both modes must produce results
consistent enough that a threshold set with one mode works with the other for
typical English text (within ~15%).
"""

from __future__ import annotations


def count_tokens_whitespace(text: str) -> int:
    """Split on whitespace and return the count. O(n) time, zero dependencies."""
    return len(text.split())


def count_tokens_tiktoken(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens using tiktoken (opt-in). Raises ImportError if not installed."""
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError(
            "tiktoken is not installed. Install it with: pip install curatorkit[tiktoken]"
        ) from e

    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


def count_tokens(text: str, use_tiktoken: bool = False) -> int:
    """Unified entry point. Defaults to whitespace tokenizer."""
    if use_tiktoken:
        return count_tokens_tiktoken(text)
    return count_tokens_whitespace(text)
