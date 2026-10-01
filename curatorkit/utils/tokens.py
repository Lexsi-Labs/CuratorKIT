"""
Token counting utilities.

Whitespace tokenizer is the default — fast, dependency-free, and predictable.
tiktoken is opt-in via use_tiktoken=True. Both modes must produce results
consistent enough that a threshold set with one mode works with the other for
typical English text (within ~15%).
"""

from __future__ import annotations

import re

# Scripts written without spaces between words: each character counts as one token.
# ponytail: fixed code-point ranges (Thai, Lao, Myanmar, Khmer, kana, CJK, Hangul);
# a real word segmenter or tiktoken is more accurate if thresholds need precision.
_UNSPACED_CHAR = re.compile(
    "[\u0e00-\u0eff"          # Thai, Lao
    "\u1000-\u109f"           # Myanmar
    "\u1780-\u17ff"           # Khmer
    "\u3040-\u30ff"           # Hiragana, Katakana
    "\u3400-\u4dbf"           # CJK Extension A
    "\u4e00-\u9fff"           # CJK Unified Ideographs
    "\uac00-\ud7af"           # Hangul syllables
    "\uf900-\ufaff"           # CJK Compatibility Ideographs
    "\U00020000-\U0003134f]"  # CJK Extensions B-G
)


def count_tokens_whitespace(text: str) -> int:
    """Count whitespace-separated words, counting each CJK/Thai/Khmer/etc.
    character as its own token. O(n) time, zero dependencies.
    """
    n_unspaced = len(_UNSPACED_CHAR.findall(text))
    if not n_unspaced:
        return len(text.split())
    return n_unspaced + len(_UNSPACED_CHAR.sub(" ", text).split())


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
