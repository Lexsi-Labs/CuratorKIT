"""
Shared regex-based text masking for deduplication.

`ignore_for_dedup` lets a user specify regex patterns for substrings that
should not influence whether two samples are treated as duplicates (e.g. a
per-sample UUID, a timestamp, a boilerplate disclaimer) — without it, a real
but unimportant difference between two otherwise-identical samples prevents
ExactDeduplicator/MinHashDeduplicator/EmbeddingDeduplicator from recognizing
them as duplicates.

Masking never touches the actual DataSample fields — it only transforms a
throwaway string built for comparison (or, for EmbeddingDeduplicator,
embedding), so source grounding for downstream generation tasks and quality
gates is unaffected.
"""

from __future__ import annotations

import re


def compile_ignore_patterns(patterns: list[str] | None) -> list[re.Pattern]:
    """Compile regex patterns eagerly so a typo raises ValueError at
    construction time, not silently (or with a cryptic re.error) mid-run."""
    if not patterns:
        return []
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error as e:
            raise ValueError(f"ignore_for_dedup: invalid regex {p!r}: {e}") from e
    return compiled


def apply_ignore_patterns(text: str, patterns: list[re.Pattern]) -> str:
    """Replace every pattern's matches with a single space, then collapse
    whitespace. Replacing (not deleting) avoids merging two words together
    across a masked span (e.g. stripping "42" from "item42end" bare would
    produce "itemend"). A no-op when there are no patterns, so default
    (unmasked) behavior — including exact whitespace — is unchanged.
    """
    if not patterns or not text:
        return text
    for pattern in patterns:
        text = pattern.sub(" ", text)
    return " ".join(text.split())
