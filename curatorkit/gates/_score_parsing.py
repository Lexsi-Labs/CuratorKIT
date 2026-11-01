"""
Shared LLM-judge score parsing for HallucinationGate and RewardGate.

Both gates ask an LLM judge to return JSON containing a specific top-level
numeric key (`grounding_score`, `overall_score`) and parse that key directly
— there is no score *computed* by CuratorKIT itself. When a custom
`prompt_template` doesn't ask the model to produce that exact key (e.g. it
asks for per-dimension scores only, or returns prose instead of JSON), the
naive `parsed.get(key, 0.0)` silently defaults to 0.0 for every sample,
which reads as "every sample failed quality review" when the judge may have
returned perfectly good scores in the wrong shape.

`extract_score()` reads the score from the reply's JSON structure, refuses
anything else, and reports whether it had to fall back to dimensions —
callers aggregate that across a run and emit one summary warning, mirroring
the pattern already used by the SFT exporters for empty rows.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


class JudgeParseError(ValueError):
    """The judge replied, but no score could be parsed from its output."""


# One fence wrapping the whole reply. Anything else around the JSON is not JSON.
_WHOLE_FENCE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL)


def parse_judge_reply(reply: str) -> tuple[str, Any]:
    """Return (text, parsed JSON or None) for a judge reply."""
    text = reply.strip()
    fence = _WHOLE_FENCE.match(text)
    if fence:
        text = fence.group(1)
    try:
        return text, json.loads(text)
    except json.JSONDecodeError:
        return text, None


def _structured(value: Any, key: str) -> float:
    """A requested JSON key's value: a real number from 0 to 1, or JudgeParseError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeParseError(f"judge {key!r} is not a number: {value!r}")
    try:
        score = float(value)
    except OverflowError as exc:
        raise JudgeParseError(f"judge {key!r} is too large") from exc
    if not 0.0 <= score <= 1.0:  # also False for NaN
        raise JudgeParseError(f"judge {key!r} is outside 0-1: {value!r}")
    return score


def _finite(value: Any) -> float:
    """float(value), rejecting NaN and infinity: `min(1.0, nan)` clamps NaN to a perfect score."""
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"non-finite score: {value!r}")
    return score


def template_mentions_key(prompt_template: str | None, primary_key: str) -> bool:
    """Cheap static check: does a custom template's text ask for `primary_key`
    anywhere? Used to warn at gate construction time — before any LLM calls
    — when a custom prompt clearly won't produce the field the parser looks
    for, instead of only discovering that after the whole pipeline runs.
    """
    if prompt_template is None:
        return True  # built-in template always includes it
    return primary_key in prompt_template


def extract_score(
    parsed: object,
    raw_text: str,
    primary_key: str,
    dimension_keys: tuple[str, ...] = (),
) -> tuple[float, bool]:
    """Read a 0-1 score from a judge's JSON reply; scores come from JSON only.

    Returns (score, used_fallback). `used_fallback` is True whenever the primary
    key wasn't read directly, so callers can count how often a custom template's
    output didn't match the expected shape.

    The score is read from the top-level structure; string values are never scanned:
      1. `parsed[primary_key]` -- the documented contract.
      2. Otherwise the average of the `dimension_keys` present.
    Every requested key present must hold a number from 0 to 1.

    Raises `JudgeParseError` for anything else: a requested key holding a value that
    is not a finite number in range, JSON with no requested key, or a reply that is
    not JSON, which includes prose and cut-off JSON. The gates turn that into a
    `judge_error:JudgeParseError` reject (or a pass with `on_judge_error="pass"`).
    """
    if isinstance(parsed, dict):
        dimensions = [_structured(parsed[k], k) for k in dimension_keys if k in parsed]
        if primary_key in parsed:
            return _structured(parsed[primary_key], primary_key), False
        if dimensions:
            return sum(dimensions) / len(dimensions), True
    raise JudgeParseError(f"judge reply has no JSON score: {raw_text[:200]!r}")
