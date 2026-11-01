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

`extract_score()` reads JSON from its structure and prose from one stated
score, refuses anything else, and reports whether it had to fall back —
callers aggregate that across a run and emit one summary warning, mirroring
the pattern already used by the SFT exporters for empty rows.
"""

from __future__ import annotations

import math
import re
from typing import Any


class JudgeParseError(ValueError):
    """The judge replied, but no score could be parsed from its output."""


# One number as a whole token: not the 5 of -5, +5, /5, 10,5 or 1e999.
_NUMBER = r"(?<![\w.,+\-−/$])(\d+(?:\.\d+)?)"
_OVER_TEN = re.compile(rf"{_NUMBER}\s*(?:/|out of)\s*10(?![\w%]|[.,]\d|\s*/)", re.IGNORECASE)
_BARE = re.compile(rf"\bscore\s*:\s*{_NUMBER}(?=[.,;!?)]?(?:\s*$|\s*\n))", re.IGNORECASE | re.MULTILINE)
# A fraction over anything but 10, or a percentage, anywhere in the reply.
_OTHER_SCALE = re.compile(
    r"\d\s*(?:/|\bof\b|\bout of\b)\s*(?!10(?!\d|\.\d))\d|\d\s*(?:%|percent\b)", re.IGNORECASE
)
_QUOTED = re.compile(r""""([^"]*)"|“([^”]*)”|`([^`]*)`|(?<!\w)'([^']*)'(?!\w)""")


def _statements(text: str) -> list[float]:
    """Every score statement in `text`, on a 0-1 scale.

    `N/10` and `N out of 10` always divide by 10. `score: N` needs the rule stated
    here: above 1 up to 10 is out of 10, a decimal from 0 to 1 is already 0-1, and
    a whole `1` could be either, so it is refused. Above 10 is on neither scale.
    """
    scores = []
    for match in _OVER_TEN.finditer(text):
        if float(match.group(1)) > 10.0:
            raise JudgeParseError(f"judge score above 10: {match.group(0)!r}")
        scores.append(float(match.group(1)) / 10.0)
    for match in _BARE.finditer(text):
        number = float(match.group(1))
        if number > 10.0 or match.group(1) == "1":
            raise JudgeParseError(f"judge score has no clear scale: {match.group(0)!r}")
        scores.append(number if number <= 1.0 else number / 10.0)
    return scores


def _prose_score(raw_text: str, keys: tuple[str, ...]) -> float:
    """The one score a non-JSON reply states, or JudgeParseError. Odd replies fail closed."""
    if raw_text.lstrip()[:1] in ("{", "["):
        raise JudgeParseError(f"judge reply is broken JSON: {raw_text[:200]!r}")
    for key in keys:
        if re.search(rf"\b{re.escape(key)}\b", raw_text, re.IGNORECASE):
            raise JudgeParseError(f"judge named {key!r} outside JSON: {raw_text[:200]!r}")
    if _OTHER_SCALE.search(raw_text):
        raise JudgeParseError(f"judge number has another scale: {raw_text[:200]!r}")
    if raw_text.count('"') % 2 or any(
        _statements(next(g for g in quoted.groups() if g is not None))
        for quoted in _QUOTED.finditer(raw_text)
    ):
        raise JudgeParseError(f"judge score is inside quoted text: {raw_text[:200]!r}")
    scores = _statements(raw_text)
    # A `score: N` that no statement took has a unit or trailing text: unclear, so refused.
    if re.search(r"\bscore\s*:\s*\S*\d", _BARE.sub("", _OVER_TEN.sub("", raw_text)), re.IGNORECASE):
        raise JudgeParseError(f"judge score is unclear: {raw_text[:200]!r}")
    if not scores:
        raise JudgeParseError(f"unparseable judge output: {raw_text[:200]!r}")
    if max(scores) - min(scores) > 1e-9:
        raise JudgeParseError(f"judge stated different scores: {raw_text[:200]!r}")
    return scores[0]


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
    """Extract a 0-1 score from a judge response.

    Returns (score, used_fallback). `used_fallback` is True whenever the primary
    key wasn't read directly, so callers can count how often a custom template's
    output didn't match the expected shape.

    A reply that parsed as JSON is read from its structure only; its string values
    are never scanned:
      1. `parsed[primary_key]` -- the documented contract.
      2. Otherwise the average of the `dimension_keys` present at the top level.
    Every requested key present must hold a number from 0 to 1.

    A reply that is not JSON must state one score: `N/10`, `N out of 10` or
    `score: N` (see `_statements`). It may repeat that score.

    Raises `JudgeParseError` for anything else: a requested key holding a value
    that is not a finite number in range, JSON with no requested key, broken JSON,
    prose naming a requested key, a number on another scale, a score inside quoted
    text, differing scores, or no score. The gates turn that into a
    `judge_error:JudgeParseError` reject (or a pass with `on_judge_error="pass"`).
    """
    if parsed is not None:
        if isinstance(parsed, dict):
            dimensions = [_structured(parsed[k], k) for k in dimension_keys if k in parsed]
            if primary_key in parsed:
                return _structured(parsed[primary_key], primary_key), False
            if dimensions:
                return sum(dimensions) / len(dimensions), True
        raise JudgeParseError(f"judge JSON has no requested key: {raw_text[:200]!r}")
    return _prose_score(raw_text, (primary_key, *dimension_keys)), True
