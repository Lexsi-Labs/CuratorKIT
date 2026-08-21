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

`extract_score()` tries progressively looser strategies before giving up,
and reports whether it had to fall back — callers aggregate that across a
run and emit one summary warning, mirroring the pattern already used by the
SFT exporters for empty rows.
"""

from __future__ import annotations

import re


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
    """Extract a 0-1 score from a judge response, trying looser strategies.

    Returns (score, used_fallback) — `used_fallback` is True whenever the
    primary key wasn't found directly, so callers can count how often a
    custom template's output didn't match the expected shape.

    Strategies, in order:
      1. `parsed[primary_key]` — the documented contract.
      2. Average of any `dimension_keys` present as top-level numeric values
         in `parsed` (covers a custom template that returns per-dimension
         scores flat, e.g. `{"truthfulness": 0.9, "creativity": 0.7}`).
      3. Any numeric value found in `raw_text` via regex (covers prose
         responses, or JSON that parsed but contains no recognizable key) —
         scaled down from a 0-10 scale if the number is > 1.0.
      4. `0.5` — a neutral default when nothing numeric can be found at all.
    """
    if isinstance(parsed, dict) and primary_key in parsed:
        try:
            return max(0.0, min(1.0, float(parsed[primary_key]))), False
        except (TypeError, ValueError):
            pass

    if isinstance(parsed, dict) and dimension_keys:
        found = []
        for dim in dimension_keys:
            if dim in parsed:
                try:
                    found.append(float(parsed[dim]))
                except (TypeError, ValueError):
                    continue
        if found:
            avg = sum(found) / len(found)
            return max(0.0, min(1.0, avg)), True

    match = re.search(r"(\d+\.?\d*)", raw_text)
    if match:
        score = float(match.group(1))
        if score > 1.0:
            score = score / 10.0
        return max(0.0, min(1.0, score)), True

    return 0.5, True
