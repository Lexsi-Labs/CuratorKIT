"""The text of a conversation that lives outside ``instruction`` and ``output``.

A conversation keeps its system prompt in ``metadata["system_prompt"]`` and every turn after the
first exchange in ``metadata["turns"]``. The exporters write both. Turns arrive in two shapes: the
generators write ShareGPT turns (``{"from", "value"}``), the connectors write normalised ones
(``{"role", "content"}``), and ``normalize_turn`` accepts either. The text itself can be a string or,
for chat data with several parts, a list of ``{"type": "text", "text": ...}`` parts (and parts that
carry a URL). A hygiene step has to read everything the exporters write, or text it was meant to
check reaches the export unseen.
"""

from __future__ import annotations

from typing import Any

from curatorkit.schema import DataSample

_TURN_TEXT_KEYS = ("content", "value")
# Keys that name a part rather than carry text; rewriting them would corrupt the record.
_STRUCTURAL_KEYS = frozenset({"type", "role", "from"})
_MAX_DEPTH = 8


def _walk(label: str, container: Any, depth: int = 0) -> list[tuple[str, Any, Any]]:
    """Every string reachable inside ``container`` (a dict or a list), as ``(label, parent, key)``."""
    slots: list[tuple[str, Any, Any]] = []
    if depth > _MAX_DEPTH:
        return slots
    items = container.items() if isinstance(container, dict) else enumerate(container)
    for key, value in items:
        if isinstance(container, dict) and key in _STRUCTURAL_KEYS:
            continue
        if isinstance(value, str):
            slots.append((f"{label}.{key}" if isinstance(container, dict) else f"{label}[{key}]", container, key))
        elif isinstance(value, (dict, list)):
            sub = f"{label}.{key}" if isinstance(container, dict) else f"{label}[{key}]"
            slots.extend(_walk(sub, value, depth + 1))
    return slots


def conversation_slots(sample: DataSample) -> list[tuple[str, Any, Any]]:
    """Return ``(label, container, key)`` for each string of conversation text outside the main fields.

    ``container[key]`` holds the text, so a step can read it or replace it in place. A string is
    returned for every text key a turn carries, and for every string nested inside list-of-parts
    content.
    """
    slots: list[tuple[str, Any, Any]] = []
    meta = sample.metadata
    system = meta.get("system_prompt")
    if isinstance(system, str):
        slots.append(("metadata.system_prompt", meta, "system_prompt"))
    elif isinstance(system, (dict, list)):
        slots.extend(_walk("metadata.system_prompt", system))
    turns = meta.get("turns")
    if isinstance(turns, list):
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            for key in _TURN_TEXT_KEYS:
                value = turn.get(key)
                if isinstance(value, str):
                    slots.append((f"metadata.turns[{i}].{key}", turn, key))
                elif isinstance(value, (dict, list)):
                    slots.extend(_walk(f"metadata.turns[{i}].{key}", value))
    return slots
