"""The text of a conversation that lives outside ``instruction`` and ``output``.

A conversation keeps its system prompt in ``metadata["system_prompt"]`` and every turn after the
first exchange in ``metadata["turns"]``. The exporters write both. Turns arrive in two shapes: the
generators write ShareGPT turns (``{"from", "value"}``), the connectors write normalised ones
(``{"role", "content"}``), and ``normalize_turn`` accepts either. A hygiene step has to read the
same keys the exporters do, or text it was meant to check reaches the export unseen.
"""

from __future__ import annotations

from curatorkit.schema import DataSample

_TURN_TEXT_KEYS = ("content", "value")


def conversation_slots(sample: DataSample) -> list[tuple[str, dict, str]]:
    """Return ``(label, container, key)`` for each string of conversation text outside the main fields.

    ``container[key]`` holds the text, so a step can read it or replace it in place. Every text key a
    turn carries is returned, not only the one a given exporter prefers.
    """
    slots: list[tuple[str, dict, str]] = []
    meta = sample.metadata
    if isinstance(meta.get("system_prompt"), str):
        slots.append(("metadata.system_prompt", meta, "system_prompt"))
    turns = meta.get("turns")
    if isinstance(turns, list):
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            for key in _TURN_TEXT_KEYS:
                if isinstance(turn.get(key), str):
                    slots.append((f"metadata.turns[{i}].{key}", turn, key))
    return slots
