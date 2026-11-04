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
# Keys that name a part rather than carry text. A step that rewrites text leaves their values alone,
# or the record would break; a step that only reads (secrets, toxicity) still scans them.
_STRUCTURAL_KEYS = frozenset({"type", "role", "from"})


def _walk(label: str, root: Any, read_only: bool) -> list[tuple[str, Any, Any]]:
    """Every string reachable inside ``root`` (a dict or a list), as ``(label, parent, key)``.

    Iterative, with no depth limit, so nothing the exporters write goes unread. With ``read_only``
    the values of structural keys are returned too, and so is every dict key, as a one-off
    container (replacing it there changes nothing, so a rewriting step must not ask for them).
    """
    slots: list[tuple[str, Any, Any]] = []
    stack: list[tuple[str, Any]] = [(label, root)]
    seen: set[int] = set()
    while stack:
        path, node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        is_dict = isinstance(node, dict)
        for key, value in node.items() if is_dict else enumerate(node):
            sub = f"{path}.{key}" if is_dict else f"{path}[{key}]"
            if is_dict and read_only and isinstance(key, str):
                slots.append((f"{sub}<key>", {"key": key}, "key"))
            if isinstance(value, str):
                if is_dict and key in _STRUCTURAL_KEYS and not read_only:
                    continue
                slots.append((sub, node, key))
            elif isinstance(value, (dict, list)):
                stack.append((sub, value))
    return slots


def conversation_slots(sample: DataSample, read_only: bool = False) -> list[tuple[str, Any, Any]]:
    """Return ``(label, container, key)`` for each string of conversation text outside the main fields.

    ``container[key]`` holds the text, so a step can read it or, unless ``read_only``, replace it in
    place. A string is returned for every text key a turn carries, and for every string nested inside
    list-of-parts content. Pass ``read_only=True`` from a step that only checks text: it also gets
    structural values such as a part's ``type`` and the dict keys, which can hold text too.
    """
    slots: list[tuple[str, Any, Any]] = []
    meta = sample.metadata
    system = meta.get("system_prompt")
    if isinstance(system, str):
        slots.append(("metadata.system_prompt", meta, "system_prompt"))
    elif isinstance(system, (dict, list)):
        slots.extend(_walk("metadata.system_prompt", system, read_only))
    turns = meta.get("turns")
    if isinstance(turns, list):
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            base = f"metadata.turns[{i}]"
            if read_only:
                # Every string anywhere in the turn: role names, extra keys, nested parts.
                slots.extend(_walk(base, turn, read_only))
                continue
            for key in _TURN_TEXT_KEYS:
                value = turn.get(key)
                if isinstance(value, str):
                    slots.append((f"{base}.{key}", turn, key))
                elif isinstance(value, (dict, list)):
                    slots.extend(_walk(f"{base}.{key}", value, read_only))
    return slots
