"""
Role alias normalizer.

Applied during DataSample construction (layer 3), after format has been committed.
Converts any known role alias to the canonical "user" / "assistant" / "system" values
and rewrites "from"/"value" turn dicts to "role"/"content".
"""

from __future__ import annotations

from typing import Any

# Canonical role values
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"

# Aliases → canonical
ROLE_ALIASES: dict[str, str] = {
    # User-side
    "human": ROLE_USER,
    "user": ROLE_USER,
    "input": ROLE_USER,
    "user_input": ROLE_USER,
    "question": ROLE_USER,
    # Assistant-side
    "gpt": ROLE_ASSISTANT,
    "assistant": ROLE_ASSISTANT,
    "model": ROLE_ASSISTANT,
    "output": ROLE_ASSISTANT,
    "chatgpt": ROLE_ASSISTANT,
    "bot": ROLE_ASSISTANT,
    "ai": ROLE_ASSISTANT,
    # System-side
    "system": ROLE_SYSTEM,
    "system_prompt": ROLE_SYSTEM,
}

# Role values to preserve as-is (future tool-call support)
_PRESERVE_ROLES = {"tool", "function", "tool_response"}


def normalize_turn(turn: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """
    Normalize a single conversation turn dict.

    Handles two schemas:
      ChatML:   {"role": "user",  "content": "..."}
      ShareGPT: {"from": "human", "value":   "..."}

    Returns (normalized_turn, warning_or_None).
    The normalized form always uses "role"/"content".
    """
    if "role" in turn and "content" in turn:
        raw_role = str(turn["role"]).lower().strip()
        canonical = ROLE_ALIASES.get(raw_role)
        if canonical is not None:
            return {**turn, "role": canonical}, None
        if raw_role in _PRESERVE_ROLES:
            return turn, None
        # Unknown role — preserve and warn
        return turn, f"unknown_role:{raw_role}"

    if "from" in turn and "value" in turn:
        raw_role = str(turn["from"]).lower().strip()
        canonical = ROLE_ALIASES.get(raw_role, raw_role)
        normalized: dict[str, Any] = {"role": canonical, "content": turn["value"]}
        # Carry over any extra keys (e.g. "weight", "token_ids")
        for k, v in turn.items():
            if k not in ("from", "value"):
                normalized[k] = v
        if (
            canonical not in (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM)
            and canonical not in _PRESERVE_ROLES
        ):
            return normalized, f"unknown_role:{canonical}"
        return normalized, None

    # Unrecognized turn structure — return as-is with a warning
    return turn, f"unrecognized_turn_schema:{sorted(turn.keys())}"


def normalize_conversations(
    convs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Normalize all turns in a conversation.
    Returns (normalized_turns, warnings).
    """
    normalized: list[dict[str, Any]] = []
    warnings: list[str] = []

    for i, turn in enumerate(convs):
        norm_turn, warning = normalize_turn(turn)
        normalized.append(norm_turn)
        if warning:
            warnings.append(f"turn_{i}:{warning}")

    return normalized, warnings


def extract_system_prompt(
    convs: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    If the first turn is a system message, extract it and return the rest.
    Returns (system_prompt, remaining_turns).
    """
    if not convs:
        return "", []

    first = convs[0]
    role = first.get("role", "").lower()
    if role == "system":
        return first.get("content", ""), convs[1:]

    return "", convs


def extract_implicit_prompt(
    chosen: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract the shared prompt from implicit-prompt preference data.

    Finds the longest common prefix of turns between chosen and rejected,
    returns (prompt_turns, chosen_remainder, rejected_remainder).

    If no common prefix exists, returns ([], chosen, rejected) and the
    caller should record this as a data quality warning.
    """
    prefix_len = 0
    for c_turn, r_turn in zip(chosen, rejected):
        c_role = c_turn.get("role", c_turn.get("from", ""))
        r_role = r_turn.get("role", r_turn.get("from", ""))
        c_content = c_turn.get("content", c_turn.get("value", ""))
        r_content = r_turn.get("content", r_turn.get("value", ""))
        if c_role == r_role and c_content == r_content:
            prefix_len += 1
        else:
            break

    prompt = chosen[:prefix_len]
    c_remain = chosen[prefix_len:]
    r_remain = rejected[prefix_len:]

    return prompt, c_remain, r_remain
