"""
FormatDetector — three-layer dataset format detection engine.

Layer 1: Column-set candidate generation using semantic equivalence classes.
Layer 2: Value-type validation per candidate.
Layer 3: Role alias normalization (applied during sample construction, not here).

Design:
  - Layer 1 generates ALL matching candidates, ranked by confidence.
  - Layer 2 validates each candidate in rank order; first passing candidate wins.
  - Unknown format after all candidates fail → emit RejectedSample upstream.
  - Confidence tiers drive how pipeline reports and handles ambiguous inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DataFormat(str, Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    PREFERENCE = "preference"
    IMPLICIT_PREFERENCE = "implicit_preference"
    UNPAIRED_PREFERENCE = "unpaired_preference"
    GRPO = "grpo"
    PROMPT_ONLY = "prompt_only"
    PRETRAIN = "pretrain"
    UNKNOWN = "unknown"


class DetectionConfidence(str, Enum):
    HIGH = "high"  # All expected columns present, no ambiguity
    MEDIUM = "medium"  # All core columns present, ancillary missing or extra
    LOW = "low"  # Partial match — best guess only
    UNKNOWN = "unknown"  # No candidate survived layer 2


# ---------------------------------------------------------------------------
# Semantic equivalence classes
# These map arbitrary real-world column names onto canonical slot names.
# ---------------------------------------------------------------------------

# "input" is listed last because it also appears in CONTEXT_COLS — specificity wins.
INSTRUCTION_COLS: frozenset[str] = frozenset(
    {
        "instruction",
        "prompt",
        "query",
        "question",
        "user_input",
        "user_query",
        "user_message",
        "human",
        "input",
    }
)
_INSTRUCTION_PRIORITY: tuple[str, ...] = (
    "instruction",
    "prompt",
    "query",
    "question",
    "user_input",
    "user_query",
    "user_message",
    "human",
    "input",
)

OUTPUT_COLS: frozenset[str] = frozenset(
    {
        "output",
        "response",
        "completion",
        "answer",
        "assistant",
        "gpt",
        "model",
        "model_response",
        "target",
        "label_text",
    }
)

CONTEXT_COLS: frozenset[str] = frozenset(
    {
        "input",
        "context",
        "background",
        "passage",
        "article",
        "document",
        "text_input",
    }
)

CONVERSATION_COLS: frozenset[str] = frozenset(
    {
        "conversations",
        "messages",
        "chat",
        "dialogue",
        "dialog",
        "turns",
        "history",
    }
)

CHOSEN_COLS: frozenset[str] = frozenset(
    {
        "chosen",
        "preferred",
        "accepted",
        "response_a",
        "response_0",
        "positive",
    }
)

REJECTED_COLS: frozenset[str] = frozenset(
    {
        "rejected",
        "refused",
        "dispreferred",
        "response_b",
        "response_1",
        "negative",
    }
)

LABEL_COLS: frozenset[str] = frozenset(
    {
        "label",
        "score",
        "rating",
        "reward",
        "quality",
        "preference",
        "rank",
        "chosen_score",
        "human_preference",
    }
)

RESPONSE_LIST_COLS: frozenset[str] = frozenset(
    {
        "responses",
        "completions",
        "candidates",
        "rollouts",
        "samples",
        "outputs",
    }
)

REWARD_LIST_COLS: frozenset[str] = frozenset(
    {
        "rewards",
        "scores",
        "reward_scores",
        "values",
        "returns",
    }
)

TEXT_COLS: frozenset[str] = frozenset(
    {
        "text",
        "content",
        "body",
        "passage",
    }
)

SYSTEM_COLS: frozenset[str] = frozenset(
    {
        "system",
        "system_prompt",
        "system_message",
        "sys_prompt",
    }
)


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """
    The resolved detection output.

    column_map maps semantic slot names to the actual column names
    found in the dataset. A None value means the slot is absent.

    Semantic slots:
      instruction, output, context, conversation,
      chosen, rejected, label, system,
      responses, rewards, text
    """

    format: DataFormat
    confidence: DetectionConfidence
    column_map: dict[str, str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    unrecognized_cols: list[str] = field(default_factory=list)

    def get(self, slot: str) -> str | None:
        return self.column_map.get(slot)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_match(
    keys: set[str],
    equivalence_class: frozenset[str],
    priority: tuple[str, ...] | None = None,
) -> str | None:
    """Return the first key in `keys` that belongs to the equivalence class.

    If `priority` is given, candidates are tested in that order first so that
    ambiguous column names (e.g. 'input' in both INSTRUCTION_COLS and
    CONTEXT_COLS) resolve predictably regardless of frozenset hash ordering.
    """
    if priority:
        for k in priority:
            if k in keys and k in equivalence_class:
                return k
        return None
    for k in equivalence_class:
        if k in keys:
            return k
    return None


def _any_match(keys: set[str], equivalence_class: frozenset[str]) -> bool:
    return bool(keys & equivalence_class)


def _is_conversational_value(value: Any) -> bool:
    """
    Return True if the value looks like a list of turn dicts.
    Accepts both role/content (ChatML) and from/value (ShareGPT).
    """
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if not isinstance(first, dict):
        return False
    return ("role" in first and "content" in first) or ("from" in first and "value" in first)


def _is_list_of_strings(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(v, str) for v in value)


def _is_list_of_numbers(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(v, (int, float)) for v in value)


# ---------------------------------------------------------------------------
# Layer 1 — candidate generation
# ---------------------------------------------------------------------------


def _generate_candidates(
    keys: set[str],
) -> list[tuple[DataFormat, DetectionConfidence, dict[str, str | None]]]:
    """
    Return all plausible format candidates as (format, confidence, column_map).
    Ordered from most specific to least specific.

    A column_map is included even at candidate stage so layer 2 knows
    which columns to sample for type validation.
    """
    candidates: list[tuple[DataFormat, DetectionConfidence, dict[str, str | None]]] = []

    has_chosen = _any_match(keys, CHOSEN_COLS)
    has_rejected = _any_match(keys, REJECTED_COLS)
    has_instr = _any_match(keys, INSTRUCTION_COLS)
    has_output = _any_match(keys, OUTPUT_COLS)
    has_conv = _any_match(keys, CONVERSATION_COLS)
    has_label = _any_match(keys, LABEL_COLS)
    has_resp_list = _any_match(keys, RESPONSE_LIST_COLS)
    has_text = _any_match(keys, TEXT_COLS)

    instr_col = _first_match(keys, INSTRUCTION_COLS, priority=_INSTRUCTION_PRIORITY)
    output_col = _first_match(keys, OUTPUT_COLS)
    conv_col = _first_match(keys, CONVERSATION_COLS)
    chosen_col = _first_match(keys, CHOSEN_COLS)
    rejected_col = _first_match(keys, REJECTED_COLS)
    label_col = _first_match(keys, LABEL_COLS)
    resp_list_col = _first_match(keys, RESPONSE_LIST_COLS)
    rewards_col = _first_match(keys, REWARD_LIST_COLS)
    text_col = _first_match(keys, TEXT_COLS)
    system_col = _first_match(keys, SYSTEM_COLS)
    ctx_col = _first_match(keys - (INSTRUCTION_COLS - {"input"}), CONTEXT_COLS)

    # --- Rule 1: Preference (DPO) ---
    if has_chosen and has_rejected:
        if has_instr:
            # Explicit prompt: highest confidence
            candidates.append(
                (
                    DataFormat.PREFERENCE,
                    DetectionConfidence.HIGH,
                    {
                        "instruction": instr_col,
                        "chosen": chosen_col,
                        "rejected": rejected_col,
                        "system": system_col,
                    },
                )
            )
        else:
            # Implicit prompt embedded in turns
            candidates.append(
                (
                    DataFormat.IMPLICIT_PREFERENCE,
                    DetectionConfidence.MEDIUM,
                    {
                        "chosen": chosen_col,
                        "rejected": rejected_col,
                    },
                )
            )

    # --- Rule 2: Unpaired preference / reward annotation ---
    if has_instr and has_output and has_label:
        candidates.append(
            (
                DataFormat.UNPAIRED_PREFERENCE,
                DetectionConfidence.HIGH,
                {
                    "instruction": instr_col,
                    "output": output_col,
                    "label": label_col,
                    "context": ctx_col,
                    "system": system_col,
                },
            )
        )

    # --- Rule 3: Conversational (ShareGPT / ChatML) ---
    # Low confidence at layer 1 — requires layer 2 to confirm value is list[dict]
    if has_conv:
        candidates.append(
            (
                DataFormat.SHAREGPT,
                DetectionConfidence.MEDIUM,
                {
                    "conversation": conv_col,
                    "system": system_col,
                },
            )
        )

    # --- Rule 4: GRPO (group rollouts) ---
    # Also low confidence until layer 2 confirms list[str] responses
    if has_instr and has_resp_list:
        candidates.append(
            (
                DataFormat.GRPO,
                DetectionConfidence.MEDIUM,
                {
                    "instruction": instr_col,
                    "responses": resp_list_col,
                    "rewards": rewards_col,
                    "system": system_col,
                },
            )
        )

    # --- Rule 5: Alpaca / SFT (instruction + output) ---
    if has_instr and has_output:
        # Exact alpaca column names → HIGH
        conf = (
            DetectionConfidence.HIGH
            if instr_col == "instruction" and output_col == "output"
            else DetectionConfidence.MEDIUM
        )
        candidates.append(
            (
                DataFormat.ALPACA,
                conf,
                {
                    "instruction": instr_col,
                    "output": output_col,
                    "context": ctx_col,
                    "system": system_col,
                },
            )
        )

    # --- Rule 6: Pre-training / language modeling ---
    # Only if no SFT/preference signals are also present
    if has_text and not has_instr and not has_output and not has_chosen:
        candidates.append(
            (
                DataFormat.PRETRAIN,
                DetectionConfidence.HIGH,
                {"text": text_col},
            )
        )

    # --- Rule 7: Prompt-only (PPO) ---
    if has_instr and not has_output and not has_conv and not has_chosen and not has_resp_list:
        candidates.append(
            (
                DataFormat.PROMPT_ONLY,
                DetectionConfidence.LOW,
                {
                    "instruction": instr_col,
                    "system": system_col,
                },
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Layer 2 — value-type validation
# ---------------------------------------------------------------------------


def _validate_candidate(
    candidate_format: DataFormat,
    column_map: dict[str, str | None],
    sample_rows: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """
    Validate that actual values match the expected types for this format.
    Returns (is_valid, warnings).
    """
    warnings: list[str] = []

    if not sample_rows:
        return True, ["no_sample_rows_for_validation"]

    row = sample_rows[0]

    def get_val(slot: str) -> Any:
        col = column_map.get(slot)
        if col is None:
            return None
        return row.get(col)

    if candidate_format == DataFormat.PREFERENCE:
        chosen_val = get_val("chosen")
        rejected_val = get_val("rejected")
        if chosen_val is None or rejected_val is None:
            return False, ["preference:chosen_or_rejected_missing"]
        # Both must be same type (str or list[dict])
        chosen_conv = _is_conversational_value(chosen_val)
        rejected_conv = _is_conversational_value(rejected_val)
        if chosen_conv != rejected_conv:
            return False, ["preference:chosen_rejected_type_mismatch"]
        return True, warnings

    elif candidate_format == DataFormat.IMPLICIT_PREFERENCE:
        chosen_val = get_val("chosen")
        rejected_val = get_val("rejected")
        if chosen_val is None or rejected_val is None:
            return False, ["implicit_preference:chosen_or_rejected_missing"]
        # For implicit prompt, turns must be lists so we can extract the prefix
        if not _is_conversational_value(chosen_val):
            warnings.append("implicit_preference:chosen_not_conversational_cannot_extract_prompt")
        return True, warnings

    elif candidate_format == DataFormat.UNPAIRED_PREFERENCE:
        instr_val = get_val("instruction")
        out_val = get_val("output")
        label_val = get_val("label")
        if instr_val is None or out_val is None:
            return False, ["unpaired_preference:instruction_or_output_missing"]
        if label_val is None:
            warnings.append("unpaired_preference:label_column_missing_in_sample")
        if not isinstance(instr_val, str):
            return False, ["unpaired_preference:instruction_not_str"]
        return True, warnings

    elif candidate_format == DataFormat.SHAREGPT:
        conv_val = get_val("conversation")
        if conv_val is None:
            return False, ["sharegpt:conversation_column_missing"]
        if not _is_conversational_value(conv_val):
            # Value is not list[dict] — this candidate is invalid
            return False, [
                f"sharegpt:conversation_value_is_{type(conv_val).__name__}_not_list_of_dict"
            ]
        return True, warnings

    elif candidate_format == DataFormat.GRPO:
        resp_val = get_val("responses")
        if resp_val is None:
            return False, ["grpo:responses_column_missing"]
        if not (
            _is_list_of_strings(resp_val)
            or (isinstance(resp_val, list) and resp_val and isinstance(resp_val[0], list))
        ):
            return False, [f"grpo:responses_value_is_{type(resp_val).__name__}_not_list"]
        return True, warnings

    elif candidate_format == DataFormat.ALPACA:
        instr_val = get_val("instruction")
        out_val = get_val("output")
        if instr_val is None or out_val is None:
            return False, ["alpaca:instruction_or_output_missing_in_sample"]
        if _is_conversational_value(instr_val):
            # Instruction column holds a conversation — this is actually sharegpt
            return False, ["alpaca:instruction_is_conversational_not_str"]
        if not isinstance(instr_val, str):
            return False, [f"alpaca:instruction_is_{type(instr_val).__name__}_not_str"]
        return True, warnings

    elif candidate_format == DataFormat.PRETRAIN:
        text_val = get_val("text")
        if text_val is None:
            return False, ["pretrain:text_column_missing"]
        if not isinstance(text_val, str):
            return False, [f"pretrain:text_is_{type(text_val).__name__}_not_str"]
        return True, warnings

    elif candidate_format == DataFormat.PROMPT_ONLY:
        instr_val = get_val("instruction")
        if instr_val is None:
            return False, ["prompt_only:instruction_column_missing"]
        return True, warnings

    return True, warnings


# ---------------------------------------------------------------------------
# Public API — FormatDetector
# ---------------------------------------------------------------------------


class FormatDetector:
    """
    Three-layer format detection engine.

    Usage:
        detector = FormatDetector()
        result = detector.detect(keys, sample_rows)

    `keys` is the set of top-level column names from the dataset.
    `sample_rows` is a small list of raw dicts (first N rows) used for
    layer 2 value-type validation. Pass an empty list to skip layer 2
    (layer 1 candidates are returned as-is with their base confidence).
    """

    def detect(
        self,
        keys: set[str],
        sample_rows: list[dict[str, Any]] | None = None,
    ) -> DetectionResult:
        if sample_rows is None:
            sample_rows = []

        # Layer 1 — generate all candidates
        candidates = _generate_candidates(keys)

        if not candidates:
            return DetectionResult(
                format=DataFormat.UNKNOWN,
                confidence=DetectionConfidence.UNKNOWN,
                unrecognized_cols=sorted(keys),
                warnings=["no_candidates_from_column_set"],
            )

        # Layer 2 — validate each candidate in priority order
        all_warnings: list[str] = []

        for fmt, base_conf, col_map in candidates:
            is_valid, layer2_warnings = _validate_candidate(fmt, col_map, sample_rows)
            all_warnings.extend(layer2_warnings)

            if not is_valid:
                continue

            # Determine final confidence
            # Downgrade if sample_rows were missing (couldn't fully validate)
            if not sample_rows and base_conf == DetectionConfidence.HIGH:
                final_conf = DetectionConfidence.MEDIUM
            else:
                final_conf = base_conf

            # Identify columns that weren't mapped to any semantic slot
            mapped_cols = {v for v in col_map.values() if v is not None}
            unrecognized = sorted(keys - mapped_cols - SYSTEM_COLS)

            return DetectionResult(
                format=fmt,
                confidence=final_conf,
                column_map=col_map,
                warnings=all_warnings + layer2_warnings,
                unrecognized_cols=unrecognized,
            )

        # All candidates failed layer 2
        return DetectionResult(
            format=DataFormat.UNKNOWN,
            confidence=DetectionConfidence.UNKNOWN,
            unrecognized_cols=sorted(keys),
            warnings=all_warnings + ["all_candidates_failed_layer2_validation"],
        )

    def detect_with_consistency_check(
        self,
        all_rows_keys: list[set[str]],
        sample_rows: list[dict[str, Any]],
    ) -> tuple[DetectionResult, bool]:
        """
        Run detection and also check whether the column set is consistent
        across the sampled rows. Returns (result, is_consistent).

        is_consistent=False means the file may have mixed schemas — the
        pipeline should warn but not hard-fail.
        """
        if not all_rows_keys:
            result = self.detect(set(), sample_rows)
            return result, True

        first_keys = all_rows_keys[0]
        is_consistent = all(k == first_keys for k in all_rows_keys)

        if not is_consistent:
            # Use the union of all keys for detection — more conservative
            union_keys = set().union(*all_rows_keys)
            result = self.detect(union_keys, sample_rows)
            result.warnings.append(
                f"inconsistent_column_sets_across_{len(all_rows_keys)}_sample_rows"
            )
        else:
            result = self.detect(first_keys, sample_rows)

        return result, is_consistent
