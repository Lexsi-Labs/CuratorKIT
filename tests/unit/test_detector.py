"""
Tests for the FormatDetector — all three layers.

Coverage:
  Layer 1: All 7 semantic rules, confidence levels, overlapping rules
  Layer 2: Value-type validation per format
  Layer 3: Role normalization (via normalizer module)
  Edge cases: empty inputs, unknown formats, consistency checks
"""

from __future__ import annotations

from curatorkit.detection.detector import (
    DataFormat,
    DetectionConfidence,
    FormatDetector,
)
from curatorkit.detection.normalizer import (
    extract_implicit_prompt,
    extract_system_prompt,
    normalize_conversations,
    normalize_turn,
)

detector = FormatDetector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect(keys, rows=None):
    return detector.detect(set(keys), rows or [])


# ---------------------------------------------------------------------------
# Layer 1 — column set → candidate generation
# ---------------------------------------------------------------------------


class TestAlpacaDetection:
    def test_exact_alpaca_columns_high_confidence(self):
        rows = [{"instruction": "What is Python?", "output": "A programming language."}]
        r = detect(["instruction", "output"], rows)
        assert r.format == DataFormat.ALPACA
        assert r.confidence == DetectionConfidence.HIGH

    def test_alpaca_with_input_field(self):
        r = detect(["instruction", "output", "input"])
        assert r.format == DataFormat.ALPACA

    def test_alpaca_with_system_field(self):
        r = detect(["instruction", "output", "system"])
        assert r.format == DataFormat.ALPACA
        assert r.column_map.get("system") == "system"

    def test_prompt_completion_variant(self):
        r = detect(["prompt", "completion"])
        assert r.format == DataFormat.ALPACA

    def test_prompt_response_variant(self):
        r = detect(["prompt", "response"])
        assert r.format == DataFormat.ALPACA

    def test_query_answer_variant(self):
        r = detect(["query", "answer"])
        assert r.format == DataFormat.ALPACA

    def test_question_response_variant(self):
        r = detect(["question", "response"])
        assert r.format == DataFormat.ALPACA

    def test_non_exact_names_medium_confidence(self):
        r = detect(["query", "answer"])
        # Not exact "instruction"/"output" but semantically equivalent
        assert r.format == DataFormat.ALPACA
        assert r.confidence in (DetectionConfidence.MEDIUM, DetectionConfidence.HIGH)

    def test_column_map_resolves_actual_names(self):
        r = detect(["query", "answer"])
        assert r.column_map.get("instruction") == "query"
        assert r.column_map.get("output") == "answer"


class TestShareGPTDetection:
    def test_conversations_key_detected(self):
        rows = [{"conversations": [{"role": "user", "content": "hi"}]}]
        r = detect(["conversations"], rows)
        assert r.format == DataFormat.SHAREGPT

    def test_messages_key_detected(self):
        rows = [{"messages": [{"role": "user", "content": "hi"}]}]
        r = detect(["messages"], rows)
        assert r.format == DataFormat.SHAREGPT

    def test_conversations_flat_string_rejected(self):
        # Value is a string, not list[dict] → sharegpt candidate fails layer 2
        rows = [{"conversations": "this is just a string"}]
        r = detect(["conversations"], rows)
        # Should fall through to unknown or alpaca, not sharegpt
        assert r.format != DataFormat.SHAREGPT

    def test_conversations_without_rows_medium_confidence(self):
        # No sample rows → can't run layer 2 → medium confidence
        r = detect(["conversations"])
        assert r.format == DataFormat.SHAREGPT
        assert r.confidence == DetectionConfidence.MEDIUM

    def test_column_map_resolved(self):
        rows = [{"messages": [{"role": "user", "content": "hi"}]}]
        r = detect(["messages"], rows)
        assert r.column_map.get("conversation") == "messages"


class TestPreferenceDetection:
    def test_explicit_preference_high_confidence(self):
        rows = [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]
        r = detect(["prompt", "chosen", "rejected"], rows)
        assert r.format == DataFormat.PREFERENCE
        assert r.confidence == DetectionConfidence.HIGH

    def test_preference_without_prompt(self):
        rows = [{"chosen": "good", "rejected": "bad"}]
        r = detect(["chosen", "rejected"], rows)
        assert r.format in (DataFormat.PREFERENCE, DataFormat.IMPLICIT_PREFERENCE)

    def test_preference_column_map(self):
        rows = [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        r = detect(["prompt", "chosen", "rejected"], rows)
        assert r.column_map.get("instruction") == "prompt"
        assert r.column_map.get("chosen") == "chosen"
        assert r.column_map.get("rejected") == "rejected"

    def test_preference_aliases(self):
        rows = [{"query": "Q", "preferred": "A", "refused": "B"}]
        r = detect(["query", "preferred", "refused"], rows)
        assert r.format in (DataFormat.PREFERENCE, DataFormat.IMPLICIT_PREFERENCE)

    def test_implicit_preference_conversational(self):
        turns = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        rows = [{"chosen": turns, "rejected": turns}]
        r = detect(["chosen", "rejected"], rows)
        assert r.format in (DataFormat.PREFERENCE, DataFormat.IMPLICIT_PREFERENCE)


class TestUnpairedPreferenceDetection:
    def test_unpaired_preference_detected(self):
        rows = [{"prompt": "Q", "response": "A", "score": 0.9}]
        r = detect(["prompt", "response", "score"], rows)
        assert r.format == DataFormat.UNPAIRED_PREFERENCE

    def test_unpaired_with_label_column(self):
        rows = [{"instruction": "Q", "output": "A", "label": True}]
        r = detect(["instruction", "output", "label"], rows)
        assert r.format == DataFormat.UNPAIRED_PREFERENCE

    def test_unpaired_with_rating(self):
        rows = [{"query": "Q", "answer": "A", "rating": 4}]
        r = detect(["query", "answer", "rating"], rows)
        assert r.format == DataFormat.UNPAIRED_PREFERENCE


class TestGRPODetection:
    def test_grpo_with_list_responses(self):
        rows = [{"prompt": "Q", "responses": ["r1", "r2"], "rewards": [0.8, 0.3]}]
        r = detect(["prompt", "responses", "rewards"], rows)
        assert r.format == DataFormat.GRPO

    def test_grpo_completions_alias(self):
        rows = [{"instruction": "Q", "completions": ["r1", "r2"]}]
        r = detect(["instruction", "completions"], rows)
        assert r.format == DataFormat.GRPO

    def test_grpo_string_responses_fails_layer2(self):
        # responses is a string → not GRPO, falls through to ALPACA
        rows = [{"prompt": "Q", "responses": "single response string"}]
        r = detect(["prompt", "responses"], rows)
        assert r.format != DataFormat.GRPO


class TestPretrainDetection:
    def test_text_column_only(self):
        rows = [{"text": "The quick brown fox"}]
        r = detect(["text"], rows)
        assert r.format == DataFormat.PRETRAIN

    def test_text_with_other_sft_signals_not_pretrain(self):
        # instruction present → ALPACA wins over PRETRAIN
        rows = [{"text": "...", "instruction": "Q", "output": "A"}]
        r = detect(["text", "instruction", "output"], rows)
        assert r.format == DataFormat.ALPACA

    def test_text_must_be_string_for_pretrain(self):
        rows = [{"text": ["list", "of", "items"]}]
        r = detect(["text"], rows)
        # text column is a list, not str → pretrain layer 2 fails
        assert r.format != DataFormat.PRETRAIN


class TestPromptOnlyDetection:
    def test_prompt_alone_low_confidence(self):
        r = detect(["prompt"])
        assert r.format == DataFormat.PROMPT_ONLY
        assert r.confidence == DetectionConfidence.LOW


class TestUnknownFormat:
    def test_no_recognizable_columns(self):
        r = detect(["foo", "bar", "baz"])
        assert r.format == DataFormat.UNKNOWN
        assert r.confidence == DetectionConfidence.UNKNOWN

    def test_empty_keys(self):
        r = detect([])
        assert r.format == DataFormat.UNKNOWN

    def test_unrecognized_cols_reported(self):
        rows = [{"instruction": "Q", "output": "A", "domain": "science"}]
        r = detect(["instruction", "output", "domain"], rows)
        assert r.format == DataFormat.ALPACA
        assert "domain" in r.unrecognized_cols


# ---------------------------------------------------------------------------
# Preference over alpaca when both signals present
# ---------------------------------------------------------------------------


class TestRulePriority:
    def test_chosen_rejected_beats_instruction_output(self):
        """When both preference and SFT columns are present, preference wins."""
        rows = [
            {"prompt": "Q", "instruction": "Q", "output": "A", "chosen": "good", "rejected": "bad"}
        ]
        r = detect(["prompt", "instruction", "output", "chosen", "rejected"], rows)
        assert r.format in (DataFormat.PREFERENCE, DataFormat.IMPLICIT_PREFERENCE)

    def test_unpaired_preference_beats_pure_alpaca(self):
        """instruction + output + label → unpaired_preference, not alpaca."""
        rows = [{"instruction": "Q", "output": "A", "label": 1.0}]
        r = detect(["instruction", "output", "label"], rows)
        assert r.format == DataFormat.UNPAIRED_PREFERENCE

    def test_grpo_beats_prompt_only(self):
        rows = [{"prompt": "Q", "responses": ["r1", "r2"]}]
        r = detect(["prompt", "responses"], rows)
        assert r.format == DataFormat.GRPO


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------


class TestConsistencyCheck:
    def test_consistent_files_no_warning(self):
        all_keys = [{"instruction", "output"}, {"instruction", "output"}]
        sample_rows = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
        ]
        result, is_consistent = detector.detect_with_consistency_check(all_keys, sample_rows)
        assert is_consistent
        assert result.format == DataFormat.ALPACA

    def test_inconsistent_files_flagged(self):
        all_keys = [
            {"instruction", "output"},
            {"conversations"},  # different schema on row 2
        ]
        sample_rows = [
            {"instruction": "Q", "output": "A"},
            {"conversations": [{"role": "user", "content": "hi"}]},
        ]
        result, is_consistent = detector.detect_with_consistency_check(all_keys, sample_rows)
        assert not is_consistent
        assert any("inconsistent" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Layer 3 — role normalization
# ---------------------------------------------------------------------------


class TestRoleNormalization:
    def test_chatml_user_role(self):
        turn, warn = normalize_turn({"role": "user", "content": "hi"})
        assert turn["role"] == "user"
        assert warn is None

    def test_sharegpt_human_role(self):
        turn, warn = normalize_turn({"from": "human", "value": "hi"})
        assert turn["role"] == "user"
        assert turn["content"] == "hi"
        assert warn is None

    def test_gpt_role_normalized(self):
        turn, warn = normalize_turn({"from": "gpt", "value": "hello"})
        assert turn["role"] == "assistant"

    def test_model_role_normalized(self):
        turn, warn = normalize_turn({"role": "model", "content": "reply"})
        assert turn["role"] == "assistant"

    def test_system_role_preserved(self):
        turn, warn = normalize_turn({"role": "system", "content": "you are helpful"})
        assert turn["role"] == "system"
        assert warn is None

    def test_unknown_role_warns(self):
        turn, warn = normalize_turn({"role": "oracle", "content": "..."})
        assert warn is not None
        assert "oracle" in warn

    def test_tool_role_preserved(self):
        turn, warn = normalize_turn({"role": "tool", "content": "result"})
        assert turn["role"] == "tool"
        assert warn is None

    def test_full_conversation_normalization(self):
        convs = [
            {"from": "system", "value": "Be helpful"},
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
        normalized, warnings = normalize_conversations(convs)
        assert normalized[0]["role"] == "system"
        assert normalized[1]["role"] == "user"
        assert normalized[2]["role"] == "assistant"
        assert not warnings

    def test_extra_keys_preserved_in_from_value(self):
        turn, _ = normalize_turn({"from": "human", "value": "hi", "weight": 1.0})
        assert turn.get("weight") == 1.0


class TestExtractSystemPrompt:
    def test_extracts_system_from_first_turn(self):
        convs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
        sys, rest = extract_system_prompt(convs)
        assert sys == "you are helpful"
        assert len(rest) == 1
        assert rest[0]["role"] == "user"

    def test_no_system_prompt_returns_empty(self):
        convs = [{"role": "user", "content": "hi"}]
        sys, rest = extract_system_prompt(convs)
        assert sys == ""
        assert len(rest) == 1

    def test_empty_conversation(self):
        sys, rest = extract_system_prompt([])
        assert sys == ""
        assert rest == []


class TestExtractImplicitPrompt:
    def test_extracts_common_prefix(self):
        chosen = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        rejected = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "5"},
        ]
        prompt, c_remain, r_remain = extract_implicit_prompt(chosen, rejected)
        assert len(prompt) == 1
        assert prompt[0]["content"] == "What is 2+2?"
        assert c_remain[0]["content"] == "4"
        assert r_remain[0]["content"] == "5"

    def test_no_common_prefix_returns_empty(self):
        chosen = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A"}]
        rejected = [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "B"}]
        prompt, c_remain, r_remain = extract_implicit_prompt(chosen, rejected)
        assert prompt == []
        assert c_remain == chosen
        assert r_remain == rejected

    def test_multi_turn_common_prefix(self):
        shared = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Tell me about Python"},
        ]
        chosen = shared + [{"role": "assistant", "content": "Python is great"}]
        rejected = shared + [{"role": "assistant", "content": "I don't know"}]
        prompt, c_remain, r_remain = extract_implicit_prompt(chosen, rejected)
        assert len(prompt) == 3
        assert len(c_remain) == 1
        assert len(r_remain) == 1
