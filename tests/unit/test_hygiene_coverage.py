"""Hygiene steps look at all of a sample's text: long texts and the later turns of a conversation."""
from types import SimpleNamespace

import pytest

from curatorkit.hygiene.pii import PIIPseudonymizer
from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.hygiene.toxicity import ToxicityGate
from curatorkit.schema import DataSample

# AWS's documented example key, assembled at run time so the diff holds no key literal.
AWS = "AKIA" + "IOSFODNN7EXAMPLE"


def _conversation(later_turn):
    return DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": [{"from": "human", "value": later_turn}, {"from": "gpt", "value": "ok"}]},
    )


def _toxicity_gate():
    gate = ToxicityGate.__new__(ToxicityGate)
    gate.text_field, gate.llm = "auto", None
    gate.classifier_pass_threshold, gate.classifier_reject_threshold = 0.3, 0.7
    gate._config_hash = lambda: "h"
    gate._load_classifier = lambda: SimpleNamespace(
        predict=lambda text: {"toxicity": 0.95 if "BADWORD" in text else 0.01}
    )
    return gate


def test_toxic_text_after_the_classifier_window_is_rejected():
    sample = DataSample(
        source_uri="x", instruction="Q?", output=("Fine text. " * 300) + "BADWORD here",
        task_type="instruction_following",
    )
    passed, rejected = _toxicity_gate().run([sample])
    assert not passed and len(rejected) == 1


def test_toxic_input_field_is_rejected():
    # the exporters append `input` to the user turn, so it is scored like the instruction
    sample = DataSample(
        source_uri="x", instruction="Summarise this.", input="BADWORD here", output="Fine.",
        task_type="instruction_following",
    )
    passed, rejected = _toxicity_gate().run([sample])
    assert not passed and len(rejected) == 1


def test_clean_long_text_still_passes():
    sample = DataSample(
        source_uri="x", instruction="Q?", output="Fine text. " * 600, task_type="instruction_following"
    )
    passed, rejected = _toxicity_gate().run([sample])
    assert len(passed) == 1 and not rejected


def test_toxic_later_turn_of_a_conversation_is_rejected():
    passed, rejected = _toxicity_gate().run([_conversation("BADWORD")])
    assert not passed and len(rejected) == 1


def test_secret_in_a_later_turn_of_a_conversation_is_rejected():
    pytest.importorskip("detect_secrets")
    passed, rejected = SecretsGate().run([_conversation(f"my key is {AWS}")])
    assert not passed and "secret_detected" in rejected[0].rejection_reason


def test_pii_in_a_later_turn_is_pseudonymized_with_the_conversation_entity_map():
    step = PIIPseudonymizer.__new__(PIIPseudonymizer)
    step._config_hash = lambda: "h"
    step._load = lambda: (None, None)
    step._fields_for_sample = lambda s: ["instruction", "output"]
    seen = []

    def fake(text, entity_map, analyzer, faker):
        seen.append(entity_map)
        return text.replace("Alice", "Zed"), ({"PERSON": 1} if "Alice" in text else {})

    step._pseudonymize = fake
    sample = _conversation("Alice again")
    out = step.run([sample])[0]
    assert out.metadata["turns"][0]["value"] == "Zed again"
    assert len({id(m) for m in seen}) == 1  # one entity map for the whole conversation


# --- Conversations ingested through the real connector --------------------------------------------
# The connector stores later turns as {"role", "content"}, the generators as {"from", "value"}.

def _ingested(tmp_path, later_turn, system="You are helpful."):
    import json

    from curatorkit.connectors.jsonl import JSONLReader

    path = tmp_path / "chat.jsonl"
    row = {"conversations": [
        {"from": "system", "value": system},
        {"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"},
        {"from": "human", "value": later_turn}, {"from": "gpt", "value": "ok"},
    ]}
    path.write_text(json.dumps(row) + "\n")
    samples, _ = JSONLReader(path).read()
    assert samples and samples[0].task_type == "conversational"
    assert samples[0].metadata["turns"][0].get("content") == later_turn  # the connector's shape
    return samples[0]


def test_toxic_turn_ingested_by_the_connector_is_rejected(tmp_path):
    passed, rejected = _toxicity_gate().run([_ingested(tmp_path, "BADWORD")])
    assert not passed and len(rejected) == 1


def test_toxic_system_prompt_ingested_by_the_connector_is_rejected(tmp_path):
    passed, rejected = _toxicity_gate().run([_ingested(tmp_path, "fine", system="BADWORD")])
    assert not passed and len(rejected) == 1


def test_secret_in_a_connector_turn_is_rejected(tmp_path):
    pytest.importorskip("detect_secrets")
    passed, rejected = SecretsGate().run([_ingested(tmp_path, f"my key is {AWS}")])
    assert not passed and "secret_detected" in rejected[0].rejection_reason


def test_secret_in_a_system_prompt_is_rejected(tmp_path):
    pytest.importorskip("detect_secrets")
    passed, rejected = SecretsGate().run([_ingested(tmp_path, "fine", system=f"token {AWS}")])
    assert not passed


def test_pii_in_a_connector_turn_and_system_prompt_is_pseudonymized(tmp_path):
    step = PIIPseudonymizer.__new__(PIIPseudonymizer)
    step._config_hash = lambda: "h"
    step._load = lambda: (None, None)
    step._fields_for_sample = lambda s: []
    step._pseudonymize = lambda text, em, an, fk: (text.replace("Alice", "Zed"), {"PERSON": 1} if "Alice" in text else {})
    out = step.run([_ingested(tmp_path, "Alice again", system="Talk to Alice")])[0]
    assert out.metadata["turns"][0]["content"] == "Zed again"
    assert out.metadata["system_prompt"] == "Talk to Zed"


def test_a_phrase_straddling_a_window_boundary_is_scored_whole():
    gate = _toxicity_gate()
    gate._load_classifier = lambda: SimpleNamespace(
        predict=lambda text: {"toxicity": 0.95 if "BAD WORD" in text else 0.01}
    )
    text = "x" * 2044 + "BAD WORD" + "y" * 2000  # "BAD WORD" crosses the first 2 048-character mark
    sample = DataSample(source_uri="x", instruction="Q?", output=text, task_type="instruction_following")
    passed, rejected = gate.run([sample])
    assert not passed and len(rejected) == 1


# --- List-of-parts content (OpenAI-style) ------------------------------------------------------

def _parts_conversation(tmp_path):
    import json

    from curatorkit.connectors.jsonl import JSONLReader

    row = {"messages": [
        {"role": "system", "content": [{"type": "text", "text": f"SYS key {AWS}"}]},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": [
            {"type": "text", "text": "mail Alice and BADWORD"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]},
        {"role": "assistant", "content": "ok"},
    ]}
    path = tmp_path / "parts.jsonl"
    path.write_text(json.dumps(row) + "\n")
    samples, _ = JSONLReader(path).read()
    assert isinstance(samples[0].metadata["turns"][0]["content"], list)  # kept as parts
    return samples[0]


def test_toxic_text_in_a_content_part_is_rejected(tmp_path):
    passed, rejected = _toxicity_gate().run([_parts_conversation(tmp_path)])
    assert not passed and len(rejected) == 1


def test_secret_in_a_system_prompt_part_is_rejected(tmp_path):
    pytest.importorskip("detect_secrets")
    passed, _ = SecretsGate().run([_parts_conversation(tmp_path)])
    assert not passed


def test_pii_in_a_content_part_is_pseudonymized_in_place_and_structure_is_kept(tmp_path):
    step = PIIPseudonymizer.__new__(PIIPseudonymizer)
    step._config_hash = lambda: "h"
    step._load = lambda: (None, None)
    step._fields_for_sample = lambda s: []
    step._pseudonymize = lambda text, em, an, fk: (text.replace("Alice", "Zed"), {"PERSON": 1} if "Alice" in text else {})
    out = step.run([_parts_conversation(tmp_path)])[0]
    part = out.metadata["turns"][0]["content"][0]
    assert part == {"type": "text", "text": "mail Zed and BADWORD"}  # "type" untouched


# --- Nothing the exporters write goes unread ----------------------------------------------------

def _nested(depth, leaf):
    node = {"text": leaf}
    for _ in range(depth):
        node = {"parts": [node]}
    return node


def test_text_nested_deeper_than_any_fixed_limit_is_still_scanned():
    sample = DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": [{"role": "user", "content": [_nested(40, "BADWORD")]}]},
    )
    passed, rejected = _toxicity_gate().run([sample])
    assert not passed and len(rejected) == 1


@pytest.mark.parametrize("where", ["type", "role", "extra_key", "dict_key"])
def test_read_only_gates_scan_structural_values_and_keys(where):
    part = {"type": "text", "text": "fine"}
    turn = {"role": "user", "content": [part]}
    if where == "type":
        part["type"] = "BADWORD"
    elif where == "role":
        turn["role"] = "BADWORD"
    elif where == "extra_key":
        turn["name"] = "BADWORD"
    else:
        part["BADWORD"] = "x"
    sample = DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": [turn]},
    )
    passed, rejected = _toxicity_gate().run([sample])
    assert not passed and len(rejected) == 1


def test_secret_in_a_structural_value_is_rejected():
    pytest.importorskip("detect_secrets")
    sample = DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": [{"role": "user", "content": [{"type": f"text {AWS}", "text": "fine"}]}]},
    )
    passed, _ = SecretsGate().run([sample])
    assert not passed


def test_the_pseudonymizer_rewrites_text_but_not_structure():
    step = PIIPseudonymizer.__new__(PIIPseudonymizer)
    step._config_hash = lambda: "h"
    step._load = lambda: (None, None)
    step._fields_for_sample = lambda s: []
    step._pseudonymize = lambda text, em, an, fk: (text.replace("Alice", "Zed"), {})
    turn = {"role": "user", "name": "Alice", "content": [{"type": "text", "text": "hi Alice"}, {"Alice": "k"}]}
    sample = DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": [turn]},
    )
    out = step.run([sample])[0].metadata["turns"][0]
    assert out["content"][0] == {"type": "text", "text": "hi Zed"}
    assert out["role"] == "user" and "Alice" in out["content"][1]  # structure untouched


def test_a_short_toxic_turn_is_not_diluted_by_clean_turns():
    seen = []
    gate = _toxicity_gate()
    gate._load_classifier = lambda: SimpleNamespace(
        predict=lambda text: seen.append(text) or {"toxicity": 0.95 if text.strip() == "BADWORD" else 0.01}
    )
    turns = [{"from": "human", "value": "clean " * 80}, {"from": "gpt", "value": "BADWORD"}, {"from": "human", "value": "clean " * 80}]
    sample = DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": turns},
    )
    passed, rejected = gate.run([sample])
    assert not passed and len(rejected) == 1
    assert "BADWORD" in seen  # scored by itself, not inside a joined text


def test_a_long_toxic_string_under_an_unexpected_key_is_scored_on_its_own():
    sample = DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": [{"role": "user", "content": "fine", "name": "BADWORD " + "pad " * 30}]},
    )
    passed, rejected = _toxicity_gate().run([sample])
    assert not passed and len(rejected) == 1


def test_the_llm_judge_reads_the_text_that_scored_worst_even_when_labels_could_collide():
    gate = _toxicity_gate()
    gate.classifier_pass_threshold, gate.classifier_reject_threshold, gate.llm_reject_threshold = 0.3, 0.99, 0.5
    gate.llm = object()
    judged = []
    gate._llm_verdict = lambda text: judged.append(text) or (0.9, "", [])
    gate._worst_window = lambda text: text
    # both strings get the label "metadata.turns[0].content[0].text" before the index is added
    turns = [{"role": "user", "content[0]": {"text": "BADWORD"}, "content": [{"text": "clean"}]}]
    sample = DataSample(
        source_uri="x", instruction="Hi", output="Hello", task_type="conversational",
        metadata={"turns": turns},
    )
    passed, rejected = gate.run([sample])
    assert not passed and judged and "BADWORD" in judged[0]
