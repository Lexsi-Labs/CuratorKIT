"""GuardModelGate: verdict parsing, decisions, routing from ToxicityGate, and the HF batching path.

Every test runs on CPU without torch, transformers or model weights: the model is a fake.
"""
from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import pytest

from curatorkit.hygiene.guard import GuardModelGate, _resolve_device, parse_guard_output
from curatorkit.hygiene.toxicity import ToxicityGate
from curatorkit.schema import DataSample

# The issue's languages; "BAD" marks the text the fake guard calls unsafe.
KO = "너는 멍청하고 쓸모없는 바보야 BAD"
ZH = "你是一个愚蠢无用的白痴 BAD"
SW = "wewe ni mjinga na sina maana BAD"


def _lm(text):
    return DataSample(source_uri="x", output=text, task_type="language_modeling")


def _fake_guard(reply_for_bad="Safety: Unsafe\nCategories: Violent", seen=None):
    def generate(conversations):
        if seen is not None:
            seen.extend(conversations)
        return [reply_for_bad if "BAD" in c[-1]["content"] else "Safety: Safe\nCategories: None"
                for c in conversations]
    return generate


# -- parsing ---------------------------------------------------------------------
@pytest.mark.parametrize("reply,expected", [
    ("Safety: Unsafe\nCategories: Violent", ("unsafe", ["Violent"])),
    ("Safety: Safe\nCategories: None\nRefusal: Yes", ("safe", [])),
    ("Safety: Controversial\nCategories: Politically Sensitive Topics, PII",
     ("controversial", ["Politically Sensitive Topics", "PII"])),
    ("\n\nunsafe\nS1,S10", ("unsafe", ["S1", "S10"])),  # Llama Guard
    ("safe", ("safe", [])),
    ("I cannot help with that.", (None, [])),
    ("", (None, [])),
])
def test_parse_guard_output(reply, expected):
    assert parse_guard_output(reply) == expected


# -- decisions and provenance ------------------------------------------------------
def test_unsafe_text_in_any_language_is_rejected_with_provenance():
    gate = GuardModelGate(generator=_fake_guard())
    passed, rejected = gate.run([_lm(KO), _lm(ZH), _lm(SW), _lm("A clean sentence.")])
    assert [s.output for s in passed] == ["A clean sentence."]
    assert len(rejected) == 3
    rec = rejected[0].provenance_chain[-1]
    assert rec.step_name == "GuardModelGate" and rec.step_version
    assert rec.notes["passed"] is False and rec.notes["worst_field"] == "output"
    assert rec.notes["categories"] == ["Violent"] and rec.notes["guard_model"] == gate.model
    assert rejected[0].rejection_reason == "unsafe_content:guard_model:unsafe"
    assert passed[0].provenance_chain[-1].notes["field_verdicts"]["output"]["verdict"] == "safe"


def test_controversial_and_unparsed_replies_follow_the_settings():
    contro = _fake_guard("Safety: Controversial\nCategories: Unethical Acts")
    assert len(GuardModelGate(generator=contro).run([_lm(KO)])[1]) == 1
    assert len(GuardModelGate(generator=contro, reject_controversial=False).run([_lm(KO)])[0]) == 1
    garbled = _fake_guard("sorry?")
    passed, rejected = GuardModelGate(generator=garbled).run([_lm(KO)])
    assert not passed and rejected[0].rejection_reason == "unsafe_content:guard_model:unparsed"
    assert len(GuardModelGate(generator=garbled, on_parse_error="pass").run([_lm(KO)])[0]) == 1
    with pytest.raises(ValueError):
        GuardModelGate(on_parse_error="ignore")


def test_reply_is_judged_with_its_prompt_and_long_text_is_windowed():
    seen = []
    sample = DataSample(source_uri="x", instruction="Write a story.",
                        output=("Fine text. " * 400) + "BAD", task_type="instruction_following")
    passed, rejected = GuardModelGate(generator=_fake_guard(seen=seen)).run([sample])
    assert not passed and len(rejected) == 1  # the unsafe part is in the last window only
    output_convs = [c for c in seen if c[0]["content"] == "Write a story." and len(c) == 2]
    assert len(output_convs) >= 2 and all(c[1]["role"] == "assistant" for c in output_convs)
    assert [c for c in seen if len(c) == 1 and c[0]["content"] == "Write a story."]  # prompt alone


def test_content_parts_wraps_message_content():
    seen = []
    GuardModelGate(generator=_fake_guard(seen=seen), content_parts=True).run([_lm("hello")])
    assert seen[0][0]["content"] == [{"type": "text", "text": "hello"}]


# -- routing from ToxicityGate ----------------------------------------------------
def _detoxify_like_gate():
    """ToxicityGate with a fake classifier that, like Detoxify, scores text it cannot read as clean."""
    gate = ToxicityGate.__new__(ToxicityGate)
    gate.text_field, gate.llm, gate.detoxify_model = "auto", None, "multilingual"
    gate.classifier_pass_threshold, gate.classifier_reject_threshold = 0.1, 0.5
    gate._config_hash = lambda: "h"
    gate._load_classifier = lambda: SimpleNamespace(predict=lambda text: {"toxicity": 0.03})
    return gate


def test_only_unscreened_screens_what_the_toxicity_gate_could_not_read():
    samples = [_lm(KO), _lm(ZH), _lm(SW), _lm("A clean sentence.")]
    passed, _ = _detoxify_like_gate().run(samples)
    assert len(passed) == 4  # Detoxify passes all of them
    seen = []
    routed = GuardModelGate(generator=_fake_guard(seen=seen), only_unscreened=True)
    ok, rejected = routed.run(passed)
    # ko and zh were flagged unscreened and are caught; Swahili is Latin script, never flagged,
    # so routing alone misses it: that is the documented limit.
    assert sorted(r.output for r in rejected) == sorted([KO, ZH])
    assert len(seen) == 2
    skipped = [s for s in ok if s.provenance_chain[-1].notes.get("skipped")]
    assert {s.output for s in skipped} == {SW, "A clean sentence."}
    assert skipped[0].provenance_chain[-1].notes["reason"] == "no_unscreened_fields"


def test_screening_every_field_catches_the_latin_script_language_routing_misses():
    passed, _ = _detoxify_like_gate().run([_lm(SW)])
    ok, rejected = GuardModelGate(generator=_fake_guard()).run(passed)
    assert not ok and rejected[0].output == SW


# -- the transformers path, with a fake tokenizer and model ------------------------
class _T:
    """Minimal 2-D tensor: shape, [:, n:] slicing and .to()."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, key):
        _, cols = key
        return _T([r[cols] for r in self.rows])

    def to(self, _device):
        return self


class _Enc(dict):
    def to(self, device):
        self.device = device
        return self


class _FakeTok:
    pad_token, eos_token, pad_token_id = None, "<eos>", 0
    padding_side = "right"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conv, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(m["content"] for m in conv)

    def __call__(self, prompts, return_tensors, padding, add_special_tokens):
        self.calls.append((list(prompts), self.padding_side))
        width = max(len(p) for p in prompts)
        # left padding: each prompt's characters as ids, padded on the left to the batch width
        return _Enc(input_ids=_T([[0] * (width - len(p)) + [1] * len(p) for p in prompts]))

    def batch_decode(self, rows, skip_special_tokens):
        assert all(len(r) == 1 for r in rows.rows)  # only the generated token, no prompt ids
        return ["Safety: Unsafe\nCategories: Violent" if r[0] == 9
                else "Safety: Safe\nCategories: None" for r in rows.rows]


class _FakeModel:
    def __init__(self, tok):
        self.tok, self.kwargs = tok, []

    def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id):
        self.kwargs.append((max_new_tokens, do_sample))
        prompts = self.tok.calls[-1][0]
        return _T([row + [9 if "BAD" in p else 5] for row, p in zip(input_ids.rows, prompts)])


def test_hf_path_loads_on_cpu_batches_left_pads_and_decodes_only_the_reply(monkeypatch):
    tok = _FakeTok()
    model = _FakeModel(tok)
    loaded = {}

    def load_model(name, revision, dtype):
        loaded.update(name=name, revision=revision, dtype=dtype)
        return SimpleNamespace(to=lambda device: loaded.setdefault("device", device) and model,
                               eval=lambda: None)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        no_grad=contextlib.nullcontext, bfloat16="bf16", float32="fp32",
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda name, revision: tok),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)))
    model.eval = lambda: None
    gate = GuardModelGate(batch_size=2, revision="abc123")
    passed, rejected = gate.run([_lm(KO), _lm("fine"), _lm("also fine"), _lm(ZH), _lm("ok")])
    assert loaded == {"name": "Qwen/Qwen3Guard-Gen-0.6B", "revision": "abc123", "dtype": "fp32",
                      "device": "cpu"}
    assert tok.pad_token == "<eos>"  # no pad token: falls back to eos
    assert len(rejected) == 2 and len(passed) == 3
    assert [len(c[0]) for c in tok.calls] == [2, 2, 1]  # batches of batch_size
    assert all(side == "left" for _, side in tok.calls)
    assert all(do_sample is False for _, do_sample in model.kwargs)


def test_auto_device_prefers_cuda_and_never_picks_mps(monkeypatch):
    def fake_torch(cuda):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                               backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch(False))
    assert _resolve_device(None) == "cpu" and _resolve_device("auto") == "cpu"
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True))
    assert _resolve_device(None) == "cuda"
    assert _resolve_device("cuda:1") == "cuda:1"


# -- pipeline wiring ---------------------------------------------------------------
def test_yaml_gate_and_curator_config_build_the_guard_gate():
    from curatorkit.config import GateConfig

    g = GateConfig(type="guard", guard_only_unscreened=True, guard_device="cpu")
    assert g.guard_model == "Qwen/Qwen3Guard-Gen-0.6B" and g.guard_only_unscreened

    from curatorkit import Curator, CuratorConfig

    steps = Curator(CuratorConfig(dataset="d.jsonl", toxicity_gate=True, guard_gate=True,
                                  guard_only_unscreened=True))._build_steps(include_exporters=False)
    names = [type(s).__name__ for s in steps]
    assert names.index("ToxicityGate") < names.index("GuardModelGate")
    guard = next(s for s in steps if isinstance(s, GuardModelGate))
    assert guard.only_unscreened is True


def test_cli_yaml_gate_builds_the_guard_gate():
    from curatorkit.cli import _build_steps
    from curatorkit.config import GateConfig, PipelineConfig, ReaderConfig

    cfg = PipelineConfig(readers=[ReaderConfig(type="jsonl", path="d.jsonl")],
                         gates=[GateConfig(type="toxicity"),
                                GateConfig(type="guard", guard_only_unscreened=True,
                                           guard_model="org/guard", guard_batch_size=4)])
    steps, _ = _build_steps(cfg, verbose=False, include_exporters=False)
    guard = next(s for s in steps if isinstance(s, GuardModelGate))
    assert (guard.model, guard.only_unscreened, guard.batch_size) == ("org/guard", True, 4)
