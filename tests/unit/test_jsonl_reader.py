"""Unit tests for JSONLReader — Alpaca, ShareGPT, raw, auto-detect, field mapping."""

from __future__ import annotations

import json
from pathlib import Path

from curatorkit.connectors.jsonl import JSONLReader


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestJSONLReaderAlpaca:
    def test_basic_alpaca(self, tmp_path):
        p = tmp_path / "data.jsonl"
        write_jsonl(
            [
                {"instruction": "What is 2+2?", "output": "4"},
                {"instruction": "Capital of France?", "output": "Paris"},
            ],
            p,
        )
        samples, _ = JSONLReader(p, format="alpaca").read()
        assert len(samples) == 2
        assert samples[0].instruction == "What is 2+2?"
        assert samples[0].output == "4"

    def test_provenance_attached(self, tmp_path):
        p = tmp_path / "data.jsonl"
        write_jsonl([{"instruction": "Hello", "output": "World"}], p)
        samples, _ = JSONLReader(p, format="alpaca").read()
        assert len(samples[0].provenance_chain) == 1
        rec = samples[0].provenance_chain[0]
        assert rec.step_name == "Connector"
        assert rec.notes["line_number"] == 1
        assert rec.notes["detected_format"] == "alpaca"

    def test_missing_output_field(self, tmp_path):
        # A row with no output column cannot be built as alpaca → becomes rejected
        p = tmp_path / "data.jsonl"
        write_jsonl([{"instruction": "Hi", "input": ""}], p)
        samples, rejected = JSONLReader(p, format="alpaca").read()
        assert len(samples) == 0
        assert len(rejected) == 1


class TestJSONLReaderShareGPT:
    def test_sharegpt_two_turn(self, tmp_path):
        p = tmp_path / "data.jsonl"
        write_jsonl(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Hello"},
                        {"from": "gpt", "value": "Hi there"},
                    ]
                }
            ],
            p,
        )
        samples, _ = JSONLReader(p, format="sharegpt").read()
        assert samples[0].instruction == "Hello"
        assert samples[0].output == "Hi there"

    def test_sharegpt_extra_turns_in_metadata(self, tmp_path):
        p = tmp_path / "data.jsonl"
        write_jsonl(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Q1"},
                        {"from": "gpt", "value": "A1"},
                        {"from": "human", "value": "Q2"},
                        {"from": "gpt", "value": "A2"},
                    ]
                }
            ],
            p,
        )
        samples, _ = JSONLReader(p, format="sharegpt").read()
        # Extra turns are normalized to role/content format before storage
        assert samples[0].metadata["turns"] == [
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]


class TestJSONLReaderAutoDetect:
    def test_auto_alpaca(self, tmp_path):
        p = tmp_path / "data.jsonl"
        write_jsonl([{"instruction": "Hi", "output": "Hello"}], p)
        samples, _ = JSONLReader(p, format="auto").read()
        assert samples[0].provenance_chain[0].notes["detected_format"] == "alpaca"

    def test_auto_sharegpt(self, tmp_path):
        p = tmp_path / "data.jsonl"
        write_jsonl(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Hi"},
                        {"from": "gpt", "value": "Hello"},
                    ]
                }
            ],
            p,
        )
        samples, _ = JSONLReader(p, format="auto").read()
        assert samples[0].provenance_chain[0].notes["detected_format"] == "sharegpt"

    def test_malformed_json_lines_become_rejected(self, tmp_path):
        p = tmp_path / "data.jsonl"
        with open(p, "w") as f:
            f.write('{"instruction": "Good", "output": "Yes"}\n')
            f.write("NOT JSON\n")
            f.write('{"instruction": "Also good", "output": "Indeed"}\n')
        samples, rejected = JSONLReader(p, format="alpaca").read()
        assert len(samples) == 2
        assert len(rejected) == 1
        assert "json_decode_error" in rejected[0].rejection_reason


class TestJSONLReaderFieldMapping:
    def test_field_mapping_override(self, tmp_path):
        p = tmp_path / "data.jsonl"
        write_jsonl([{"prompt": "Tell me about Python", "completion": "Python is great"}], p)
        samples, _ = JSONLReader(
            p,
            format="alpaca",
            field_mapping={"prompt": "instruction", "completion": "output"},
        ).read()
        assert samples[0].instruction == "Tell me about Python"
        assert samples[0].output == "Python is great"


class TestJSONLReaderEdgeCases:
    def test_latin1_chars_in_utf8_file(self, tmp_path):
        p = tmp_path / "data.jsonl"
        with open(p, "w", encoding="utf-8", errors="replace") as f:
            f.write('{"instruction": "caf\ufffd", "output": "yes"}\n')
        samples, _ = JSONLReader(p, format="alpaca").read()
        assert len(samples) == 1

    def test_empty_instruction_produces_sample(self, tmp_path):
        # Empty instruction passes through reader; SchemaGate rejects it
        p = tmp_path / "data.jsonl"
        write_jsonl([{"instruction": "", "output": "something"}], p)
        samples, _ = JSONLReader(p, format="alpaca").read()
        assert len(samples) == 1
        assert samples[0].instruction == ""

    def test_empty_file(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text("")
        samples, rejected = JSONLReader(p).read()
        assert samples == []
        assert rejected == []
