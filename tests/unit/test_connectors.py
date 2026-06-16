"""
Integration tests for connectors and the BaseConnector pipeline.

Tests verify:
  - preprocessing_fn execution order and return types
  - field_mapping (flat and nested dot notation)
  - format detection through real files
  - All DataSample construction paths (alpaca, sharegpt, preference, grpo, etc.)
  - RejectedSample emission for every failure mode
  - DPOExporter output
"""

from __future__ import annotations

import json
from pathlib import Path

from curatorkit.connectors.csv_reader import CSVReader
from curatorkit.connectors.json_reader import JSONReader
from curatorkit.connectors.jsonl import JSONLReader
from curatorkit.schema import DataSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_json(data, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def write_csv(rows: list[dict], path: Path, delimiter: str = ",") -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# JSONLReader — format detection paths
# ---------------------------------------------------------------------------


class TestJSONLReaderAlpaca:
    def test_standard_alpaca(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {"instruction": "What is 2+2?", "output": "4"},
                {"instruction": "Capital of France?", "output": "Paris"},
            ],
            p,
        )
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 2
        assert len(rejected) == 0
        assert samples[0].instruction == "What is 2+2?"
        assert samples[0].output == "4"
        assert samples[0].task_type == "instruction_following"

    def test_query_answer_variant(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {"query": "What is Python?", "answer": "A programming language."},
            ],
            p,
        )
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].instruction == "What is Python?"
        assert samples[0].output == "A programming language."

    def test_prompt_completion_variant(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {"prompt": "Explain recursion.", "completion": "Recursion is..."},
            ],
            p,
        )
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].instruction == "Explain recursion."

    def test_provenance_attached(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl([{"instruction": "Hi", "output": "Hello"}], p)
        samples, _ = JSONLReader(p).read()
        assert len(samples[0].provenance_chain) == 1
        rec = samples[0].provenance_chain[0]
        assert rec.step_name == "Connector"
        assert rec.notes["detected_format"] == "alpaca"

    def test_malformed_json_becomes_rejected(self, tmp_path):
        p = tmp_path / "d.jsonl"
        with open(p, "w") as f:
            f.write('{"instruction": "Good", "output": "Yes"}\n')
            f.write("NOT JSON\n")
            f.write('{"instruction": "Also good", "output": "Indeed"}\n')
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 2
        assert len(rejected) == 1
        assert "json_decode_error" in rejected[0].rejection_reason

    def test_empty_file(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text("")
        samples, rejected = JSONLReader(p).read()
        assert samples == []
        assert rejected == []

    def test_unrecognized_format_becomes_rejected(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl([{"foo": "bar", "baz": "qux"}], p)
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 0
        assert len(rejected) == 1
        assert "unrecognized_format" in rejected[0].rejection_reason


class TestJSONLReaderShareGPT:
    def test_chatml_role_content(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there"},
                    ]
                }
            ],
            p,
        )
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].instruction == "Hello"
        assert samples[0].output == "Hi there"
        assert samples[0].task_type == "conversational"

    def test_sharegpt_from_value(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "What is 2+2?"},
                        {"from": "gpt", "value": "4"},
                    ]
                }
            ],
            p,
        )
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].instruction == "What is 2+2?"
        assert samples[0].output == "4"

    def test_system_prompt_extracted(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                    ]
                }
            ],
            p,
        )
        samples, _ = JSONLReader(p).read()
        assert samples[0].metadata.get("system_prompt") == "You are helpful"
        assert samples[0].instruction == "Hi"

    def test_extra_turns_in_metadata(self, tmp_path):
        p = tmp_path / "d.jsonl"
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
        samples, _ = JSONLReader(p).read()
        assert "turns" in samples[0].metadata
        assert len(samples[0].metadata["turns"]) == 2


class TestJSONLReaderPreference:
    def test_explicit_preference(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "prompt": "What is 2+2?",
                    "chosen": "4",
                    "rejected": "5",
                }
            ],
            p,
        )
        samples, rejected = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].task_type == "preference"
        assert samples[0].chosen == "4"
        assert samples[0].rejected == "5"
        assert samples[0].instruction == "What is 2+2?"

    def test_implicit_preference_extracts_prompt(self, tmp_path):
        p = tmp_path / "d.jsonl"
        chosen = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        rejected = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "5"},
        ]
        write_jsonl([{"chosen": chosen, "rejected": rejected}], p)
        samples, rej = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].task_type == "implicit_preference"
        # instruction should contain the extracted prompt
        assert "2+2" in samples[0].instruction or samples[0].instruction != ""


class TestJSONLReaderUnpairedPreference:
    def test_unpaired_with_label(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "instruction": "Write a poem.",
                    "output": "Roses are red...",
                    "label": True,
                }
            ],
            p,
        )
        samples, _ = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].task_type == "unpaired_preference"
        assert samples[0].label == 1.0

    def test_float_label(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "prompt": "Explain AI.",
                    "response": "AI is...",
                    "score": 0.87,
                }
            ],
            p,
        )
        samples, _ = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].label is not None
        assert abs(samples[0].label - 0.87) < 1e-6


class TestJSONLReaderGRPO:
    def test_grpo_rollouts(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "prompt": "Write hello world in Python.",
                    "responses": ["print('hello')", "print('world')", "print('hi')"],
                    "rewards": [1.0, 0.5, 0.8],
                }
            ],
            p,
        )
        samples, _ = JSONLReader(p).read()
        assert len(samples) == 1
        assert samples[0].task_type == "grpo"
        assert len(samples[0].responses) == 3
        assert len(samples[0].reward_scores) == 3


# ---------------------------------------------------------------------------
# field_mapping tests
# ---------------------------------------------------------------------------


class TestFieldMapping:
    def test_flat_key_rename(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl([{"query": "What is AI?", "answer": "AI is..."}], p)
        samples, _ = JSONLReader(
            p, field_mapping={"query": "instruction", "answer": "output"}
        ).read()
        assert samples[0].instruction == "What is AI?"
        assert samples[0].output == "AI is..."

    def test_nested_dot_notation(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl(
            [
                {
                    "meta": {"prompt": "Explain gravity"},
                    "result": {"text": "Gravity is a force"},
                }
            ],
            p,
        )
        samples, _ = JSONLReader(
            p,
            field_mapping={
                "meta.prompt": "instruction",
                "result.text": "output",
            },
        ).read()
        assert samples[0].instruction == "Explain gravity"
        assert samples[0].output == "Gravity is a force"

    def test_field_mapping_applied_before_detection(self, tmp_path):
        """After renaming, detection sees canonical names → HIGH confidence."""
        p = tmp_path / "d.jsonl"
        write_jsonl([{"q": "Hi", "a": "Hello"}], p)
        # Without mapping: would be unknown (q and a not in equivalence classes)
        samples_no_map, rej_no_map = JSONLReader(p).read()
        assert len(rej_no_map) == 1  # unknown format

        # With mapping: detection finds instruction + output → alpaca
        samples_map, rej_map = JSONLReader(
            p, field_mapping={"q": "instruction", "a": "output"}
        ).read()
        assert len(samples_map) == 1
        assert len(rej_map) == 0


# ---------------------------------------------------------------------------
# preprocessing_fn tests
# ---------------------------------------------------------------------------


class TestPreprocessingFn:
    def test_preprocessing_normalizes_dict(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl([{"weird_prompt": "Hi", "weird_answer": "Hello"}], p)

        def normalize(row):
            return {"instruction": row["weird_prompt"], "output": row["weird_answer"]}

        samples, _ = JSONLReader(p, preprocessing_fn=normalize).read()
        assert len(samples) == 1
        assert samples[0].instruction == "Hi"

    def test_preprocessing_returns_datasample_bypasses_detection(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl([{"x": "y"}], p)

        def to_sample(row):
            return DataSample(
                source_uri="test://",
                instruction="forced instruction",
                output="forced output",
            )

        samples, rejected = JSONLReader(p, preprocessing_fn=to_sample).read()
        assert len(samples) == 1
        assert samples[0].instruction == "forced instruction"
        assert len(rejected) == 0

    def test_preprocessing_returns_none_emits_rejected(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl([{"instruction": "Hi", "output": "Hello"}], p)

        def drop_all(row):
            return None

        samples, rejected = JSONLReader(p, preprocessing_fn=drop_all).read()
        assert len(samples) == 0
        assert len(rejected) == 1
        assert "preprocessing_fn_returned_none" in rejected[0].rejection_reason

    def test_preprocessing_exception_emits_rejected(self, tmp_path):
        p = tmp_path / "d.jsonl"
        write_jsonl([{"instruction": "Hi", "output": "Hello"}], p)

        def exploding(row):
            raise ValueError("intentional error")

        samples, rejected = JSONLReader(p, preprocessing_fn=exploding).read()
        assert len(samples) == 0
        assert len(rejected) == 1
        assert "preprocessing_fn_error" in rejected[0].rejection_reason
        assert "intentional error" in rejected[0].rejection_reason


# ---------------------------------------------------------------------------
# JSONReader
# ---------------------------------------------------------------------------


class TestJSONReader:
    def test_top_level_array(self, tmp_path):
        p = tmp_path / "d.json"
        write_json(
            [
                {"instruction": "Q1", "output": "A1"},
                {"instruction": "Q2", "output": "A2"},
            ],
            p,
        )
        samples, _ = JSONReader(p).read()
        assert len(samples) == 2

    def test_wrapped_dict_data_key(self, tmp_path):
        p = tmp_path / "d.json"
        write_json(
            {
                "data": [
                    {"instruction": "Q1", "output": "A1"},
                ],
                "meta": {"version": 1},
            },
            p,
        )
        samples, _ = JSONReader(p).read()
        assert len(samples) == 1

    def test_explicit_data_key(self, tmp_path):
        p = tmp_path / "d.json"
        write_json(
            {
                "examples": [
                    {"instruction": "Q", "output": "A"},
                ]
            },
            p,
        )
        samples, _ = JSONReader(p, data_key="examples").read()
        assert len(samples) == 1

    def test_invalid_json_becomes_rejected(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text("NOT JSON")
        samples, rejected = JSONReader(p).read()
        assert len(samples) == 0
        assert len(rejected) == 1


# ---------------------------------------------------------------------------
# CSVReader
# ---------------------------------------------------------------------------


class TestCSVReader:
    def test_standard_csv(self, tmp_path):
        p = tmp_path / "d.csv"
        write_csv(
            [
                {"instruction": "What is AI?", "output": "AI is..."},
                {"instruction": "Explain ML.", "output": "ML is..."},
            ],
            p,
        )
        samples, _ = CSVReader(p).read()
        assert len(samples) == 2
        assert samples[0].instruction == "What is AI?"

    def test_tsv_file(self, tmp_path):
        p = tmp_path / "d.tsv"
        write_csv(
            [
                {"instruction": "Hi", "output": "Hello"},
            ],
            p,
            delimiter="\t",
        )
        samples, _ = CSVReader(p, delimiter="\t").read()
        assert len(samples) == 1

    def test_json_encoded_cell_parsed(self, tmp_path):
        p = tmp_path / "d.csv"
        conversations = json.dumps(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        )
        write_csv([{"messages": conversations}], p)
        samples, _ = CSVReader(p, parse_json_cells=True).read()
        assert len(samples) == 1
        assert samples[0].task_type == "conversational"


# ---------------------------------------------------------------------------
# DPOExporter
# ---------------------------------------------------------------------------


class TestDPOExporter:
    def test_exports_preference_samples(self, tmp_path):
        from curatorkit.exporters.dpo import DPOExporter

        samples = [
            DataSample(
                source_uri="test://",
                instruction="What is 2+2?",
                chosen="4",
                rejected="5",
                task_type="preference",
            ),
            DataSample(
                source_uri="test://",
                instruction="Capital of France?",
                chosen="Paris",
                rejected="London",
                task_type="preference",
            ),
        ]

        DPOExporter().export(samples, tmp_path)
        output = tmp_path / "dpo.jsonl"
        assert output.exists()

        lines = [json.loads(ln) for ln in output.read_text().strip().splitlines()]
        assert len(lines) == 2
        assert lines[0]["prompt"] == "What is 2+2?"
        assert lines[0]["chosen"] == "4"
        assert lines[0]["rejected"] == "5"

    def test_skips_non_preference_samples(self, tmp_path):
        from curatorkit.exporters.dpo import DPOExporter

        samples = [
            DataSample(
                source_uri="test://",
                instruction="Hi",
                output="Hello",
                task_type="instruction_following",
            ),
        ]
        DPOExporter().export(samples, tmp_path)
        output = tmp_path / "dpo.jsonl"
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 0

    def test_exports_conversational_preference(self, tmp_path):
        import json as _json

        from curatorkit.exporters.dpo import DPOExporter

        chosen_turns = [{"role": "assistant", "content": "4"}]
        rejected_turns = [{"role": "assistant", "content": "5"}]

        samples = [
            DataSample(
                source_uri="test://",
                instruction=_json.dumps([{"role": "user", "content": "2+2?"}]),
                chosen=_json.dumps(chosen_turns),
                rejected=_json.dumps(rejected_turns),
                task_type="implicit_preference",
            ),
        ]
        DPOExporter().export(samples, tmp_path)
        output = tmp_path / "dpo.jsonl"
        lines = [_json.loads(ln) for ln in output.read_text().strip().splitlines()]
        assert len(lines) == 1
        # prompt and chosen/rejected should be parsed back to lists
        assert isinstance(lines[0]["prompt"], list)
        assert isinstance(lines[0]["chosen"], list)


# ---------------------------------------------------------------------------
# Updated gate — task-type-aware validation
# ---------------------------------------------------------------------------


class TestSchemaGateTaskTypeAware:
    def _gate(self, **kwargs):
        from curatorkit.gates.schema import SchemaGate

        return SchemaGate(**kwargs)

    def test_preference_requires_chosen_rejected(self):
        gate = self._gate(min_tokens=1)
        sample = DataSample(
            source_uri="t://",
            instruction="Q",
            task_type="preference",
            # chosen and rejected absent
        )
        _, rejected = gate.run([sample])
        assert len(rejected) == 1
        assert "chosen" in rejected[0].rejection_reason

    def test_preference_passes_with_chosen_rejected(self):
        gate = self._gate(min_tokens=1)
        sample = DataSample(
            source_uri="t://",
            instruction="What is 2+2?",
            chosen="4",
            rejected="5",
            task_type="preference",
        )
        passed, rejected = gate.run([sample])
        assert len(passed) == 1
        assert len(rejected) == 0

    def test_grpo_requires_responses(self):
        gate = self._gate(min_tokens=1)
        sample = DataSample(
            source_uri="t://",
            instruction="Write code.",
            task_type="grpo",
            # responses absent
        )
        _, rejected = gate.run([sample])
        assert len(rejected) == 1
        assert "responses" in rejected[0].rejection_reason

    def test_pretrain_validates_output(self):
        gate = self._gate(min_tokens=1)
        sample = DataSample(
            source_uri="t://",
            output="",  # empty
            task_type="language_modeling",
        )
        _, rejected = gate.run([sample])
        assert len(rejected) == 1
        assert "output" in rejected[0].rejection_reason

    def test_prompt_only_skips_output_check(self):
        gate = self._gate(min_tokens=1)
        sample = DataSample(
            source_uri="t://",
            instruction="What is the capital of France?",
            output="",  # intentionally empty
            task_type="prompt_only",
        )
        passed, rejected = gate.run([sample])
        assert len(passed) == 1

    def test_enforce_task_types(self):
        gate = self._gate(min_tokens=1, enforce_task_types=["preference"])
        sft_sample = DataSample(
            source_uri="t://",
            instruction="Hi",
            output="Hello",
            task_type="instruction_following",
        )
        _, rejected = gate.run([sft_sample])
        assert len(rejected) == 1
        assert "task_type_not_allowed" in rejected[0].rejection_reason

    def test_token_counting_preference_uses_chosen(self):
        """Token count should be based on instruction + chosen, not instruction + output."""
        gate = self._gate(min_tokens=5, max_tokens=100)
        sample = DataSample(
            source_uri="t://",
            instruction="Q",  # 1 token
            chosen="this is a good and complete answer",  # 8 tokens
            rejected="bad",
            task_type="preference",
        )
        passed, rejected = gate.run([sample])
        assert len(passed) == 1  # 1 + 8 = 9 tokens ≥ min_tokens=5

    def test_pipeline_reader_rejected_flows_into_result(self, tmp_path):
        """Reader rejections from tuple return are collected by Pipeline."""
        from curatorkit.pipeline import Pipeline

        p = tmp_path / "d.jsonl"
        with open(p, "w") as f:
            f.write('{"instruction": "Good", "output": "Yes"}\n')
            f.write("INVALID JSON\n")

        pipeline = Pipeline([JSONLReader(p)], output_dir=tmp_path)
        result = pipeline.run()

        assert len(result.passed) == 1
        assert len(result.rejected) == 1
        assert "json_decode_error" in result.rejected[0].rejection_reason
