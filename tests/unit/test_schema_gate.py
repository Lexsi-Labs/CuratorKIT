"""Unit tests for SchemaGate — field validation, token limits, encoding."""

from __future__ import annotations

from curatorkit.gates.schema import SchemaGate
from curatorkit.schema import DataSample


def make_sample(**kwargs) -> DataSample:
    defaults = {
        "source_uri": "test://",
        "instruction": "What is Python and how does it work in practice?",
        "output": "Python is a high-level general-purpose programming language.",
    }
    return DataSample(**{**defaults, **kwargs})


class TestSchemaGateFieldValidation:
    def test_passes_valid_sample(self):
        gate = SchemaGate()
        passed, rejected = gate.run([make_sample()])
        assert len(passed) == 1
        assert len(rejected) == 0

    def test_rejects_empty_instruction(self):
        gate = SchemaGate(required_fields=["instruction", "output"])
        passed, rejected = gate.run([make_sample(instruction="")])
        assert len(passed) == 0
        assert len(rejected) == 1
        assert "missing_field:instruction" in rejected[0].rejection_reason

    def test_rejects_none_output(self):
        gate = SchemaGate(required_fields=["instruction", "output"])
        # output defaults to "" which is falsy — should be rejected
        passed, rejected = gate.run([make_sample(output="")])
        assert len(rejected) == 1
        assert "missing_field:output" in rejected[0].rejection_reason

    def test_missing_required_field(self):
        gate = SchemaGate(required_fields=["instruction", "output", "input"])
        # input defaults to "" — should fail
        passed, rejected = gate.run([make_sample()])
        assert len(rejected) == 1


class TestSchemaGateTokenLimits:
    def test_token_underflow(self):
        gate = SchemaGate(min_tokens=100, max_tokens=2048)
        passed, rejected = gate.run([make_sample(instruction="Hi", output="Yes")])
        assert len(rejected) == 1
        assert "below_min_tokens" in rejected[0].rejection_reason

    def test_token_overflow(self):
        gate = SchemaGate(min_tokens=1, max_tokens=5)
        long_text = " ".join(["word"] * 100)
        passed, rejected = gate.run([make_sample(instruction=long_text, output="x")])
        assert len(rejected) == 1
        assert "above_max_tokens" in rejected[0].rejection_reason

    def test_at_boundary_passes(self):
        gate = SchemaGate(min_tokens=3, max_tokens=10)
        # "What is Python" = 3 tokens + "A programming language" = 3 tokens = 6 total
        passed, rejected = gate.run(
            [
                make_sample(
                    instruction="What is Python",
                    output="A programming language",
                )
            ]
        )
        assert len(passed) == 1


class TestSchemaGateEncoding:
    def test_null_byte_rejected(self):
        gate = SchemaGate()
        passed, rejected = gate.run(
            [make_sample(instruction="Hello\x00World", output="test answer here")]
        )
        assert len(rejected) == 1
        assert "encoding_error" in rejected[0].rejection_reason


class TestSchemaGateSidecarContract:
    def test_rejected_sample_carries_full_data(self):
        gate = SchemaGate()
        sample = make_sample(instruction="")
        _, rejected = gate.run([sample])
        assert rejected[0].source_uri == "test://"
        assert rejected[0].rejecting_step == "SchemaGate"

    def test_all_failed_items_are_rejected_samples(self):
        gate = SchemaGate(min_tokens=1000)
        samples = [make_sample() for _ in range(5)]
        passed, rejected = gate.run(samples)
        assert len(passed) + len(rejected) == 5
        assert len(rejected) == 5
