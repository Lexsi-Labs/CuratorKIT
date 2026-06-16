"""Unit tests for schema.py — DataSample, ProvenanceRecord, RejectedSample."""

from __future__ import annotations

from datetime import UTC, datetime

from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample


def make_provenance(**kwargs) -> ProvenanceRecord:
    defaults = {
        "step_name": "TestStep",
        "step_version": "0.1.0",
        "timestamp": datetime.now(UTC).replace(tzinfo=None),
        "config_hash": "abc123",
        "notes": {},
    }
    return ProvenanceRecord(**{**defaults, **kwargs})


class TestDataSample:
    def test_auto_uuid(self):
        s1 = DataSample(source_uri="s3://test", instruction="Hello")
        s2 = DataSample(source_uri="s3://test", instruction="Hello")
        assert s1.id != s2.id

    def test_defaults(self):
        s = DataSample(source_uri="s3://test", instruction="Hi")
        assert s.input == ""
        assert s.output == ""
        assert s.responses == []
        assert s.reward_scores == []
        assert s.task_type == "instruction_following"
        assert s.metadata == {}
        assert s.provenance_chain == []

    def test_provenance_append_only(self):
        s = DataSample(source_uri="s3://test", instruction="Hi")
        p1 = make_provenance(step_name="Step1")
        p2 = make_provenance(step_name="Step2")
        s.append_provenance(p1)
        s.append_provenance(p2)
        assert len(s.provenance_chain) == 2
        assert s.provenance_chain[0].step_name == "Step1"
        assert s.provenance_chain[1].step_name == "Step2"

    def test_provenance_immutability_of_prior_records(self):
        s = DataSample(source_uri="s3://test", instruction="Hi")
        p1 = make_provenance(step_name="OriginalStep")
        s.append_provenance(p1)
        original_name = s.provenance_chain[0].step_name
        # Appending more records should not change existing ones
        s.append_provenance(make_provenance(step_name="LaterStep"))
        assert s.provenance_chain[0].step_name == original_name


class TestRejectedSample:
    def test_rejected_has_reason_and_step(self):
        r = RejectedSample(
            source_uri="s3://test",
            instruction="Hi",
            rejection_reason="missing_field:output",
            rejecting_step="SchemaGate",
        )
        assert r.rejection_reason == "missing_field:output"
        assert r.rejecting_step == "SchemaGate"

    def test_rejected_inherits_datasample_fields(self):
        r = RejectedSample(
            source_uri="s3://test",
            instruction="Hi",
            rejection_reason="below_min_tokens:2",
            rejecting_step="SchemaGate",
        )
        assert r.task_type == "instruction_following"
        assert r.provenance_chain == []
