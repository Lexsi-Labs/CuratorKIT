"""Tests for curatorkit/diagnostic/diagnostics.py"""

import json
import tempfile
from pathlib import Path

from curatorkit.diagnostic.diagnostics import PipelineDiagnostics
from curatorkit.diagnostic.failure_modes import FailureDiagnosis, FailureMode
from curatorkit.schema import DataSample, RejectedSample


def _make_diagnosed(mode: FailureMode, recovered: bool = False) -> RejectedSample:
    s = RejectedSample(
        source_uri="test/doc",
        instruction="What?",
        output="An answer.",
        rejection_reason="below_threshold",
        rejecting_step="HallucinationGate",
    )
    recovered_sample = None
    if recovered:
        recovered_sample = DataSample(
            source_uri="test/doc",
            instruction="What?",
            output="A regenerated passing answer.",
        )
    s.diagnosis = FailureDiagnosis.from_mode(
        mode, evidence=[True], probe_calls=2, recovered_sample=recovered_sample
    )
    return s


def _make_undiagnosed() -> RejectedSample:
    return RejectedSample(
        source_uri="test/doc",
        instruction="What?",
        output="An answer.",
        rejection_reason="below_threshold",
        rejecting_step="HallucinationGate",
    )


class TestPipelineDiagnostics:
    def test_empty_probe_recovery(self):
        d = PipelineDiagnostics()
        assert d.probe_recovery_count() == 0
        assert d.to_dict()["probe_recovery_pct"] == 0.0

    def test_probe_recovery_count_and_pct(self):
        d = PipelineDiagnostics()
        d.record(_make_diagnosed(FailureMode.GENERATOR_TEMPERATURE, recovered=True))
        d.record(_make_diagnosed(FailureMode.SOURCE_AMBIGUOUS))  # not recovered
        assert d.probe_recovery_count() == 1
        assert d.to_dict()["probe_recovery_pct"] == 0.5

    def test_mode_counts_aggregates_correctly(self):
        d = PipelineDiagnostics()
        d.record(_make_diagnosed(FailureMode.GENERATOR_TEMPERATURE))
        d.record(_make_diagnosed(FailureMode.GENERATOR_TEMPERATURE))
        d.record(_make_diagnosed(FailureMode.SOURCE_AMBIGUOUS))
        counts = d.mode_counts()
        assert counts["generator_temperature"] == 2
        assert counts["source_ambiguous"] == 1

    def test_mode_counts_undiagnosed(self):
        d = PipelineDiagnostics()
        d.record(_make_undiagnosed())
        counts = d.mode_counts()
        assert counts.get("undiagnosed", 0) == 1

    def test_recovered_samples_counted(self):
        d = PipelineDiagnostics()
        d.record(_make_diagnosed(FailureMode.GENERATOR_TEMPERATURE, recovered=True))
        d.record(_make_diagnosed(FailureMode.NEAR_DUPLICATE))  # not recovered
        d.record(_make_undiagnosed())  # no diagnosis at all
        assert d.probe_recovery_count() == 1

    def test_undiagnosed_never_counted_as_recovered(self):
        d = PipelineDiagnostics()
        d.record(_make_undiagnosed())
        d.record(_make_undiagnosed())
        assert d.probe_recovery_count() == 0

    def test_total_probe_calls(self):
        d = PipelineDiagnostics()
        s1 = _make_diagnosed(FailureMode.GENERATOR_TEMPERATURE)
        s1.diagnosis.probe_calls = 3
        s2 = _make_diagnosed(FailureMode.SOURCE_AMBIGUOUS)
        s2.diagnosis.probe_calls = 6
        d.record(s1)
        d.record(s2)
        assert d.total_probe_calls() == 9

    def test_to_dict_structure(self):
        d = PipelineDiagnostics()
        d.record(_make_diagnosed(FailureMode.GENERATOR_TEMPERATURE))
        result = d.to_dict()
        assert "total_diagnosed" in result
        assert "probe_recovered" in result
        assert "probe_recovery_pct" in result
        assert "total_probe_calls" in result
        assert "mode_counts" in result
        assert result["total_diagnosed"] == 1

    def test_write_summary_produces_valid_json(self):
        d = PipelineDiagnostics()
        d.record(_make_diagnosed(FailureMode.DOMAIN_MISMATCH))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "summary.json"
            d.write_summary(path)
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["total_diagnosed"] == 1
