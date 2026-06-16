"""
Integration tests: Pipeline.run() / run_async() with mock probe attached to a gate.
"""

import asyncio
from unittest.mock import MagicMock

from curatorkit.diagnostic.diagnostics import PipelineDiagnostics
from curatorkit.diagnostic.failure_modes import FailureDiagnosis, FailureMode
from curatorkit.interfaces import BaseGate, BaseReader
from curatorkit.pipeline import Pipeline
from curatorkit.schema import DataSample, RejectedSample


def _make_sample(i: int = 0) -> DataSample:
    return DataSample(
        source_uri=f"test/doc/{i}",
        instruction=f"Question {i}?",
        input="Source text here.",
        output=f"Answer {i}.",
    )


def _make_rejected_sample() -> RejectedSample:
    return RejectedSample(
        source_uri="test/doc/rejected",
        instruction="Rejected question?",
        input="Source.",
        output="Bad answer.",
        rejection_reason="below_threshold:0.4",
        rejecting_step="MockGate",
    )


class _FixedGate(BaseGate):
    """Concrete BaseGate subclass — returns pre-configured pass/reject lists."""

    def __init__(self, passes: list, rejected_list: list) -> None:
        self._passes = passes
        self._rejected_list = rejected_list
        self.probe = None

    def run(self, samples):
        return self._passes, self._rejected_list


class _FixedReader(BaseReader):
    """Concrete BaseReader subclass — returns pre-configured samples."""

    def __init__(self, samples: list) -> None:
        self._samples = samples

    def read(self):
        return self._samples, []


def _make_mock_probe(
    mode: FailureMode = FailureMode.THRESHOLD_MARGINAL,
    recovered_sample: DataSample | None = None,
):
    """Build a mock probe whose diagnose_batch returns one fixed diagnosis per sample."""
    probe = MagicMock()

    def _diagnose_batch(rejected, concurrency=32):
        return [
            FailureDiagnosis.from_mode(
                mode,
                evidence=[True],
                probe_calls=3,
                recovered_sample=recovered_sample,
            )
            for _ in rejected
        ]

    probe.diagnose_batch = MagicMock(side_effect=_diagnose_batch)
    return probe


class TestPipelineRunWithProbe:
    def test_probe_called_for_each_rejected(self):
        samples = [_make_sample(i) for i in range(3)]
        rejected = [_make_rejected_sample(), _make_rejected_sample()]

        probe = _make_mock_probe()
        gate = _FixedGate(passes=samples, rejected_list=rejected)
        gate.probe = probe

        pipeline = Pipeline([_FixedReader(samples), gate])
        result = pipeline.run()

        # One batched call covering every rejected sample
        assert probe.diagnose_batch.call_count == 1
        (batch_arg,) = probe.diagnose_batch.call_args.args
        assert list(batch_arg) == rejected
        assert all(r.diagnosis is not None for r in result.rejected)

    def test_rejected_samples_carry_diagnosis(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]

        probe = _make_mock_probe(FailureMode.GENERATOR_TEMPERATURE)
        gate = _FixedGate(passes=samples, rejected_list=rejected)
        gate.probe = probe

        result = Pipeline([_FixedReader(samples), gate]).run()

        assert result.rejected[0].diagnosis is not None
        assert result.rejected[0].diagnosis.mode is FailureMode.GENERATOR_TEMPERATURE

    def test_recovered_sample_reenters_accepted_pool(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]
        recovered = _make_sample(99)

        probe = _make_mock_probe(FailureMode.GENERATOR_TEMPERATURE, recovered_sample=recovered)
        gate = _FixedGate(passes=samples, rejected_list=rejected)
        gate.probe = probe

        result = Pipeline([_FixedReader(samples), gate]).run()

        assert recovered in result.passed
        assert result.stage_counts["_FixedGate"]["probe_recovered"] == 1

    def test_without_probe_no_diagnosis_set(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]

        gate = _FixedGate(passes=samples, rejected_list=rejected)
        result = Pipeline([_FixedReader(samples), gate]).run()

        assert result.rejected[0].diagnosis is None

    def test_diagnostics_accumulator_populated(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]

        probe = _make_mock_probe()
        gate = _FixedGate(passes=samples, rejected_list=rejected)
        gate.probe = probe

        diag_acc = PipelineDiagnostics()
        result = Pipeline([_FixedReader(samples), gate], diagnostics=diag_acc).run()

        assert result.diagnostics is diag_acc
        assert len(diag_acc._diagnosed) == 1

    def test_pipeline_result_diagnostics_none_without_probe(self):
        samples = [_make_sample()]
        gate = _FixedGate(passes=samples, rejected_list=[])

        result = Pipeline([_FixedReader(samples), gate]).run()
        assert result.diagnostics is None

    def test_backwards_compat_no_probe_rejected_unchanged(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]

        gate = _FixedGate(passes=samples, rejected_list=rejected)
        result = Pipeline([_FixedReader(samples), gate]).run()

        assert len(result.rejected) == 1
        assert result.rejected[0].diagnosis is None


class TestPipelineRunAsyncWithProbe:
    def test_async_probe_called_for_rejected(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]

        probe = _make_mock_probe()
        gate = _FixedGate(passes=samples, rejected_list=rejected)
        gate.probe = probe

        result = asyncio.run(Pipeline([_FixedReader(samples), gate]).run_async())

        assert probe.diagnose_batch.call_count == 1
        assert result.rejected[0].diagnosis is not None

    def test_async_recovered_sample_reenters_accepted_pool(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]
        recovered = _make_sample(99)

        probe = _make_mock_probe(FailureMode.GENERATOR_TEMPERATURE, recovered_sample=recovered)
        gate = _FixedGate(passes=samples, rejected_list=rejected)
        gate.probe = probe

        result = asyncio.run(Pipeline([_FixedReader(samples), gate]).run_async())

        assert recovered in result.passed
        assert result.stage_counts["_FixedGate"]["probe_recovered"] == 1

    def test_async_without_probe_no_diagnosis(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]

        gate = _FixedGate(passes=samples, rejected_list=rejected)
        result = asyncio.run(Pipeline([_FixedReader(samples), gate]).run_async())

        assert result.rejected[0].diagnosis is None

    def test_async_diagnostics_accumulator_populated(self):
        samples = [_make_sample()]
        rejected = [_make_rejected_sample()]

        probe = _make_mock_probe()
        gate = _FixedGate(passes=samples, rejected_list=rejected)
        gate.probe = probe

        diag_acc = PipelineDiagnostics()
        result = asyncio.run(
            Pipeline([_FixedReader(samples), gate], diagnostics=diag_acc).run_async()
        )

        assert result.diagnostics is diag_acc
        assert len(diag_acc._diagnosed) == 1

    def test_async_pipeline_result_diagnostics_none_without_probe(self):
        samples = [_make_sample()]
        gate = _FixedGate(passes=samples, rejected_list=[])

        result = asyncio.run(Pipeline([_FixedReader(samples), gate]).run_async())
        assert result.diagnostics is None
