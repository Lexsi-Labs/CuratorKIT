"""
Regression test: RewardRefiner-recovered samples must land in exported
files, not just in the returned CuratorResult.

Reported bug: 18 samples generated -> RewardGate passes 13, rejects 5 ->
RewardRefiner recovers 2 of the 5 -> CuratorResult.passed has 15, but
sft_alpaca.jsonl on disk only has 13 rows.

Root cause: Curator._build_steps() appends exporters as the last pipeline
step, so they run *inside* Pipeline.run()/run_async() using the pre-recovery
`passed` list. RewardRefiner recovery only happens in Curator.run(), *after*
Pipeline.run()/run_async() has already returned (and already exported) —
so exported files reflect the stale 13-sample snapshot. Curator.run_async()
additionally never invoked the refiner at all, so enable_reward_refiner had
zero effect when a caller used that entry point directly.

The fix: Curator._build_steps() skips the inline exporter step whenever a
RewardRefiner was built for this run, and Curator.run()/run_async() run the
refiner first and export afterwards (via the new _run_exporters helper),
using the post-recovery sample list in both places.
"""

from __future__ import annotations

import asyncio
import json

from curatorkit.curator import Curator, CuratorConfig
from curatorkit.exporters.alpaca import AlpacaExporter
from curatorkit.interfaces import BaseGate, BaseReader
from curatorkit.schema import DataSample, RejectedSample

_NUM_SAMPLES = 18
_NUM_REWARD_PASSED = 13
_NUM_RECOVERED = 2
_NUM_TOTAL_EXPECTED = _NUM_REWARD_PASSED + _NUM_RECOVERED  # 15


class _FakeReader(BaseReader):
    def read(self):
        samples = [
            DataSample(source_uri="t://fake", instruction=f"q{i}", output=f"a{i}")
            for i in range(_NUM_SAMPLES)
        ]
        return samples, []


class _FakeRewardGate(BaseGate):
    """Mirrors the reported split: 13 pass, 5 rejected by RewardGate."""

    def run(self, samples):
        passed = samples[:_NUM_REWARD_PASSED]
        rejected = [
            RejectedSample(
                **s.model_dump(exclude={"rejection_reason", "rejecting_step"}),
                rejection_reason="below_reward_threshold:0.4",
                rejecting_step="RewardGate",
            )
            for s in samples[_NUM_REWARD_PASSED:]
        ]
        return passed, rejected


class _FakeRefiner:
    """Mirrors the reported split: 2 of the 5 rejects recover, 3 stay rejected."""

    def refine(self, rejected):
        recovered = [
            DataSample(
                source_uri=r.source_uri,
                instruction=r.instruction,
                output=f"refined-{r.output}",
            )
            for r in rejected[:_NUM_RECOVERED]
        ]
        still_rejected = rejected[_NUM_RECOVERED:]
        return recovered, still_rejected


def _rig_curator(tmp_path):
    """Build a Curator whose _build_steps is swapped for a minimal fake
    pipeline (reader -> reward gate -> inline exporter), with a fake
    RewardRefiner attached.

    The inline AlpacaExporter step is deliberately unconditional here (unlike
    the real, fixed _build_steps, which now skips it whenever a refiner is
    active) so this harness reproduces the reported symptom precisely: an
    exporter that runs inline on the pre-recovery `passed` list, writing 13
    rows to disk. It isolates what actually changed for this fix — the
    ordering/dispatch in Curator.run()/run_async() — from _build_steps'
    own exporter-skipping logic, which is exercised separately by the
    existing _build_steps-level tests.
    """
    cfg = CuratorConfig(
        dataset="unused",
        output_dir=str(tmp_path),
        export_formats=["alpaca"],
        enable_reward_refiner=True,
    )
    curator = Curator(cfg)

    def _fake_build_steps(self, include_exporters=True):
        self._reward_refiner = _FakeRefiner()
        steps = [_FakeReader(), _FakeRewardGate()]
        if include_exporters:
            steps.append(AlpacaExporter())
        return steps

    curator._build_steps = _fake_build_steps.__get__(curator, Curator)
    return curator


class TestRewardRefinerExportOrderSync:
    def test_result_passed_includes_recovered_samples(self, tmp_path):
        curator = _rig_curator(tmp_path)
        result = curator.run()
        assert len(result.passed) == _NUM_TOTAL_EXPECTED

    def test_exported_file_includes_recovered_samples(self, tmp_path):
        curator = _rig_curator(tmp_path)
        curator.run()

        alpaca_path = tmp_path / "sft_alpaca.jsonl"
        assert alpaca_path.exists()
        lines = alpaca_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == _NUM_TOTAL_EXPECTED

        records = [json.loads(line) for line in lines]
        outputs = {r["output"] for r in records}
        assert "refined-a13" in outputs
        assert "refined-a14" in outputs

    def test_stage_counts_records_reward_refiner_recovery(self, tmp_path):
        """Recovery is otherwise invisible: the RewardGate's own stage_counts
        entry doesn't shrink when the refiner later saves some of its
        rejects, and RewardRefiner isn't a Pipeline step, so without this
        entry nothing in the manifest/dataset card would show that 2 of the
        5 RewardGate rejects were recovered after the fact.
        """
        curator = _rig_curator(tmp_path)
        result = curator.run()

        assert "RewardRefiner" in result.stage_counts
        refiner_counts = result.stage_counts["RewardRefiner"]
        assert refiner_counts["input_count"] == 5
        assert refiner_counts["probe_recovered"] == _NUM_RECOVERED
        assert refiner_counts["rejected_count"] == 5


class TestRewardRefinerExportOrderAsync:
    def test_run_async_applies_refiner_and_exports_recovered_samples(self, tmp_path):
        curator = _rig_curator(tmp_path)
        result = asyncio.run(curator.run_async())

        assert len(result.passed) == _NUM_TOTAL_EXPECTED

        alpaca_path = tmp_path / "sft_alpaca.jsonl"
        assert alpaca_path.exists()
        lines = alpaca_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == _NUM_TOTAL_EXPECTED
