"""
PipelineDiagnostics — run-level accumulator for failure diagnoses.

Held by the Pipeline instance when the probe is active. Passed through
PipelineResult to Curator, then accessible to the caller via
result.diagnostics.

Recovery is INLINE: probe_recovery_count() reports samples where the
DiagnosticProbe actually produced a passing re-generation. This replaces
the old hypothetical recovery_rate() which counted RECOVERABLE dict flags.

Typical uses:
  mode_counts() and probe_recovery_count() feed the per-mode rejection
  breakdown written to diagnostic_summary.json; total_probe_calls()
  tracks the LLM budget the probe consumed, so recovery yield can be
  cost-normalised. by_stage() breaks the same summary down per
  rejecting_step (gate name) — the top-level totals pool every
  probe-enabled gate together, which hides which gate's probe is actually
  doing the recovering whenever more than one gate has a probe attached.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from curatorkit.schema import RejectedSample


class PipelineDiagnostics:
    def __init__(self) -> None:
        self._diagnosed: list[RejectedSample] = []

    def record(self, sample: RejectedSample) -> None:
        self._diagnosed.append(sample)

    @staticmethod
    def _summarize(samples: list[RejectedSample]) -> dict[str, Any]:
        """The to_dict()-shaped summary for one group of diagnosed samples —
        shared by the pooled top-level totals and each by_stage() group so
        the two can never drift out of sync."""
        total = len(samples)
        recovered = sum(
            1 for s in samples if s.diagnosis is not None and s.diagnosis.recovered_sample is not None
        )
        mode_counter: Counter = Counter()
        for s in samples:
            mode_counter[s.diagnosis.mode.value if s.diagnosis else "undiagnosed"] += 1
        probe_calls = sum(s.diagnosis.probe_calls for s in samples if s.diagnosis)
        return {
            "total_diagnosed": total,
            "probe_recovered": recovered,
            "probe_recovery_pct": round(recovered / total, 4) if total else 0.0,
            "total_probe_calls": probe_calls,
            "mode_counts": dict(mode_counter),
        }

    def probe_recovery_count(self) -> int:
        """Number of samples where the probe produced an inline passing re-generation."""
        return sum(
            1
            for s in self._diagnosed
            if s.diagnosis is not None and s.diagnosis.recovered_sample is not None
        )

    def mode_counts(self) -> dict[str, int]:
        counter: Counter = Counter()
        for s in self._diagnosed:
            counter[s.diagnosis.mode.value if s.diagnosis else "undiagnosed"] += 1
        return dict(counter)

    def total_probe_calls(self) -> int:
        return sum(s.diagnosis.probe_calls for s in self._diagnosed if s.diagnosis)

    def by_stage(self) -> dict[str, dict[str, Any]]:
        """Per-rejecting_step (gate name) breakdown, each shaped like to_dict().

        When both HallucinationGate and RewardGate have a probe attached,
        the pooled totals can't tell you which gate's probe is actually
        recovering samples — this splits the same summary by the gate that
        rejected each sample.
        """
        stages: dict[str, list[RejectedSample]] = {}
        for s in self._diagnosed:
            stages.setdefault(s.rejecting_step, []).append(s)
        return {stage: self._summarize(samples) for stage, samples in stages.items()}

    def to_dict(self) -> dict[str, Any]:
        summary = self._summarize(self._diagnosed)
        summary["by_stage"] = self.by_stage()
        return summary

    def write_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
