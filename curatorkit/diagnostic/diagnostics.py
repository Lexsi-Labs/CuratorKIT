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
  cost-normalised.
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

    def to_dict(self) -> dict[str, Any]:
        total = len(self._diagnosed)
        recovered = self.probe_recovery_count()
        return {
            "total_diagnosed": total,
            "probe_recovered": recovered,
            "probe_recovery_pct": round(recovered / total, 4) if total else 0.0,
            "total_probe_calls": self.total_probe_calls(),
            "mode_counts": self.mode_counts(),
        }

    def write_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
