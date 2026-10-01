"""
DiversityMonitor — pipeline normalizer that runs CoverageAnalyser per batch.

Sits anywhere in the normalizer slot — typically immediately after a
generation step so the report reflects what was actually generated.

The monitor:
  1. Calls CoverageAnalyser.analyse() on the batch.
  2. Attaches a ProvenanceRecord with the full report to every sample, so
     manifest.json.diversity_stats picks it up automatically (it scans
     provenance for the most recent DiversityMonitor record).
  3. Maintains a `previous_report` across runs of the same instance, so the
     collapse-warning logic can compare batch N to batch N-1.
  4. Issues a Python `warnings.warn()` if the analyser flags collapse —
     callers can wrap the run in `warnings.catch_warnings(error=True)`
     to treat it as a hard fail.

The monitor never removes samples — it is read-only over the batch.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import UTC, datetime

from curatorkit.coverage.analyser import CoverageAnalyser, CoverageReport
from curatorkit.interfaces import BaseNormalizer
from curatorkit.schema import DataSample, ProvenanceRecord

STEP_VERSION = "1.0.0"


class DiversityMonitor(BaseNormalizer):
    """
    BaseNormalizer wrapper around CoverageAnalyser.

    Parameters
    ----------
    analyser : Optional[CoverageAnalyser]
        Pass a pre-built analyser, or None to construct one from the kwargs
        below. Useful for sharing the embedding model across multiple
        monitors in one pipeline.
    k_clusters, thin_threshold, embedding_model, text_field
        Forwarded to CoverageAnalyser when `analyser` is None.
    raise_on_collapse : bool
        If True, raise RuntimeError when the analyser flags collapse.
        Default False (warning only).
    attach_to_first_only : bool
        If True (default), attach the full report ProvenanceRecord only to
        the first sample in the batch — avoids N copies of the same payload.
        Set to False if downstream tooling expects every sample to carry it.
    """

    def __init__(
        self,
        analyser: CoverageAnalyser | None = None,
        k_clusters: int = 8,
        thin_threshold: float = 0.05,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_field: str = "auto",
        raise_on_collapse: bool = False,
        attach_to_first_only: bool = True,
    ) -> None:
        self.analyser = analyser or CoverageAnalyser(
            k_clusters=k_clusters,
            thin_threshold=thin_threshold,
            embedding_model=embedding_model,
            text_field=text_field,
        )
        self.raise_on_collapse = raise_on_collapse
        self.attach_to_first_only = attach_to_first_only
        # `_last_report` is the analyser's view of the previous batch.
        # We compare batch N to batch N-1 by passing `_last_report` as the
        # previous on each call.
        self._last_report: CoverageReport | None = None

    # ─────────────────────────────────────────────────────────────────────

    @property
    def last_report(self) -> CoverageReport | None:
        """The most recent report, for callers that want programmatic access."""
        return self._last_report

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "k_clusters": self.analyser.k_clusters,
                "thin_threshold": self.analyser.thin_threshold,
                "embedding_model": self.analyser.embedding_model_name,
                "text_field": self.analyser.text_field,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        if not samples:
            return samples

        report = self.analyser.analyse(samples, previous_report=self._last_report)
        self._last_report = report

        if report.collapse_warning:
            if self.raise_on_collapse:
                raise RuntimeError(
                    f"DiversityMonitor: collapse detected — {report.collapse_warning}"
                )
            warnings.warn(
                f"DiversityMonitor: {report.collapse_warning}",
                stacklevel=2,
            )

        # Attach provenance
        ts = datetime.now(UTC).replace(tzinfo=None)
        cfg = self._config_hash()
        notes = {
            "diversity_stats": report.to_dict(),
            "monitor_version": STEP_VERSION,
        }
        record = ProvenanceRecord(
            step_name="DiversityMonitor",
            step_version=STEP_VERSION,
            timestamp=ts,
            config_hash=cfg,
            notes=notes,
        )

        if self.attach_to_first_only:
            samples[0].append_provenance(record)
        else:
            for s in samples:
                s.append_provenance(record)

        return samples
