"""
CoverageDirectedGenerator — close the coverage feedback loop.

The CoverageAnalyser writes a list of `thin_clusters` and `recommendations`
to manifest.diversity_stats after each batch. This wrapper reads that
report (either from a previous run's manifest.json or from an in-memory
CoverageReport) and *conditions the next generation batch* toward
under-represented regions.

Two complementary mechanisms:

  1. Sample re-weighting — when the input samples carry a `cluster_id` in
     their metadata (e.g., from a previous DiversityMonitor pass), we
     over-represent thin-cluster samples in the input the wrapped
     generator sees.

  2. Prompt enrichment — the wrapped generator's `_build_messages` output
     gets an extra system-style hint listing the thin-cluster topics. The
     generator can use that hint to drift its output distribution toward
     those topics.

Both mechanisms are opt-in via constructor flags. The wrapper is
transparent: it delegates `flush_rejected()` and provenance behavior to
the inner generator.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curatorkit.coverage.analyser import CoverageReport
from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _load_diversity_stats_from_manifest(path: Path | str) -> dict[str, Any] | None:
    """Read manifest.json and return its `diversity_stats` field, or None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("diversity_stats")


def _thin_clusters_from_stats(stats: dict[str, Any]) -> list[int]:
    if not stats:
        return []
    out: list[int] = []
    for v in stats.get("thin_clusters", []) or []:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _recommendations_from_stats(stats: dict[str, Any]) -> list[str]:
    return [str(r) for r in (stats.get("recommendations", []) or [])]


def _task_type_distribution(stats: dict[str, Any]) -> dict[str, int]:
    raw = stats.get("task_type_distribution", {}) or {}
    return {str(k): int(v) for k, v in raw.items()}


# ───────────────────────────────────────────────────────────────────────────
# CoverageDirectedGenerator
# ───────────────────────────────────────────────────────────────────────────


class CoverageDirectedGenerator(BaseGenerationTask):
    """
    Wrap any BaseGenerationTask so its next batch is biased toward
    under-represented regions.

    Parameters
    ----------
    inner : BaseGenerationTask
        The generator to wrap. CoverageDirectedGenerator delegates
        `_build_messages` / `_parse_response` / `run` to it.
    coverage_report : CoverageReport | None
        Pre-built report from a previous CoverageAnalyser.analyse() run.
        Use either this or `manifest_path` — not both.
    manifest_path : str | Path | None
        Path to a previous run's manifest.json. The wrapper reads
        `diversity_stats.thin_clusters` and `recommendations` from it.
    oversample_factor : float
        Multiplier applied to input samples whose `metadata["cluster_id"]`
        falls in the thin-cluster list. e.g., 2.0 doubles their presence
        in the input that the inner generator sees.
        Set to 1.0 to disable sample re-weighting. Default 2.0.
    enrich_prompts : bool
        If True (default), prepend a thin-region hint string to the inner
        generator's prompt messages so the generator can read it.
    rebalance_task_types : bool
        If True, under-represented task_types from the previous run get
        the same oversampling treatment. Useful for cross-paradigm balance.
    rng_seed : int
        Seed for the oversampling RNG.
    """

    def __init__(
        self,
        inner: BaseGenerationTask,
        coverage_report: CoverageReport | None = None,
        manifest_path: str | Path | None = None,
        oversample_factor: float = 2.0,
        enrich_prompts: bool = True,
        rebalance_task_types: bool = False,
        rng_seed: int = 42,
    ) -> None:
        # Inherit the inner's LLM / template / concurrency so the wrapper
        # is a drop-in for BaseGenerationTask.
        super().__init__(
            llm=inner.llm,
            prompt_template=inner.prompt_template,
            concurrency=inner.concurrency,
        )
        self.inner = inner
        self.oversample_factor = max(1.0, float(oversample_factor))
        self.enrich_prompts = enrich_prompts
        self.rebalance_task_types = rebalance_task_types
        self.rng_seed = int(rng_seed)

        if coverage_report is not None and manifest_path is not None:
            raise ValueError("Pass coverage_report OR manifest_path, not both.")

        self._stats: dict[str, Any]
        if coverage_report is not None:
            self._stats = coverage_report.to_dict()
        elif manifest_path is not None:
            loaded = _load_diversity_stats_from_manifest(manifest_path)
            self._stats = loaded or {}
        else:
            self._stats = {}

        self.thin_clusters = _thin_clusters_from_stats(self._stats)
        self.recommendations = _recommendations_from_stats(self._stats)
        self.task_type_dist = _task_type_distribution(self._stats)

    # ─────────────────────────────────────────────────────────────────────
    # Identity-mode passthroughs
    # ─────────────────────────────────────────────────────────────────────

    @property
    def task_name(self) -> str:
        return f"CoverageDirected({type(self.inner).__name__})"

    @property
    def rejected(self) -> list[RejectedSample]:
        return self.inner.rejected

    def flush_rejected(self) -> list[RejectedSample]:
        return self.inner.flush_rejected()

    # _build_messages / _parse_response satisfy the ABC; we override
    # run/run_async to apply the oversampling + prompt enrichment.

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        return self.inner._build_messages(sample)

    def _parse_response(
        self,
        sample: DataSample,
        response: LLMResponse,
    ) -> list[DataSample]:
        return self.inner._parse_response(sample, response)

    # ─────────────────────────────────────────────────────────────────────
    # Input transformation
    # ─────────────────────────────────────────────────────────────────────

    def _oversample(self, samples: list[DataSample]) -> list[DataSample]:
        """Duplicate thin-cluster (or thin-task-type) samples by oversample_factor."""
        if not samples:
            return samples
        if self.oversample_factor <= 1.0:
            return samples
        if not self.thin_clusters and not (self.rebalance_task_types and self.task_type_dist):
            return samples

        # Identify under-represented task_types (below mean fraction)
        thin_tts: set[str] = set()
        if self.rebalance_task_types and self.task_type_dist:
            total = sum(self.task_type_dist.values())
            mean = total / max(1, len(self.task_type_dist))
            thin_tts = {tt for tt, c in self.task_type_dist.items() if c < mean}

        rng = random.Random(self.rng_seed)
        extra: list[DataSample] = []
        # How many extra copies per matching sample
        n_extra = int(self.oversample_factor) - 1
        if n_extra < 1:
            n_extra = 1  # at least one duplicate when factor > 1.0

        for s in samples:
            in_thin_cluster = s.metadata.get("cluster_id") in self.thin_clusters
            in_thin_tt = s.task_type in thin_tts
            if in_thin_cluster or in_thin_tt:
                # `n_extra` deep copies (re-id'd) to be transparent to dedup
                for _ in range(n_extra):
                    dup = s.model_copy(deep=True)
                    # Tag the copy so downstream knows it was oversampled
                    dup.metadata = {
                        **dup.metadata,
                        "coverage_oversampled": True,
                    }
                    extra.append(dup)

        if not extra:
            return samples
        # Shuffle the union deterministically so order doesn't carry signal
        merged = list(samples) + extra
        rng.shuffle(merged)
        return merged

    def _enrichment_hint(self) -> str | None:
        """Build a short hint string the inner generator can prepend."""
        if not self.enrich_prompts:
            return None
        parts: list[str] = []
        if self.thin_clusters:
            parts.append(
                f"Coverage feedback: focus on cluster ids "
                f"{sorted(self.thin_clusters)} (under-represented in the previous batch)."
            )
        if self.recommendations:
            parts.append(
                "Recommendations from prior batch:\n  - " + "\n  - ".join(self.recommendations[:5])
            )
        if not parts:
            return None
        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline interface — delegate to inner with augmented input + prompt
    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        prepared = self._prepare_samples(samples)
        results = self.inner.run(prepared)
        self._stamp_provenance(results)
        return results

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        prepared = self._prepare_samples(samples)
        results = await self.inner.run_async(prepared)
        self._stamp_provenance(results)
        return results

    # ─────────────────────────────────────────────────────────────────────

    def _prepare_samples(self, samples: list[DataSample]) -> list[DataSample]:
        prepared = self._oversample(samples)
        hint = self._enrichment_hint()
        if hint:
            # Stash the hint on each sample so the inner generator's
            # _build_messages can incorporate it via the existing
            # metadata channel. Generators that don't read this key are
            # unaffected.
            for s in prepared:
                s.metadata = {
                    **s.metadata,
                    "coverage_hint": hint,
                }
        return prepared

    def _stamp_provenance(self, results: list[DataSample]) -> None:
        if not results:
            return
        ts = datetime.now(UTC).replace(tzinfo=None)
        notes = {
            "wrapped_generator": type(self.inner).__name__,
            "thin_clusters": self.thin_clusters,
            "oversample_factor": self.oversample_factor,
            "rebalance_task_types": self.rebalance_task_types,
            "enrich_prompts": self.enrich_prompts,
        }
        for s in results:
            s.append_provenance(
                ProvenanceRecord(
                    step_name="CoverageDirectedGenerator",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=self.inner.llm.config_hash(),
                    notes=notes,
                )
            )
