"""
CoverageAnalyser — embed batch, cluster, identify thin regions.

Runs between generation batches, not as a post-hoc analysis, so coverage
can be steered batch by batch (see CoverageDirectedGenerator).

The analyser is *not* a gate — it does not reject samples. It produces a
report and (optionally) attaches per-cluster recommendations to a manifest
slot the next generation batch can consume.

K-means is implemented from scratch (~60 lines). sentence-transformers is
required only for the embedding step and lazy-imported.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from curatorkit.coverage.metrics import (
    diversity_metrics,
    normalized_entropy,
    pairwise_cosine_stats,
)
from curatorkit.schema import DataSample

# ───────────────────────────────────────────────────────────────────────────
# Lazy embedding loader (shared with DiversityGate)
# ───────────────────────────────────────────────────────────────────────────


def _ensure_sentence_transformers() -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. Install with: "
            "pip install curatorkit[embedding]"
        ) from e


# ───────────────────────────────────────────────────────────────────────────
# From-scratch k-means (cosine distance)
# ───────────────────────────────────────────────────────────────────────────


def _l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]


def _kmeans_cosine(
    vectors: list[list[float]],
    k: int,
    max_iter: int = 50,
    seed: int = 42,
) -> tuple[list[int], list[list[float]]]:
    """
    Spherical k-means (cosine distance).

    Implemented here (no clustering dependency). Returns
    (cluster_id_per_vector, centroids).

    Empty input or k <= 0 → ([], []). k > n collapses to n clusters.
    """
    n = len(vectors)
    if n == 0 or k <= 0:
        return [], []

    # L2-normalise so cosine distance reduces to (1 - dot product).
    vectors = [_l2_normalize(v) for v in vectors]
    k = min(k, n)
    rng = random.Random(seed)

    # k-means++ initialization (the difference between k-means++ and random
    # seeding shows up at small k — important here because the analyser
    # often runs with k=8-16 to match task_type vocabulary size).
    centroids: list[list[float]] = [vectors[rng.randrange(n)]]
    while len(centroids) < k:
        # Distance from each point to its nearest existing centroid
        dists = []
        for v in vectors:
            d = min(1.0 - sum(x * y for x, y in zip(v, c)) for c in centroids)
            dists.append(max(d, 0.0))
        total = sum(dists)
        if total == 0:
            centroids.append(vectors[rng.randrange(n)])
            continue
        # Weighted draw
        r = rng.random() * total
        cum = 0.0
        for i, d in enumerate(dists):
            cum += d
            if cum >= r:
                centroids.append(vectors[i])
                break

    assignments = [0] * n

    for _ in range(max_iter):
        changed = False
        # Assign
        for i, v in enumerate(vectors):
            best = 0
            best_sim = -2.0
            for c_idx, c in enumerate(centroids):
                sim = sum(x * y for x, y in zip(v, c))
                if sim > best_sim:
                    best_sim = sim
                    best = c_idx
            if assignments[i] != best:
                changed = True
                assignments[i] = best

        if not changed:
            break

        # Update centroids (mean-of-assigned, then re-normalize)
        new_centroids: list[list[float]] = []
        for c_idx in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == c_idx]
            if not members:
                # Re-seed empty cluster on a random point
                new_centroids.append(vectors[rng.randrange(n)])
                continue
            dim = len(members[0])
            mean = [sum(v[d] for v in members) / len(members) for d in range(dim)]
            new_centroids.append(_l2_normalize(mean))
        centroids = new_centroids

    return assignments, centroids


# ───────────────────────────────────────────────────────────────────────────
# CoverageReport
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class CoverageReport:
    """
    What the analyser produces for one batch.

    `n_samples`             — input batch size.
    `k_clusters`            — chosen k (may differ from requested if n < k).
    `cluster_sizes`         — {cluster_id: count}.
    `cluster_normalized_entropy` — 0..1 occupancy uniformity (1 = perfectly
                              uniform clusters; <0.5 → strongly skewed).
    `thin_clusters`         — cluster ids with < `thin_threshold` fraction
                              of total samples; recommended generation
                              targets for the next batch.
    `embedding_stats`       — pairwise cosine-distance summary.
    `text_diversity`        — distinct-N + type-token-ratio bundle from
                              metrics.diversity_metrics.
    `recommendations`       — human-readable advice attached to manifest.
    `task_type_distribution`— observed task_type counts (from samples), for
                              cross-checking against any target_distribution.
    `collapse_warning`      — None, or a short string flagging early-warning
                              signs of model collapse (sharp drop in
                              distinct-2, dominant cluster, …).
    """

    n_samples: int
    k_clusters: int
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    cluster_normalized_entropy: float = 0.0
    thin_clusters: list[int] = field(default_factory=list)
    embedding_stats: dict[str, float] = field(default_factory=dict)
    text_diversity: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    task_type_distribution: dict[str, int] = field(default_factory=dict)
    collapse_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "k_clusters": self.k_clusters,
            "cluster_sizes": self.cluster_sizes,
            "cluster_normalized_entropy": self.cluster_normalized_entropy,
            "thin_clusters": self.thin_clusters,
            "embedding_stats": self.embedding_stats,
            "text_diversity": self.text_diversity,
            "recommendations": self.recommendations,
            "task_type_distribution": self.task_type_distribution,
            "collapse_warning": self.collapse_warning,
        }


# ───────────────────────────────────────────────────────────────────────────
# CoverageAnalyser
# ───────────────────────────────────────────────────────────────────────────


class CoverageAnalyser:
    """
    Coverage diagnostic for a generation batch.

    Parameters
    ----------
    k_clusters : int
        K for spherical k-means. Defaults to 8 — a small number is the right
        default for instruction datasets where task_type vocabulary is also
        in the 8-12 range. Set to 0 to skip clustering (text + embedding
        stats only).
    thin_threshold : float
        Clusters smaller than `thin_threshold * n_samples` are recommended
        as generation targets. Default 0.05 (5%).
    embedding_model : str
        sentence-transformers model id.
    text_field : str
        Which DataSample field to embed. "auto" picks per task_type.
    sample_pair_cap : int | None
        Max pairs for pairwise_cosine_stats. None → all pairs.
    """

    def __init__(
        self,
        k_clusters: int = 8,
        thin_threshold: float = 0.05,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_field: str = "auto",
        sample_pair_cap: int | None = 5000,
    ) -> None:
        self.k_clusters = int(k_clusters)
        self.thin_threshold = float(thin_threshold)
        self.embedding_model_name = embedding_model
        self.text_field = text_field
        self.sample_pair_cap = sample_pair_cap
        self._model: Any = None

    # ─────────────────────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        if self._model is None:
            SentenceTransformer = _ensure_sentence_transformers()
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _get_text(self, sample: DataSample) -> str:
        if self.text_field != "auto":
            return getattr(sample, self.text_field, "") or ""
        task = sample.task_type
        if task == "language_modeling":
            return sample.output or ""
        if task in ("preference", "implicit_preference"):
            return f"{sample.instruction} {sample.chosen}".strip()
        if task == "grpo" and sample.responses:
            return f"{sample.instruction} {sample.responses[0]}".strip()
        return f"{sample.instruction} {sample.output}".strip()

    # ─────────────────────────────────────────────────────────────────────

    def analyse(
        self,
        samples: list[DataSample],
        previous_report: CoverageReport | None = None,
    ) -> CoverageReport:
        """
        Run the full diagnostic.

        Passing `previous_report` enables cross-batch comparison — the
        report's `collapse_warning` is populated if distinct-2 drops by
        more than 10% from the previous run.
        """
        if not samples:
            return CoverageReport(n_samples=0, k_clusters=0)

        texts = [self._get_text(s) for s in samples]
        labels = [s.task_type for s in samples]

        # Text-only metrics (cheap, always run)
        text_div = diversity_metrics(texts, labels=labels)

        report = CoverageReport(
            n_samples=len(samples),
            k_clusters=0,
            text_diversity=text_div,
            task_type_distribution=dict(Counter(labels)),
        )

        # Embedding + clustering (skipped if k_clusters <= 0)
        if self.k_clusters <= 0:
            return self._finalize(report, previous_report)

        try:
            model = self._load_model()
        except ImportError as e:
            report.recommendations.append(
                f"Embedding skipped — install curatorkit[embedding] to enable: {e}"
            )
            return self._finalize(report, previous_report)

        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emb_list = [e.tolist() for e in embeddings]

        report.embedding_stats = pairwise_cosine_stats(
            emb_list,
            sample_pairs=self.sample_pair_cap,
        )

        # Spherical k-means
        assignments, _ = _kmeans_cosine(emb_list, self.k_clusters)
        cluster_sizes = Counter(assignments)
        report.k_clusters = max(cluster_sizes.keys()) + 1 if cluster_sizes else 0
        report.cluster_sizes = dict(cluster_sizes)
        report.cluster_normalized_entropy = round(
            normalized_entropy(str(a) for a in assignments),
            4,
        )

        # Identify thin clusters
        threshold = self.thin_threshold * len(samples)
        report.thin_clusters = sorted(
            [cid for cid, size in cluster_sizes.items() if size < threshold]
        )

        # Build recommendations
        if report.thin_clusters:
            report.recommendations.append(
                f"Direct next batch toward {len(report.thin_clusters)} thin clusters: "
                f"{report.thin_clusters}. These clusters hold less than "
                f"{100 * self.thin_threshold:.0f}% of the batch each."
            )
        if report.cluster_normalized_entropy < 0.6:
            report.recommendations.append(
                f"Cluster occupancy is skewed (normalized entropy = "
                f"{report.cluster_normalized_entropy:.2f}). A handful of clusters "
                f"dominate — coverage is uneven."
            )

        return self._finalize(report, previous_report)

    # ─────────────────────────────────────────────────────────────────────

    def _finalize(
        self,
        report: CoverageReport,
        previous: CoverageReport | None,
    ) -> CoverageReport:
        """Cross-batch collapse detection."""
        if previous is None:
            return report

        # Distinct-2 collapse: > 10% drop is a hard early-warning signal
        prev_d2 = float(previous.text_diversity.get("distinct_2", 0.0))
        cur_d2 = float(report.text_diversity.get("distinct_2", 0.0))
        if prev_d2 > 0 and (prev_d2 - cur_d2) / prev_d2 > 0.10:
            report.collapse_warning = (
                f"distinct-2 dropped from {prev_d2:.3f} to {cur_d2:.3f} "
                f"({100 * (prev_d2 - cur_d2) / prev_d2:.1f}% relative). "
                f"Possible model collapse — check the next batch before training."
            )

        # Cluster-entropy collapse
        prev_ent = previous.cluster_normalized_entropy
        cur_ent = report.cluster_normalized_entropy
        if prev_ent > 0 and (prev_ent - cur_ent) > 0.10:
            warning = (
                f"Cluster normalized entropy dropped from {prev_ent:.3f} to "
                f"{cur_ent:.3f} — output is becoming more topically concentrated."
            )
            report.collapse_warning = (
                f"{report.collapse_warning}\n{warning}" if report.collapse_warning else warning
            )

        return report
