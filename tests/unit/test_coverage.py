"""
Tests for the coverage-aware generation module.

Covers:
  metrics        — distinct-N, type-token-ratio, entropy, pairwise cosine
  analyser       — k-means clustering edge cases, report shape, cross-batch
  monitor        — pipeline integration, collapse-warning detection
"""

from __future__ import annotations

import math
import warnings

import pytest

from curatorkit.coverage.analyser import CoverageAnalyser, _kmeans_cosine, _l2_normalize
from curatorkit.coverage.metrics import (
    distinct_n,
    diversity_metrics,
    normalized_entropy,
    pairwise_cosine_stats,
    shannon_entropy,
    type_token_ratio,
)
from curatorkit.coverage.monitor import DiversityMonitor
from curatorkit.schema import DataSample

# ───────────────────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────────────────


class TestDistinctN:
    def test_empty(self):
        assert distinct_n([], n=2) == 0.0

    def test_repeated_text_collapses_distinct_2(self):
        texts = ["the cat sat"] * 5
        d2 = distinct_n(texts, n=2)
        # 5 samples × 2 bigrams = 10 total, 2 unique → 0.2
        assert d2 == pytest.approx(0.2)

    def test_all_unique_2grams_score_1(self):
        texts = ["alpha beta", "gamma delta", "epsilon zeta"]
        assert distinct_n(texts, n=2) == 1.0

    def test_handles_text_shorter_than_n(self):
        # Single-token text → no n-grams contributed at n=2
        assert distinct_n(["hello", "alpha beta"], n=2) == 1.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            distinct_n(["x"], n=0)


class TestTypeTokenRatio:
    def test_empty(self):
        assert type_token_ratio([]) == 0.0

    def test_single_unique_word(self):
        assert type_token_ratio(["foo foo foo"]) == pytest.approx(1 / 3)

    def test_all_unique(self):
        assert type_token_ratio(["alpha beta gamma"]) == 1.0


class TestShannonEntropy:
    def test_single_class_is_zero(self):
        assert shannon_entropy(["a", "a", "a"]) == 0.0

    def test_two_balanced_classes(self):
        # 50/50 → 1 bit
        assert shannon_entropy(["a", "b"]) == pytest.approx(1.0)

    def test_max_for_k_classes(self):
        # 3 equal classes → log2(3) ≈ 1.585
        labels = ["a", "b", "c"]
        assert shannon_entropy(labels) == pytest.approx(math.log2(3))

    def test_normalized_entropy_bounded(self):
        labels = ["a", "b", "c", "a"]
        assert 0.0 <= normalized_entropy(labels) <= 1.0


class TestPairwiseCosineStats:
    def test_empty(self):
        stats = pairwise_cosine_stats([])
        assert stats["n_pairs"] == 0

    def test_identical_vectors_zero_distance(self):
        v = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        stats = pairwise_cosine_stats(v)
        assert stats["mean_distance"] == pytest.approx(0.0)

    def test_orthogonal_vectors_one_distance(self):
        v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        stats = pairwise_cosine_stats(v)
        assert stats["mean_distance"] == pytest.approx(1.0)

    def test_subsampling_works(self):
        v = [[float(i), 0.0, 0.0] for i in range(100)]
        stats = pairwise_cosine_stats(v, sample_pairs=50)
        assert stats["n_pairs"] <= 50


class TestDiversityMetrics:
    def test_bundle_keys(self):
        b = diversity_metrics(["the cat sat", "a dog ran"], labels=["a", "b"])
        for key in (
            "n_samples",
            "type_token_ratio",
            "distinct_1",
            "distinct_2",
            "distinct_3",
            "label_entropy_bits",
        ):
            assert key in b


# ───────────────────────────────────────────────────────────────────────────
# K-means
# ───────────────────────────────────────────────────────────────────────────


class TestKMeans:
    def test_empty_input(self):
        assignments, centroids = _kmeans_cosine([], k=3)
        assert assignments == []
        assert centroids == []

    def test_zero_k(self):
        assignments, _ = _kmeans_cosine([[1.0, 0.0]], k=0)
        assert assignments == []

    def test_k_clamped_to_n(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        assignments, centroids = _kmeans_cosine(vectors, k=10)
        # Can't have more than n clusters
        assert len(centroids) == 2

    def test_separates_two_obvious_clusters(self):
        # 5 vectors near [1,0], 5 near [0,1]
        a = [[1.0 + i * 0.01, 0.0] for i in range(5)]
        b = [[0.0, 1.0 + i * 0.01] for i in range(5)]
        assignments, _ = _kmeans_cosine(a + b, k=2, seed=0)
        # All five of `a` should land in one cluster, all five of `b` in the other
        a_assignments = set(assignments[:5])
        b_assignments = set(assignments[5:])
        assert len(a_assignments) == 1
        assert len(b_assignments) == 1
        assert a_assignments != b_assignments

    def test_l2_normalize_zero_vector(self):
        # Zero vector should not divide by zero
        v = _l2_normalize([0.0, 0.0, 0.0])
        assert v == [0.0, 0.0, 0.0]


# ───────────────────────────────────────────────────────────────────────────
# CoverageAnalyser
# ───────────────────────────────────────────────────────────────────────────


class TestCoverageAnalyser:
    def test_empty_input(self):
        report = CoverageAnalyser().analyse([])
        assert report.n_samples == 0

    def test_k_clusters_zero_skips_embeddings(self):
        samples = [
            DataSample(source_uri="x", instruction=f"q{i}", output=f"a{i}") for i in range(5)
        ]
        report = CoverageAnalyser(k_clusters=0).analyse(samples)
        assert report.n_samples == 5
        assert report.k_clusters == 0
        # Text-only metrics still populated
        assert "distinct_2" in report.text_diversity
        assert report.task_type_distribution

    def test_task_type_distribution(self):
        samples = [
            DataSample(source_uri="x", instruction="q", output="a", task_type="coding"),
            DataSample(source_uri="x", instruction="q", output="a", task_type="coding"),
            DataSample(source_uri="x", instruction="q", output="a", task_type="general"),
        ]
        report = CoverageAnalyser(k_clusters=0).analyse(samples)
        assert report.task_type_distribution == {"coding": 2, "general": 1}

    def test_report_serializable_to_dict(self):
        samples = [DataSample(source_uri="x", instruction="q", output="a")]
        report = CoverageAnalyser(k_clusters=0).analyse(samples)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "n_samples" in d

    def test_cross_batch_detects_distinct_2_collapse(self):
        # Build two pre-cooked reports manually
        analyser = CoverageAnalyser(k_clusters=0)
        b1 = [
            DataSample(source_uri="x", instruction=f"q{i} unique words", output=f"a{i}")
            for i in range(10)
        ]
        b2 = [
            DataSample(source_uri="x", instruction="same question repeated", output="same answer")
            for _ in range(10)
        ]
        r1 = analyser.analyse(b1)
        r2 = analyser.analyse(b2, previous_report=r1)
        assert r2.collapse_warning is not None
        assert "distinct-2" in r2.collapse_warning


# ───────────────────────────────────────────────────────────────────────────
# DiversityMonitor (pipeline integration)
# ───────────────────────────────────────────────────────────────────────────


class TestDiversityMonitor:
    def test_returns_input_unchanged(self):
        samples = [DataSample(source_uri="x", instruction=f"q{i}") for i in range(3)]
        mon = DiversityMonitor(k_clusters=0)
        out = mon.run(samples)
        # Identity check (we should not have replaced any sample)
        assert out is samples

    def test_attaches_provenance_to_first_sample(self):
        samples = [DataSample(source_uri="x", instruction=f"q{i}") for i in range(3)]
        mon = DiversityMonitor(k_clusters=0, attach_to_first_only=True)
        mon.run(samples)
        steps_first = [r.step_name for r in samples[0].provenance_chain]
        steps_others = [r.step_name for r in samples[1].provenance_chain]
        assert "DiversityMonitor" in steps_first
        assert "DiversityMonitor" not in steps_others

    def test_attaches_to_every_sample_when_requested(self):
        samples = [DataSample(source_uri="x", instruction=f"q{i}") for i in range(3)]
        mon = DiversityMonitor(k_clusters=0, attach_to_first_only=False)
        mon.run(samples)
        for s in samples:
            steps = [r.step_name for r in s.provenance_chain]
            assert "DiversityMonitor" in steps

    def test_empty_batch_returns_empty(self):
        mon = DiversityMonitor(k_clusters=0)
        assert mon.run([]) == []

    def test_collapse_warning_fires(self):
        b1 = [
            DataSample(source_uri="x", instruction=f"q{i} unique words", output=f"a{i}")
            for i in range(10)
        ]
        b2 = [
            DataSample(source_uri="x", instruction="same question repeated", output="same answer")
            for _ in range(10)
        ]
        mon = DiversityMonitor(k_clusters=0)
        mon.run(b1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mon.run(b2)
        assert any("collapse" in str(x.message).lower() for x in w)

    def test_raise_on_collapse_mode(self):
        b1 = [
            DataSample(source_uri="x", instruction=f"q{i} unique words", output=f"a{i}")
            for i in range(10)
        ]
        b2 = [
            DataSample(source_uri="x", instruction="same question repeated", output="same answer")
            for _ in range(10)
        ]
        mon = DiversityMonitor(k_clusters=0, raise_on_collapse=True)
        mon.run(b1)
        with pytest.raises(RuntimeError, match="collapse"):
            mon.run(b2)

    def test_last_report_accessor(self):
        mon = DiversityMonitor(k_clusters=0)
        assert mon.last_report is None
        mon.run([DataSample(source_uri="x", instruction="q", output="a")])
        assert mon.last_report is not None
        assert mon.last_report.n_samples == 1
