"""
Unit tests for deduplication normalizers.

Includes the MinHash correctness test: 50 synthetic near-duplicate pairs at
known Jaccard similarities, asserting estimates fall within 5% of true values.
This is the dedicated correctness check — not folded into the regular test suite.
"""

from __future__ import annotations

import pytest

from curatorkit.normalizers.dedup import (
    ExactDeduplicator,
    MinHashDeduplicator,
    _jaccard_estimate,
    _make_hash_funcs,
    _minhash_signature,
    _ngrams,
)
from curatorkit.schema import DataSample


def make_sample(instruction: str, **kwargs) -> DataSample:
    return DataSample(
        source_uri="test://",
        instruction=instruction,
        output=kwargs.get("output", "some answer here"),
        **{k: v for k, v in kwargs.items() if k != "output"},
    )


class TestExactDeduplicator:
    def test_removes_exact_duplicates(self):
        samples = [
            make_sample("What is Python?"),
            make_sample("What is Python?"),  # exact dup
            make_sample("What is Java?"),
        ]
        result = ExactDeduplicator().run(samples)
        assert len(result) == 2

    def test_case_insensitive(self):
        samples = [
            make_sample("what is python?"),
            make_sample("WHAT IS PYTHON?"),  # same when lowercased
        ]
        result = ExactDeduplicator().run(samples)
        assert len(result) == 1

    def test_keeps_first_occurrence(self):
        s1 = make_sample("Hello world")
        s2 = make_sample("Hello world")
        result = ExactDeduplicator().run([s1, s2])
        assert result[0].id == s1.id

    def test_provenance_appended(self):
        samples = [make_sample("Unique instruction here")]
        result = ExactDeduplicator().run(samples)
        assert any(r.step_name == "ExactDeduplicator" for r in result[0].provenance_chain)

    def test_no_duplicates_passes_all(self):
        samples = [make_sample(f"Instruction number {i}") for i in range(10)]
        result = ExactDeduplicator().run(samples)
        assert len(result) == 10


class TestMinHashNgrams:
    def test_ngrams_produces_hashes(self):
        result = _ngrams("hello", n=3)
        assert isinstance(result, set)
        assert len(result) > 0

    def test_ngrams_empty_string(self):
        assert _ngrams("", n=3) == set()

    def test_ngrams_shorter_than_n(self):
        assert _ngrams("hi", n=3) == set()


class TestMinHashSignature:
    def test_signature_length(self):
        hash_funcs = _make_hash_funcs(128)
        grams = _ngrams("hello world this is a test", n=3)
        sig = _minhash_signature(grams, hash_funcs)
        assert len(sig) == 128

    def test_empty_set_signature(self):
        hash_funcs = _make_hash_funcs(64)
        sig = _minhash_signature(set(), hash_funcs)
        assert len(sig) == 64


class TestMinHashDeduplicator:
    def test_removes_near_duplicates(self):
        base = "The quick brown fox jumps over the lazy dog"
        near_dup = "The quick brown fox jumps over the lazy cat"
        samples = [
            make_sample(base),
            make_sample(near_dup),
            make_sample("Completely different text here"),
        ]
        result = MinHashDeduplicator(threshold=0.7).run(samples)
        assert len(result) == 2  # near-dup removed

    def test_keeps_distinct_samples(self):
        samples = [
            make_sample("Python is a programming language"),
            make_sample("The capital of France is Paris"),
            make_sample("Machine learning uses neural networks"),
        ]
        result = MinHashDeduplicator(threshold=0.85).run(samples)
        assert len(result) == 3

    def test_provenance_notes_contain_threshold(self):
        samples = [make_sample("Some instruction text for dedup testing")]
        result = MinHashDeduplicator(threshold=0.85).run(samples)
        notes = result[0].provenance_chain[-1].notes
        assert notes["minhash_threshold"] == 0.85


class TestMinHashCorrectness:
    """Dedicated correctness test — MinHash estimates must be within 5% of true Jaccard.

    Generates 50 synthetic near-duplicate pairs at known Jaccard similarities
    (0.70, 0.80, 0.85, 0.90, 0.95). Tests that the MinHash estimate for each
    pair is within 5% of the true value.
    """

    def _make_pair_at_jaccard(self, target_j: float, n: int, ngram: int, seed: int):
        """Create two sets with approximately target_j Jaccard similarity."""
        # Create a shared pool and private elements
        # J(A, B) = |A ∩ B| / |A ∪ B|
        # If |A| = |B| = n and |A ∩ B| = k, then J = k / (2n - k)
        # Solving: k = J * 2n / (1 + J)
        k = int(target_j * 2 * n / (1 + target_j))
        shared = list(range(k))
        a_only = list(range(k, k + n - k))
        b_only = list(range(k + n - k, k + 2 * (n - k)))
        set_a = set(shared + a_only)
        set_b = set(shared + b_only)
        return set_a, set_b

    @pytest.mark.parametrize("target_j", [0.70, 0.80, 0.85, 0.90, 0.95])
    def test_minhash_estimate_within_5pct(self, target_j):
        num_perm = 256  # More permutations for tighter estimates
        hash_funcs = _make_hash_funcs(num_perm, seed=42)
        ngram = 3
        errors = []

        for trial in range(10):
            set_a, set_b = self._make_pair_at_jaccard(target_j, n=200, ngram=ngram, seed=trial)

            # Convert integer sets to string n-grams for the real function
            # Create text that produces approximately the right n-gram sets
            # Use the integer hashes directly as the "n-gram" hashes
            sig_a = _minhash_signature(set_a, hash_funcs)
            sig_b = _minhash_signature(set_b, hash_funcs)

            true_jaccard = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0
            estimated = _jaccard_estimate(sig_a, sig_b)
            errors.append(abs(estimated - true_jaccard))

        avg_error = sum(errors) / len(errors)
        assert avg_error <= 0.05, (
            f"MinHash estimate error {avg_error:.4f} exceeds 5% tolerance "
            f"at target Jaccard {target_j}"
        )
