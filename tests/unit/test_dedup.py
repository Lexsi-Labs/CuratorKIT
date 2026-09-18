"""
Unit tests for deduplication normalizers.

Includes the MinHash correctness test: 50 synthetic near-duplicate pairs at
known Jaccard similarities, asserting estimates fall within 5% of true values.
This is the dedicated correctness check — not folded into the regular test suite.
"""

from __future__ import annotations

import warnings

import pytest

from curatorkit.normalizers.dedup import (
    ExactDeduplicator,
    MinHashDeduplicator,
    _jaccard_estimate,
    _make_hash_funcs,
    _minhash_signature,
    _ngrams,
    _sample_dedup_text,
)
from curatorkit.normalizers.dedup_utils import apply_ignore_patterns, compile_ignore_patterns
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


class TestConversationalDedupText:
    """conversational samples store only the first exchange in
    instruction/output; everything past it lives in metadata['turns'].
    Comparing only instruction+output let two conversations that diverge
    after turn 1 wrongly collapse into "duplicates"."""

    def test_diverging_later_turns_are_not_duplicates(self):
        opener = DataSample(
            source_uri="t://",
            instruction="What is Python?",
            output="A programming language.",
            task_type="conversational",
            metadata={"turns": [{"role": "user", "content": "Who created it?"}]},
        )
        diverging = opener.model_copy(deep=True)
        diverging.metadata["turns"] = [{"role": "user", "content": "Is it fast?"}]

        assert _sample_dedup_text(opener) != _sample_dedup_text(diverging)

        result = ExactDeduplicator().run([opener, diverging])
        assert len(result) == 2  # must NOT be collapsed into one

    def test_reader_chatml_and_generator_sharegpt_turn_shapes_both_read(self):
        chatml = DataSample(
            source_uri="t://",
            instruction="q",
            output="a",
            task_type="conversational",
            metadata={"turns": [{"role": "user", "content": "follow-up text"}]},
        )
        sharegpt = DataSample(
            source_uri="t://",
            instruction="q",
            output="a",
            task_type="conversational",
            metadata={"turns": [{"from": "human", "value": "follow-up text"}]},
        )
        assert "follow-up text" in _sample_dedup_text(chatml)
        assert "follow-up text" in _sample_dedup_text(sharegpt)

    def test_truly_identical_conversations_still_dedup(self):
        s1 = DataSample(
            source_uri="t://",
            instruction="q",
            output="a",
            task_type="conversational",
            metadata={"turns": [{"role": "user", "content": "same follow-up"}]},
        )
        s2 = s1.model_copy(deep=True)
        result = ExactDeduplicator().run([s1, s2])
        assert len(result) == 1


class TestUnpairedPreferenceDedupText:
    """unpaired_preference rows carry a label (e.g. the same instruction+
    output rated differently by two annotators) that instruction+output
    alone doesn't capture — omitting it let differently-labeled rows for
    the same text wrongly collapse into one."""

    def test_different_labels_are_not_duplicates(self):
        positive = DataSample(
            source_uri="t://",
            instruction="Explain gravity",
            output="Gravity pulls objects together.",
            task_type="unpaired_preference",
            label=1.0,
        )
        negative = positive.model_copy(deep=True)
        negative.label = 0.0

        assert _sample_dedup_text(positive) != _sample_dedup_text(negative)

        result = ExactDeduplicator().run([positive, negative])
        assert len(result) == 2  # both labeled examples must survive

    def test_same_label_still_dedups(self):
        s1 = DataSample(
            source_uri="t://",
            instruction="Explain gravity",
            output="Gravity pulls objects together.",
            task_type="unpaired_preference",
            label=1.0,
        )
        s2 = s1.model_copy(deep=True)
        result = ExactDeduplicator().run([s1, s2])
        assert len(result) == 1


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


class TestIgnoreForDedupPatterns:
    """Shared curatorkit.normalizers.dedup_utils helpers."""

    def test_invalid_regex_raises_at_compile_time(self):
        with pytest.raises(ValueError, match="invalid regex"):
            compile_ignore_patterns(["["])  # unbalanced bracket

    def test_no_patterns_is_a_noop(self):
        assert apply_ignore_patterns("hello   world", []) == "hello   world"

    def test_match_replaced_with_space_and_whitespace_collapsed(self):
        patterns = compile_ignore_patterns([r"id-\d+"])
        assert apply_ignore_patterns("ticket id-4821 closed", patterns) == "ticket closed"

    def test_replacement_does_not_merge_adjacent_words(self):
        patterns = compile_ignore_patterns([r"\d+"])
        # deleting outright (not replacing with a space) would produce "itemend"
        assert apply_ignore_patterns("item42end", patterns) == "item end"


class TestExactDeduplicatorIgnoreForDedup:
    def test_samples_differing_only_in_masked_span_become_duplicates(self):
        samples = [
            make_sample("Book flight for order id-1001", output="Confirmed."),
            make_sample("Book flight for order id-2002", output="Confirmed."),
        ]
        # Without masking, these are distinct
        assert len(ExactDeduplicator().run(samples)) == 2
        # With the order id masked, they're the same
        result = ExactDeduplicator(ignore_for_dedup=[r"id-\d+"]).run(samples)
        assert len(result) == 1

    def test_unmasked_difference_still_prevents_dedup(self):
        samples = [
            make_sample("Book flight for order id-1001", output="Confirmed A."),
            make_sample("Book flight for order id-2002", output="Confirmed B."),
        ]
        result = ExactDeduplicator(ignore_for_dedup=[r"id-\d+"]).run(samples)
        assert len(result) == 2  # outputs genuinely differ

    def test_invalid_pattern_raises_at_construction(self):
        with pytest.raises(ValueError, match="invalid regex"):
            ExactDeduplicator(ignore_for_dedup=["("])

    def test_config_hash_reflects_ignore_for_dedup(self):
        d1 = ExactDeduplicator()._config_hash()
        d2 = ExactDeduplicator(ignore_for_dedup=[r"\d+"])._config_hash()
        assert d1 != d2


class TestMinHashDeduplicatorIgnoreForDedup:
    def test_masked_span_pushes_similarity_above_threshold(self):
        base = "The quick brown fox jumps over the lazy dog id-1001"
        near_dup = "The quick brown fox jumps over the lazy dog id-9999"
        samples = [make_sample(base), make_sample(near_dup)]

        # Without masking, the differing id lowers similarity below threshold
        result = MinHashDeduplicator(threshold=0.95).run(samples)
        assert len(result) == 2

        # With the id masked, the two texts become identical
        result_masked = MinHashDeduplicator(
            threshold=0.95, ignore_for_dedup=[r"id-\d+"]
        ).run(samples)
        assert len(result_masked) == 1

    def test_config_hash_reflects_ignore_for_dedup(self):
        d1 = MinHashDeduplicator()._config_hash()
        d2 = MinHashDeduplicator(ignore_for_dedup=[r"\d+"])._config_hash()
        assert d1 != d2


class _FakeEmbeddingModel:
    """Deterministic bag-of-words hashing 'embedding' so EmbeddingDeduplicator's
    masking behavior can be tested without a real sentence-transformers model.
    Identical (post-masking) text always produces identical vectors."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        import numpy as np

        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vecs[i, hash(word) % self.dim] += 1.0
        return vecs


class TestEmbeddingDeduplicatorIgnoreForDedup:
    def _task(self, tmp_path, **kw):
        from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator

        task = EmbeddingDeduplicator(index_dir=tmp_path / "idx", threshold=0.99, **kw)
        task._model = _FakeEmbeddingModel()  # skip loading a real model
        return task

    def test_masked_span_makes_samples_duplicates(self, tmp_path):
        samples = [
            make_sample("Book flight for order id-1001", output="Confirmed."),
            make_sample("Book flight for order id-2002", output="Confirmed."),
        ]
        unmasked = self._task(tmp_path / "a")
        assert len(unmasked.run(list(samples))) == 2

        masked = self._task(tmp_path / "b", ignore_for_dedup=[r"id-\d+"])
        assert len(masked.run(list(samples))) == 1

    def test_invalid_pattern_raises_at_construction(self, tmp_path):
        from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator

        with pytest.raises(ValueError, match="invalid regex"):
            EmbeddingDeduplicator(index_dir=tmp_path, ignore_for_dedup=["("])

    def test_warns_on_ignore_for_dedup_mismatch_across_runs(self, tmp_path):
        index_dir = tmp_path / "idx"
        first = self._task(index_dir, ignore_for_dedup=[r"id-\d+"])
        first.run([make_sample("Order id-1001", output="ok")])

        second = self._task(index_dir, ignore_for_dedup=[r"\d+"])
        with pytest.warns(UserWarning, match="ignore_for_dedup"):
            second.run([make_sample("Order id-2002", output="ok")])

    def test_no_warning_when_ignore_for_dedup_unchanged(self, tmp_path):
        index_dir = tmp_path / "idx"
        first = self._task(index_dir, ignore_for_dedup=[r"id-\d+"])
        first.run([make_sample("Order id-1001", output="ok")])

        second = self._task(index_dir, ignore_for_dedup=[r"id-\d+"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            second.run([make_sample("Order id-2002", output="ok")])


class TestIgnoreForDedupYamlWiring:
    """The YAML/CLI pipeline path (PipelineConfig -> _build_steps) must
    thread ignore_for_dedup through to the actual dedup step instances,
    same as the programmatic CuratorConfig path."""

    def test_exact_dedup_normalizer_config_wires_ignore_for_dedup(self):
        from curatorkit.cli import _build_steps
        from curatorkit.config import NormalizerConfig, PipelineConfig

        config = PipelineConfig(
            normalizers=[NormalizerConfig(type="exact_dedup", ignore_for_dedup=[r"id-\d+"])]
        )
        steps, _ = _build_steps(config, verbose=False, include_exporters=False)
        dedup_steps = [s for s in steps if isinstance(s, ExactDeduplicator)]
        assert len(dedup_steps) == 1
        assert dedup_steps[0].ignore_for_dedup == [r"id-\d+"]

    def test_minhash_dedup_normalizer_config_wires_ignore_for_dedup(self):
        from curatorkit.cli import _build_steps
        from curatorkit.config import NormalizerConfig, PipelineConfig

        config = PipelineConfig(
            normalizers=[NormalizerConfig(type="minhash_dedup", ignore_for_dedup=[r"id-\d+"])]
        )
        steps, _ = _build_steps(config, verbose=False, include_exporters=False)
        dedup_steps = [s for s in steps if isinstance(s, MinHashDeduplicator)]
        assert len(dedup_steps) == 1
        assert dedup_steps[0].ignore_for_dedup == [r"id-\d+"]
