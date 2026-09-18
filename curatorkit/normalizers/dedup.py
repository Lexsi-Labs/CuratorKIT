"""
Deduplication normalizers — both implemented from scratch.

ExactDeduplicator: SHA-256 hash of normalised instruction text.

MinHashDeduplicator: Full MinHash implementation without datasketch.
  - Character n-gram tokenisation (default n=3)
  - k independent hash functions via (a*x + b) mod p for random (a, b)
  - Jaccard estimation as fraction of matching signature positions
  - Threshold is a tunable config value — not a code constant

The MinHash threshold is a direct data quality lever. A threshold that works
for a general instruction dataset will incorrectly flag valid near-identical
legal clauses as duplicates. Document this in every run's manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import UTC, datetime

from curatorkit.interfaces import BaseNormalizer
from curatorkit.normalizers.dedup_utils import apply_ignore_patterns, compile_ignore_patterns
from curatorkit.schema import DataSample, ProvenanceRecord

STEP_VERSION = "0.1.0"


def _format_extra_turns(turns: list) -> str:
    """Flatten metadata['turns'] (everything past the first exchange) into
    plain text for dedup comparison. Handles both shapes found in the
    codebase: readers normalize ingested conversations to ChatML
    {"role": ..., "content": ...} (curatorkit.detection.normalizer), while
    MultiTurnTask's own generation output uses ShareGPT-style
    {"from": ..., "value": ...} (generators/multiturn_gen.py)."""
    parts = []
    for t in turns:
        if isinstance(t, dict):
            parts.append(str(t.get("content", t.get("value", ""))))
    return "".join(parts)


def _dedup_fields(sample: DataSample) -> tuple[list[str], str]:
    """Task-type-aware field selection for duplicate comparison — the single
    shared implementation for ExactDeduplicator and MinHashDeduplicator.
    Returns (maskable components, unmasked suffix): components are kept
    unjoined so an ignore_for_dedup pattern is applied per-field, before
    concatenation, and can't bleed across a field boundary; the suffix
    (currently only unpaired_preference's label) is a semantic tag, not
    user text, and is never subject to masking.

    preference / implicit_preference → instruction + chosen + rejected
    grpo                             → instruction + all responses
    language_modeling                → output
    conversational                   → instruction + output (first exchange)
                                        + every later turn in metadata['turns']
    unpaired_preference              → instruction + output, plus a
                                        never-masked label suffix
    everything else (SFT)            → instruction + output

    conversational and unpaired_preference need their own branches, not the
    generic SFT fallback: a conversational sample's first-exchange fields
    (instruction/output) don't capture the rest of the conversation, which
    lives in metadata['turns'] — comparing only the opener let two
    conversations that diverge after turn 1 collapse into "duplicates".
    unpaired_preference samples carry a label (e.g. the same instruction+
    output rated differently by two annotators) that instruction+output
    alone doesn't reflect — omitting it let differently-labeled rows for the
    same text wrongly collapse into one, silently dropping a label.
    """
    task = sample.task_type
    if task in ("preference", "implicit_preference"):
        return [sample.instruction, sample.chosen, sample.rejected], ""
    if task == "grpo":
        return [sample.instruction, *sample.responses], ""
    if task == "language_modeling":
        return [sample.output], ""
    if task == "conversational":
        extra = _format_extra_turns(sample.metadata.get("turns", []))
        return [sample.instruction, sample.output, extra], ""
    if task == "unpaired_preference":
        return [sample.instruction, sample.output], f"|label={sample.label}"
    return [sample.instruction, sample.output], ""


def _sample_dedup_text(sample: DataSample, ignore_patterns: list[re.Pattern] | None = None) -> str:
    """Build the final comparison string for `sample`, applying
    ignore_for_dedup patterns (if any) to each field before concatenation."""
    fields, suffix = _dedup_fields(sample)
    if ignore_patterns:
        fields = [apply_ignore_patterns(f, ignore_patterns) for f in fields]
    return "".join(fields) + suffix


# Large prime used in the universal hash family (a*x + b) mod p mod 2^32
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


# ---------------------------------------------------------------------------
# MinHash primitives
# ---------------------------------------------------------------------------


def _ngrams(text: str, n: int = 3) -> set[int]:
    """Character n-grams hashed to integers."""
    tokens = [text[i : i + n] for i in range(len(text) - n + 1)]
    return {int(hashlib.md5(t.encode()).hexdigest(), 16) & _MAX_HASH for t in tokens}


def _make_hash_funcs(num_perm: int, seed: int = 42) -> list[tuple[int, int]]:
    """Generate num_perm (a, b) pairs for universal hashing."""
    rng = random.Random(seed)
    return [
        (rng.randint(1, _MERSENNE_PRIME), rng.randint(0, _MERSENNE_PRIME)) for _ in range(num_perm)
    ]


def _minhash_signature(ngram_hashes: set[int], hash_funcs: list[tuple[int, int]]) -> list[int]:
    """Compute the MinHash signature vector for a set of n-gram hashes."""
    sig = []
    for a, b in hash_funcs:
        min_val = _MAX_HASH
        for h in ngram_hashes:
            val = int((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH
            if val < min_val:
                min_val = val
        sig.append(min_val if ngram_hashes else _MAX_HASH)
    return sig


def _jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity as fraction of equal signature positions."""
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


# ---------------------------------------------------------------------------
# LSH banding helpers
# ---------------------------------------------------------------------------


def _pick_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """
    Pick (bands, rows) such that bands * rows == num_perm and the LSH curve
    s ↦ 1 - (1 - s^rows)^bands has its 0.5 inflection point near `threshold`.

    For each divisor pair, compute the inflection point ≈ (1/bands)^(1/rows)
    and pick the pair whose inflection is closest to `threshold`.
    """
    best = (1, num_perm)
    best_dist = float("inf")
    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        inflection = (1.0 / b) ** (1.0 / r)
        dist = abs(inflection - threshold)
        if dist < best_dist:
            best_dist = dist
            best = (b, r)
    return best


def _band_keys(sig: list[int], bands: int, rows: int) -> list[tuple]:
    """Split a signature into `bands` tuples of `rows` ints each (used as dict keys)."""
    return [tuple(sig[i * rows : (i + 1) * rows]) for i in range(bands)]


# ---------------------------------------------------------------------------
# ExactDeduplicator
# ---------------------------------------------------------------------------


class ExactDeduplicator(BaseNormalizer):
    """Remove exact duplicates by SHA-256 - hash key is task-type-aware.

    See _dedup_fields() for the exact field selection per task_type.

    Parameters
    ----------
    ignore_for_dedup : list[str] | None
        Regex patterns. Matches are masked out (replaced with a space) from
        each field's text before it's hashed, so a real but unimportant
        difference between two samples (a per-sample UUID, a timestamp) does
        not itself prevent them from being recognized as duplicates. Never
        mutates the actual sample — only the throwaway string built for
        comparison.
    """

    def __init__(self, ignore_for_dedup: list[str] | None = None) -> None:
        self.ignore_for_dedup = list(ignore_for_dedup) if ignore_for_dedup else []
        self._ignore_patterns = compile_ignore_patterns(self.ignore_for_dedup)

    def _config_hash(self) -> str:
        payload = json.dumps({"ignore_for_dedup": self.ignore_for_dedup}, sort_keys=True)
        return hashlib.sha256(f"ExactDeduplicator:0.2.0:{payload}".encode()).hexdigest()[:16]

    def _sample_key(self, sample: DataSample) -> str:
        text = _sample_dedup_text(sample, self._ignore_patterns)
        return " ".join(text.lower().split())

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        from tqdm import tqdm

        seen: dict[str, str] = {}  # hash -> sample id
        passed: list[DataSample] = []
        removed = 0

        for sample in tqdm(samples, desc="ExactDeduplicator", unit="sample"):
            key = self._sample_key(sample)
            h = hashlib.sha256(key.encode()).hexdigest()
            if h in seen:
                removed += 1
            else:
                seen[h] = sample.id
                passed.append(sample)

        cfg_hash = self._config_hash()
        ts = datetime.now(UTC).replace(tzinfo=None)
        for sample in passed:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="ExactDeduplicator",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "exact_duplicates_removed": removed,
                        "surviving_count": len(passed),
                    },
                )
            )

        return passed


# ---------------------------------------------------------------------------
# MinHashDeduplicator
# ---------------------------------------------------------------------------


class MinHashDeduplicator(BaseNormalizer):
    """Remove near-duplicate samples using MinHash + LSH banding.

    Algorithm:
      1. Tokenise each instruction into character n-grams (default n=3).
      2. Compute a k-dimensional MinHash signature per sample using k
         independent universal hash functions (default k=128).
      3. Split each signature into `bands` rows of `rows` ints; bands × rows = k.
         The (bands, rows) pair is auto-tuned so the LSH s-curve's 0.5
         inflection sits near `threshold`.
      4. Two signatures collide in a band iff their `rows`-int sub-vector is
         equal. We only verify Jaccard for collision candidates — turning the
         O(n²) brute-force into ~O(n) under typical thresholds.
      5. If estimated Jaccard >= threshold, the later sample is a near-dup.

    The threshold is a tunable config value — not a code constant.
    A threshold of 0.85 is a starting point, not a universal answer.

    Parameters
    ----------
    ignore_for_dedup : list[str] | None
        Regex patterns. Matches are masked out (replaced with a space) from
        each field's text before n-gram tokenisation, so a real but
        unimportant difference between two samples doesn't itself push their
        estimated Jaccard similarity below `threshold`. Never mutates the
        actual sample — only the throwaway string built for comparison.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        ngram: int = 3,
        num_perm: int = 128,
        seed: int = 42,
        bands: int | None = None,
        rows: int | None = None,
        ignore_for_dedup: list[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.ngram = ngram
        self.num_perm = num_perm
        self.seed = seed
        self.ignore_for_dedup = list(ignore_for_dedup) if ignore_for_dedup else []
        self._ignore_patterns = compile_ignore_patterns(self.ignore_for_dedup)
        self._hash_funcs = _make_hash_funcs(num_perm, seed)
        # Auto-pick LSH (bands, rows) unless explicitly provided
        if bands is None or rows is None:
            self.bands, self.rows = _pick_bands(num_perm, threshold)
        else:
            if bands * rows != num_perm:
                raise ValueError(f"bands * rows ({bands * rows}) must equal num_perm ({num_perm})")
            self.bands, self.rows = bands, rows

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "threshold": self.threshold,
                "ngram": self.ngram,
                "num_perm": self.num_perm,
                "seed": self.seed,
                "bands": self.bands,
                "rows": self.rows,
                "ignore_for_dedup": self.ignore_for_dedup,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        # signatures stores one entry per non-duplicate sample only.
        signatures: list[list[int]] = []
        passed_indices: list[int] = []
        removed = 0

        # LSH buckets: (band_index, band_key) -> list of indices into `signatures`
        # Candidates for sample i are the union of bucket members across all bands.
        from collections import defaultdict

        buckets: dict[tuple[int, tuple], list[int]] = defaultdict(list)

        from tqdm import tqdm

        for i, sample in enumerate(tqdm(samples, desc="MinHashDeduplicator", unit="sample")):
            text = _sample_dedup_text(sample, self._ignore_patterns)
            grams = _ngrams(text.lower(), self.ngram)
            sig = _minhash_signature(grams, self._hash_funcs)

            # Collect candidate indices via band collisions
            keys = _band_keys(sig, self.bands, self.rows)
            candidate_indices: set[int] = set()
            for band_idx, key in enumerate(keys):
                bucket = buckets.get((band_idx, key))
                if bucket:
                    candidate_indices.update(bucket)

            # Verify only collision candidates with full Jaccard
            is_dup = False
            for c_idx in candidate_indices:
                if _jaccard_estimate(sig, signatures[c_idx]) >= self.threshold:
                    is_dup = True
                    break

            if is_dup:
                removed += 1
            else:
                new_idx = len(signatures)
                signatures.append(sig)
                passed_indices.append(i)
                # Index this sample's bands so future samples can find it
                for band_idx, key in enumerate(keys):
                    buckets[(band_idx, key)].append(new_idx)

        passed = [samples[i] for i in passed_indices]
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC).replace(tzinfo=None)

        for sample in passed:
            sample.append_provenance(
                ProvenanceRecord(
                    step_name="MinHashDeduplicator",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "minhash_threshold": self.threshold,
                        "ngram": self.ngram,
                        "num_perm": self.num_perm,
                        "lsh_bands": self.bands,
                        "lsh_rows": self.rows,
                        "near_duplicates_removed": removed,
                        "surviving_count": len(passed),
                        "threshold_caveat": (
                            "0.85 is a starting point. Adjust per domain. "
                            "Legal/medical data may need a lower threshold."
                        ),
                    },
                )
            )

        return passed
