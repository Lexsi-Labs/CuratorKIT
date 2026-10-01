"""
Diversity primitives — pure Python, no embedding dependency.

These three classical metrics give a fast first read on coverage:

  distinct-N        — fraction of unique N-grams over total N-grams.
                      Distinct-1 ≈ vocabulary diversity. Distinct-2 is the
                      standard NLG diversity reference (Li et al. 2016).
                      Collapse signal: distinct-2 < 0.40 across a batch.

  type-token ratio  — unique-tokens / total-tokens. Naïve but cheap. Drops
                      under model collapse before distinct-N moves.

  Shannon entropy   — over an arbitrary categorical distribution (clusters,
                      task_types). Higher → more uniform → better coverage.
                      Used by the analyser to compare cluster occupancy.

These are NOT a replacement for embedding-space diversity (mean pairwise
cosine, cluster-occupancy KL). They are the floor a generation pipeline
must clear before paying for embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ───────────────────────────────────────────────────────────────────────────
# n-gram diversity
# ───────────────────────────────────────────────────────────────────────────


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """
    Distinct-N: unique n-grams / total n-grams across the batch.

    Returns 0.0 for an empty batch and for texts that produce no n-grams
    (e.g., shorter than n tokens after tokenization).
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    total = 0
    seen: set[tuple[str, ...]] = set()

    for text in texts:
        toks = _tokenize(text)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i : i + n])
            seen.add(gram)
            total += 1

    if total == 0:
        return 0.0
    return len(seen) / total


def type_token_ratio(texts: Iterable[str]) -> float:
    """Unique tokens / total tokens across the batch."""
    total = 0
    types: set[str] = set()
    for text in texts:
        for t in _tokenize(text):
            types.add(t)
            total += 1
    if total == 0:
        return 0.0
    return len(types) / total


# ───────────────────────────────────────────────────────────────────────────
# Cluster / category entropy
# ───────────────────────────────────────────────────────────────────────────


def shannon_entropy(labels: Iterable[str]) -> float:
    """
    Shannon entropy of a categorical distribution, in bits (log base 2).

    Returns 0.0 for empty / single-class input. Maximum entropy for K
    classes is log2(K). Use `entropy / log2(K)` if you want a 0..1
    "occupancy uniformity" reading.
    """
    counts = Counter(labels)
    n = sum(counts.values())
    if n == 0 or len(counts) <= 1:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def normalized_entropy(labels: Iterable[str]) -> float:
    """Shannon entropy divided by log2(K). 0..1 uniformity reading."""
    counts = Counter(labels)
    if len(counts) <= 1:
        return 0.0
    return shannon_entropy(counts.elements()) / math.log2(len(counts))


# ───────────────────────────────────────────────────────────────────────────
# Embedding-space metrics (called with numeric vectors)
# ───────────────────────────────────────────────────────────────────────────


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pairwise_cosine_stats(
    embeddings: list[Sequence[float]],
    sample_pairs: int | None = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Mean / min / max pairwise cosine *distance* (1 - similarity) over a batch.

    For batches larger than ~100 samples computing all C(n, 2) pairs is
    quadratic and slow; pass `sample_pairs=N` to subsample uniformly.

    Returns {"mean_distance", "min_distance", "max_distance", "n_pairs"}.
    Empty input or n < 2 → all zeros.
    """
    n = len(embeddings)
    if n < 2:
        return {"mean_distance": 0.0, "min_distance": 0.0, "max_distance": 0.0, "n_pairs": 0}

    import random

    rng = random.Random(seed)

    # Generate pair indices
    total_pairs = n * (n - 1) // 2
    if sample_pairs is None or total_pairs <= sample_pairs:
        pair_iter: Iterable[tuple[int, int]] = ((i, j) for i in range(n) for j in range(i + 1, n))
    else:
        # Reservoir-free uniform sample: draw `sample_pairs` (i, j) tuples.
        # Duplicates are vanishingly rare for the recommended cap.
        pair_iter = (tuple(sorted(rng.sample(range(n), 2))) for _ in range(sample_pairs))

    total = 0.0
    mn = 1.0
    mx = -1.0
    actual = 0
    for i, j in pair_iter:
        d = 1.0 - _cosine(embeddings[i], embeddings[j])
        total += d
        mn = min(mn, d)
        mx = max(mx, d)
        actual += 1

    if actual == 0:
        return {"mean_distance": 0.0, "min_distance": 0.0, "max_distance": 0.0, "n_pairs": 0}

    return {
        "mean_distance": round(total / actual, 4),
        "min_distance": round(mn, 4),
        "max_distance": round(mx, 4),
        "n_pairs": actual,
    }


# ───────────────────────────────────────────────────────────────────────────
# Bundled report
# ───────────────────────────────────────────────────────────────────────────


def diversity_metrics(
    texts: list[str],
    labels: list[str] | None = None,
    embeddings: list[Sequence[float]] | None = None,
    distinct_ns: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float | dict[str, float]]:
    """
    One-shot diversity bundle. Skips embedding-dependent metrics if no
    embeddings are passed.
    """
    out: dict[str, float | dict[str, float]] = {
        "n_samples": len(texts),
        "type_token_ratio": round(type_token_ratio(texts), 4),
    }
    for n in distinct_ns:
        out[f"distinct_{n}"] = round(distinct_n(texts, n), 4)

    if labels:
        out["label_entropy_bits"] = round(shannon_entropy(labels), 4)
        out["label_normalized_entropy"] = round(normalized_entropy(labels), 4)
        out["unique_labels"] = len(set(labels))

    if embeddings:
        out["embedding_stats"] = pairwise_cosine_stats(embeddings)

    return out
