"""Train/val/test splits, kept apart by group, and a check for leakage between them.

A per-sample shuffle can put the translations or paraphrases of one item on both sides of
a split, so a test score then measures memorisation. With a group key, every sample of a
group lands in the same split.

A group's split comes from a hash of its id, so it depends only on the group and the seed:
an item stays in its split when the corpus grows, which a shuffle can't promise. The cost
is that split sizes only approximate the fractions, more loosely with few groups.
"""

from __future__ import annotations

import hashlib
import math
import random
import warnings
from collections import defaultdict
from collections.abc import Iterable

from curatorkit.detection.normalizer import ROLE_USER, normalize_turn
from curatorkit.normalizers.dedup import (
    _band_keys,
    _jaccard_estimate,
    _make_hash_funcs,
    _minhash_signature,
    _ngrams,
    _sample_dedup_text,
)
from curatorkit.schema import DataSample


def _unit(seed: int, group: str) -> float:
    """A stable number in [0, 1) for ``group``. ``hash()`` changes between processes."""
    digest = hashlib.sha256(f"split:{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _group_of(sample: DataSample, group_key: str) -> str | None:
    value = sample.metadata.get(group_key)
    return None if value is None or value == "" else str(value)


def _grouped_split(
    samples: list[DataSample], fractions: dict[str, float], seed: int, group_key: str
) -> dict[str, list[DataSample]]:
    missing = sum(_group_of(s, group_key) is None for s in samples)
    if samples and missing == len(samples):
        raise ValueError(
            f"output_split_group: no sample has metadata[{group_key!r}]. "
            f"Check the key; an input column not mapped to a field is kept in metadata, "
            f"except `id` and `source_uri`, which become the sample's own."
        )
    if missing:
        warnings.warn(
            f"output_split_group: {missing} of {len(samples)} samples have no "
            f"metadata[{group_key!r}], so each of those is split on its own text.",
            stacklevel=3,
        )

    names = list(fractions)
    bounds, total = [], 0.0
    for name in names[:-1]:
        total += fractions[name]
        bounds.append(total)

    out: dict[str, list[DataSample]] = {name: [] for name in names}
    for sample in samples:
        # Ids are random by default, so an ungrouped sample is keyed on its text instead.
        group = _group_of(sample, group_key) or f"text:{_sample_dedup_text(sample)}"
        unit = _unit(seed, group)
        out[names[sum(unit >= bound for bound in bounds)]].append(sample)

    empty = [name for name, rows in out.items() if not rows]
    if samples and empty:
        warnings.warn(
            f"output_split_group: split(s) {empty} are empty; there are too few groups "
            f"for these fractions.",
            stacklevel=3,
        )
    return out


def split_samples(
    samples: Iterable[DataSample],
    fractions: dict[str, float],
    *,
    seed: int = 42,
    group_key: str | None = None,
) -> dict[str, list[DataSample]]:
    """Split ``samples`` into ``{split name: samples}``.

    Without ``group_key``, a seeded shuffle gives exact sizes, and the last split takes
    the rounding remainder. With it, every sample sharing ``metadata[group_key]`` lands in
    the same split (see the module docstring). A sample with no value there is split on
    its own text, with a warning. No sample having one is taken for a wrong key and
    raises ``ValueError``, as do fractions that don't sum to 1.0.
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"output_split fractions must sum to 1.0, got {total:.4f}. Adjust your split ratios."
        )
    rows = list(samples)
    if group_key:
        return _grouped_split(rows, fractions, seed, group_key)

    random.Random(seed).shuffle(rows)
    out: dict[str, list[DataSample]] = {}
    start = 0
    items = list(fractions.items())
    for i, (name, fraction) in enumerate(items):
        end = len(rows) if i == len(items) - 1 else start + math.floor(len(rows) * fraction)
        out[name] = rows[start:end]
        start = end
    return out


def _prompt_text(sample: DataSample) -> str:
    """What the model is shown, normalised. One prompt with two answers still leaks, so
    this leaves out the answer that ``_sample_dedup_text`` includes, unless there's no
    prompt to compare."""
    parts = [sample.instruction, sample.input]
    if sample.task_type == "conversational":
        # The opener alone would make every chat that starts with "Hi" a duplicate. The system
        # prompt is left out: one shared by a whole corpus would make every chat a near duplicate.
        for turn in sample.metadata.get("turns") or []:
            if isinstance(turn, dict):
                turn, _ = normalize_turn(turn)
                if turn.get("role") == ROLE_USER:
                    parts.append(str(turn.get("content", "")))
    text = " ".join(parts).strip() or _sample_dedup_text(sample)
    return " ".join(text.split()).casefold()


def _recall_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """(bands, rows) whose LSH curve passes 0.5 at or below ``threshold``.

    Dedup's ``_pick_bands`` takes the nearest point on either side; for 0.8 with 128
    permutations that is 0.88, which finds under half of the pairs at 0.85. A missed leak
    costs more here than a candidate pair, which the Jaccard estimate then rejects."""
    pairs = [(b, num_perm // b) for b in range(1, num_perm + 1) if num_perm % b == 0]
    below = [(b, r) for b, r in pairs if (1 / b) ** (1 / r) <= threshold]
    return min(below or pairs, key=lambda br: abs((1 / br[0]) ** (1 / br[1]) - threshold))


def _near_duplicates(
    rows: list[tuple[str, DataSample, str]], threshold: float, num_perm: int, seed: int
) -> list[tuple[str, str, float]]:
    funcs = _make_hash_funcs(num_perm, seed)
    bands, width = _recall_bands(num_perm, threshold)
    signatures: dict[int, list[int]] = {}
    # One representative per split in each bucket keeps this linear in the samples: on a
    # templated corpus most prompts share a bucket, and comparing all pairs would be quadratic.
    buckets: dict[tuple, dict[str, int]] = defaultdict(dict)
    found: dict[tuple[int, int], float] = {}
    for i, (split, _, text) in enumerate(rows):
        grams = _ngrams(text)
        if not grams:  # shorter than an n-gram: an empty signature would match every other
            continue
        signature = signatures[i] = _minhash_signature(grams, funcs)
        for band, key in enumerate(_band_keys(signature, bands, width)):
            bucket = buckets[(band, key)]
            for other_split, j in bucket.items():
                if other_split == split or rows[j][2] == text or (j, i) in found:
                    continue  # same split, or an exact duplicate, which is reported on its own
                score = _jaccard_estimate(signatures[j], signature)
                if score >= threshold:
                    found[(j, i)] = score
            bucket.setdefault(split, i)
    return [(rows[a][1].id, rows[b][1].id, score) for (a, b), score in sorted(found.items())]


def leakage_report(
    splits: dict[str, list[DataSample]],
    *,
    group_key: str | None = None,
    near_dup_threshold: float | None = 0.8,
    num_perm: int = 128,
    seed: int = 42,
) -> dict:
    """Check that nothing appears in more than one split.

    Returns ``groups_in_multiple_splits`` (only with ``group_key``), ``exact_duplicates``
    and ``near_duplicates`` (sample-id pairs, near ones with their estimated Jaccard
    similarity), and ``ok``. Prompts are compared after folding case and whitespace.
    Text matching can't see a translation, so only a group key catches those.

    Near duplicates come from MinHash, so they are approximate, and a templated corpus
    reports some of its pairs rather than all of them. The pass costs about as much as
    MinHash dedup; ``near_dup_threshold=None`` skips it.
    """
    rows = [(name, s, _prompt_text(s)) for name, samples in splits.items() for s in samples]

    groups: dict[str, set[str]] = defaultdict(set)
    if group_key:
        for name, sample, _ in rows:
            group = _group_of(sample, group_key)
            if group is not None:
                groups[group].add(name)
    in_several = sorted(group for group, names in groups.items() if len(names) > 1)

    first_seen: dict[str, dict[str, str]] = defaultdict(dict)
    for name, sample, text in rows:
        first_seen[text].setdefault(name, sample.id)
    exact = [
        (ids[a], ids[b])
        for ids in (first_seen[text] for text in first_seen if text)
        for a, b in zip(list(ids), list(ids)[1:])
    ]

    near = (
        _near_duplicates(rows, near_dup_threshold, num_perm, seed)
        if near_dup_threshold is not None
        else []
    )
    return {
        "groups_in_multiple_splits": in_several,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "ok": not (in_several or exact or near),
    }


def warn_on_leakage(
    splits: dict[str, list[DataSample]],
    group_key: str | None = None,
    near_dup_threshold: float | None = 0.8,
) -> dict:
    """Run :func:`leakage_report` and warn when it finds anything."""
    report = leakage_report(splits, group_key=group_key, near_dup_threshold=near_dup_threshold)
    if not report["ok"]:
        warnings.warn(
            f"output_split leakage: {len(report['groups_in_multiple_splits'])} groups, "
            f"{len(report['exact_duplicates'])} exact and "
            f"{len(report['near_duplicates'])} near-duplicate prompts are in more than one "
            f"split. Use curatorkit.splits.leakage_report for the sample ids.",
            stacklevel=2,
        )
    return report
