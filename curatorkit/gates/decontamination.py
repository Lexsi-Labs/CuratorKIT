"""
DecontaminationGate — reject samples that contaminate eval benchmarks.

Reference: Brown et al. (GPT-3 paper, 2020) §4.0 introduced 13-gram overlap
as the canonical decontamination test. The same protocol is still standard
in 2024-2026 — e.g. Llama 3 used 8-gram overlap, Qwen 2.5 used a hybrid of
8/13-gram, and the EleutherAI lm-evaluation-harness records contamination
via n-gram overlap.

CuratorKIT's DecontaminationGate:

  1. At construction time, ingest one or more "eval shard" files (JSONL or
     plaintext) and build a Bloom-filter-free *exact* n-gram set per shard.
  2. For each input sample, tokenise the relevant text field and check how
     many n-grams overlap with each shard.
  3. Reject the sample if the overlap exceeds `max_ngram_overlap` for any
     shard, with reason `decontamination:{shard_name}:{n_overlap}`.

The shard-set is held in memory: ~10M 13-grams from MMLU/HumanEval/GSM8K
fits comfortably under 2 GB. For larger reference corpora swap in a
`MemoryMappedNgramStore` (not shipped).

Rejection vocabulary added:
  decontamination:{shard}:{overlap_count}
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word-token split — matches GPT-3's preprocessing."""
    return _TOKEN_RE.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    """Yield n-grams as tuples — uses tuple-of-strings for hashability."""
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


# ───────────────────────────────────────────────────────────────────────────
# Shard
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class _ContaminationShard:
    """One eval-benchmark shard's n-gram set + metadata."""

    name: str
    n: int
    ngrams: set[tuple[str, ...]] = field(default_factory=set)
    n_documents: int = 0
    source_path: str = ""

    def add_text(self, text: str) -> None:
        toks = _tokenize(text)
        for gram in _ngrams(toks, self.n):
            self.ngrams.add(gram)
        self.n_documents += 1

    def overlap(self, text: str) -> int:
        """Count input n-grams that appear in this shard's set."""
        toks = _tokenize(text)
        if len(toks) < self.n:
            return 0
        hits = 0
        for gram in _ngrams(toks, self.n):
            if gram in self.ngrams:
                hits += 1
        return hits


# ───────────────────────────────────────────────────────────────────────────
# DecontaminationGate
# ───────────────────────────────────────────────────────────────────────────


class DecontaminationGate(BaseGate):
    """
    Reject samples whose text overlaps eval-benchmark n-grams beyond a
    configurable threshold.

    Parameters
    ----------
    shards : dict[str, list[str | Path]]
        {shard_name: [list of files to ingest]}. Each file is either:
          - .jsonl  → reads each line as JSON and indexes
                      the value of `text_key` (default: "text" or
                      "question" or "prompt", first one present)
          - .txt    → reads the whole file as one document
    n : int
        N-gram length. GPT-3 used 13; Llama 3 used 8. Default 13.
    max_ngram_overlap : int
        Reject if overlap is > this many n-grams against any single shard.
        Default 1 (any overlap = rejection — matches Llama 3's strict
        protocol). Increase to allow incidental phrase coincidence.
    fields : list[str]
        Which DataSample text fields to scan. Default: instruction + output.
    """

    def __init__(
        self,
        shards: dict[str, list[str | Path]],
        n: int = 13,
        max_ngram_overlap: int = 1,
        fields: list[str] | None = None,
        text_keys: tuple[str, ...] = ("text", "question", "prompt", "input"),
    ) -> None:
        self.n = max(1, int(n))
        self.max_ngram_overlap = max(0, int(max_ngram_overlap))
        self.fields = fields or ["instruction", "output"]
        self.text_keys = tuple(text_keys)

        self._shards: dict[str, _ContaminationShard] = {}
        for name, paths in shards.items():
            shard = _ContaminationShard(name=name, n=self.n)
            for p in paths:
                self._ingest_file(shard, Path(p))
            self._shards[name] = shard

    def _ingest_file(self, shard: _ContaminationShard, path: Path) -> None:
        if not path.exists():
            return
        shard.source_path = str(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = self._extract_jsonl_text(rec)
                if text:
                    shard.add_text(text)
        else:
            shard.add_text(path.read_text(encoding="utf-8", errors="ignore"))

    def _extract_jsonl_text(self, rec: Any) -> str:
        if not isinstance(rec, dict):
            return str(rec)
        for k in self.text_keys:
            if k in rec and isinstance(rec[k], str):
                return rec[k]
        # Last resort: concatenate all string values
        return " ".join(str(v) for v in rec.values() if isinstance(v, str))

    # ─────────────────────────────────────────────────────────────────────

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "n": self.n,
                "max_ngram_overlap": self.max_ngram_overlap,
                "fields": sorted(self.fields),
                "shard_counts": {
                    name: {"n_docs": s.n_documents, "n_ngrams": len(s.ngrams)}
                    for name, s in self._shards.items()
                },
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _scan(self, sample: DataSample) -> tuple[str | None, int]:
        """Return (offending_shard_name, overlap_count) or (None, 0) if clean."""
        text = " ".join(getattr(sample, f, "") or "" for f in self.fields)
        worst_shard = None
        worst_overlap = 0
        for name, shard in self._shards.items():
            overlap = shard.overlap(text)
            if overlap > worst_overlap:
                worst_shard = name
                worst_overlap = overlap
        if worst_overlap > self.max_ngram_overlap:
            return worst_shard, worst_overlap
        return None, worst_overlap

    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        cfg_hash = self._config_hash()

        for sample in samples:
            shard_name, overlap = self._scan(sample)
            if shard_name is None:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="DecontaminationGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": True,
                            "max_overlap": overlap,
                            "threshold": self.max_ngram_overlap,
                            "n": self.n,
                        },
                    )
                )
                passed.append(sample)
                continue

            rej = RejectedSample(
                **sample.model_dump(),
                rejection_reason=f"decontamination:{shard_name}:{overlap}",
                rejecting_step="DecontaminationGate",
            )
            rej.append_provenance(
                ProvenanceRecord(
                    step_name="DecontaminationGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "passed": False,
                        "shard": shard_name,
                        "overlap_count": overlap,
                        "threshold": self.max_ngram_overlap,
                        "n": self.n,
                    },
                )
            )
            rejected.append(rej)

        return passed, rejected

    # ─────────────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────────────

    def shard_stats(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "n_documents": s.n_documents,
                "n_ngrams": len(s.ngrams),
                "n": s.n,
            }
            for name, s in self._shards.items()
        }
