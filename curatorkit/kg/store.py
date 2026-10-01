"""
KnowledgeGraph — in-memory triples + adjacency + JSON persistence.

Owned implementation (no networkx import) so we can extend / inspect /
serialize without indirection. Three primary operations the rest of the
KG module needs:

  add_triple(s, p, o)        — insert (subject, predicate, object)
  neighbors(entity, k=1)     — k-hop neighbors (BFS, returns set)
  paths(src, dst, max_hops)  — enumerate paths of length ≤ max_hops
                                between two entities (used by multi-hop QA)
  entities() / relations()   — vocabularies (used by coverage analyser)

The triple set is stored as `dict[subject, dict[predicate, set[object]]]`.
For lookups in the reverse direction the class maintains a parallel
`dict[object, set[(subject, predicate)]]` index so walking is O(1) per step.

A graph is persisted as a JSON file with two top-level keys:
  - triples:   list of {s, p, o, source_uri, source_chunk_hash, confidence}
  - meta:      {entity_count, relation_count, triple_count, built_at}
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ───────────────────────────────────────────────────────────────────────────
# Triple
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Triple:
    """One graph edge: subject -[predicate]-> object, with provenance."""

    subject: str
    predicate: str
    object: str
    source_uri: str = ""
    source_chunk_hash: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "s": self.subject,
            "p": self.predicate,
            "o": self.object,
            "source_uri": self.source_uri,
            "source_chunk_hash": self.source_chunk_hash,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Triple:
        return cls(
            subject=d["s"],
            predicate=d["p"],
            object=d["o"],
            source_uri=d.get("source_uri", ""),
            source_chunk_hash=d.get("source_chunk_hash", ""),
            confidence=float(d.get("confidence", 1.0)),
        )


# ───────────────────────────────────────────────────────────────────────────
# KnowledgeGraph
# ───────────────────────────────────────────────────────────────────────────


class KnowledgeGraph:
    """
    Dict-of-dict-of-set triple store.

    Construction is empty by default; populate via `add_triple()` or load
    from disk via `KnowledgeGraph.load()`.

    Thread-safety: not thread-safe. Build the graph in one thread (or
    serialize updates externally); reads are safe once construction is done.
    """

    def __init__(self) -> None:
        # Forward adjacency:   {subject: {predicate: set(objects)}}
        self._adj: dict[str, dict[str, set[str]]] = {}
        # Reverse adjacency:   {object: set((predicate, subject))}
        self._inv: dict[str, set[tuple[str, str]]] = {}
        # All triples, in insertion order (used by serialization + provenance).
        self._triples: list[Triple] = []
        # Predicate frequency cache (used by coverage analyser).
        self._predicate_counts: dict[str, int] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Insertion
    # ─────────────────────────────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_uri: str = "",
        source_chunk_hash: str = "",
        confidence: float = 1.0,
    ) -> bool:
        """
        Insert one triple. Returns True if the (s,p,o) was new, False if it
        was a duplicate. Duplicates do not raise — the call is idempotent —
        but the highest-confidence provenance wins.
        """
        subject = subject.strip()
        predicate = predicate.strip()
        obj = obj.strip()
        if not subject or not predicate or not obj:
            return False

        slot = self._adj.setdefault(subject, {}).setdefault(predicate, set())
        was_new = obj not in slot
        slot.add(obj)
        self._inv.setdefault(obj, set()).add((predicate, subject))

        if was_new:
            self._triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    source_uri=source_uri,
                    source_chunk_hash=source_chunk_hash,
                    confidence=confidence,
                )
            )
            self._predicate_counts[predicate] = self._predicate_counts.get(predicate, 0) + 1
        return was_new

    def add_triples(self, triples: Iterable[Triple | tuple[str, str, str]]) -> int:
        """Bulk insert. Returns count of NEW triples added."""
        added = 0
        for t in triples:
            if isinstance(t, Triple):
                if self.add_triple(
                    t.subject,
                    t.predicate,
                    t.object,
                    t.source_uri,
                    t.source_chunk_hash,
                    t.confidence,
                ):
                    added += 1
            elif isinstance(t, tuple) and len(t) == 3:
                if self.add_triple(*t):
                    added += 1
        return added

    # ─────────────────────────────────────────────────────────────────────
    # Queries
    # ─────────────────────────────────────────────────────────────────────

    def entities(self) -> set[str]:
        """Return the union of subjects and objects (the vocabulary)."""
        return set(self._adj.keys()) | set(self._inv.keys())

    def relations(self) -> set[str]:
        """All distinct predicates."""
        return set(self._predicate_counts.keys())

    def triples(self) -> list[Triple]:
        return list(self._triples)

    def predicate_counts(self) -> dict[str, int]:
        return dict(self._predicate_counts)

    def __len__(self) -> int:
        return len(self._triples)

    # ─────────────────────────────────────────────────────────────────────

    def neighbors(self, entity: str, k: int = 1) -> set[str]:
        """
        Return all entities reachable from `entity` in ≤ k hops (directed +
        reverse). Excludes `entity` itself.
        """
        if k <= 0:
            return set()
        seen: set[str] = {entity}
        frontier: set[str] = {entity}
        for _ in range(k):
            next_frontier: set[str] = set()
            for node in frontier:
                # Forward edges
                for objs in self._adj.get(node, {}).values():
                    next_frontier |= objs
                # Reverse edges
                for _pred, subj in self._inv.get(node, set()):
                    next_frontier.add(subj)
            frontier = next_frontier - seen
            seen |= frontier
            if not frontier:
                break
        return seen - {entity}

    def successors(self, entity: str) -> list[tuple[str, str]]:
        """Return [(predicate, object)] for entity as subject."""
        out: list[tuple[str, str]] = []
        for pred, objs in self._adj.get(entity, {}).items():
            for o in objs:
                out.append((pred, o))
        return out

    def predecessors(self, entity: str) -> list[tuple[str, str]]:
        """Return [(predicate, subject)] for entity as object."""
        return sorted(self._inv.get(entity, set()))

    def paths(
        self,
        src: str,
        dst: str,
        max_hops: int = 3,
    ) -> list[list[tuple[str, str, str]]]:
        """
        Enumerate all paths from `src` to `dst` with at most `max_hops`
        edges. Each path is a list of (s, p, o) tuples. Returns empty list
        if no path exists.

        Used by multi-hop QA: a path of length k generates a k-hop question.
        BFS over the undirected version of the graph; per-path edge direction
        is preserved in the output tuples.
        """
        if src == dst or src not in self.entities() or dst not in self.entities():
            return []

        results: list[list[tuple[str, str, str]]] = []
        # (current_node, path_so_far)
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque([(src, [])])
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            # Forward edges
            for pred, objs in self._adj.get(node, {}).items():
                for o in objs:
                    if any(o == p[0] or o == p[2] for p in path):
                        continue  # avoid revisits
                    new_path = path + [(node, pred, o)]
                    if o == dst:
                        results.append(new_path)
                    else:
                        queue.append((o, new_path))
            # Reverse edges (treat as undirected for path finding)
            for pred, subj in self._inv.get(node, set()):
                if subj == node:
                    continue
                if any(subj == p[0] or subj == p[2] for p in path):
                    continue
                new_path = path + [(subj, pred, node)]
                if subj == dst:
                    results.append(new_path)
                else:
                    queue.append((subj, new_path))
        return results

    def sample_paths(
        self,
        max_hops: int = 2,
        sample_size: int = 100,
        seed: int = 42,
    ) -> list[list[tuple[str, str, str]]]:
        """
        Uniformly sample paths from the graph for multi-hop QA generation.

        Each sample is a sequence of (s,p,o) tuples of length in [1, max_hops].
        Used by MultiHopQATask to drive generation without enumerating every
        possible path (which is exponential).
        """
        import random

        rng = random.Random(seed)
        if not self._triples:
            return []

        results: list[list[tuple[str, str, str]]] = []
        n_attempts = sample_size * 4  # may reject self-cycle paths
        attempts = 0
        while len(results) < sample_size and attempts < n_attempts:
            attempts += 1
            start = rng.choice(self._triples)
            path = [(start.subject, start.predicate, start.object)]
            current = start.object
            target_hops = rng.randint(1, max_hops)
            for _ in range(target_hops - 1):
                successors = self.successors(current)
                # Filter back-edges
                successors = [
                    (p, o)
                    for (p, o) in successors
                    if not any(o == step[0] or o == step[2] for step in path)
                ]
                if not successors:
                    break
                pred, obj = rng.choice(successors)
                path.append((current, pred, obj))
                current = obj
            results.append(path)
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "triples": [t.to_dict() for t in self._triples],
            "meta": {
                "entity_count": len(self.entities()),
                "relation_count": len(self.relations()),
                "triple_count": len(self._triples),
                "built_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            },
        }

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> KnowledgeGraph:
        kg = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for t in data.get("triples", []):
            kg.add_triple(
                t["s"],
                t["p"],
                t["o"],
                source_uri=t.get("source_uri", ""),
                source_chunk_hash=t.get("source_chunk_hash", ""),
                confidence=float(t.get("confidence", 1.0)),
            )
        return kg

    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(entities={len(self.entities())}, "
            f"relations={len(self.relations())}, "
            f"triples={len(self._triples)})"
        )
