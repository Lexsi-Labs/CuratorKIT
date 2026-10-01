"""
GraphCoverageAnalyser — knowledge-gap detection over a KnowledgeGraph.

Reference: GraphGen (Yang et al., 2025) §3 — "identifies knowledge gaps in
LLMs using the expected calibration error metric, prioritizing the generation
of QA pairs that target high-value, long-tail knowledge."

The paper's ECE-based gap detection requires access to the trainee model's
output probabilities, which CuratorKIT does not have in-pipeline. We use a
graph-structural analogue that captures the same intent — under-covered
entities and predicates that the next generation batch should target. The
two views are complementary; ECE catches model-side gaps, structural
coverage catches data-side gaps.

Surface:
  GraphCoverageAnalyser.analyse(kg) -> CoverageReport
    .long_tail_entities  — entities below a degree percentile
    .rare_predicates     — predicates with count < threshold
    .disconnected_components — small isolated subgraphs (probable noise OR
                              rare specialised topics worth targeting)
    .density_per_entity  — entity-level statistics
    .recommendations     — list of "generate more QA about <entity>" hints
                            ready to feed back into MultiHopQATask.

The output is also written into the dataset manifest's diversity_stats
slot so the human-readable card surfaces it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from curatorkit.kg.store import KnowledgeGraph

# ───────────────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class GraphCoverageReport:
    """One coverage diagnostic over a KnowledgeGraph."""

    n_entities: int = 0
    n_predicates: int = 0
    n_triples: int = 0
    avg_entity_degree: float = 0.0
    degree_percentiles: dict[str, int] = field(default_factory=dict)
    long_tail_entities: list[str] = field(default_factory=list)
    rare_predicates: dict[str, int] = field(default_factory=dict)
    disconnected_components: list[list[str]] = field(default_factory=list)
    density_per_entity: dict[str, int] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_entities": self.n_entities,
            "n_predicates": self.n_predicates,
            "n_triples": self.n_triples,
            "avg_entity_degree": round(self.avg_entity_degree, 3),
            "degree_percentiles": self.degree_percentiles,
            "long_tail_entities": self.long_tail_entities[:50],  # cap for sanity
            "rare_predicates": self.rare_predicates,
            "n_disconnected_components": len(self.disconnected_components),
            "disconnected_component_sizes": [len(c) for c in self.disconnected_components],
            "recommendations": self.recommendations,
        }


# ───────────────────────────────────────────────────────────────────────────
# Analyser
# ───────────────────────────────────────────────────────────────────────────


class GraphCoverageAnalyser:
    """
    Compute structural coverage statistics + generation recommendations.

    Parameters
    ----------
    long_tail_percentile : float
        Entities below this percentile of degree are flagged "long-tail".
        Default 0.20 (lowest-degree 20%).
    rare_predicate_min_count : int
        Predicates with strictly fewer than this many triples are flagged
        "rare". Default 2.
    min_component_size : int
        Disconnected components with ≤ this many entities are reported in
        the warning list. Default 3 — flags small orphan subgraphs while
        ignoring the main connected component of a healthy KG.
    max_recommendations : int
        Cap the recommendations list at this many entries so the manifest
        stays readable. Default 20.
    """

    def __init__(
        self,
        long_tail_percentile: float = 0.20,
        rare_predicate_min_count: int = 2,
        min_component_size: int = 3,
        max_recommendations: int = 20,
    ) -> None:
        self.long_tail_percentile = float(long_tail_percentile)
        self.rare_predicate_min_count = int(rare_predicate_min_count)
        self.min_component_size = int(min_component_size)
        self.max_recommendations = int(max_recommendations)

    # ─────────────────────────────────────────────────────────────────────

    def analyse(self, kg: KnowledgeGraph) -> GraphCoverageReport:
        if len(kg) == 0:
            return GraphCoverageReport()

        # Entity degree (undirected: incoming + outgoing)
        density = self._entity_degree(kg)
        degrees = list(density.values())
        avg_deg = sum(degrees) / len(degrees) if degrees else 0.0

        # Percentile cuts (manual implementation — no numpy dep needed)
        ordered = sorted(degrees)

        def pct(p: float) -> int:
            if not ordered:
                return 0
            idx = max(0, min(len(ordered) - 1, int(p * (len(ordered) - 1))))
            return int(ordered[idx])

        percentiles = {
            "p10": pct(0.10),
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "p99": pct(0.99),
        }
        cutoff = pct(self.long_tail_percentile)
        long_tail = sorted(
            [e for e, d in density.items() if d <= cutoff],
            key=lambda e: density[e],
        )

        # Predicate rarity
        pred_counts = kg.predicate_counts()
        rare_preds = {p: c for p, c in pred_counts.items() if c < self.rare_predicate_min_count}

        # Connected components (undirected view)
        components = self._connected_components(kg)
        small_components = [c for c in components if len(c) <= self.min_component_size]

        # Build recommendations
        recs = self._recommendations(
            long_tail,
            rare_preds,
            small_components,
            density,
            kg,
        )

        return GraphCoverageReport(
            n_entities=len(density),
            n_predicates=len(pred_counts),
            n_triples=len(kg),
            avg_entity_degree=avg_deg,
            degree_percentiles=percentiles,
            long_tail_entities=long_tail,
            rare_predicates=rare_preds,
            disconnected_components=small_components,
            density_per_entity=density,
            recommendations=recs,
        )

    # ─────────────────────────────────────────────────────────────────────

    def _entity_degree(self, kg: KnowledgeGraph) -> dict[str, int]:
        """Undirected degree per entity."""
        deg: dict[str, int] = defaultdict(int)
        for t in kg.triples():
            deg[t.subject] += 1
            deg[t.object] += 1
        return dict(deg)

    def _connected_components(self, kg: KnowledgeGraph) -> list[list[str]]:
        """
        Undirected connected components (BFS).

        Returns components sorted by size (descending) so the analyser can
        report "small / isolated" tails quickly.
        """
        # Build undirected adjacency
        adj: dict[str, set[str]] = defaultdict(set)
        for t in kg.triples():
            adj[t.subject].add(t.object)
            adj[t.object].add(t.subject)

        visited: set[str] = set()
        components: list[list[str]] = []
        for start in adj.keys():
            if start in visited:
                continue
            queue = deque([start])
            comp: list[str] = []
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                queue.extend(adj[node] - visited)
            components.append(comp)

        # Singletons that appear nowhere in `adj` get added if any KG entity
        # is referenced without successors. Already covered: any entity that
        # appears in a triple is in adj.
        components.sort(key=lambda c: -len(c))
        return components

    def _recommendations(
        self,
        long_tail: list[str],
        rare_preds: dict[str, int],
        small_components: list[list[str]],
        density: dict[str, int],
        kg: KnowledgeGraph,
    ) -> list[str]:
        recs: list[str] = []

        if long_tail:
            tail_top = long_tail[: min(5, len(long_tail))]
            recs.append(
                f"{len(long_tail)} long-tail entities (degree ≤ "
                f"{density.get(tail_top[-1], 0)}). Examples: {tail_top}. "
                "Target the next batch toward QA mentioning these."
            )

        if rare_preds:
            rare_top = sorted(rare_preds.items(), key=lambda x: x[1])
            top_preds = [p for p, _ in rare_top[:5]]
            recs.append(
                f"{len(rare_preds)} rare predicates "
                f"(< {self.rare_predicate_min_count} triples each): {top_preds}. "
                "Aggregated QA with these predicates will boost their coverage."
            )

        if small_components:
            recs.append(
                f"{len(small_components)} isolated subgraphs (≤ "
                f"{self.min_component_size} entities each). These are "
                f"either rare specialised topics worth targeting OR "
                f"extraction noise — inspect before generating from them."
            )

        # Density inversion — flag if top-1 entity holds > 25% of all degree.
        if density:
            total_degree = sum(density.values())
            top_entity, top_deg = max(density.items(), key=lambda x: x[1])
            if total_degree > 0 and top_deg / total_degree > 0.25:
                recs.append(
                    f"Coverage skew: '{top_entity}' carries "
                    f"{100 * top_deg / total_degree:.1f}% of all entity "
                    "degree. The next batch should under-sample this entity "
                    "and over-sample long-tail."
                )

        return recs[: self.max_recommendations]
