"""
Coverage-aware generation.

Components:
  metrics.py    — diversity primitives (n-gram, type-token ratio, cluster
                  entropy). Pure-python; no embedding dependency.
  analyser.py   — CoverageAnalyser: embed → cluster → identify thin regions →
                  recommend categories for the next generation batch.
  monitor.py    — DiversityMonitor: BaseNormalizer that runs after each
                  generation batch, attaches metrics to provenance, raises
                  a collapse warning if diversity drops sharply between
                  successive runs.

Public API:
  from curatorkit.coverage import (
      diversity_metrics,
      CoverageAnalyser,
      DiversityMonitor,
  )
"""

from curatorkit.coverage.metrics import (
    distinct_n,
    diversity_metrics,
    pairwise_cosine_stats,
    shannon_entropy,
    type_token_ratio,
)

__all__ = [
    "diversity_metrics",
    "distinct_n",
    "type_token_ratio",
    "shannon_entropy",
    "pairwise_cosine_stats",
]


def __getattr__(name: str):
    if name == "CoverageAnalyser":
        from curatorkit.coverage.analyser import CoverageAnalyser

        return CoverageAnalyser
    if name == "DiversityMonitor":
        from curatorkit.coverage.monitor import DiversityMonitor

        return DiversityMonitor
    if name == "CoverageDirectedGenerator":
        from curatorkit.coverage.directed import CoverageDirectedGenerator

        return CoverageDirectedGenerator
    raise AttributeError(f"module 'curatorkit.coverage' has no attribute {name!r}")
