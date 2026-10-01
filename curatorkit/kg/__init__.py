"""
Knowledge-graph-driven generation.

Reference: Yang et al., "GraphGen: Enhancing Supervised Fine-Tuning for LLMs
with Knowledge-Driven Synthetic Data Generation" (2025) —
  Paper: https://arxiv.org/abs/2505.20416
  Repo:  https://github.com/open-sciencelab/GraphGen

Components:
  store.py      — KnowledgeGraph: in-memory triples + adjacency + persistence
  extractor.py  — KGExtractor: LLM-based entity+relation extraction
  multihop_qa.py — MultiHopQATask: atomic / aggregated / k-hop QA over the graph
  coverage.py   — GraphCoverageAnalyser: under-represented entities/relations

Implemented from scratch — no external graph library imported (the plan
forbids opaque dependencies). Triple set is a plain dict-of-dict-of-set.

Public API:
  from curatorkit.kg import (
      KnowledgeGraph, Triple,
      KGExtractor,
      MultiHopQATask,
      GraphCoverageAnalyser,
  )
"""

from curatorkit.kg.store import KnowledgeGraph, Triple

__all__ = ["KnowledgeGraph", "Triple"]


def __getattr__(name: str):
    if name == "KGExtractor":
        from curatorkit.kg.extractor import KGExtractor

        return KGExtractor
    if name == "MultiHopQATask":
        from curatorkit.kg.multihop_qa import MultiHopQATask

        return MultiHopQATask
    if name == "GraphCoverageAnalyser":
        from curatorkit.kg.coverage import GraphCoverageAnalyser

        return GraphCoverageAnalyser
    raise AttributeError(f"module 'curatorkit.kg' has no attribute {name!r}")
