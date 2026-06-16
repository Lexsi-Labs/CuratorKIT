# Roadmap

CuratorKIT 1.0 is the first public release. The pipeline architecture (readers,
normalizers, generators, gates, and exporters behind stable base classes) is designed
so each area below extends without breaking existing configs.

Directions under active consideration, in no particular order:

**Sources and ingestion**

- Additional connectors: web pages, object storage, SQL databases
- Incremental ingestion for growing corpora

**Generation and gates**

- Additional generation tasks and multi-document grounding
- Configurable gate ensembles and per-domain judge rubrics
- Cost tracking and budget caps per run

**Outputs and integrations**

- Additional export formats as trainer ecosystems evolve
- Direct dataset publishing to the HuggingFace Hub with generated dataset cards
- Deeper [AlignTune](https://github.com/Lexsi-Labs/aligntune) integration beyond the
  documented [curate-then-train workflow](../guides/train-with-aligntune.md), such as
  one-call handoff without a Hub round-trip

Priorities are driven by usage. Propose or upvote items in
[GitHub Discussions](https://github.com/Lexsi-Labs/CuratorKIT/discussions); concrete
proposals are welcome as [feature requests](https://github.com/Lexsi-Labs/CuratorKIT/issues).
No dates are committed here; the [changelog](changelog.md) records what ships.
