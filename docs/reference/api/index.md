# API reference

Generated from the source docstrings. The classes most users touch:

| Import | Purpose |
|---|---|
| `from curatorkit import Curator, CuratorConfig` | Configure and run a pipeline |
| `from curatorkit import DataSample, RejectedSample` | The sample data models |
| `curatorkit.connectors` | Readers for every supported source |
| `curatorkit.generators` | LLM generation tasks |
| `curatorkit.gates` | Quality gates |
| `curatorkit.hygiene` | Secrets, PII, and toxicity stages |
| `curatorkit.exporters` | Output format writers |

Optional-dependency classes (generation, embedding, hygiene) import lazily; the
[installation guide](../../getting-started/installation.md) maps each to its extra.
