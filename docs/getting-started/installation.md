# Installation

CuratorKIT requires Python 3.11 or newer and runs on Linux, macOS, and Windows.
The `hygiene` and `pdf` extras pull large model stacks (torch, MinerU) with their
own platform notes; the core package and connectors are pure Python.

```bash
pip install "curatorkit[all]"
```

The `all` extra installs connectors, LLM generation, embedding, and the data hygiene
gates. The core package alone covers cleaning and deduplication:

```bash
pip install curatorkit
```

## Selecting extras

Install only what you need. Extras compose: `pip install "curatorkit[generation,hf]"`.

| Extra | Adds | Install when you need |
|---|---|---|
| `hf` | datasets, huggingface_hub | HuggingFace Hub datasets |
| `parquet` | pyarrow | Parquet files |
| `connectors` | hf + parquet | All file/Hub readers in one extra |
| `tiktoken` | tiktoken | Exact LLM token counts in the schema gate |
| `generation` | litellm, tenacity, nest-asyncio | Synthetic data generation with any LLM API |
| `embedding` | sentence-transformers, numpy | Diversity gate, cross-run dedup |
| `embedding-faiss` | embedding + faiss-cpu | Fast ANN for large dedup indexes |
| `generation-full` | generation + embedding-faiss | Generation with all gates |
| `hygiene` | detect-secrets, presidio, detoxify, spacy, faker | Secrets, PII, and toxicity gates (spaCy PII backbone) |
| `hygiene-transformers` | presidio-analyzer[transformers] (torch, transformers) | Transformer-based NER backbone for `PIIPseudonymizer` — best recall for clinical/legal PII. Stack on top of `hygiene`. |
| `hygiene-stanza` | presidio-analyzer[stanza] | Stanza NER backbone for `PIIPseudonymizer` — use for languages spaCy has no good model for. Stack on top of `hygiene`. |
| `pdf` | mineru | Layout-aware PDF parsing |
| `all` | connectors + tiktoken + generation-full + hygiene | The full pipeline (excludes `pdf`, `trl`, and both `hygiene-*` NER backbones) |
| `docs`, `dev`, `trl` | site/tooling/integration-test deps | Contributing |

The `pdf` extra is excluded from `all` because it pulls a large model stack. It runs
on CPU anywhere; for CUDA acceleration install a CUDA build of torch first. MinerU is
licensed AGPL-3.0, so confirm that suits your use before installing.

`hygiene-transformers` and `hygiene-stanza` are likewise excluded from `all` — they
pull in the torch/transformers or stanza stack on top of `hygiene`'s spaCy default, and
most users only need one of the three PII backbones. Install them explicitly alongside
`hygiene`:

```bash
pip install "curatorkit[all,hygiene-transformers]"   # clinical/legal PII, best recall
pip install "curatorkit[all,hygiene-stanza]"          # non-English PII, no good spaCy model
```

See [Data hygiene](../guides/data-hygiene.md#pii-ner-backbones) for which model to pick
per backbone and domain.

## License

CuratorKIT is released under the **Lexsi Labs Source Available License (LSAL) v1.1** —
free for research, education, and non-commercial use. Commercial use requires a separate
license. See [LICENSE](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/LICENSE.md)
or contact support@lexsi.ai.

## From source

```bash
pip install "curatorkit[all] @ git+https://github.com/Lexsi-Labs/CuratorKIT.git"
```

## Verify the install

```bash
curatorkit --version
```

Next: [Quickstart](quickstart.md)
