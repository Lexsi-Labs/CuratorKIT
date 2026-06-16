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
| `hygiene` | detect-secrets, presidio, detoxify, spacy, faker | Secrets, PII, and toxicity gates |
| `pdf` | mineru | Layout-aware PDF parsing |
| `all` | connectors + tiktoken + generation-full + hygiene | The full pipeline (excludes `pdf` and `trl`) |
| `docs`, `dev`, `trl` | site/tooling/integration-test deps | Contributing |

The `pdf` extra is excluded from `all` because it pulls a large model stack. It runs
on CPU anywhere; for CUDA acceleration install a CUDA build of torch first. MinerU is
licensed AGPL-3.0, so confirm that suits your use before installing.

## From source

```bash
pip install "curatorkit[all] @ git+https://github.com/Lexsi-Labs/CuratorKIT.git"
```

## Verify the install

```bash
curatorkit --version
```

Next: [Quickstart](quickstart.md)
