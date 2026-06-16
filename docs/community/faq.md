# FAQ

## Which extra do I need?

| You want to… | Install |
|---|---|
| Clean and deduplicate local files | `pip install curatorkit` (core) |
| Read HuggingFace Hub datasets | `curatorkit[hf]` |
| Read Parquet files | `curatorkit[parquet]` |
| All file/Hub readers | `curatorkit[connectors]` |
| Exact LLM token counts | `curatorkit[tiktoken]` |
| Parse PDFs (layout-aware, MinerU) | `curatorkit[pdf]` |
| Generate synthetic data with an LLM | `curatorkit[generation]` |
| Diversity gate / cross-run dedup (embeddings) | `curatorkit[embedding]` or `[embedding-faiss]` |
| Generation + embeddings + FAISS together | `curatorkit[generation-full]` |
| Secrets / PII / toxicity gates | `curatorkit[hygiene]` |
| Everything except PDF and TRL | `curatorkit[all]` |

The [installation page](../getting-started/installation.md) lists what each extra pulls in.

Quote the extras in zsh: `pip install "curatorkit[all]"`.

## Where do API keys go?

Environment variables are the recommended place. CuratorKIT calls LLMs through
LiteLLM, which reads the standard variable for each provider (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, …). `llm_api_key` in the config overrides the variable when
you need it. Keys are never written into manifests, dataset cards, or
logs. See the [generation guide](../guides/generation.md) for backend configuration.

## Can I use a local model instead of an API?

Yes, in two ways:

- **Ollama**: set the model to your local Ollama model and use the Ollama backend.
- **vLLM / any OpenAI-compatible server**: point the LiteLLM backend at your server's
  base URL with an `openai/...` model string.

The [generation guide](../guides/generation.md) and the tutorial notebooks show both setups.

## Why is my PDF parse slow?

The `pdf` extra runs MinerU layout detection and OCR. On CPU this can take minutes
per large document; with a CUDA GPU it is roughly 10-20× faster. Install a CUDA
build of torch before the extra if you have a GPU. Model weights download
automatically on first parse. `curatorkit setup-pdf --check` verifies the install.

## What gets written to the output directory?

Every run writes `manifest.json` (config hash, per-stage counts, rejection
breakdown), `rejected.jsonl` (each rejected sample with a structured reason),
`dataset_card.md`, and `checksums.txt`, plus one file per requested export format
(`sft_alpaca.jsonl`, `dpo.jsonl`, …). Exporters **overwrite** files from previous
runs, so use a fresh `output_dir` to keep prior outputs. See
[exporters](../guides/exporters.md).

## How do I disable a gate?

Gates are controlled by their config fields: `schema_gate=False` disables schema
checking, and the hallucination/reward/diversity gates run only when their
thresholds are set. In YAML pipelines, simply omit the gate from the `gates:` list.
See [quality filtering](../guides/quality-gates.md).

## Why were all my samples rejected?

Check `rejected.jsonl`: every rejection carries a structured reason string, and
`manifest.json` has the per-stage breakdown. Common causes: thresholds set too high
(try 0.6-0.7 to start), source chunks too thin for grounded QA, or a judge model
that is too weak. The [adaptive recovery guide](../guides/adaptive-recovery.md) covers
diagnosing and recovering rejects.

## Which Python versions are supported?

Python 3.11, 3.12, and 3.13. CI tests all three.

## Does CuratorKIT work with my trainer?

The exports are standard formats: Alpaca and ShareGPT for SFT, DPO pairs, GRPO
rollouts, and PPO prompts, directly loadable by TRL and
[AlignTune](https://github.com/Lexsi-Labs/aligntune), and by anything else that
reads JSONL. See the [exporter compatibility matrix](../guides/exporters.md).
