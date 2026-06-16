# Quickstart

`CuratorConfig` holds all configuration; `Curator.run()` executes the pipeline.
Three patterns cover most uses.

## Clean and deduplicate an existing dataset

No LLM or API key required. Reads any supported format, deduplicates, cleans, exports.

```python
from curatorkit import Curator, CuratorConfig

result = Curator(CuratorConfig(
    dataset    = {"name": "tatsu-lab/alpaca",   # HF Hub name, local file, or list
                  "max_samples": 2000},         # cap how much is read
    dedup      = "minhash",              # "exact" | "minhash" | "none"
    clean      = True,
    export_formats = ["alpaca", "sharegpt"],
    output_dir = "output/clean",
)).run()

result.print_summary()
```

HuggingFace Hub sources need the `connectors` extra (included in `all`); local
JSONL, JSON, and CSV files run on the core install. Drop `max_samples` to process
the full dataset.

## Generate synthetic data from a document

Reads a PDF, generates QA pairs, and verifies each answer is grounded in the source
chunk it was generated from. Requires the `generation` and `pdf` extras, plus an API
key in the provider's standard environment variable (`OPENAI_API_KEY` here); any
LiteLLM backend or local Ollama/vLLM server works.

```python
result = Curator(CuratorConfig(
    dataset                 = "docs/handbook.pdf",
    llm_model               = "openai/gpt-4o-mini",
    generation_task         = "qa",
    num_questions           = 3,
    hallucination_threshold = 0.7,     # drop answers that aren't grounded
    export_formats          = ["alpaca"],
    output_dir              = "output/qa",
)).run()
```

## Generate, filter, and recover failures

Adds a reward quality gate plus the two recovery mechanisms: the diagnostic probe
classifies each rejection into a failure mode and retries the fixable ones during
the run, and the reward refiner re-scores borderline rejects afterwards. The
[adaptive recovery guide](../guides/adaptive-recovery.md) explains both.

```python
result = Curator(CuratorConfig(
    dataset                 = "docs/handbook.pdf",
    llm_model               = "openai/gpt-4o-mini",
    judge_llm_model         = "openai/gpt-4o",    # separate judge avoids self-leniency
    generation_task         = "qa",
    hallucination_threshold = 0.7,
    reward_threshold        = 0.7,
    enable_diagnostic_probe = True,
    enable_reward_refiner   = True,
    export_formats          = ["alpaca", "sharegpt"],
    output_dir              = "output/qa_full",
)).run()

print(result.diagnostics.to_dict())
```

## From the command line

The same pipelines run declaratively from YAML:

```bash
curatorkit run pipeline.yaml --output-dir output/
```

The repository ships a runnable example that needs no API key:
[`examples/quickstart/`](https://github.com/Lexsi-Labs/CuratorKIT/tree/main/examples/quickstart).
See the [CLI reference](../reference/cli.md) for the YAML schema.

Next: [Reading the output](outputs.md)
