# Getting started

CuratorKIT builds post-training datasets as a gated pipeline: ingest from any
source, clean and deduplicate, generate synthetic data with an LLM, verify every
generated sample against its source, recover salvageable rejects, and export in the
format your trainer expects.

## Install

```bash
pip install "curatorkit[all]"        # connectors + generation + embedding + hygiene
pip install curatorkit               # core only: cleaning and dedup
```

Requires Python 3.11+. The [installation page](installation.md) maps every extra to
what it unlocks.

## First run: clean a dataset

No LLM or API key needed. Reads any supported format, deduplicates, cleans, exports:

```python
from curatorkit import Curator, CuratorConfig

result = Curator(CuratorConfig(
    dataset        = {"name": "tatsu-lab/alpaca", "max_samples": 2000},
    dedup          = "minhash",
    clean          = True,
    export_formats = ["alpaca", "sharegpt"],
    output_dir     = "output/clean",
)).run()

result.print_summary()
```

HuggingFace Hub sources need the `connectors` extra (included in `all`); local
JSONL, JSON, and CSV files run on the core install.

## Second run: generate gated synthetic data

Set an API key (any LiteLLM provider, or point at local Ollama/vLLM), then:

```python
result = Curator(CuratorConfig(
    dataset                 = "handbook.pdf",      # needs the [pdf] extra
    llm_model               = "openai/gpt-4o-mini",
    generation_task         = "qa",
    hallucination_threshold = 0.7,                 # verify answers against the source
    reward_threshold        = 0.7,                 # LLM-judge quality gate
    export_formats          = ["alpaca"],
    output_dir              = "output/qa",
)).run()
```

The [quickstart](quickstart.md) extends this with adaptive recovery and the CLI.

## What a run produces

Every run writes the export files plus four provenance artifacts:

```
output/
  sft_alpaca.jsonl     exported training data (one file per requested format)
  manifest.json        config hash, per-stage counts, rejection breakdown
  rejected.jsonl       every rejected sample with a structured reason
  dataset_card.md      human-readable run summary
  checksums.txt        SHA-256 for all output files
```

`result.passed`, `result.rejected`, and `result.stage_counts` expose the same
information in code. [Reading the output](outputs.md) walks through each file.

## Go deeper

| | |
|---|---|
| Each pipeline stage in depth | [Guides](../guides/index.md) |
| Every `CuratorConfig` parameter | [Configuration reference](../reference/configuration.md) |
| YAML pipelines and CLI flags | [CLI reference](../reference/cli.md) |
| Runnable notebooks | [Tutorials](../tutorials/index.md) |
| Classes and functions | [API reference](../reference/api/index.md) |
