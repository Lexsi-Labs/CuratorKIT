---
title: CuratorKIT
hide:
  - navigation
  - toc
---

<div class="ck-hero" markdown>

<img class="ck-hero-mark" src="assets/icon.png" alt="">

<h1 class="ck-hero-title">Post-training data, <em>curated with proof</em>.</h1>

<p class="ck-tagline">
CuratorKIT builds LLM training datasets as a gated pipeline: ingest from any source,
generate with any LLM, verify every sample against its source, recover what fails,
and export trainer-ready formats, with a provenance manifest on every run.
</p>

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Lexsi-Labs/CuratorKIT){ .md-button }

<p class="ck-chips">
<span>v1.0</span>
<span>MIT</span>
<span>Python 3.11+</span>
<span>core runs CPU-only</span>
<span>any LiteLLM backend</span>
</p>

</div>

```mermaid
flowchart LR
    A[Ingest] --> B[Clean + dedup]
    B --> C[Hygiene]
    C --> D[Generate]
    D --> E{Quality gates}
    E -->|pass| F[Export]
    E -->|reject| G[Adaptive recovery]
    G -->|recovered| E
    G -->|unrecoverable| H[rejected.jsonl]
    F --> I[manifest · dataset card · checksums]
```

<div class="grid cards" markdown>

-   :material-source-branch-check:{ .lg .middle } **Grounded hallucination gate**

    ---

    Each generated answer is verified against the exact source chunk it was
    generated from, not against the judge model's general knowledge.

-   :material-backup-restore:{ .lg .middle } **Adaptive recovery**

    ---

    Rejections are diagnosed against a failure-mode taxonomy; the recoverable
    ones are repaired and re-gated instead of discarded.

-   :material-shield-lock-outline:{ .lg .middle } **Data hygiene**

    ---

    Secrets detection, PII pseudonymisation, and toxicity filtering run as
    pipeline stages, before any sample reaches a training file.

-   :material-database-import:{ .lg .middle } **Any source**

    ---

    JSONL, JSON, CSV, Parquet, HuggingFace Hub, and layout-parsed PDFs.
    Multi-source runs support per-source field mapping.

-   :material-robot-outline:{ .lg .middle } **Eight generation tasks**

    ---

    QA, preference pairs, GRPO rollouts, multi-turn, Evol-Instruct,
    chain-of-thought, and adversarial variants, on any LiteLLM backend or
    local Ollama/vLLM.

-   :material-export:{ .lg .middle } **Trainer-ready exports**

    ---

    Alpaca, ShareGPT, DPO, GRPO, and PPO with train/val/test splits, consumed
    directly by TRL and [AlignTune](https://github.com/Lexsi-Labs/aligntune).

</div>

## Sixty seconds

=== "Python"

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

    No LLM or API key needed for cleaning and dedup. Add `llm_model` and
    `generation_task` for gated synthetic generation; the
    [generation guide](guides/generation.md) covers it.

=== "CLI"

    ```bash
    pip install "curatorkit[all]"
    curatorkit run pipeline.yaml --output-dir output/
    ```

    Pipelines are declarative YAML, validated before anything runs. A runnable
    no-API-key example ships in
    [`examples/quickstart/`](https://github.com/Lexsi-Labs/CuratorKIT/tree/main/examples/quickstart);
    the schema is in the [CLI reference](reference/cli.md).

Every run writes `manifest.json`, `rejected.jsonl`, `dataset_card.md`, and
`checksums.txt` alongside the export files.

## Where next

<div class="ck-wide" markdown>

| | |
|---|---|
| **[Getting started](getting-started/index.md)** | Install, the three usage patterns, reading output |
| **[Guides](guides/index.md)** | Each pipeline stage in depth |
| **[Configuration](reference/configuration.md)** | Every `CuratorConfig` parameter |
| **[API reference](reference/api/index.md)** | Generated from the source docstrings |
| **[Tutorials](tutorials/index.md)** | Eight notebooks, each runnable in Colab |
| **[Roadmap](community/roadmap.md)** | Where 1.0 goes from here |

</div>

## Run the tutorials in Colab

<div class="ck-wide" markdown>

| | | |
|---|---|---|
| **01** Generate an SFT dataset from a PDF | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/01_generate_sft_dataset.ipynb) | LLM endpoint |
| **02** Generate DPO preference pairs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/02_generate_dpo_pairs.ipynb) | LLM endpoint |
| **03** Generate GRPO rollouts | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/03_generate_grpo_rollouts.ipynb) | LLM endpoint |
| **04** Ingest multiple sources | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/04_ingest_multiple_sources.ipynb) | no LLM |
| **05** Clean and deduplicate a dataset | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/05_clean_and_dedup.ipynb) | no LLM |
| **06** Adaptive recovery | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/06_adaptive_recovery.ipynb) | LLM endpoint |
| **07** Adversarial generation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/07_adversarial_generation.ipynb) | LLM endpoint |
| **08** Data hygiene pipeline | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/08_data_hygiene_pipeline.ipynb) | no LLM |

</div>

New to the library? Start with **05**, then **04**, then the generation notebooks.
The [tutorials index](tutorials/index.md) has full descriptions.

---

CuratorKIT is built by [Lexsi Labs](https://lexsi.ai) alongside
[AlignTune](https://github.com/Lexsi-Labs/aligntune), which consumes its exports
natively: curate here, train there.

<div class="ck-lexsi-footer" markdown>
<a href="https://www.lexsi.ai">
  <img class="ck-lexsi-on-light" src="assets/lexsi-logo-dark.png" alt="Lexsi Labs" width="240">
  <img class="ck-lexsi-on-dark" src="assets/lexsi-logo-white.png" alt="Lexsi Labs" width="240">
</a>
<p><a href="https://www.lexsi.ai">https://www.lexsi.ai</a></p>
<p>Paris 🇫🇷 · Mumbai 🇮🇳 · London 🇬🇧</p>
</div>
