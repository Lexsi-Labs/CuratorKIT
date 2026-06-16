# Guides

Each guide covers one stage of the pipeline in depth. All examples are runnable as
written; configuration shown in Python applies equally to YAML pipelines.

<div class="grid cards" markdown>

-   :material-database-import:{ .lg .middle } **[Data sources](data-sources.md)**

    ---

    Readers for JSONL, JSON, CSV, Parquet, HuggingFace Hub, and PDF. Field mapping,
    format detection, preprocessing functions, multi-source runs.

-   :material-robot-outline:{ .lg .middle } **[Generation](generation.md)**

    ---

    The eight generation tasks (QA, preference, GRPO, multi-turn, Evol-Instruct,
    chain-of-thought, and the adversarial variants) and their prompt structures.

-   :material-filter-check:{ .lg .middle } **[Quality gates](quality-gates.md)**

    ---

    The schema, hallucination, reward, and diversity gates: what each checks,
    rejection reasons, and threshold tuning.

-   :material-backup-restore:{ .lg .middle } **[Adaptive recovery](adaptive-recovery.md)**

    ---

    How rejected samples are diagnosed and repaired: the inline diagnostic probe,
    the failure-mode taxonomy, and the reward refiner.

-   :material-shield-lock-outline:{ .lg .middle } **[Data hygiene](data-hygiene.md)**

    ---

    Secrets detection, PII pseudonymisation, and toxicity filtering as pipeline
    stages, in Python, YAML, and the CLI.

-   :material-export:{ .lg .middle } **[Exporters](exporters.md)**

    ---

    Alpaca, ShareGPT, DPO, GRPO, PPO, and corpus formats; trainer compatibility;
    train/val/test splits.

-   :material-school-outline:{ .lg .middle } **[Train with AlignTune](train-with-aligntune.md)**

    ---

    The curate-then-train workflow: export with CuratorKIT, publish to the Hub,
    fine-tune with AlignTune.

-   :material-tune:{ .lg .middle } **[Customisation](customisation.md)**

    ---

    Custom prompt templates, LLM backends, reward rubrics, preprocessing functions,
    and extension points.

</div>
