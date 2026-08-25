# Changelog

All notable changes to CuratorKIT are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added
- YAML/CLI pipeline config now mirrors `CuratorConfig`'s per-role LLM override shape
  (`generator_llm:`/`judge_llm:` buckets, nested `<role>_llm:` blocks), plus `grpo_scoring_llm`
  and `enable_reward_refiner`/`refiner_llm` YAML support. Legacy fields still work unchanged.
- `LLMOverride` now accepts `temperature`, `max_tokens`, `timeout`, `max_retries`, `extra_body`,
  `drop_params`, and `concurrency` per role (previously only `model`/`api_base`/`api_key`).
- SFT exporters (Alpaca, ShareGPT) warn on rows with empty instruction/output.
- `LiteLLMBackend` raises a clear `ImportError` when the `generation` extra is missing.

### Fixed
- Generation tasks discarded valid samples when a model nested a string field in an object
  (e.g. `{"chosen": {"poem": "..."}}`) — added a shared `coerce_text()` helper to unwrap these
  instead of failing downstream.
- `PreferenceGenerationTask.run_async()` ignored `preference_mode: "two_pass"`, always falling
  back to the single-call path under async invocation — it now dispatches on `mode` like `run()`.
- `max_retries` on any `*_max_retries` param was used as the total attempt count instead of
  "retries after the first attempt" — `max_retries=0` skipped the call entirely and raised a
  misleading "failed" error. Total calls is now `max_retries + 1` (0 = one attempt, no retries).
- `reward_prompt_template`/`hallucination_prompt_template` custom overrides that didn't return the
  expected `overall_score`/`grounding_score` key silently scored every sample 0.0. Now falls back
  to averaging dimension scores or extracting a number from the raw response, warns immediately if
  a custom template omits the key, and warns again after a run if the fallback was actually used.
- `enable_reward_refiner` recovery ran *after* exporters had already written files, so recovered
  samples showed up in `CuratorResult.passed` but not in the exported `sft_alpaca.jsonl`/etc. (e.g.
  13 initial passes + 2 recovered = 15 in memory, but 13 rows on disk). `Curator.run_async()` also
  never invoked the refiner at all, so `enable_reward_refiner` had no effect when called directly
  instead of through `run()`. Exporting is now deferred until after refiner recovery (and, when
  configured, `output_split`) completes in both `run()` and `run_async()`.

## 1.0.0 - 2026-06-12

First public release.

### Added
- Data hygiene gates: `SecretsGate` (credential/API-key detection), `ToxicityGate`
  (local classifier with optional LLM-judge escalation), and `PIIPseudonymizer`
  (Presidio-based entity replacement), available in Python, YAML, and CLI channels.
- Adversarial generation tasks: `adversarial_qa` and `adversarial_preference`.
- Layout-aware PDF ingestion via the MinerU 3.x SDK (`pdf` extra).
- Tutorial notebooks covering generation, ingestion, cleaning, recovery, adversarial
  data, and hygiene, each runnable in Colab.
- Streaming ingestion support for large HuggingFace datasets.
- Documentation site with guides, config reference, architecture notes, tutorials, and FAQ.
- CI (lint, test matrix on Python 3.11-3.13, wheel-build validation, quickstart e2e,
  docs build), docs deployment, and PyPI publishing workflows.
- `py.typed` marker: the package ships its type annotations (PEP 561).

### Fixed
- Async event-loop handling in notebooks/Jupyter; missing exporter imports in split exports.

## 0.2.0 - 2026-04

### Added
- LLM generation tasks: QA, preference pairs, GRPO rollouts, multi-turn, Evol-Instruct,
  and chain-of-thought, via LiteLLM-compatible APIs and local Ollama.
- Quality gates: provenance-grounded hallucination gate, multi-dimension reward gate,
  and embedding-based diversity gate.
- Adaptive recovery: inline diagnostic probe, failure-mode taxonomy, and reward refiner.
- Trainer-ready exporters: Alpaca, ShareGPT, DPO, GRPO, PPO with train/val/test splits.
- Declarative YAML pipelines and the `curatorkit` CLI.
- Provenance manifest, dataset card, rejection log, and checksums on every run.

## 0.1.0 - 2026-03

### Added
- Core ingestion connectors: JSONL, JSON, CSV, Parquet, HuggingFace datasets, PDF.
- Cleaning and deduplication: text cleaner, exact and MinHash dedup, stratified sampling.
- Schema gate and the `DataSample` / provenance record data model.
