# Changelog

All notable changes to CuratorKIT are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added
- YAML/CLI pipeline config (`PipelineConfig`) now mirrors `CuratorConfig`'s per-role LLM override
  shape: new `generator_llm:`/`judge_llm:` mid-tier buckets, and a nested `<role>_llm:` block
  (model/temperature/max_tokens/timeout/max_retries/extra_body/drop_params/concurrency) on
  `hallucination`/`reward`/`toxicity` gates, generators, and the diagnostic probe — previously YAML
  only had a single global `llm:` block plus a bare model-string override per role. Legacy bare
  `*_llm_model` fields still work unchanged. Also closes two YAML-only feature gaps: `grpo_scoring_llm`
  (GRPO rollouts previously reused the same LLM for scoring as for generation) and
  `enable_reward_refiner`/`refiner_llm` (RewardRefiner had no YAML/CLI wiring at all). Quality gates
  built via YAML also now receive a resolved `concurrency` (previously `HallucinationGate`/`RewardGate`
  silently used their class defaults regardless of config).
- `LLMOverride` (the per-role override dict passed to `generator_llm`, `judge_llm`,
  `hallucination_llm`, `reward_llm`, `toxicity_llm`, `grpo_scoring_llm`, `probe_llm`,
  `refiner_llm`) now accepts `temperature`, `max_tokens`, `timeout`, `max_retries`,
  `extra_body`, `drop_params`, and `concurrency` — previously only `model`/`api_base`/
  `api_key` were configurable per role, and the six non-generator/judge roles silently
  ignored any sampling-param override because their call sites hardcoded literal
  `temperature`/`max_tokens` values. All 8 roles now resolve through a consistent 3-tier
  cascade (role override → mid-tier `judge_llm`/`generator_llm` → role default or global
  bucket); default behavior is unchanged for anyone not setting a new field. `probe_llm`
  and `refiner_llm` also gain a `concurrency` override (previously hardcoded to `32` with
  no config path at all), and their forced `enable_thinking=False` extra_body injection now
  respects an explicit user override instead of always clobbering it. `toxicity_llm.concurrency`
  is accepted but has no effect yet — `ToxicityGate`'s LLM-judge path runs sequentially with
  no executor.
- SFT exporters (Alpaca, ShareGPT) warn when exported rows have empty
  instruction/output, catching task-type/format mismatches that previously
  produced silent empty datasets.
- `LiteLLMBackend` raises a helpful `ImportError` at construction when the
  `generation` extra is missing, instead of rejecting every sample mid-run.

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
