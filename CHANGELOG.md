# Changelog

All notable changes to CuratorKIT are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added
- `ignore_for_dedup`: regex patterns whose matches are masked out (replaced with a space, then
  whitespace-collapsed) from a sample's text before deduplication compares or embeds it — lets a
  real but unimportant difference (a per-sample UUID, a timestamp, a boilerplate disclaimer) not
  itself prevent two otherwise-identical samples from being recognized as duplicates. Masking is
  applied per constituent field, before concatenation, so a pattern can't bleed across a field
  boundary (e.g. match trailing `instruction` chars and leading `chosen` chars as one span); it
  only ever transforms a throwaway comparison string, never the actual `DataSample` fields, so
  source grounding for downstream generation tasks and quality gates is unaffected. Available on
  all three dedup methods — `ExactDeduplicator`, `MinHashDeduplicator`, and
  `EmbeddingDeduplicator` (constructor param), plus `CuratorConfig.dedup_ignore_patterns` and
  YAML `NormalizerConfig.ignore_for_dedup` for the programmatic and CLI paths respectively. For
  `EmbeddingDeduplicator`, masking happens before encoding (not just before comparison), so a
  masked span is invisible to the embedding model too — since that changes what the persisted
  cross-run index actually contains, the index's `ignore_for_dedup` is recorded in a `config.json`
  sidecar and `run()` warns (without failing) if a later run's patterns don't match what the
  on-disk index was built with. `ExactDeduplicator`/`MinHashDeduplicator`'s task-type field
  selection (previously duplicated verbatim between the two classes) is now the single shared
  `_sample_dedup_text()`/`_dedup_fields()` helper in `curatorkit.normalizers.dedup`, and
  `ExactDeduplicator._config_hash()` — previously a hardcoded literal that never reflected actual
  config — now hashes real config including `ignore_for_dedup`.
- YAML/CLI pipeline config now mirrors `CuratorConfig`'s per-role LLM override shape
  (`generator_llm:`/`judge_llm:` buckets, nested `<role>_llm:` blocks), plus `grpo_scoring_llm`
  and `enable_reward_refiner`/`refiner_llm` YAML support. Legacy fields still work unchanged.
- `LLMOverride` now accepts `temperature`, `max_tokens`, `timeout`, `max_retries`, `extra_body`,
  `drop_params`, and `concurrency` per role (previously only `model`/`api_base`/`api_key`).
- `LiteLLMBackend` raises a clear `ImportError` when the `generation` extra is missing.
- `CuratorConfig.export_formats` now defaults to `None` ("auto") instead of a fixed
  `["alpaca", "sharegpt", "dpo"]`: it resolves to whichever export format(s) are actually
  designed for the task_type `generation_task` produces (e.g. `grpo`→`["grpo"]`,
  `preference`→`["dpo"]`), or — with no `generation_task` — to the task_type(s) the readers
  actually emit, resolved once they've run. See `curatorkit.exporters.compatibility`. An
  explicit `export_formats` is always honored as given; combining it with a `generation_task`
  it doesn't suit (e.g. `["dpo"]` with `generation_task="qa"`) now warns instead of silently
  writing a near-empty file.
- Every exporter now enforces `curatorkit.exporters.compatibility.TASK_TYPE_EXPORT_FORMATS` as
  a hard per-sample gate: a sample whose task_type isn't designed for that format is skipped and
  counted in a summary warning (one warning per run, not per row). Previously only `DPOExporter`
  did this — `AlpacaExporter`/`ShareGPTExporter` wrote every sample regardless of task_type
  (silently keeping only the first turn of `conversational` data and dropping the rest, or
  writing an empty `output` for `preference`/`grpo`/`prompt_only`/pretrain data), and
  `GRPOExporter`/`PPOExporter`/`CorpusExporter` accepted *any* task_type with a non-empty
  `instruction`/`output` (e.g. `instruction_following` data silently "passed" for GRPO/PPO/Corpus
  even though none of those are what it's for). `GRPOExporter` still never warns on empty
  `responses`/`rewards` alone within task_type `"grpo"` — that's the documented, intentional case
  of exporting seed prompts before `GRPORolloutTask` has run.

### Fixed
- `ExactDeduplicator`/`MinHashDeduplicator` used the generic instruction+output comparison for
  `conversational` and `unpaired_preference` samples, both of which lost information that
  distinguishes genuinely different samples. A `conversational` sample only stores its first
  user/assistant exchange in `instruction`/`output` — every later turn lives in
  `metadata["turns"]` — so two conversations that diverge after turn 1 were wrongly collapsed into
  "duplicates". An `unpaired_preference` sample's `label` (e.g. the same instruction+output rated
  differently by two annotators) was ignored entirely, so differently-labeled rows for the same
  text also collapsed into one, silently dropping a label. Both dedup classes now use a single
  shared `_sample_dedup_text()` helper (previously this task_type branching was duplicated
  verbatim in each class) that adds dedicated branches: `conversational` folds in every turn from
  `metadata["turns"]` (handling both the ChatML `{"role","content"}` shape readers normalize to
  and the ShareGPT `{"from","value"}` shape `MultiTurnTask`'s own generation output uses), and
  `unpaired_preference` includes `label` in the comparison key.
- `PreferenceGenerationTask`'s `two_pass` mode (both `run()` and `run_async()`) crashed the whole
  curation run — instead of cleanly rejecting the one affected sample — whenever chosen/rejected
  generation failed for a sample (empty completion, or corpus-mode parse/incomplete failure). The
  rejection path built `RejectedSample(**sample.model_dump(), ..., metadata={...})`, but
  `sample.model_dump()` already includes `metadata`, so the explicit `metadata=` kwarg collided
  with it and raised `TypeError: RejectedSample() got multiple values for keyword argument
  'metadata'` — surfacing as a hard curation failure rather than a rejected-sample entry. Fixed by
  excluding `metadata` from the `model_dump()` call (`exclude={"metadata"}`), matching the pattern
  already used elsewhere in the codebase for this exact reason.
- `AdversarialPreferenceTask`'s naive (non-adversarial) rejected-answer path just re-asked the
  faithful QA prompt at `temperature=1.1` with no instruction to actually make the answer worse —
  relying entirely on sampling noise to produce something worse, which isn't guaranteed. It now
  uses an explicit degradation-instruction prompt (vague/incomplete/shallow, but still plausible)
  at `temperature=0.9`. Separately, this task had its own instance of the `run_async()` gap above:
  its `_parse_response()` — reused unchanged by the inherited generic async path — made *blocking*
  `self.llm.generate()` calls for every rejected-answer generation (both the adversarial and naive
  branches), which froze the entire event loop for that call's duration on every single sample,
  serializing what should have been `concurrency`-bounded concurrent work. Added a proper
  `run_async()` using `await self.llm.agenerate()` throughout, and factored the JSON-parsing and
  DataSample-construction logic (no I/O) into shared `_extract_pairs()`/`_build_pair_sample()`
  helpers so `run()` and `run_async()` can't drift apart again.
- `GRPORolloutTask`, `EvolInstructTask`, `MultiTurnTask`, and `AdversarialQAGenerationTask` each
  override the synchronous `run()` with real per-task logic (multi-rollout generation,
  strategy-cycling expansion, turn-by-turn conversation, adversarial injection planning) but
  never overrode `run_async()` — since `Curator.run()` always executes asynchronously whenever
  any `generation_task` is configured, and `Pipeline.run_async()` dispatches to `run_async()`
  whenever `hasattr(step, "run_async")` is true (which it always was, via inheritance from
  `BaseGenerationTask`), every one of these tasks silently used the wrong, generic
  single-response async path on every real invocation through `Curator` — this bug has been
  present since the library's first public release. For `GRPORolloutTask` this was total and
  loud: its `_parse_response()` is an intentional no-op (`run()` handles multi-response
  generation directly), so **100% of samples were rejected with `generation_parse_failed` and
  zero rollouts were ever produced** — `generation_task="grpo"` was completely non-functional
  through the standard API. For the other three it was silent and partial: `EvolInstructTask`
  silently skipped `num_evolutions` expansion and the `generate_answers` pass; `MultiTurnTask`
  always behaved like `single_call` regardless of `multiturn_mode`/its `turn_by_turn` default;
  `AdversarialQAGenerationTask` silently produced zero adversarial samples regardless of
  `injection_rate`, since its injection plan is only built inside `run()`. All four now have a
  proper `run_async()` mirroring their sync counterpart's logic with `agenerate`/`asyncio`
  concurrency. (`PreferenceGenerationTask` already had this fixed in a previous release —
  see its own `run_async()` — but the fix wasn't recognized as one instance of a general
  pattern at the time, so it wasn't applied to these four sibling classes.)
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
- `EvolInstructTask` required a custom `evol_prompt_template` to contain `{context}` (it's used in
  the with-context branch), but the no-context branch called `.format(instruction=, strategy=)`
  without passing `context=` at all — any template that passed validation still raised
  `KeyError('context')` the moment a sample had no source text (e.g. plain instruction-only data).
  Fixed by always passing `context=""` in that branch.
- Added the missing upfront placeholder validation (construction-time `ValueError` instead of a
  silently-broken prompt at generation time) to `QAGenerationTask.prompt_template`/
  `table_prompt_template`, `AdversarialPreferenceTask.faithful_prompt_template`/
  `adversarial_prompt_template`, `HallucinationGate.prompt_template`, `RewardGate.prompt_template`,
  `RewardRefiner.refine_prompt_template`/`instruction_refine_template`, and
  `DiagnosticProbe.extra_templates` — previously only `cot_prompt_template`, `evol_prompt_template`,
  `preference_prompt_template`, and `grpo_prompt_template`/`scoring_prompt` validated their
  placeholders; the rest silently dropped the real data from the prompt on a typo instead of
  raising. Extracted the shared check into `curatorkit.utils.prompt_validation` so gates and the
  recovery modules (not `BaseGenerationTask` subclasses) can use it too.
- `multiturn_prompt_template` had no effect at all: `MultiTurnTask` only reads it in
  `mode="single_call"`, but `Curator` never passed a `mode`, so it always used the constructor's
  `"turn_by_turn"` default. Added `CuratorConfig.multiturn_mode` (mirroring the existing
  `preference_mode`/`cot_mode` pattern) so `multiturn_prompt_template` can actually take effect.
- `CuratorConfig.apply_patch({"prompt_template": ...})` (used by `AdaptivePass2Runner` for offline
  patch-sweep analysis) set `llm_prompt_template` on the patched config, but nothing ever read that
  field back — the patch had zero effect on the actual regeneration call, only `llm_temperature`
  did. `_regenerate_sample` now swaps the generator's `prompt_template` for the duration of the
  call too, restoring it afterward.
- `grpo_prompt_template` and `cot_prompt_template` (`cot_mode="generate"`) were silently dropped
  the moment a sample carried source context — even with an instruction present — in favor of a
  different, non-customizable built-in prompt, so a custom template had no effect at all on any
  PDF/corpus-derived pipeline. Both now always use the custom template, with context threaded in
  through a new required `{context_section}` placeholder (empty string when there's no context) —
  the same pattern `preference_prompt_template`/`evol_prompt_template` already used. Added
  `BaseGenerationTask._context_section()` as the shared implementation (previously duplicated only
  in `PreferenceGenerationTask`). `qa_prompt_template`, `preference_prompt_template`,
  `evol_prompt_template`, `multiturn_prompt_template` (`single_call`), and the adversarial
  templates were already always honored regardless of context and needed no change.

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
