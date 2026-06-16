# CuratorKIT Architecture

## DataSample contract

`DataSample` is the canonical unit of data moving through the pipeline.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `id` | `str` | auto UUID4 | Stable identifier |
| `source_uri` | `str` | required | URI of the source document or file |
| `instruction` | `str` | `""` | Prompt / instruction (SFT, preference prompt) |
| `input` | `str` | `""` | Optional Alpaca-style context; also stores source chunk text for QA-generated samples |
| `output` | `str` | `""` | Target response (SFT); full text (pretrain) |
| `chosen` | `str` | `""` | Preferred completion (DPO preference) |
| `rejected` | `str` | `""` | Dispreferred completion (DPO preference) |
| `label` | `float \| None` | `None` | Quality label (unpaired preference / reward) |
| `responses` | `list[str]` | `[]` | GRPO group rollouts |
| `reward_scores` | `list[float]` | `[]` | Paired with `responses` |
| `task_type` | `str` | `"instruction_following"` | See vocabulary below |
| `metadata` | `dict` | `{}` | Unmapped columns and step-specific data |
| `provenance_chain` | `list[ProvenanceRecord]` | `[]` | **Append-only** |

### task_type vocabulary

| Value | Active fields | Typical use |
|-------|--------------|-------------|
| `instruction_following` | instruction, input, output | Alpaca SFT |
| `conversational` | instruction, output, metadata["turns"] | ShareGPT / ChatML SFT |
| `preference` | instruction, chosen, rejected | DPO with explicit prompt |
| `implicit_preference` | chosen, rejected (instruction = extracted prefix) | DPO implicit prompt |
| `unpaired_preference` | instruction, output, label | Reward model training |
| `grpo` | instruction, responses, reward_scores | GRPO rollouts |
| `prompt_only` | instruction | PPO prompt collection |
| `language_modeling` | output | Continued pre-training; PDF chunk mode |
| `source_chunk` | input | Raw corpus chunks awaiting generation |

---

## RejectedSample contract

`RejectedSample` extends `DataSample` and is emitted by every step that drops a sample. No step silently discards — every rejection produces a `RejectedSample` with a structured reason.

| Field | Type | Notes |
|-------|------|-------|
| `rejection_reason` | `str` | Structured string, e.g. `"hallucination_contract_failed:0.42"` |
| `rejecting_step` | `str` | Class name of the step that rejected the sample |
| `diagnosis` | `FailureDiagnosis \| None` | Populated by `DiagnosticProbe` when `enable_diagnostic_probe=True`; `None` otherwise |

The `diagnosis` field is serialised to `rejected.jsonl` via an overridden `model_dump()` — `FailureDiagnosis` is a plain dataclass, not a Pydantic model, so the override calls `.to_dict()` before serialisation.

---

## Plugin interface guide

All pipeline components implement one of four ABCs in `curatorkit/interfaces.py`:

```
BaseReader     → read() -> tuple[list[DataSample], list[RejectedSample]]
BaseGate       → run(samples) -> tuple[list[DataSample], list[RejectedSample]]
BaseNormalizer → run(samples) -> list[DataSample]
BaseExporter   → export(samples, output_dir) -> None
```

**Note:** `BaseReader.read()` returns a tuple identical to `BaseGate.run()`.
Reader-level parse failures are first-class `RejectedSample` objects.

---

## Connector layer

All connectors inherit from `BaseConnector` in `curatorkit/connectors/base.py`.
Subclasses only implement `_iter_rows()` which yields `(line_no, raw_dict)` tuples.
The base class owns the full pipeline for each row:

```
raw_dict
  → preprocessing_fn   (optional user callable — structural normalisation)
  → field_mapping      (flat key renames; dot notation for nested paths)
  → FormatDetector     (3-layer detection, committed from first N rows)
  → DataSample construction (format-specific, with role normalisation)
  → RejectedSample     (for every failure — no silent drops)
```

### Supported connector types

| Type | Class | File | Optional dep |
|------|-------|------|--------------|
| `jsonl` | `JSONLReader` | `connectors/jsonl.py` | none |
| `json` | `JSONReader` | `connectors/json_reader.py` | none |
| `csv` | `CSVReader` | `connectors/csv_reader.py` | none |
| `parquet` | `ParquetReader` | `connectors/parquet_reader.py` | `pyarrow` |
| `huggingface` | `HuggingFaceReader` | `connectors/huggingface.py` | `datasets` |
| `pdf` | `PDFReader` | `connectors/pdf.py` | `mineru` (install extra: `pdf`) |

### PDFReader

`PDFReader` chunks PDF documents and can operate in two modes controlled by `output_mode`:

| `output_mode` | Behaviour |
|---|---|
| `"chunk"` | Emits one `DataSample` per text chunk (`output` = chunk text) |
| `"qa"` / `"preference"` / `"grpo"` / `"multiturn"` | Internally runs an LLM generation pass and emits generated samples directly |

Chunks use a heading-aware strategy with configurable `chunk_max_tokens` (default 512) and `chunk_overlap_tokens` (default 50). Each chunk sample carries `metadata["chunk_index"]`, `metadata["page"]`, `metadata["parent_heading"]`, and `metadata["source_file"]` — these are forwarded to generated samples so chunk-level provenance survives generation.

---

## Format detection system

`curatorkit/detection/detector.py` — `FormatDetector`

**Layer 1 — column-set candidate generation**

Uses semantic equivalence classes rather than exact string matches:
- `INSTRUCTION_COLS`: instruction, prompt, query, question, input, ...
- `OUTPUT_COLS`: output, response, completion, answer, ...
- `CHOSEN_COLS`: chosen, preferred, accepted, response_a, ...
- `REJECTED_COLS`: rejected, refused, dispreferred, response_b, ...

Generates **all** matching candidates in priority order. Does not stop at first match.

**Layer 2 — value-type validation**

Validates actual cell values against the expected type for each candidate format.
A `sharegpt` candidate requires the conversations column value to be `list[dict]`,
not a flat string. Candidates that fail layer 2 are dropped; the next candidate is tried.

**Layer 3 — role alias normalisation**

Applied during `DataSample` construction after format is committed.
Normalises `from`/`value` → `role`/`content` and expands role aliases:
- `human`, `user`, `input` → `"user"`
- `gpt`, `assistant`, `model`, `output` → `"assistant"`
- `system` → `"system"`

### Detection result confidence

| Level | Meaning |
|-------|---------|
| `HIGH` | Exact or near-exact column-set match, layer 2 passed |
| `MEDIUM` | Partial match or layer 2 passed without sample rows |
| `LOW` | Best guess only — pipeline warns, emits provenance note |
| `UNKNOWN` | No candidate survived — rows become `RejectedSample` |

---

## SchemaGate

`SchemaGate` is task-type-aware.

Default field validation per `task_type`:

| task_type | Fields checked |
|-----------|---------------|
| SFT family | instruction + output non-empty |
| preference / implicit_preference | chosen + rejected non-empty |
| grpo | instruction + responses (non-empty list) |
| language_modeling | output non-empty |
| prompt_only | instruction non-empty; output may be empty |
| source_chunk | input non-empty; instruction/output generated later |

Token counting is also task-type-aware:
- SFT: `instruction + output`
- Preference: `instruction` + the longer of `chosen` / `rejected`
- GRPO: `instruction + longest response`
- Pretrain: `output`
- Source chunk: `input`

---

## LLM abstraction layer

`curatorkit/llm/` provides a provider-agnostic LLM interface.

```
BaseLLM (ABC)
  .generate(messages, **kwargs) -> LLMResponse     # sync
  .agenerate(messages, **kwargs) -> LLMResponse    # async
  .config_hash() -> str                            # stable hash for provenance
  .temperature: float                              # mutable default; callers can override per call

LiteLLMBackend   → routes to OpenAI, Anthropic, Gemini, Mistral, vLLM, etc.
OllamaBackend    → routes to local Ollama server
```

`LLMResponse` carries `.text`, `.model`, `.usage` (token counts), and `.to_provenance_dict()` which serialises model name, token usage, and finish reason into the `ProvenanceRecord.notes` dict.

---

## Generation tasks

`curatorkit/generators/` — all tasks extend `BaseGenerationTask(BaseNormalizer)`.

```
BaseGenerationTask
  ._build_messages(sample) -> list[dict]   # abstract — build the LLM prompt
  ._parse_response(sample, response) -> list[DataSample]  # abstract — parse output
  .run(samples) -> list[DataSample]        # sync, one LLM call per sample
  .run_async(samples) -> list[DataSample]  # async, concurrency-controlled
  .flush_rejected() -> list[RejectedSample]  # return and clear accumulated failures
```

**Parse failure handling:** if `_parse_response()` returns `[]` (parse failed), the source sample is emitted as a `RejectedSample` with `rejection_reason="generation_parse_failed:{task_name}"`. No silent drops.

### Available generation tasks

| Task | Class | Output `task_type` | Notes |
|------|-------|-------------------|-------|
| QA generation | `QAGenerationTask` | `instruction_following` | Stores source chunk in `sample.input`; forwards `chunk_index`, `page`, `parent_heading` to metadata |
| Preference pairs | `PreferenceGenerationTask` | `preference` | `mode="single_call"` (1 LLM call) or `mode="two_pass"` (2 calls, higher quality contrast) |
| GRPO rollouts | `GRPORolloutTask` | `grpo` | Generates N responses per prompt; optional reward scoring via separate LLM call |
| Multi-turn | `MultiTurnTask` | `conversational` | Generates N turn pairs in a simulated dialogue |
| Evol-Instruct | `EvolInstructTask` | `instruction_following` | Rewrites instructions at increasing complexity; optionally generates answers |
| Chain-of-thought | `ChainOfThoughtTask` | `instruction_following` | `mode="generate"` (new CoT) or `mode="wrap"` (wrap existing answer) |
| Adversarial preference | `AdversarialPreferenceTask` | `preference` | Faithful `chosen` + adversarially corrupted `rejected`; corruption controlled by `injection_rate` / `injection_types` |
| Adversarial QA | `AdversarialQAGenerationTask` | `instruction_following` | QA pairs with a controlled fraction of injected failure modes; marks `metadata["injected_failure"]` and `metadata["injection_type"]` |

Each generated sample carries a `ProvenanceRecord` with the LLM's model name, token usage, prompt hash, and source sample ID.

---

## Quality gates

Quality gates sit after the generation step and filter on content rather than structure.

### HallucinationGate

`curatorkit/gates/hallucination.py`

Uses an LLM judge to verify that each generated answer is grounded in its source chunk. The source is read from `sample.input` — the exact chunk text stored by `QAGenerationTask`. This is **provenance-exact**: the judge always sees the same chunk that generated the sample, not a retrieved approximation.

Scoring prompt asks the judge to rate grounding on a 0–1 scale. Samples below `threshold` (default 0.7) become `RejectedSample` with `rejection_reason="hallucination_contract_failed:{score:.2f}"`. Samples with no source context pass through silently by default (`skip_if_no_context=True`); with `skip_if_no_context=False` they are rejected with `rejection_reason="hallucination_gate:no_source_context"`.

The `ProvenanceRecord` includes `grounding_score`, `unsupported_claims`, and the judge verdict — the `DiagnosticProbe` reads these for score-conditioned routing and fallback classification.

**Probe attachment point:** when `enable_diagnostic_probe=True`, `Curator._build_steps()` sets `gate.probe = DiagnosticProbe(...)` on both the `HallucinationGate` and the `RewardGate` instances. The pipeline's gate block checks `getattr(step, "probe", None)` and calls `probe.diagnose_batch(rejected)` on each gate's rejections — see the gate block below.

### RewardGate

`curatorkit/gates/reward.py`

Uses an LLM judge to score sample quality across configurable dimensions (default: helpfulness, honesty, instruction_following). Scores are averaged; samples below `threshold` are rejected. The judge does not receive the source chunk — this is purely instruction-following quality, not faithfulness.

### DiversityGate

`curatorkit/gates/diversity.py`

Embedding-based near-duplicate filter. Uses `sentence-transformers` to encode each sample and reject those with cosine similarity above `similarity_threshold` against already-accepted samples in the current batch.

---

## Cross-run embedding deduplication

`curatorkit/normalizers/embedding_dedup.py` — `EmbeddingDeduplicator`

Extends within-batch deduplication with a **persistent embedding index** that survives across pipeline runs. The index is stored as `embeddings.npy` + `metadata.json` in `embedding_index_dir`.

- **Cross-run dedup:** new samples are checked against the full index before being checked against each other.
- **Within-run dedup:** samples in the current batch that pass the index check are still deduplicated against each other.
- **FAISS-optional:** uses FAISS for ANN if installed (`pip install "curatorkit[embedding-faiss]"`); falls back to NumPy brute-force otherwise.
- The `text_field="auto"` mode picks the embedding target based on `task_type` (e.g. `chosen` for preference, `instruction + output` for SFT).

---

## Pipeline orchestration

`curatorkit/pipeline.py` — `Pipeline`

```
Pipeline(steps, output_dir, diagnostics=None)
  .run()        -> PipelineResult    # sync
  .run_async()  -> PipelineResult    # async
```

The runner dispatches on step type. Generation tasks (`BaseGenerationTask` subclasses) are `BaseNormalizer` instances with an additional `run_async()` and `flush_rejected()`. The pipeline calls `flush_rejected()` after each generation step and merges the failures into `all_rejected`.

### Gate block — with diagnostic probe

Both `run()` and `run_async()` contain identical gate blocks. When a gate has a `probe` attribute attached, the block diagnoses every rejection in one concurrent batch and merges probe-recovered samples back into the stream:

```
passed, rejected = gate.run(samples)

probe = getattr(step, "probe", None)
probe_recovered = []
if probe is not None and rejected:
    diagnoses = probe.diagnose_batch(rejected)        # ← concurrent inline probe
    for r, diag in zip(rejected, diagnoses):
        r.diagnosis = diag
        if self._diagnostics is not None:
            self._diagnostics.record(r)
        if diag.recovered_sample is not None:
            probe_recovered.append(diag.recovered_sample)

all_rejected.extend(rejected)
samples = passed + probe_recovered      # recovered samples flow into the next step
```

Pipelines without a probe are unaffected — `getattr(step, "probe", None)` returns `None` and the block is skipped. Each gate's `stage_counts` entry records the number of probe-recovered samples.

### PipelineResult

```python
@dataclass
class PipelineResult:
    passed:             list[DataSample]
    rejected:           list[RejectedSample]
    stage_counts:       dict[str, dict[str, int]]
    wall_clock_seconds: float
    diagnostics:        PipelineDiagnostics | None   # None when probe is inactive
```

`stage_counts` has one entry per step: `{input_count, output_count, probe_recovered, rejected_count}` for gates; `{output_count, rejected_count}` for readers; `{input_count, output_count}` for normalizers (generation tasks add `rejected_count`); `{exported_count}` for exporters.

---

## CuratorConfig and Curator entry point

`curatorkit/curator.py`

`CuratorConfig` is a flat `@dataclass` (not Pydantic). All pipeline behaviour is driven by its fields — no nested sub-objects.

### Key config field groups

| Group | Fields |
|---|---|
| Source | `dataset`, `split`, `subset`, `streaming`, `max_samples` |
| LLM | `llm_model`, `llm_temperature`, `llm_max_tokens`, `llm_api_key`, `llm_api_base`, `llm_concurrency` |
| Generation | `generation_task`, `num_questions`, `num_responses`, `num_turns`, `num_evolutions`, `difficulty`, `preference_mode`, `cot_mode` |
| Quality gates | `hallucination_threshold`, `reward_threshold`, `reward_dimensions`, `diversity_threshold` |
| Cross-run dedup | `embedding_dedup`, `embedding_index_dir`, `embedding_dedup_threshold`, `embedding_model` |
| PDF | `pdf_output_mode`, `pdf_extract_tables` |
| Adaptive recovery | `enable_diagnostic_probe`, `probe_temperatures`, `probe_score_split`, `probe_generator_model`, `probe_extra_templates`, `enable_reward_refiner` |
| Export | `output_dir`, `export_formats` |

### apply_patch()

`CuratorConfig.apply_patch(patch: dict) -> CuratorConfig` returns a **shallow copy** with only the patched fields changed. The original config is never mutated. Used by `AdaptivePass2Runner` (an offline utility — see the diagnostic module section) to apply config patches when re-running rejected samples:

| Patch key | Config field modified |
|---|---|
| `llm_temperature` | `config.llm_temperature` |
| `prompt_template` | `config.llm_prompt_template` |
| `context_window` | consumed by `AdaptivePass2Runner` directly, not stored |
| `regenerate_field` | consumed by `AdaptivePass2Runner` directly, not stored |

### Curator._build_steps() — step construction order

```
1.  Readers            (one per dataset in config.dataset)
2.  SchemaGate
3.  ExactDeduplicator  (if dedup != "none")
4.  MinHashDeduplicator (if dedup == "minhash")
5.  TextCleaner        (if clean=True)
── Data hygiene (pre-generation) ──────────────────────────────
6.  SecretsGate        (if secrets_gate=True)
7.  PIIPseudonymizer   (if pii_pseudonymize=True)
8.  ToxicityGate       (if toxicity_gate=True)
────────────────────────────────────────────────────────────────
9.  Generation task    (if generation_task + llm_model set)
10. HallucinationGate  (if hallucination_threshold is not None)
11. RewardGate         (if reward_threshold set)
    └── DiagnosticProbe attached to gates 10 and 11 (if enable_diagnostic_probe=True)
12. DiversityGate      (if diversity_threshold set)
13. EmbeddingDeduplicator (if embedding_dedup=True)
14. StratifiedSampler  (if resample=True)
15. MaxSamplesTruncator (if max_samples set) — applied after resampling so the final distribution is preserved
16. Exporters          (one per format in export_formats)
```

`Curator._build_llm()` routes to `OllamaBackend` for `ollama/` prefixed models and `LiteLLMBackend` for everything else.

When `enable_reward_refiner=True`, a `RewardRefiner` is held on the `Curator` instance (it is not a pipeline step) and runs post-pipeline on `RewardGate` rejections.

---

## Diagnostic module (adaptive recovery)

`curatorkit/diagnostic/` — diagnoses *why* a gate rejected a sample and attempts an inline, targeted repair before the sample is declared lost.

### FailureMode taxonomy

`curatorkit/diagnostic/failure_modes.py` defines nine modes:

```
FailureMode (9 modes)
├── HallucinationGate causes
│   ├── SOURCE_AMBIGUOUS       source/response relationship unclear
│   ├── GENERATOR_TEMPERATURE  high temperature causes source drift
│   ├── GENERATOR_PARAMETRIC   model ignores source, uses prior knowledge
│   └── THRESHOLD_MARGINAL     score just below threshold, unstable
├── RewardGate causes
│   ├── INSTRUCTION_QUALITY    generated question is poor quality
│   ├── RESPONSE_QUALITY       generated answer is poor quality
│   └── DOMAIN_MISMATCH        generation prompt wrong for this domain
├── DiversityGate cause
│   └── NEAR_DUPLICATE         too similar to an already-accepted sample
└── Fallback
    └── UNKNOWN                probe inconclusive
```

The same module defines `PROMPT_TEMPLATES` — the named re-generation prompts (`strict_grounding`, `domain_specific`, `generate_question`, `default`) — and the `FailureDiagnosis` dataclass attached to `RejectedSample.diagnosis`:

| Field | Meaning |
|---|---|
| `mode` | Diagnosed `FailureMode` |
| `evidence` | Temperature-sweep pass/fail pattern, e.g. `[True, False]` |
| `probe_calls` | Total LLM calls consumed by the probes |
| `notes` | Extra info (e.g. which prompt variant succeeded) |
| `recovered_sample` | The passing re-generation, or `None` if all probes were exhausted |

Recovery is **inline**: when a probe path produces a re-generation that passes the gate, the passing sample is stored in `recovered_sample` and the pipeline routes it forward immediately. There is no separate second pass.

### DiagnosticProbe

`curatorkit/diagnostic/probe.py`

Score-conditioned probe sequence, run by the pipeline gate block via `diagnose_batch()`. Routing is conditioned on the `grounding_score` the `HallucinationGate` wrote to the sample's provenance chain:

```
grounding_score >= score_split (default 0.5)   — near-boundary path
  Probe 1 — temperature sweep (default [0.3, 0.5])              max 2 calls
    all pass        → THRESHOLD_MARGINAL
    low-T pass only → GENERATOR_TEMPERATURE
    mixed           → THRESHOLD_MARGINAL
    all fail        → continue to prompt variants
  Probe 2a — strict_grounding prompt                            max 1 call
    passes gate     → GENERATOR_PARAMETRIC
  Probe 2b — domain prompt (domain_specific, or the template    max 1 call
             named by metadata["domain_prompt_key"])
    passes gate     → DOMAIN_MISMATCH
  Probe 2c — instruction re-generation (generate_question)     max 1 call
    passes gate     → INSTRUCTION_QUALITY

grounding_score < score_split                  — clearly-failing path
  Probe 2a (strict_grounding) runs first, then the temperature
  sweep, then the remaining prompt variants.

All probes exhausted → SOURCE_AMBIGUOUS (provenance notes present)
                       or UNKNOWN (no provenance notes at all)
```

Worst case: **5 LLM calls** per rejected sample (2 temperature sweep + 3 prompt variants). DPO pairs rejected with `rejected_above_threshold` exit immediately as `RESPONSE_QUALITY` with 0 calls — re-generating the chosen answer cannot fix an insufficient quality contrast.

`diagnose()` never raises — it catches all exceptions and returns `UNKNOWN`. `diagnose_batch(rejected, concurrency=32)` diagnoses samples in parallel; each sample's probe sequence stays sequential internally because each result conditions the next probe.

Source context is read from `rejected.input` (the exact chunk text stored by the generation task). Context extension with adjacent chunks is deliberately excluded: generator and judge operate on the same fixed context, so cross-chunk probing would produce coincidental rather than causal diagnoses.

Every recovered sample receives a `DiagnosticProbe` `ProvenanceRecord` recording the successful probe path and the diagnosed mode, so the provenance chain shows exactly how the sample was recovered.

### RewardRefiner

`curatorkit/diagnostic/reward_refine.py`

Targeted single-retry recovery for `RewardGate` rejections, run post-pipeline by `Curator` when `enable_reward_refiner=True`. For each rejected sample it reads the lowest-scoring dimension and the judge's weakness note from provenance, prompts the generator to rewrite the answer to improve that specific dimension (one LLM call per sample, concurrent), then re-evaluates all candidates in a single bulk `RewardGate` call. With `refine_instruction=True` it also rewrites poorly formed questions. `rejected_above_threshold` pairs are skipped. Refined samples carry `metadata["reward_refined"]`, `metadata["refinement_axis"]`, `metadata["refinement_type"]`, and a `RewardRefiner` provenance record; for DPO pairs the refined answer becomes the new `chosen` while the original `rejected` response is preserved.

### PipelineDiagnostics

`curatorkit/diagnostic/diagnostics.py`

Run-level accumulator attached to `Pipeline` when `enable_diagnostic_probe=True`. Provides:
- `mode_counts()` — `dict[str, int]` of failure mode frequencies
- `probe_recovery_count()` — number of samples where the probe produced an inline passing re-generation
- `total_probe_calls()` — total LLM calls consumed by all probes
- `to_dict()` / `write_summary(path)` — summary dict / writes `diagnostic_summary.json`

Accessible as `CuratorResult.diagnostics` after a run.

### Offline utilities

Two modules in `curatorkit/diagnostic/` are **not** used by production pipelines:

- `pass2.py` — `AdaptivePass2Runner` re-runs diagnosed rejections against a gate with an explicit `force_patch` config patch (applied via `CuratorConfig.apply_patch()`), for offline what-if analysis of repair strategies. Returns a `Pass2Result` with `accepted`, `rejected`, `conversion_rate`, and per-mode rates in `summary`; writes `rejected_pass2.jsonl` and `pass2_summary.json` to `output_dir`.
- `retriever.py` — `PostHocRetriever` retrieves the most similar chunk for a query by embedding cosine similarity, for comparing exact provenance tracking against post-hoc retrieval.

---

## Provenance manifest

`curatorkit/manifest.py`

`manifest.json` is always written after a run. Key fields:

| Field | Description |
|---|---|
| `pipeline_config_hash` | SHA-256 of key config fields |
| `run_timestamp` | ISO-8601 UTC |
| `stage_counts` | Per-step input / output / rejected counts |
| `rejected_breakdown` | Rejection reason → count |
| `dedup_stats` | Extracted from `MinHashDeduplicator` provenance |
| `diagnostic_stats` | `PipelineDiagnostics.to_dict()` — `null` when the probe is inactive |
| `diagnostic_files` | `["diagnostic_summary.json"]` when the probe is active, else `[]` |
| `tool_versions` | `curatorkit` and `python` version strings |

`rejected.jsonl` is always written (even when empty). Each line is a `RejectedSample` serialised via `model_dump_json()`. When a `diagnosis` is present it appears as a nested `diagnosis` object with `mode`, `was_recovered`, `evidence`, `probe_calls`, `notes`.

Checksums (`checksums.txt`) cover both `*.jsonl` and `*.json` output files.

---

## Exporter registry

| Exporter | Output file | Format |
|----------|-------------|--------|
| `AlpacaExporter` | `sft_alpaca.jsonl` | `{instruction, input, output}` |
| `ShareGPTExporter` | `sft_sharegpt.jsonl` | `{conversations: [{from, value}]}` |
| `DPOExporter` | `dpo.jsonl` | `{prompt, chosen, rejected}` |
| `GRPOExporter` | `grpo.jsonl` | `{prompt, responses, rewards}` |
| `PPOExporter` | `ppo.jsonl` | `{prompt}` |
| `CorpusExporter` | `corpus.jsonl` | Raw corpus chunks with full chunk metadata and provenance |

---

## YAML schema reference

Validated by `PipelineConfig` in `curatorkit/config.py` (pydantic) at CLI startup — a malformed config fails immediately with a clear error, not mid-pipeline. See [CLI and YAML](cli.md) for the complete per-key reference.

```yaml
name: curatorkit_pipeline
version: "0.2.0"

max_samples: null              # cap on total samples, applied right after readers
output_split: null             # e.g. {train: 0.8, val: 0.1, test: 0.1}; fractions sum to 1.0
output_split_seed: 42

readers:
  - type: jsonl | json | csv | parquet | huggingface | pdf
    path: data/file.jsonl

    format: auto | alpaca | sharegpt | preference | implicit_preference |
            unpaired_preference | grpo | prompt_only | pretrain

    field_mapping:
      source_key: canonical_key
      nested.source.key: canonical_key

    preprocessing_fn: "mymodule.my_fn"
    detection_sample_size: 10
    source_uri: null                 # provenance override

    # JSON-specific
    json_data_key: null
    # CSV-specific
    csv_delimiter: null
    csv_parse_json_cells: true
    # Parquet-specific
    parquet_columns: null
    parquet_batch_size: 1000

    # HuggingFace-specific
    hf_split: train
    hf_subset: null
    hf_streaming: false
    hf_token: "${HF_TOKEN}"
    hf_columns: null

    # PDF-specific
    chunk_strategy: heading | sentence | fixed
    chunk_max_tokens: 512
    chunk_overlap_tokens: 50
    extract_tables: false
    ocr: false
    min_section_tokens: 30
    output_mode: chunk | qa | preference | grpo | multiturn
    llm_model: null                  # per-reader LLM override

llm:                                 # global LLM config, shared by generators and gates
  model: openai/gpt-4o-mini
  temperature: 0.7
  max_tokens: 1024
  api_key: null
  api_base: null
  concurrency: 10
  timeout: 120.0
  max_retries: 3
  drop_params: true
  extra_body: {}

generators:
  - type: qa | evol_instruct | preference | grpo | multiturn | cot |
          adversarial_preference | adversarial_qa

    # qa
    num_questions: 3
    difficulty: medium
    # evol_instruct
    num_evolutions: 1
    generate_answers: true
    # preference
    preference_mode: single_call | two_pass
    # grpo
    num_responses: 4
    score_responses: true
    temperature_spread: 0.6
    # multiturn
    num_turns: 3
    # cot
    cot_mode: generate | wrap
    # adversarial_preference / adversarial_qa
    injection_rate: 0.5
    injection_types: []              # empty = all types
    injection_seed: 42
    high_temp: 1.4

    prompt_template: null            # custom prompt override
    llm_model: null                  # per-generator LLM override

gates:
  - type: schema
    min_tokens: 10
    max_tokens: 2048
    use_tiktoken: false

  # ── Hygiene gates (before generators) ──
  - type: secrets
    secrets_code_corpus_mode: false  # true for code datasets

  - type: toxicity
    toxicity_classifier_pass_threshold: 0.1
    toxicity_classifier_reject_threshold: 0.5
    toxicity_detoxify_model: unbiased
    # toxicity_llm_model: openai/gpt-4o-mini   # enable Stage 2 LLM judge for borderline

  # ── Generation-time gates ─────────────────────
  - type: hallucination
    hallucination_threshold: 0.7
    skip_if_no_context: true
    hallucination_llm_model: null    # override global LLM

  - type: reward
    reward_threshold: 0.7
    reward_dimensions: [helpfulness, honesty, instruction_following]
    store_score_in_label: true
    reward_llm_model: null           # override global LLM

  - type: diversity
    similarity_threshold: 0.92
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    embedding_device: null           # cuda | cpu | mps | null (auto)

normalizers:
  - type: exact_dedup | minhash_dedup | text_cleaner | stratified_sampler |
          embedding_dedup | pii_pseudonymizer

  # PIIPseudonymizer (hygiene — after SecretsGate, before generators)
  - type: pii_pseudonymizer
    pii_score_threshold: 0.7
    pii_spacy_model: en_core_web_lg
    pii_faker_seed: 42
    # pii_entity_types: []           # empty = default types; extend for clinical/legal

  - type: embedding_dedup
    embedding_index_dir: output/embedding_index
    embedding_threshold: 0.92

diagnostic:                          # diagnostic probe (opt-in)
  enable_probe: true
  probe_temperatures: [0.3, 0.5]
  probe_generator_model: null        # null = gate's LLM model
  score_split: 0.5
  extra_templates: {}                # override built-in probe prompt templates

exporters:
  - type: alpaca | sharegpt | dpo | grpo | ppo | corpus
```

---

## Output file reference

| File | Always written | Description |
|---|---|---|
| `manifest.json` | yes | Pipeline config hash, stage counts, rejection breakdown |
| `rejected.jsonl` | yes | All `RejectedSample` objects (includes `diagnosis` when probe active) |
| `checksums.txt` | yes | SHA-256 for all `.jsonl` and `.json` output files |
| `dataset_card.md` | yes | Human-readable pipeline summary |
| `sft_alpaca.jsonl` | if exported | Alpaca-format SFT data |
| `sft_sharegpt.jsonl` | if exported | ShareGPT-format SFT data |
| `dpo.jsonl` | if exported | DPO preference data |
| `grpo.jsonl` | if exported | GRPO rollout data |
| `ppo.jsonl` | if exported | PPO prompt data |
| `corpus.jsonl` | if exported | Raw corpus chunks with provenance metadata |
| `diagnostic_summary.json` | when probe active | Mode counts, recovery counts, probe call totals |
| `rejected_pass2.jsonl` | offline utility only | Written by `AdaptivePass2Runner` — samples that failed the gate re-check |
| `pass2_summary.json` | offline utility only | Written by `AdaptivePass2Runner` — per-mode conversion rates |
