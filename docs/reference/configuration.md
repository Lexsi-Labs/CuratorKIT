# CuratorConfig Parameter Reference

All parameters for the `CuratorConfig` dataclass, grouped by category. Every field has a default — none are required. The minimal working config is just `dataset` + `llm_model` + `generation_task`.

```python
from curatorkit import Curator, CuratorConfig
```

---

## Data source

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str \| dict \| list` | `""` | Input path, HuggingFace dataset name, dict with reader options, or list for multi-source |
| `split` | `str` | `"train"` | Dataset split when loading from HuggingFace (`"train"`, `"test"`, etc.) |
| `subset` | `str \| None` | `None` | HuggingFace dataset configuration name (e.g. `"en"` for multilingual datasets) |
| `streaming` | `bool` | `False` | Stream the dataset instead of loading into memory (HuggingFace only) |
| `hf_token` | `str \| None` | `None` | HuggingFace API token for gated datasets |
| `hf_subset` | `str \| None` | `None` | Alias for `subset` |
| `hf_columns` | `list[str] \| None` | `None` | Load only these columns (reduces memory for large HF datasets) |
| `max_samples` | `int \| None` | `None` | Hard cap on the final sample count, applied late (after gates, dedup, and resampling) so distributions are preserved. To cap how much is *read* from a large source, use the per-reader form: `dataset={"name": "...", "max_samples": N}`. |

---

## Column mapping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | `"auto"` | Force a specific input format: `"alpaca"`, `"sharegpt"`, `"preference"`, `"implicit_preference"`, `"unpaired_preference"`, `"grpo"`, `"prompt_only"`, `"pretrain"`. `"auto"` detects from column names. |
| `field_mapping` | `dict[str, str]` | `{}` | Rename source columns to DataSample fields, as `{source_column: datasample_field}` — keys are *your* columns. E.g. `{"user_query": "instruction", "response": "output"}`. Keys may use dot notation for nested dict values: `{"meta.prompt": "instruction"}`. |
| `preprocessing_fn` | `Callable \| list \| None` | `None` | Function `(dict) -> dict \| None` applied to every raw row. Return `None` to drop. |

---

## Quality filters (pre-generation)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_tokens` | `int` | `10` | Minimum token count. Samples below this are rejected by SchemaGate. |
| `max_tokens` | `int` | `2048` | Maximum token count. Samples above this are rejected by SchemaGate. |
| `use_tiktoken` | `bool` | `False` | Use tiktoken for token counting instead of whitespace-based estimation |
| `schema_use_tiktoken` | `bool` | `False` | Use tiktoken specifically for SchemaGate (overrides `use_tiktoken` for gate only) |
| `schema_enforce_task_types` | `list[str]` | `[]` | Only pass samples with these `task_type` values. Empty = pass all. |
| `schema_gate` | `bool` | `True` | Enable SchemaGate. Set `False` to skip all schema checks. |
| `schema_required_fields` | `list[str]` | `[]` | Additional DataSample fields that must be non-empty to pass SchemaGate |

---

## Deduplication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dedup` | `str` | `"exact"` | Deduplication strategy: `"exact"`, `"minhash"`, `"none"` |
| `minhash_threshold` | `float` | `0.85` | Jaccard similarity threshold for MinHash dedup. Higher = more aggressive. |
| `minhash_ngram` | `int` | `3` | N-gram size for MinHash shingle construction |
| `minhash_num_perm` | `int` | `128` | Number of hash permutations. Higher = more accurate but slower. |
| `minhash_seed` | `int` | `42` | Random seed for MinHash permutations |

---

## Text cleaning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clean` | `bool` | `True` | Enable text cleaning |
| `clean_transforms` | `dict[str, bool]` | `{}` | Toggle specific transforms. Keys: `"normalise_unicode"`, `"fix_encoding_artifacts"`, `"collapse_whitespace"`, `"remove_control_chars"`, `"strip_html"`. Default: all enabled. |
| `clean_fields` | `list[str]` | `[]` | Apply cleaning to these DataSample fields. Empty = applies to `instruction`, `input`, and `output`. |

---

## Data hygiene (pre-generation)

Hygiene steps run after text cleaning and before any generation task. Execution order is
fixed: `SecretsGate → PIIPseudonymizer → ToxicityGate`. See [Data hygiene](../guides/data-hygiene.md)
for full usage examples.

### SecretsGate

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `secrets_gate` | `bool` | `False` | Enable `SecretsGate`. Rejects samples containing API keys, tokens, private keys, or high-entropy secrets detected by the `detect-secrets` battery. |
| `secrets_code_corpus_mode` | `bool` | `False` | Enable `KeywordDetector` in addition to entropy-based detectors. Set `True` for code datasets; leave `False` for prose (too noisy). |

### PIIPseudonymizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pii_pseudonymize` | `bool` | `False` | Enable `PIIPseudonymizer`. Replaces detected PII entities with consistent Faker-generated values. Modifies samples in-place; does not reject. |
| `pii_entity_types` | `list[str]` | `[]` | Presidio entity types to detect. Empty list = default set (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`). Use `ENTITY_TYPES_CLINICAL` from `curatorkit.hygiene.pii` for medical/legal corpora. |
| `pii_score_threshold` | `float` | `0.7` | Presidio confidence threshold. Detections below this score are ignored. Lower values catch more PII at the cost of higher false-positive rate. |
| `pii_spacy_model` | `str` | `"en_core_web_lg"` | spaCy model for NER. `"en_core_web_lg"` for production; `"en_core_web_sm"` for dev/CI. |
| `pii_faker_seed` | `int` | `42` | Seed for Faker replacements. Same seed = same fake entities across runs for reproducibility. |

### ToxicityGate

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `toxicity_gate` | `bool` | `False` | Enable `ToxicityGate`. Two-stage: Stage 1 (local Detoxify classifier) first, then optional Stage 2 (LLM judge) for borderline samples. |
| `toxicity_classifier_pass_threshold` | `float` | `0.1` | Max-dimension Detoxify score below this → immediate pass (no LLM call). |
| `toxicity_classifier_reject_threshold` | `float` | `0.5` | Max-dimension Detoxify score above this → immediate reject (no LLM call). Scores between the two thresholds are escalated to the LLM judge if `toxicity_llm_judge=True`. |
| `toxicity_detoxify_model` | `str` | `"unbiased"` | Detoxify model variant. `"unbiased"` for general corpora; `"original"` for informal web text; `"multilingual"` for non-English. |
| `toxicity_llm_judge` | `bool` | `False` | Enable LLM second opinion for borderline samples. Requires `llm_model` to be set. |
| `toxicity_llm_reject_threshold` | `float` | `0.5` | LLM judge score at or above which a borderline sample is rejected (only used when `toxicity_llm_judge=True`). |

---

## Export

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str \| Path` | `"output"` | Directory for all output files |
| `export_formats` | `list[str]` | `["alpaca", "sharegpt", "dpo"]` | Which exporters to run. Options: `"alpaca"`, `"sharegpt"`, `"dpo"`, `"grpo"`, `"ppo"`, `"corpus"` |
| `output_split` | `dict[str, float] \| None` | `None` | Split accepted samples into subdirectories. E.g. `{"train": 0.8, "val": 0.1, "test": 0.1}`. Must sum to 1.0. |
| `output_split_seed` | `int` | `42` | Seed for the pre-split shuffle. Set the same value across runs to get identical train/val/test assignments. |

---

## Misc

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"curatorkit_run"` | Pipeline name, written to `manifest.json` and `dataset_card.md` |
| `detection_sample_size` | `int` | `10` | Number of rows to inspect for auto-format detection |

---

## Resampling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resample` | `bool` | `False` | Enable stratified resampling to match `target_distribution` |
| `target_distribution` | `dict[str, float]` | `{}` | Target class proportions. E.g. `{"finance": 0.5, "legal": 0.3, "general": 0.2}`. Must sum to 1.0. |
| `resample_field` | `str` | `"source_dataset"` | DataSample metadata field used as the stratification key |
| `resample_seed` | `int` | `42` | Random seed for resampling |

---

## Generator LLM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_model` | `str \| None` | `None` | LiteLLM model string. E.g. `"openai/gpt-4o-mini"`, `"ollama/llama3.1:8b"`. Required for generation tasks. |
| `llm_temperature` | `float` | `0.7` | Sampling temperature for generation |
| `llm_max_tokens` | `int` | `1024` | Max tokens per LLM response |
| `llm_api_key` | `str \| None` | `None` | API key. Falls back to environment variable (e.g. `OPENAI_API_KEY`) if `None`. |
| `llm_api_base` | `str \| None` | `None` | Custom base URL for OpenAI-compatible endpoints (vLLM, Ollama, etc.) |
| `llm_concurrency` | `int` | `10` | Max parallel LLM requests for generation |
| `llm_timeout` | `float` | `120.0` | Request timeout in seconds |
| `llm_max_retries` | `int` | `3` | Retry attempts on transient errors |
| `llm_drop_params` | `bool` | `True` | Silently drop unsupported params rather than raising (LiteLLM behaviour) |
| `llm_extra_body` | `dict` | `{}` | Extra fields forwarded verbatim in the API request body (e.g. `{"chat_template_kwargs": {"enable_thinking": True}}`) |

---

## Judge LLM

Used by HallucinationGate and RewardGate. Defaults to the generator LLM when not set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `judge_llm_model` | `str \| None` | `None` | Model for quality scoring. `None` = use `llm_model`. **Recommended**: set a separate model to avoid self-leniency bias. |
| `judge_llm_api_base` | `str \| None` | `None` | Base URL for judge model endpoint |
| `judge_llm_temperature` | `float` | `0.1` | Low temperature for deterministic scoring |
| `judge_llm_max_tokens` | `int` | `512` | Max tokens for judge responses (scores are short) |
| `judge_llm_timeout` | `float` | `120.0` | Judge request timeout in seconds |
| `judge_llm_max_retries` | `int` | `3` | Retry attempts for judge requests |
| `judge_llm_extra_body` | `dict` | `{}` | Extra body params for judge model (e.g. disable thinking mode for structured output) |

---

## Generation task

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generation_task` | `str \| None` | `None` | Which task to run: `"qa"`, `"preference"`, `"grpo"`, `"multiturn"`, `"evol"`, `"cot"`, `"adversarial_preference"`, `"adversarial_qa"`. `None` = no generation, cleaning/filtering only. |
| `num_questions` | `int` | `3` | Number of QA pairs or questions to generate per source chunk (`qa`, `adversarial_qa`) |
| `num_responses` | `int` | `4` | Number of rollout responses per prompt (`grpo`) |
| `num_turns` | `int` | `3` | Number of turns in a multi-turn conversation (`multiturn`) |
| `num_evolutions` | `int` | `1` | Number of evolution rounds per instruction (`evol`) |
| `difficulty` | `str` | `"medium"` | Question difficulty level: `"easy"`, `"medium"`, `"hard"` (`qa`, `adversarial_qa`) |
| `score_responses` | `bool` | `True` | Score each rollout response after generation (`grpo`) |
| `generate_answers` | `bool` | `True` | Generate answers for evolved instructions (`evol`) |
| `cot_mode` | `str` | `"generate"` | Chain-of-thought mode: `"generate"` (LLM produces CoT from scratch) or `"wrap"` (wrap an existing answer with reasoning) |
| `cot_marker` | `str \| None` | `None` | Separator inserted between the reasoning block and the final answer in `cot` output. `None` = the built-in separator (`"\n\n## Answer\n"`). |
| `preference_mode` | `str` | `"single_call"` | Preference pair generation: `"single_call"` (one call for both chosen/rejected) or `"two_pass"` (separate calls) |
| `generation_concurrency` | `int \| None` | `None` | Concurrency for the generation task. `None` = use `llm_concurrency`. |
| `judge_concurrency` | `int \| None` | `None` | Concurrency for scoring/judging calls. `None` = use `llm_concurrency`. |

---

## GRPO-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grpo_temperature_spread` | `float` | `0.0` | Spread applied around `llm_temperature` to create a range of rollout temperatures. `0.0` = all rollouts use the same temperature. |
| `grpo_temperatures` | `list[float] \| None` | `None` | Explicit per-rollout temperatures. Overrides `grpo_temperature_spread`. Cycled if shorter than `num_responses`. |
| `grpo_scoring_llm_model` | `str \| None` | `None` | Model for scoring GRPO responses. `None` = use `llm_model` (the generator). |

---

## Prompt templates

Override the default LLM prompt for each task. If you provide a template, all required placeholder variables must be present or construction raises `ValueError`.

| Parameter | Required variables | Task |
|-----------|-------------------|------|
| `qa_prompt_template` | `{context}`, `{num_questions}` (optional: `{difficulty}` — if absent and `difficulty != "medium"`, a difficulty hint is appended as a suffix) | `qa` |
| `qa_table_prompt_template` | `{context}`, `{num_questions}` (optional: `{difficulty}`) | `qa` — separate prompt used for table-derived chunks |
| `evol_prompt_template` | `{instruction}`, `{strategy}`, `{context}` | `evol` |
| `evol_answer_prompt_template` | `{instruction}` | `evol` — answer pass for evolved instructions without source context (requires `generate_answers=True`) |
| `preference_prompt_template` | `{instruction}`, `{context_section}` | `preference` (`single_call` mode) |
| `preference_chosen_prompt` | `{instruction}`, `{context_section}` | `preference` (`two_pass` mode) — chosen-response generation |
| `preference_rejected_prompt` | `{instruction}`, `{context_section}` | `preference` (`two_pass` mode) — rejected-response generation |
| `grpo_prompt_template` | `{instruction}` | `grpo` |
| `multiturn_prompt_template` | `{num_turns}`, `{context_section}`, `{initial_question}` | `multiturn` (only used in `single_call` mode — not active via `CuratorConfig`, which always uses `turn_by_turn`; see the [customisation guide](../guides/customisation.md)) |
| `cot_prompt_template` | `{instruction}` (generate mode); `{instruction}`, `{answer}` (wrap mode) | `cot` |
| `adversarial_prompt_template` | `{context}`, `{question}`, `{injection_type}` | `adversarial_preference` (template for the adversarially-corrupted rejected response) |
| `llm_prompt_template` | task-dependent | fallback for unlisted tasks |

All default to `None` (built-in template used).

---

## Adversarial generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `injection_rate` | `float` | `0.5` | Fraction of questions to inject adversarial hallucinations into (0.0–1.0) |
| `injection_types` | `list[str]` | `[]` | Which injection strategies to use. For `adversarial_preference`: `"contradicts_source"`, `"parametric_drift"`, `"domain_mismatch"`, `"instruction_quality"`. For `adversarial_qa`: same four plus `"high_temperature_drift"`. Empty = all types for the active task. |
| `injection_seed` | `int` | `42` | Random seed for injection sampling |
| `high_temp` | `float` | `1.4` | Sampling temperature used for the `high_temperature_drift` injection type (`adversarial_qa` only) |

---

## Quality gates (post-generation)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hallucination_threshold` | `float \| None` | `None` | Enable HallucinationGate. Samples below this score are rejected. Range 0.0–1.0. `None` = gate disabled. |
| `hallucination_prompt_template` | `str \| None` | `None` | Custom hallucination judge prompt. Required variables: `{source}`, `{claim}`. |
| `reward_threshold` | `float \| None` | `None` | Enable RewardGate. Samples below this score are rejected. Range 0.0–1.0. `None` = gate disabled. |
| `reward_dimensions` | `list[str]` | `["helpfulness", "honesty", "instruction_following"]` | Built-in scoring axes. Valid values: `"helpfulness"`, `"honesty"`, `"instruction_following"`, `"truthfulness"`, `"depth"`, `"creativity"`, `"coherence"`. |
| `reward_prompt_template` | `str \| None` | `None` | Replace the entire reward judge prompt with a custom rubric. Required variables: `{instruction}`, `{response}`. Must return `{"score": 0.XX, "reasoning": "..."}`. |
| `reward_store_score` | `bool` | `True` | Write the overall reward score to each sample's `label` field (used by the reward refiner and downstream filtering) |
| `diversity_threshold` | `float \| None` | `None` | Enable DiversityGate. Samples with cosine similarity above this to any accepted sample are rejected. Range 0.0–1.0. `None` = gate disabled. |

---

## Inline recovery (adaptive probe)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_diagnostic_probe` | `bool` | `False` | Enable the inline probe. Fires after each gate's rejects, before the next gate sees samples. |
| `probe_temperatures` | `list[float]` | `[0.3, 0.5]` | Temperature values to try during the temperature sweep recovery path |
| `probe_generator_model` | `str \| None` | `None` | Model for probe re-generation. `None` = use `llm_model`. |
| `probe_score_split` | `float` | `0.5` | Score boundary for routing. Samples above this go to the temperature path; below go to the strict grounding path. |
| `probe_extra_templates` | `dict[str, str]` | `{}` | Override built-in probe templates (`"default"`, `"strict_grounding"`, `"domain_specific"`) or add new keys, selected per sample via `metadata["domain_prompt_key"]`. See [Customisation](../guides/customisation.md#custom-probe-templates) for routing details. |
| `enable_reward_refiner` | `bool` | `False` | Enable the post-pipeline reward refiner (rewrites answers targeting the weakest quality dimension) |
| `reward_refine_prompt_template` | `str \| None` | `None` | Custom refiner prompt. `None` = built-in template. |
| `reward_instruction_refine_template` | `str \| None` | `None` | Custom template for instruction rewrites (used when `instruction_quality` failure mode is detected). `None` = built-in. |

---

## Embedding (diversity gate + dedup)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | SentenceTransformer model for diversity gate and embedding dedup |
| `diversity_embedding_model` | `str \| None` | `None` | Override `embedding_model` for the diversity gate only |
| `embedding_dedup_model` | `str \| None` | `None` | Override `embedding_model` for the embedding dedup normalizer only |
| `embedding_device` | `str \| None` | `None` | Device for embedding inference: `"cpu"`, `"cuda"`, `"mps"`. `None` = auto-detect. |
| `embedding_batch_size` | `int` | `64` | Batch size for embedding model inference |

---

## Cross-run embedding dedup

Persists an FAISS index across runs to reject samples already seen in previous runs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dedup` | `bool` | `False` | Enable cross-run embedding dedup |
| `embedding_index_dir` | `str` | `"output/embedding_index"` | Directory where the FAISS index is stored and loaded |
| `embedding_dedup_threshold` | `float` | `0.92` | Cosine similarity threshold. Samples above this are rejected as near-duplicates of previously indexed samples. |
| `embedding_reset_index` | `bool` | `False` | Delete and rebuild the index at the start of this run |

---

## PDF extraction

Controls how PDFs are chunked when `dataset` points to a `.pdf` file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_output_mode` | `str` | `"chunk"` | How to yield content: `"chunk"` (raw source chunks), `"qa"`, `"preference"`, `"grpo"`, or `"multiturn"` (LLM generation runs inline in the reader). Use `"chunk"` and set `generation_task` separately — inline modes are legacy. |
| `pdf_chunk_strategy` | `str` | `"heading"` | Chunking strategy: `"heading"` (split on heading boundaries), `"sentence"` (sentence boundaries), or `"fixed"` (fixed token size) |
| `pdf_chunk_max_tokens` | `int` | `512` | Target max tokens per chunk |
| `pdf_chunk_overlap_tokens` | `int` | `50` | Token overlap between adjacent chunks |
| `pdf_extract_tables` | `bool` | `False` | Include table cells as text in chunks |
| `pdf_ocr` | `bool` | `False` | Run OCR on scanned pages (requires MinerU GPU setup) |
| `pdf_min_section_tokens` | `int` | `30` | Discard sections shorter than this (headers, footers, captions) |

---

## See also

- [CLI and YAML pipeline](cli.md) — equivalent config via YAML
- [Data sources](../guides/data-sources.md) — reader-specific options
- [Generation](../guides/generation.md) — generation task details and prompt templates
- [Quality filtering](../guides/quality-gates.md) — gate behaviour and tuning
- [Adaptive recovery](../guides/adaptive-recovery.md) — probe and refiner details
- [Customisation](../guides/customisation.md) — custom prompts, backends, extensions
