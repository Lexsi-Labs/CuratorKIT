# CLI and YAML Pipeline

CuratorKIT has two entry points:

| Channel | When to use |
|---------|------------|
| **Python API** (`CuratorConfig` + `Curator`) | Scripting, notebooks, programmatic control |
| **CLI + YAML** (`curatorkit run pipeline.yaml`) | Reproducible runs, CI/CD, no-code configuration |

Both produce identical output. The YAML config maps to the same pipeline steps as `CuratorConfig`.

---

## CLI commands

```bash
curatorkit --help
```

### `run` — execute a pipeline

```bash
curatorkit run pipeline.yaml
curatorkit run pipeline.yaml --output-dir output/run1
curatorkit run pipeline.yaml --dry-run          # validate config and print plan, no execution
curatorkit run pipeline.yaml --async            # async runner (faster for generation tasks)
curatorkit run pipeline.yaml --verbose          # show per-stage counts during run
curatorkit run pipeline.yaml --reset-index      # clear persistent embedding index before run
```

**`--dry-run`** is useful before an expensive generation run — it validates the YAML, prints every step in order with its config, and exits. No LLM calls are made.

### `setup-pdf` — verify the PDF parsing setup

PDF parsing uses MinerU, installed via the `pdf` extra. Model weights download automatically on the first PDF parse — there is no manual download step.

```bash
pip install "curatorkit[pdf]"
curatorkit setup-pdf            # verify MinerU is installed and importable
curatorkit setup-pdf --check    # same verification; --check is the only flag
```

If MinerU is missing, the command exits non-zero and prints the install instructions above.

---

## YAML pipeline config

The YAML file is parsed and validated by Pydantic before any step runs. A malformed config produces a clear error immediately — not a crash mid-pipeline.

### Minimal example — clean and deduplicate

```yaml
name: clean_alpaca
version: "1.0"

readers:
  - type: huggingface
    path: tatsu-lab/alpaca
    hf_split: train

gates:
  - type: schema
    min_tokens: 10
    max_tokens: 2048

normalizers:
  - type: exact_dedup
  - type: text_cleaner

exporters:
  - type: alpaca
  - type: sharegpt
```

### Full example — QA generation with filtering and recovery

```yaml
name: qa_handbook
version: "1.0"

# Global LLM config — shared by all generation tasks and gates unless overridden
llm:
  model: openai/gpt-4o-mini
  temperature: 0.7
  max_tokens: 1024
  concurrency: 10
  api_base: null                            # null = use provider default

readers:
  - type: pdf
    path: docs/handbook.pdf
    chunk_strategy: heading
    chunk_max_tokens: 512
    chunk_overlap_tokens: 50

gates:
  - type: schema
    min_tokens: 20
    max_tokens: 4096
  - type: hallucination
    hallucination_threshold: 0.7
    hallucination_llm_model: null           # null = use global llm.model
  - type: reward
    reward_threshold: 0.7
    reward_dimensions: [helpfulness, honesty, instruction_following, depth]
    reward_llm_model: null

normalizers:
  - type: exact_dedup
  - type: minhash_dedup
    minhash_threshold: 0.85
  - type: text_cleaner

generators:
  - type: qa
    num_questions: 3
    difficulty: medium

# Adaptive recovery — attaches to HallucinationGate
diagnostic:
  enable_probe: true
  probe_temperatures: [0.3, 0.5]
  score_split: 0.5

exporters:
  - type: alpaca
  - type: sharegpt

output_split:
  train: 0.8
  val: 0.1
  test: 0.1
```

### DPO preference pair example

```yaml
name: dpo_from_pdf
version: "1.0"

llm:
  model: openai/gpt-4o-mini
  temperature: 0.7
  max_tokens: 2048
  concurrency: 16

readers:
  - type: pdf
    path: docs/policy.pdf
    chunk_strategy: heading

# One gates list — the schema gate runs before generators, the others after
gates:
  - type: schema
    min_tokens: 30
    max_tokens: 4096
  - type: hallucination
    hallucination_threshold: 0.7
  - type: reward
    reward_threshold: 0.75
    reward_dimensions: [helpfulness, honesty, instruction_following, depth]

normalizers:
  - type: exact_dedup
  - type: text_cleaner

generators:
  - type: preference
    preference_mode: single_call

exporters:
  - type: dpo
```

### YAML for a local vLLM endpoint

```yaml
llm:
  model: openai/Qwen/Qwen3-8B
  api_base: http://localhost:8000/v1
  api_key: token-abc123
  temperature: 0.7
  concurrency: 32
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
```

---

## YAML schema reference

### Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `"curatorkit_pipeline"` | Pipeline name (written to manifest) |
| `version` | string | `"0.2.0"` | Config version (written to manifest) |
| `readers` | list[ReaderConfig] | `[]` | Input sources |
| `gates` | list[GateConfig] | `[]` | Quality filters (schema runs before generators; others run after) |
| `normalizers` | list[NormalizerConfig] | `[]` | Dedup, cleaning, sampling |
| `generators` | list[GenerationConfig] | `[]` | LLM generation tasks |
| `exporters` | list[ExporterConfig] | `[]` | Output formats |
| `llm` | LLMConfig | null | Global LLM config |
| `diagnostic` | DiagnosticConfig | null | Inline probe config |
| `max_samples` | int | null | Cap total samples after all readers |
| `output_split` | dict[str, float] | null | Train/val/test split fractions (must sum to 1.0) |

### Reader types

| `type` | Required fields | Notes |
|--------|----------------|-------|
| `jsonl` | `path` | — |
| `json` | `path` | Use `json_data_key` if records are nested |
| `csv` | `path` | `csv_delimiter` auto-detected for TSV |
| `parquet` | `path` | Requires `[connectors]` extra |
| `huggingface` | `path` (dataset name) | Use `hf_split`, `hf_subset`, `hf_streaming`, `hf_token` |
| `pdf` | `path` | Use `chunk_strategy`, `chunk_max_tokens`, `output_mode` |

All readers accept:
- `format`: `"auto"` (default) or explicit format name
- `field_mapping`: dict remapping source column names to DataSample fields
- `detection_sample_size`: how many rows to inspect for auto-detection

### Gate types

| `type` | Key fields |
|--------|-----------|
| `schema` | `min_tokens`, `max_tokens`, `required_fields`, `enforce_task_types` |
| `hallucination` | `hallucination_threshold` (0.0–1.0), `hallucination_llm_model` |
| `reward` | `reward_threshold`, `reward_dimensions`, `reward_llm_model` |
| `diversity` | `similarity_threshold`, `embedding_model`, `embedding_device` |

### Normalizer types

| `type` | Key fields |
|--------|-----------|
| `exact_dedup` | — |
| `minhash_dedup` | `minhash_threshold`, `minhash_ngram`, `minhash_num_perm` |
| `text_cleaner` | `transforms` dict, `clean_fields` list |
| `embedding_dedup` | `embedding_index_dir`, `embedding_model`, `embedding_threshold` |
| `stratified_sampler` | `category_field`, `target_distribution`, `sampler_seed` |

### Generator types

| `type` | Key fields |
|--------|-----------|
| `qa` | `num_questions`, `difficulty`, `prompt_template` |
| `preference` | `preference_mode` (`single_call`/`two_pass`), `prompt_template` |
| `grpo` | `num_responses`, `score_responses`, `temperature_spread` (default `0.6`) |
| `multiturn` | `num_turns`, `include_context`, `prompt_template` |
| `evol_instruct` | `num_evolutions`, `strategies`, `generate_answers` |
| `cot` | `cot_mode` (`generate`/`wrap`), `prompt_template` |
| `adversarial_preference` | `injection_rate`, `injection_types`, `injection_seed` |
| `adversarial_qa` | `injection_rate`, `injection_types`, `injection_seed`, `high_temp` |

In YAML, `temperature_spread` defaults to `0.6` — GRPO rollouts are sampled at temperatures spread around the base LLM temperature. Note that the Python API equivalent (`grpo_temperature_spread` on `CuratorConfig`) defaults to `0.0`, so set it explicitly if you need identical behaviour across both channels.

### DiagnosticConfig

```yaml
diagnostic:
  enable_probe: true
  probe_temperatures: [0.3, 0.5]
  score_split: 0.5
  probe_generator_model: null       # null = use global llm.model
  extra_templates:
    strict_grounding: "Answer using only the passage.\n\nPassage:\n{source}\n\nQuestion:\n{question}"
    domain_specific:  "You are a legal analyst. Answer from the passage only.\n\nPassage:\n{source}\n\nQuestion:\n{question}"
```

---

## Python API vs YAML — feature parity

Most features available in `CuratorConfig` (Python) are also available in the YAML config. Key differences:

- **`preprocessing_fn`**: Python API accepts a callable; YAML accepts a dotted module path string (`"mymodule.my_fn"`) that is imported at runtime.
- **Data hygiene** (`secrets_gate`, `pii_pseudonymize`, `toxicity_gate`): Python API flags only. YAML equivalents use `type: secrets`, `type: toxicity` in `gates` and `type: pii_pseudonymizer` in `normalizers`.

For full control, use the Python API. For reproducible scheduled runs and CI, use the YAML CLI.

---

## Next: [Config reference →](configuration.md)
