# Data Hygiene

Data hygiene components run during ingestion and cleaning — **before any LLM call**. This means:

- Credentials and API keys never reach an external API.
- PII in source documents is pseudonymized before a model generates continuations, so the output inherits clean entities rather than leaking real ones.
- Toxic source material is discarded before you pay for generation.

Three components are available:

| Component | Type | What it does |
|-----------|------|--------------|
| `SecretsGate` | Gate (rejects) | Rejects samples containing credentials, API keys, or high-entropy secrets |
| `PIIPseudonymizer` | Normalizer (modifies) | Replaces PII entities with consistent fake values (per-sample scope) |
| `ToxicityGate` | Gate (rejects) | Rejects toxic content in two stages: Stage 1 (local classifier), Stage 2 (optional LLM judge) |

All three are **task-aware**: they automatically select the correct fields to scan or pseudonymize based on `sample.task_type` — checking `chosen`/`rejected` for preference data, `responses` for GRPO rollouts, and `output` for source chunks.

---

## Install

```bash
pip install "curatorkit[hygiene]"
```

This installs: `detoxify`, `detect-secrets`, `presidio-analyzer`, `presidio-anonymizer`, `spacy`, and `faker`.

For `PIIPseudonymizer`, also download the spaCy model:

```bash
python -m spacy download en_core_web_lg   # default (~800 MB, highest accuracy)
# or for dev/CI:
python -m spacy download en_core_web_sm   # ~12 MB, adequate for standard PII types
```

---

## Channel 1 — `CuratorConfig` Python API

Set flags directly on `CuratorConfig`. The hygiene steps are inserted automatically in the right order.

### SecretsGate

```python
from curatorkit import Curator, CuratorConfig

result = Curator(CuratorConfig(
    dataset             = "data/raw.jsonl",
    secrets_gate        = True,
    # Enable KeywordDetector for code corpora (off by default — too noisy for prose)
    secrets_code_corpus_mode = False,
)).run()
```

### PIIPseudonymizer

```python
from curatorkit import Curator, CuratorConfig

result = Curator(CuratorConfig(
    dataset             = "data/medical_notes.jsonl",
    pii_pseudonymize    = True,
    # pii_entity_types  = []       # empty = default types (no DATE_TIME)
    # pii_score_threshold = 0.7    # Presidio confidence threshold
    # pii_spacy_model   = "en_core_web_lg"
    # pii_faker_seed    = 42       # reproducible replacements
)).run()
```

For clinical or legal corpora where dates and locations are also PII:

```python
from curatorkit.hygiene.pii import ENTITY_TYPES_CLINICAL

result = Curator(CuratorConfig(
    dataset          = "data/clinical_notes.jsonl",
    pii_pseudonymize = True,
    pii_entity_types = ENTITY_TYPES_CLINICAL,   # adds DATE_TIME, LOCATION, MEDICAL_LICENSE
    pii_spacy_model  = "en_core_web_lg",
)).run()
```

### ToxicityGate

```python
from curatorkit import Curator, CuratorConfig

result = Curator(CuratorConfig(
    dataset                              = "data/raw.jsonl",
    toxicity_gate                        = True,
    # Stage 1 (local classifier) thresholds:
    #   score < pass_threshold  → pass immediately (no LLM call)
    #   score > reject_threshold → reject immediately (no LLM call)
    #   in between               → Stage 2, LLM judge (only when toxicity_llm_judge=True)
    toxicity_classifier_pass_threshold   = 0.1,
    toxicity_classifier_reject_threshold = 0.5,
    toxicity_detoxify_model              = "unbiased",  # or "original" / "multilingual"
    # Enable LLM second-opinion for borderline samples:
    toxicity_llm_judge                   = False,
    toxicity_llm_reject_threshold        = 0.5,   # LLM judge rejects at or above this score
    llm_model                            = "openai/gpt-4o-mini",  # required when llm_judge=True
)).run()
```

### All three together

```python
result = Curator(CuratorConfig(
    dataset                              = "data/raw.jsonl",
    secrets_gate                         = True,
    pii_pseudonymize                     = True,
    toxicity_gate                        = True,
    toxicity_classifier_pass_threshold   = 0.15,  # raised for academic corpora
    toxicity_classifier_reject_threshold = 0.5,
    llm_model                            = "openai/gpt-4o-mini",
    generation_task                      = "qa",
)).run()
```

**Execution order** (fixed): `SecretsGate → PIIPseudonymizer → ToxicityGate → [generation]`

---

## Channel 2 — YAML pipeline config (CLI)

### Running a pipeline

```bash
curatorkit run pipeline.yaml
curatorkit run pipeline.yaml --dry-run   # validate config, print step plan, exit
```

### SecretsGate in YAML

Add a gate with `type: secrets` to the `gates` list. It runs before generators.

```yaml
name: hygiene_pipeline
version: "0.1"

readers:
  - type: jsonl
    path: data/raw.jsonl

gates:
  - type: secrets
    secrets_code_corpus_mode: false   # set true for datasets that include source code

generators:
  - type: qa
    num_questions: 3

exporters:
  - type: alpaca
```

### PIIPseudonymizer in YAML

Add a normalizer with `type: pii_pseudonymizer`. It runs before generators, after dedup and text cleaning.

```yaml
normalizers:
  - type: exact_dedup
  - type: text_cleaner
  - type: pii_pseudonymizer
    pii_score_threshold: 0.7
    pii_spacy_model: en_core_web_lg
    pii_faker_seed: 42
    # pii_entity_types: []   # empty = default types; add DATE_TIME, LOCATION for clinical corpora
```

For clinical corpora, list entity types explicitly:

```yaml
normalizers:
  - type: pii_pseudonymizer
    pii_entity_types:
      - PERSON
      - EMAIL_ADDRESS
      - PHONE_NUMBER
      - US_SSN
      - CREDIT_CARD
      - IP_ADDRESS
      - DATE_TIME
      - MEDICAL_LICENSE
      - LOCATION
```

### ToxicityGate in YAML

Add a gate with `type: toxicity`. Runs before generators.

```yaml
gates:
  - type: schema
  - type: secrets
  - type: toxicity
    toxicity_classifier_pass_threshold: 0.1
    toxicity_classifier_reject_threshold: 0.5
    toxicity_detoxify_model: unbiased
    # toxicity_llm_model: openai/gpt-4o-mini   # enable LLM judge for borderline band
```

### Full hygiene pipeline YAML example

```yaml
name: full_hygiene_pipeline
version: "0.1"

llm:
  model: openai/gpt-4o-mini
  temperature: 0.7
  concurrency: 10

readers:
  - type: jsonl
    path: data/raw_instructions.jsonl

gates:
  - type: schema
    min_tokens: 10
    max_tokens: 2048
  - type: secrets
    secrets_code_corpus_mode: false
  - type: toxicity
    toxicity_classifier_pass_threshold: 0.1
    toxicity_classifier_reject_threshold: 0.5
    toxicity_detoxify_model: unbiased

normalizers:
  - type: exact_dedup
  - type: text_cleaner
  - type: pii_pseudonymizer
    pii_score_threshold: 0.7
    pii_spacy_model: en_core_web_sm   # use sm for faster CI runs

generators:
  - type: qa
    num_questions: 3

exporters:
  - type: alpaca
  - type: sharegpt
```

---

## Channel 3 — Direct module import

For use in custom pipelines, scripts, or when you need to compose steps manually outside of `Curator`.

### SecretsGate

```python
from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.schema import DataSample

gate = SecretsGate(
    code_corpus_mode=False,
    # fields=None  → task-aware auto-selection (recommended)
    # fields=["instruction", "output"]  → explicit override
)

samples: list[DataSample] = [...]
passed, rejected = gate.run(samples)

for r in rejected:
    print(r.rejection_reason)   # e.g. "secret_detected:AWS Access Key,PrivateKeyDetector"
```

### PIIPseudonymizer

```python
from curatorkit.hygiene.pii import PIIPseudonymizer, ENTITY_TYPES_CLINICAL

pseudonymizer = PIIPseudonymizer(
    entity_types=None,           # None = default types
    score_threshold=0.7,
    spacy_model="en_core_web_lg",
    faker_seed=42,
)

samples: list[DataSample] = [...]
samples = pseudonymizer.run(samples)  # modifies in-place, returns same list
```

Cross-field consistency is guaranteed within each sample: if "John Smith" appears in both `instruction` and `output`, both get the same fake name.

### ToxicityGate

```python
from curatorkit.hygiene.toxicity import ToxicityGate

# Classifier-only (no LLM)
gate = ToxicityGate(
    classifier_pass_threshold=0.1,
    classifier_reject_threshold=0.5,
    detoxify_model="unbiased",
)

# With LLM judge for borderline samples
from curatorkit.llm.litellm import LiteLLMBackend

llm = LiteLLMBackend(model="openai/gpt-4o-mini")
gate = ToxicityGate(
    classifier_pass_threshold=0.1,
    classifier_reject_threshold=0.5,
    llm=llm,
    llm_reject_threshold=0.5,
)

passed, rejected = gate.run(samples)
```

### Composing in a custom pipeline

```python
from curatorkit.pipeline import Pipeline
from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.hygiene.pii import PIIPseudonymizer
from curatorkit.hygiene.toxicity import ToxicityGate
from curatorkit.connectors.jsonl import JSONLReader
from curatorkit.exporters.alpaca import AlpacaExporter
from pathlib import Path

steps = [
    JSONLReader("data/raw.jsonl"),
    SecretsGate(),
    PIIPseudonymizer(spacy_model="en_core_web_sm"),
    ToxicityGate(),
    AlpacaExporter(),
]

pipeline = Pipeline(steps, output_dir=Path("output/"))
result = pipeline.run()
```

---

## Task awareness

All three components automatically select the right fields per `task_type`. You never need to tell them about DPO pairs or GRPO rollouts — they detect the task type from the sample.

| `task_type` | Fields checked / pseudonymized |
|-------------|-------------------------------|
| `preference`, `implicit_preference` | `instruction`, `input`, `chosen`, `rejected` |
| `grpo` | `instruction`, `input`, `responses` (each rollout scanned independently) |
| `language_modeling`, `source_chunk` | `output` |
| `prompt_only` | `instruction`, `input` |
| `unpaired_preference` | `instruction`, `input`, `output` |
| `conversational`, `instruction_following` | `instruction`, `input`, `output` |
| Unknown / None | All configured fields |

**Override the field list** for any component by passing an explicit `fields=` argument. When an explicit list is provided, task-aware selection is disabled entirely.

```python
# Only scan the output field, regardless of task type
gate = SecretsGate(fields=["output"])
```

---

## Rejection reasons and provenance

### SecretsGate

Rejected samples have `rejection_reason = "secret_detected:{type_list}"` where `type_list` is a sorted comma-separated list of detected secret types.

```
secret_detected:AWSKeyDetector,GitHubTokenDetector
secret_detected:Base64HighEntropyString
secret_detected:PrivateKeyDetector
```

The provenance record on each sample (passed or rejected) includes:
```json
{
  "passed": false,
  "secret_type_counts": {"AWSKeyDetector": 1},
  "fields_scanned": ["instruction", "input", "output"],
  "total_findings": 1
}
```

### PIIPseudonymizer

Provenance records log entity type counts — never the original or replaced values.

```json
{
  "entities_replaced": {"PERSON": 3, "EMAIL_ADDRESS": 1},
  "fields_processed": ["instruction", "input", "output"],
  "total_replacements": 4
}
```

### ToxicityGate

```
toxic_content:classifier:0.621          # rejected at Stage 1 (local classifier), score=0.621
toxic_content:llm_judge:0.710           # borderline at Stage 1, escalated to Stage 2 (LLM judge), rejected
```

Provenance on passing samples (the `phase` key records which stage decided — `"classifier"` or `"llm_judge"`):
```json
{
  "passed": true,
  "max_score": 0.042,
  "worst_field": "instruction",
  "phase": "classifier"
}
```

---

## Tuning guide

### SecretsGate false positives

`code_corpus_mode=False` disables `KeywordDetector` by default. If you still see false positives in prose corpora, check which plugin is triggering using `result.rejected[i].provenance_chain[-1].notes["secret_type_counts"]`. High entropy strings that aren't secrets (base64 images, encoded payloads) can be addressed by raising the entropy thresholds:

```python
SecretsGate(plugins=[
    {"name": "AWSKeyDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "Base64HighEntropyString", "base64_limit": 5.5},  # raised from 4.5
    {"name": "HexHighEntropyString",    "hex_limit":    4.0},  # raised from 3.0
])
```

### PIIPseudonymizer over-redaction

Lower `score_threshold` (e.g. `0.5`) catches more PII but also mislabels common nouns as entities. Start at `0.7` and lower only if real PII is slipping through. Use `en_core_web_lg` over `en_core_web_sm` for higher precision.

### ToxicityGate thresholds for academic corpora

Academic text discussing crime, medication, or social issues typically scores 0.1–0.25 on `toxicity` even when completely clean. If you see excessive LLM escalations, raise `classifier_pass_threshold` to `0.2`:

```python
CuratorConfig(
    toxicity_gate                      = True,
    toxicity_classifier_pass_threshold = 0.2,   # raised for academic/legal/medical corpora
    toxicity_classifier_reject_threshold = 0.6,
)
```

Use `detoxify_model="multilingual"` for non-English corpora. Use `"unbiased"` (default) over `"original"` for any corpus with legitimate discussion of sensitive topics — the unbiased model suppresses false positives.

---

## Parameter reference

### `CuratorConfig` hygiene fields

| Field | Default | Description |
|-------|---------|-------------|
| `secrets_gate` | `False` | Enable SecretsGate |
| `secrets_code_corpus_mode` | `False` | Enable KeywordDetector (for code datasets) |
| `pii_pseudonymize` | `False` | Enable PIIPseudonymizer |
| `pii_entity_types` | `[]` | Presidio entity types; `[]` = default set (no DATE_TIME) |
| `pii_score_threshold` | `0.7` | Presidio detection confidence threshold |
| `pii_spacy_model` | `"en_core_web_lg"` | spaCy model name |
| `pii_faker_seed` | `42` | Faker seed for reproducible replacements |
| `toxicity_gate` | `False` | Enable ToxicityGate |
| `toxicity_classifier_pass_threshold` | `0.1` | Score below this → immediate pass |
| `toxicity_classifier_reject_threshold` | `0.5` | Score above this → immediate reject |
| `toxicity_detoxify_model` | `"unbiased"` | `"unbiased"` \| `"original"` \| `"multilingual"` |
| `toxicity_llm_judge` | `False` | Use LLM for borderline band (requires `llm_model`) |
| `toxicity_llm_reject_threshold` | `0.5` | LLM judge score at or above which a borderline sample is rejected (only when `toxicity_llm_judge=True`) |

### YAML `GateConfig` fields (`type: toxicity`, `type: secrets`)

| Field | Default | Description |
|-------|---------|-------------|
| `toxicity_classifier_pass_threshold` | `0.1` | Classifier pass threshold |
| `toxicity_classifier_reject_threshold` | `0.5` | Classifier reject threshold |
| `toxicity_detoxify_model` | `"unbiased"` | Detoxify model variant |
| `toxicity_llm_model` | `null` | LLM model for Stage 2 (LLM judge); `null` = no judge |
| `secrets_code_corpus_mode` | `false` | Enable KeywordDetector |
| `secrets_fields` | `[]` | Fields to scan; `[]` = task-aware auto-selection |

### YAML `NormalizerConfig` fields (`type: pii_pseudonymizer`)

| Field | Default | Description |
|-------|---------|-------------|
| `pii_entity_types` | `[]` | Presidio entity types; `[]` = default set |
| `pii_score_threshold` | `0.7` | Presidio confidence threshold |
| `pii_spacy_model` | `"en_core_web_lg"` | spaCy model name |
| `pii_faker_seed` | `42` | Faker seed |
| `pii_language` | `"en"` | Analysis language |
| `pii_fields` | `[]` | Fields to process; `[]` = task-aware auto-selection |

---

This is the last guide. For the full parameter listing, see the [Config reference](../reference/configuration.md).
