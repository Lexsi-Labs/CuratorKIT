"""
Pipeline YAML configuration models.

Validated at CLI startup before any step runs. A malformed config produces a
clear pydantic error immediately — not a crash mid-pipeline.

Covers the full pipeline surface:
  - Readers (jsonl, json, csv, parquet, huggingface, pdf)
  - LLM configuration (model, temperature, concurrency, etc.)
  - Generation task configuration (qa, preference, grpo, multiturn, evol, cot, adversarial)
  - Quality gate configurations (schema, hallucination, reward, diversity, secrets, toxicity)
  - Normalizers (dedup, cleaning, sampling, PII pseudonymisation)
  - Exporters and the diagnostic probe
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReaderConfig(BaseModel):
    """One input source in the pipeline YAML (`readers:` list).

    `type` selects the reader: "jsonl", "json", "csv", "parquet",
    "huggingface", or "pdf". `path` is the file path for file-based
    readers and the Hub dataset name for the huggingface reader.
    Prefixed field groups (`json_*`, `csv_*`, `parquet_*`, `hf_*`, and the
    chunking/`output_mode` fields for PDF) apply only to their reader type
    and are ignored by the others.

    `format` controls column detection ("auto" samples
    `detection_sample_size` rows), `field_mapping` renames columns before
    detection, and `preprocessing_fn` is a dotted import path
    (e.g. "mymodule.my_fn") applied to each raw record.
    """

    # ---- Source type ----
    type: Literal["jsonl", "json", "csv", "parquet", "huggingface", "pdf"]

    # ---- Common to all file-based readers ----
    path: Path | None = None

    # ---- Format detection ----
    format: Literal[
        "auto",
        "alpaca",
        "sharegpt",
        "preference",
        "implicit_preference",
        "unpaired_preference",
        "grpo",
        "prompt_only",
        "pretrain",
    ] = "auto"

    # ---- Column remapping (applied before detection) ----
    field_mapping: dict[str, str] = Field(default_factory=dict)

    # ---- Optional user preprocessing function ----
    preprocessing_fn: str | None = None

    # ---- Detection tuning ----
    detection_sample_size: int = 10

    # ---- source_uri override for provenance ----
    source_uri: str | None = None

    # =========================================================
    # JSONL-specific
    # =========================================================
    # (none beyond common fields)

    # =========================================================
    # JSON-specific
    # =========================================================
    json_data_key: str | None = None

    # =========================================================
    # CSV-specific
    # =========================================================
    csv_delimiter: str | None = None
    csv_parse_json_cells: bool = True

    # =========================================================
    # Parquet-specific
    # =========================================================
    parquet_columns: list[str] | None = None
    parquet_batch_size: int = 1000

    # =========================================================
    # HuggingFace-specific
    # =========================================================
    hf_split: str = "train"
    hf_subset: str | None = None
    hf_streaming: bool = False
    hf_token: str | None = None
    hf_columns: list[str] | None = None

    # =========================================================
    # PDF fields
    # =========================================================
    chunk_strategy: Literal["heading", "sentence", "fixed"] = "heading"
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    extract_tables: bool = False
    ocr: bool = False
    min_section_tokens: int = 30

    # ---- PDF output mode (non-"chunk" modes trigger LLM generation) ----
    output_mode: Literal["chunk", "qa", "preference", "grpo", "multiturn"] = "chunk"
    llm_model: str | None = None  # per-reader LLM override


# =========================================================================
# LLM configuration
# =========================================================================


class LLMConfig(BaseModel):
    """Global LLM configuration shared across generation tasks and gates."""

    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str | None = None
    api_base: str | None = None
    concurrency: int = 10
    timeout: float = 120.0
    max_retries: int = 3
    drop_params: bool = True
    extra_body: dict = Field(default_factory=dict)


# =========================================================================
# Diagnostic probe configuration
# =========================================================================


class DiagnosticConfig(BaseModel):
    """Configuration for the DiagnosticProbe diagnostic loop.

    When `enable_probe` is True and a HallucinationGate is configured, every
    rejected sample is run through DiagnosticProbe to classify the failure
    mode. Diagnoses are attached to RejectedSample.diagnosis and a summary
    is written to diagnostic_summary.json.
    """

    enable_probe: bool = False
    probe_temperatures: list[float] = Field(default_factory=lambda: [0.3, 0.5])
    # Override generator model for re-generation in the probe.
    # If None, uses the gate's LLM model.
    probe_generator_model: str | None = None
    # Grounding score threshold below which the probe routes to strict-grounding first.
    score_split: float = 0.5
    # Extra prompt templates merged with built-ins; keys used as template names in probe routing.
    # Must have {source} and {question} placeholders. Override built-ins by using their keys:
    # strict_grounding | domain_specific | generate_question | default
    extra_templates: dict[str, str] = Field(default_factory=dict)


# =========================================================================
# Generation task configuration
# =========================================================================


class GenerationConfig(BaseModel):
    """Configuration for LLM-powered generation tasks."""

    type: Literal[
        "qa",
        "evol_instruct",
        "preference",
        "grpo",
        "multiturn",
        "cot",
        "adversarial_preference",
        "adversarial_qa",
    ]

    # ---- QA-specific ----
    num_questions: int = 3
    difficulty: str = "medium"
    table_prompt_template: str | None = None  # separate prompt for table chunks

    # ---- EvolInstruct-specific ----
    num_evolutions: int = 1
    strategies: list[str] = Field(
        default_factory=lambda: [
            "add_constraints",
            "deepen",
            "concretize",
            "increase_reasoning",
            "broaden",
        ]
    )
    generate_answers: bool = True
    answer_prompt_template: str | None = None  # separate prompt for the answer pass

    # ---- Preference-specific ----
    preference_mode: Literal["single_call", "two_pass"] = "single_call"
    chosen_prompt_template: str | None = None  # two_pass: prompt for chosen generation
    rejected_prompt_template: str | None = None  # two_pass: prompt for rejected generation

    # ---- GRPO-specific ----
    num_responses: int = 4
    score_responses: bool = True
    temperature_spread: float = 0.6

    # ---- MultiTurn-specific ----
    num_turns: int = 3
    include_context: bool = True

    # ---- CoT-specific ----
    cot_mode: Literal["generate", "wrap"] = "generate"
    cot_marker: str | None = None  # separator string between reasoning and answer

    # ---- Custom prompts (override defaults) ----
    prompt_template: str | None = None

    # ---- Adversarial-specific (adversarial_preference and adversarial_qa) ----
    injection_rate: float = 0.5
    injection_types: list[str] = Field(default_factory=list)  # empty = all types
    injection_seed: int = 42
    adversarial_prompt_template: str | None = None
    high_temp: float = 1.4  # adversarial_qa: temperature for high_temperature_drift injection

    # ---- Gate prompt overrides ----
    hallucination_prompt_template: str | None = None
    reward_prompt_template: str | None = None

    # ---- LLM override (uses global LLM config if None) ----
    llm_model: str | None = None


# =========================================================================
# Gate configurations
# =========================================================================


class GateConfig(BaseModel):
    """One quality gate in the pipeline YAML (`gates:` list).

    `type` selects the gate: "schema", "hallucination", "reward",
    "diversity", "secrets", or "toxicity". Prefixed field groups apply only
    to the matching gate type (`hallucination_*`, `reward_*`, `secrets_*`,
    `toxicity_*`; the embedding/similarity fields belong to the diversity
    gate, and `required_fields`/`min_tokens`/`max_tokens` to the schema
    gate) — fields for other types are ignored.

    The hallucination and reward gates call an LLM judge; they use the
    pipeline's global `llm:` config unless overridden via
    `hallucination_llm_model` / `reward_llm_model`.
    """

    type: Literal["schema", "hallucination", "reward", "diversity", "secrets", "toxicity"]

    # ---- Schema gate ----
    required_fields: list[str] = Field(default_factory=lambda: ["instruction", "output"])
    min_tokens: int = 10
    max_tokens: int = 2048
    use_tiktoken: bool = False
    enforce_task_types: list[str] = Field(default_factory=list)

    # ---- Hallucination gate ----
    hallucination_threshold: float = 0.7
    skip_if_no_context: bool = True
    hallucination_llm_model: str | None = None  # override global LLM
    hallucination_prompt_template: str | None = None

    # ---- Reward gate ----
    reward_threshold: float = 0.7
    reward_dimensions: list[str] = Field(
        default_factory=lambda: ["helpfulness", "honesty", "instruction_following"]
    )
    store_score_in_label: bool = True
    reward_llm_model: str | None = None  # override global LLM
    reward_prompt_template: str | None = None

    # ---- Diversity gate ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.92
    diversity_text_field: str = "auto"
    coverage_field: str | None = None
    embedding_device: str | None = None  # "cuda" | "cpu" | "mps" | None (auto)
    embedding_batch_size: int = 64

    # ---- SecretsGate (data hygiene) ----
    secrets_code_corpus_mode: bool = False
    secrets_fields: list[str] = Field(default_factory=list)
    secrets_hex_limit: float = 3.0
    secrets_base64_limit: float = 4.5

    # ---- ToxicityGate (data hygiene) ----
    toxicity_classifier_pass_threshold: float = 0.1
    toxicity_classifier_reject_threshold: float = 0.5
    toxicity_detoxify_model: str = "unbiased"
    toxicity_llm_model: str | None = None
    toxicity_llm_reject_threshold: float = 0.5
    toxicity_text_field: str = "auto"


# =========================================================================
# Normalizer configurations
# =========================================================================


class NormalizerConfig(BaseModel):
    """One normalizer in the pipeline YAML (`normalizers:` list).

    `type` selects the step: "exact_dedup", "minhash_dedup",
    "text_cleaner", "stratified_sampler", "embedding_dedup", or
    "pii_pseudonymizer". Prefixed field groups (`minhash_*`,
    `embedding_*`, `pii_*`) and the `transforms`/`clean_fields`
    (text_cleaner) and `category_field`/`target_distribution`/
    `sampler_seed` (stratified_sampler) fields apply only to the matching
    type — fields for other types are ignored.

    Normalizers transform or drop samples without LLM calls; embedding
    dedup persists its index in `embedding_index_dir` so duplicates are
    caught across runs.
    """

    type: Literal[
        "exact_dedup",
        "minhash_dedup",
        "text_cleaner",
        "stratified_sampler",
        "embedding_dedup",
        "pii_pseudonymizer",
    ]

    # MinHash-specific
    minhash_threshold: float = 0.85
    minhash_ngram: int = 3
    minhash_num_perm: int = 128
    minhash_seed: int = 42

    # TextCleaner-specific
    transforms: dict[str, bool] = Field(
        default_factory=lambda: {
            "strip_html": True,
            "normalise_unicode": True,
            "fix_encoding_artifacts": True,
            "collapse_whitespace": True,
            "remove_control_chars": True,
        }
    )
    clean_fields: list[str] = Field(default_factory=list)  # empty = default fields

    # StratifiedSampler-specific
    category_field: str = "source_dataset"
    target_distribution: dict[str, float] = Field(default_factory=dict)
    sampler_seed: int = 42

    # ---- Embedding dedup ----
    embedding_index_dir: str = "output/embedding_index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_threshold: float = 0.92
    embedding_text_field: str = "auto"
    embedding_device: str | None = None  # "cuda" | "cpu" | "mps" | None (auto)
    embedding_batch_size: int = 64

    # ---- PIIPseudonymizer (data hygiene) ----
    pii_entity_types: list[str] = Field(default_factory=list)
    pii_score_threshold: float = 0.7
    pii_spacy_model: str = "en_core_web_lg"
    pii_faker_seed: int = 42
    pii_language: str = "en"
    pii_fields: list[str] = Field(default_factory=list)


class ExporterConfig(BaseModel):
    """One output format in the pipeline YAML (`exporters:` list).

    `type` selects the exporter: "alpaca", "sharegpt", "grpo", "ppo",
    "dpo", or "corpus". Each exporter writes accepted samples to its own
    JSONL file in the output directory.
    """

    type: Literal["alpaca", "sharegpt", "grpo", "ppo", "dpo", "corpus"]


class PipelineConfig(BaseModel):
    """Root model for a pipeline YAML file.

    Maps the top-level YAML keys to typed sub-configs: `readers`, `gates`,
    `normalizers`, `exporters` (lists, run in order within each stage),
    plus the optional `llm:` block (global LLM settings), `generators:`
    (LLM generation tasks), and `diagnostic:` (failure-mode probe attached
    to the hallucination gate). `max_samples` caps the total sample count
    after reading, and `output_split` shuffles accepted samples into split
    subdirectories (fractions must sum to 1.0). Unrecognised settings can
    be stashed in `extra`.

    Build one with `PipelineConfig.from_yaml(path)` or directly via
    keyword arguments.
    """

    name: str = "curatorkit_pipeline"
    version: str = "0.2.0"
    readers: list[ReaderConfig] = Field(default_factory=list)
    gates: list[GateConfig] = Field(default_factory=list)
    normalizers: list[NormalizerConfig] = Field(default_factory=list)
    exporters: list[ExporterConfig] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    # ---- Cap on total sample count, applied right after readers ----
    max_samples: int | None = None

    # ---- Output split: shuffle accepted samples and write to split subdirs ----
    # Fractions must sum to 1.0. None = single unsplit output (default).
    # Example: {train: 0.8, val: 0.1, test: 0.1}
    output_split: dict[str, float] | None = None
    output_split_seed: int = 42  # seed for the pre-split shuffle

    # ---- Global LLM config ----
    llm: LLMConfig | None = None

    # ---- Generation tasks ----
    generators: list[GenerationConfig] = Field(default_factory=list)

    # ---- Diagnostic probe (attached to HallucinationGate) ----
    diagnostic: DiagnosticConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load and validate a pipeline YAML file.

        Parses the file with `yaml.safe_load` and validates it eagerly
        against this model, so a malformed config raises
        `pydantic.ValidationError` before any pipeline step runs.
        """
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
