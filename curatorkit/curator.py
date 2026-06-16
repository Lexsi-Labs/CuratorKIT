"""
curator.py — The unified CuratorKIT entry point.

Mirrors the TRL / Unsloth trainer pattern:

    config = CuratorConfig(dataset="Anthropic/hh-rlhf")
    curator = Curator(config)
    curator.run()

LLM-powered features:
    - Synthetic generation (qa, preference, grpo, multiturn, evol, cot)
    - Quality gates (hallucination, reward, diversity)
    - Cross-run embedding deduplication
    - Async pipeline execution via curator.run_async()

CuratorConfig holds everything. Curator.run() does everything.
No reader objects, no pipeline construction, no step wiring — all of that
happens internally based on what you put in the config.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running.

    Jupyter kernels run a persistent event loop, so asyncio.run() raises
    RuntimeError there. nest_asyncio patches the loop to allow nesting.
    """
    try:
        asyncio.get_running_loop()
        # Inside a running loop (Jupyter, IPython, FastAPI, etc.)
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            raise RuntimeError(
                "Curator.run() was called from inside a running event loop "
                "(e.g. Jupyter). Install nest_asyncio: pip install nest_asyncio"
            ) from None
    except RuntimeError as exc:
        if "no running event loop" not in str(exc) and "no current event loop" not in str(exc):
            raise
    return asyncio.run(coro)


try:
    from curatorkit.diagnostic.diagnostics import PipelineDiagnostics
except ImportError:
    PipelineDiagnostics = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# CuratorConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CuratorConfig:
    """
    All configuration for a CuratorKIT pipeline in one place.

    Minimal usage — just the dataset name:
        config = CuratorConfig(dataset="Anthropic/hh-rlhf")

    LLM generation usage:
        config = CuratorConfig(
            dataset="docs/handbook.pdf",
            llm_model="openai/gpt-4o-mini",
            generation_task="qa",
        )

    The rest has sensible defaults. Override only what you need.

    ── Source ──────────────────────────────────────────────────────────────────
    dataset         HF Hub name, local file path, or list of either.
                    Supported file types: .jsonl  .json  .csv  .tsv  .parquet  .pdf
                    Examples:
                        "Anthropic/hh-rlhf"
                        "data/my_data.jsonl"
                        "docs/handbook.pdf"
                        ["tatsu-lab/alpaca", "data/extra.jsonl"]

    ── LLM configuration ───────────────────────────────────────────────────────
    llm_model       LiteLLM model string. Default None (no generation).
                    Examples: "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-20250514",
                              "ollama/llama3"
    llm_temperature Temperature for generation. Default 0.7.
    llm_max_tokens  Max tokens per LLM call. Default 1024.
    llm_api_key     API key override. Falls back to env var.
    llm_api_base    Custom API base URL (for vLLM, Ollama, etc.)
    llm_concurrency Async worker pool size. Default 10.

    ── LLM generation ──────────────────────────────────────────────────────────
    generation_task Task type: "qa" | "preference" | "grpo" | "multiturn"
                    | "evol" | "cot" | None (no generation).
    num_questions   QA: questions per chunk. Default 3.
    num_responses   GRPO: responses per prompt. Default 4.
    num_turns       MultiTurn: turn pairs. Default 3.
    num_evolutions  EvolInstruct: evolutions per instruction. Default 1.
    difficulty      QA: "easy" | "medium" | "hard". Default "medium".

    ── Quality gates ───────────────────────────────────────────────────────────
    hallucination_threshold  Grounding score threshold. None = gate off (default).
                             Set a float (e.g. 0.7) to enable the hallucination gate.
    reward_threshold    Drop samples below this quality score. Default None (off).
    reward_dimensions   Quality dimensions to evaluate.
    diversity_threshold Drop samples above this similarity. Default None (off).

    ── Cross-run dedup ─────────────────────────────────────────────────────────
    embedding_dedup     Enable cross-run embedding dedup. Default False.
    embedding_index_dir Directory for persistent index. Default "output/embedding_index".
    embedding_threshold Similarity threshold. Default 0.92.
    """

    # ── Source ──────────────────────────────────────────────────────────────
    # dataset accepts:
    #   "name"                                    — string shorthand (global split/subset)
    #   {"name": "...", "split": "...", "subset": "..."}  — per-source overrides
    #   list of either of the above
    dataset: str | dict | list = ""
    split: str = "train"
    subset: str | None = None
    streaming: bool = False
    hf_token: str | None = None
    hf_subset: str | None = None
    hf_columns: list[str] | None = None
    max_samples: int | None = None

    # ── Column mapping ──────────────────────────────────────────────────────
    format: str = "auto"
    field_mapping: dict[str, str] = field(default_factory=dict)
    preprocessing_fn: Callable | list[Callable | None] | None = None

    # ── Quality filters ─────────────────────────────────────────────────────
    min_tokens: int = 10
    max_tokens: int = 2048
    use_tiktoken: bool = False
    schema_use_tiktoken: bool = False
    schema_enforce_task_types: list[str] = field(default_factory=list)
    schema_gate: bool = True
    schema_required_fields: list[str] = field(default_factory=list)  # empty = auto per task_type

    # ── Deduplication ───────────────────────────────────────────────────────
    dedup: str = "exact"  # "exact" | "minhash" | "none"
    minhash_threshold: float = 0.85
    minhash_ngram: int = 3
    minhash_num_perm: int = 128
    minhash_seed: int = 42

    # ── Text cleaning ───────────────────────────────────────────────────────
    clean: bool = True
    clean_transforms: dict[str, bool] = field(default_factory=dict)
    clean_fields: list[str] = field(
        default_factory=list
    )  # empty = default [instruction, input, output]

    # ── Export ──────────────────────────────────────────────────────────────
    output_dir: str | Path = "output"
    export_formats: list[str] = field(default_factory=lambda: ["alpaca", "sharegpt", "dpo"])
    # output_split: if set, the accepted samples are split before export.
    # Fractions must sum to 1.0. None = single unsplit output (default).
    # Example: {"train": 0.8, "val": 0.1, "test": 0.1}
    output_split: dict[str, float] | None = None
    output_split_seed: int = 42  # seed for the pre-split shuffle

    # ── Misc ────────────────────────────────────────────────────────────────
    name: str = "curatorkit_run"
    detection_sample_size: int = 10

    # ── Resampling ──────────────────────────────────────────────────────────
    resample: bool = False
    target_distribution: dict[str, float] = field(default_factory=dict)
    resample_field: str = "source_dataset"
    resample_seed: int = 42

    # ═══════════════════════════════════════════════════════════════════════
    # LLM configuration
    # ═══════════════════════════════════════════════════════════════════════
    llm_model: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_api_key: str | None = None
    llm_api_base: str | None = None
    llm_concurrency: int = 10
    llm_timeout: float = 120.0
    llm_max_retries: int = 3
    llm_drop_params: bool = True  # silently drop unsupported params per provider
    llm_extra_body: dict = field(
        default_factory=dict
    )  # e.g. {"chat_template_kwargs": {"enable_thinking": False}}

    # Separate judge model for hallucination/reward gates.
    # If None, falls back to llm_model. Use a stronger/different model to
    # avoid same-model leniency bias when generator == judge.
    judge_llm_model: str | None = None
    judge_llm_api_base: str | None = None
    judge_llm_temperature: float = 0.1  # low-temp for deterministic judgements
    judge_llm_max_tokens: int = 512
    judge_llm_timeout: float = 120.0
    judge_llm_max_retries: int = 3
    judge_llm_extra_body: dict = field(default_factory=dict)

    # ── Generation task ─────────────────────────────────────────────────────
    generation_task: str | None = (
        None  # qa | preference | grpo | multiturn | evol | cot | adversarial_preference
    )
    num_questions: int = 3
    num_responses: int = 4
    num_turns: int = 3
    num_evolutions: int = 1
    difficulty: str = "medium"
    score_responses: bool = True
    generate_answers: bool = True
    cot_mode: str = "generate"  # generate | wrap
    preference_mode: str = "single_call"  # single_call | two_pass

    # ── Per-task concurrency overrides (None = use llm_concurrency for all) ──
    generation_concurrency: int | None = None  # overrides llm_concurrency for generation task
    judge_concurrency: int | None = None  # overrides llm_concurrency for gates

    # ── GRPO-specific ────────────────────────────────────────────────────────
    grpo_temperature_spread: float = 0.0
    grpo_temperatures: list[float] | None = None  # explicit per-rollout temps; overrides spread
    grpo_scoring_llm_model: str | None = None  # separate scoring LLM; None = use llm_model

    # ── Per-task prompt templates (None = use built-in default) ─────────────
    qa_prompt_template: str | None = None
    qa_table_prompt_template: str | None = None  # separate prompt for table chunks
    evol_prompt_template: str | None = None
    evol_answer_prompt_template: str | None = None  # separate prompt for the answer pass
    preference_prompt_template: str | None = None
    preference_chosen_prompt: str | None = None  # two_pass: prompt for chosen generation
    preference_rejected_prompt: str | None = None  # two_pass: prompt for rejected generation
    grpo_prompt_template: str | None = None
    multiturn_prompt_template: str | None = None
    cot_prompt_template: str | None = None
    cot_marker: str | None = None  # separator string between reasoning and answer
    adversarial_prompt_template: str | None = None  # for adversarial_preference task

    # ── Adversarial generation settings ─────────────────────────────────────
    injection_rate: float = 0.5  # adversarial_preference: fraction of pairs to inject
    injection_types: list[str] = field(default_factory=list)  # empty = all types
    injection_seed: int = 42
    high_temp: float = 1.4  # adversarial_qa: temperature for high_temperature_drift

    # ── Quality gates ───────────────────────────────────────────────────────
    hallucination_threshold: float | None = None  # None = off; set a float to enable
    hallucination_prompt_template: str | None = None  # custom CheckEval judge prompt
    reward_threshold: float | None = None
    reward_dimensions: list[str] = field(
        default_factory=lambda: ["helpfulness", "honesty", "instruction_following"]
    )
    reward_prompt_template: str | None = None  # custom UltraFeedback judge prompt
    reward_store_score: bool = True  # write overall_score to DataSample.label
    diversity_threshold: float | None = None

    # ── Inline recovery ──────────────────────────────────────────────────────
    enable_reward_refiner: bool = False  # run RewardRefiner on RewardGate rejects
    reward_refine_prompt_template: str | None = None
    reward_instruction_refine_template: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    diversity_embedding_model: str | None = None  # overrides embedding_model for DiversityGate
    embedding_dedup_model: str | None = None  # overrides embedding_model for EmbeddingDeduplicator
    embedding_device: str | None = None  # "cuda" | "cpu" | "mps" | None (auto-detect)
    embedding_batch_size: int = 64  # batch size for DiversityGate + EmbeddingDedup

    # ── Cross-run embedding dedup ───────────────────────────────────────────
    embedding_dedup: bool = False
    embedding_index_dir: str = "output/embedding_index"
    embedding_dedup_threshold: float = 0.92
    embedding_reset_index: bool = False

    # ── PDF extraction ───────────────────────────────────────────────────────
    pdf_output_mode: str = "chunk"  # chunk | qa | preference | grpo | multiturn
    pdf_chunk_strategy: str = "heading"  # heading | sentence | fixed
    pdf_chunk_max_tokens: int = 512
    pdf_chunk_overlap_tokens: int = 50
    pdf_extract_tables: bool = False
    pdf_ocr: bool = False
    pdf_min_section_tokens: int = 30  # heading strategy: merge sections shorter than this

    # ── Diagnostic probe ─────────────────────────────────────────────────────
    enable_diagnostic_probe: bool = False
    probe_temperatures: list[float] = field(default_factory=lambda: [0.3, 0.5])
    probe_generator_model: str | None = None  # if None, uses llm_model
    probe_score_split: float = 0.5
    # Extra probe templates: merged with built-in PROMPT_TEMPLATES; can also override built-ins.
    # Keys: any string used as template name; values: prompt strings with {source}/{question} placeholders.
    probe_extra_templates: dict[str, str] = field(default_factory=dict)
    llm_prompt_template: str | None = None

    # ── Data hygiene ─────────────────────────────────────────────────────────
    # All three run before generation so the LLM never sees raw credentials
    # or PII, and toxic source material is discarded before any API cost is incurred.
    #
    # secrets_gate             Reject samples containing API keys or credentials.
    # secrets_code_corpus_mode Enable KeywordDetector for code corpora. Default False.
    # secrets_fields           Fields to scan. [] = task-aware default (all relevant fields).
    # secrets_hex_limit        Shannon entropy threshold for hex strings. Default 3.0.
    #                          Raise to 4.5+ for prose corpora to reduce false positives.
    # secrets_base64_limit     Shannon entropy threshold for base64 strings. Default 4.5.
    #                          Raise to 5.5+ for prose corpora to reduce false positives.
    # pii_pseudonymize         Replace PII with consistent fake values (Presidio+Faker).
    # pii_entity_types         Presidio entity types. [] = default (no DATE_TIME).
    #                          Pass ENTITY_TYPES_CLINICAL for medical/legal corpora.
    # pii_fields               Fields to pseudonymize. [] = task-aware default.
    # pii_score_threshold      Presidio confidence threshold. Default 0.7.
    # pii_spacy_model          spaCy model. Default "en_core_web_lg" (~800 MB).
    # pii_faker_seed           Reproducibility seed for Faker. Default 42.
    # pii_language             Presidio analysis language. Default "en".
    # toxicity_gate            Reject samples with toxic content (Detoxify classifier).
    # toxicity_classifier_pass_threshold    Below this → pass immediately. Default 0.1.
    # toxicity_classifier_reject_threshold  Above this → reject immediately. Default 0.5.
    # toxicity_detoxify_model  "unbiased" | "original" | "multilingual". Default "unbiased".
    # toxicity_llm_judge       Enable LLM second-opinion for borderline band. Default False.
    # toxicity_llm_reject_threshold  LLM judge rejection threshold for borderline band. Default 0.5.
    # toxicity_text_field      Field to score. "auto" = task-aware; or a specific field name.
    secrets_gate: bool = False
    secrets_code_corpus_mode: bool = False
    secrets_fields: list[str] = field(default_factory=list)
    secrets_hex_limit: float = 3.0
    secrets_base64_limit: float = 4.5
    pii_pseudonymize: bool = False
    pii_entity_types: list[str] = field(default_factory=list)
    pii_fields: list[str] = field(default_factory=list)
    pii_score_threshold: float = 0.7
    pii_spacy_model: str = "en_core_web_lg"
    pii_faker_seed: int = 42
    pii_language: str = "en"
    toxicity_gate: bool = False
    toxicity_classifier_pass_threshold: float = 0.1
    toxicity_classifier_reject_threshold: float = 0.5
    toxicity_detoxify_model: str = "unbiased"
    toxicity_llm_judge: bool = False
    toxicity_llm_reject_threshold: float = 0.5
    toxicity_text_field: str = "auto"

    def apply_patch(self, patch: dict) -> CuratorConfig:
        """
        Return a copy of this config with patch fields applied.

        Supported patch keys:
          llm_temperature  → self.llm_temperature
          prompt_template  → self.llm_prompt_template

        Returns a new CuratorConfig — the original is unchanged.
        """
        import copy as _copy

        patched = _copy.copy(self)

        for key, value in patch.items():
            if key == "llm_temperature":
                patched.llm_temperature = value
            elif key == "prompt_template":
                patched.llm_prompt_template = value
        return patched


# ─────────────────────────────────────────────────────────────────────────────
# CuratorResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CuratorResult:
    """
    Everything the pipeline produced, in one object.

    passed          — list of DataSample objects that cleared every filter
    rejected        — list of RejectedSample objects with structured reasons
    stage_counts    — per-step input/output/rejected counts
    output_dir      — where the files were written
    wall_clock_seconds — total run time
    diagnostics     — PipelineDiagnostics when enable_diagnostic_probe=True, else None
    """

    passed: list
    rejected: list
    stage_counts: dict[str, dict[str, int]]
    output_dir: Path
    wall_clock_seconds: float
    diagnostics: Any = None

    def summary(self) -> str:
        lines = [
            f"{'─' * 44}",
            f"  passed   : {len(self.passed):>8,}",
            f"  rejected : {len(self.rejected):>8,}",
            f"  time     : {self.wall_clock_seconds:>7.1f}s",
            f"  output   : {self.output_dir}",
            f"{'─' * 44}",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def sample(self, n: int = 3) -> None:
        """Print the first n passed samples."""
        for i, s in enumerate(self.passed[:n]):
            print(f"\n── Sample {i + 1} ({s.task_type}) ──")
            if s.instruction:
                print(f"  instruction : {s.instruction[:120]!r}")
            if s.output:
                print(f"  output      : {s.output[:120]!r}")
            if s.chosen:
                print(f"  chosen      : {s.chosen[:120]!r}")
            if s.rejected:
                print(f"  rejected    : {s.rejected[:120]!r}")
            if s.responses:
                print(f"  responses   : {len(s.responses)} rollouts")


# ─────────────────────────────────────────────────────────────────────────────
# _CappedReader — wraps any reader to cap its own output before pipeline concat
# ─────────────────────────────────────────────────────────────────────────────

from curatorkit.interfaces import BaseReader as _BaseReader


class _CappedReader(_BaseReader):
    """Wraps a reader and caps its output to max_samples during read().

    Inherits BaseReader so isinstance checks in the pipeline pass correctly.
    The cap applies only to this reader's own output, not to the accumulated
    list from all previous readers.
    """

    def __init__(self, reader: _BaseReader, max_samples: int) -> None:
        self._reader = reader
        self._max = max_samples
        self._display_name = type(reader).__name__

    def read(self):
        samples, rejected = self._reader.read()
        return samples[: self._max], rejected


# ─────────────────────────────────────────────────────────────────────────────
# Curator
# ─────────────────────────────────────────────────────────────────────────────


class Curator:
    """
    CuratorKIT pipeline runner.

    Curation-only usage (no LLM):
        curator = Curator(CuratorConfig(dataset="Anthropic/hh-rlhf"))
        result  = curator.run()

    LLM generation usage:
        curator = Curator(CuratorConfig(
            dataset="docs/handbook.pdf",
            llm_model="openai/gpt-4o-mini",
            generation_task="qa",
            hallucination_threshold=0.7,
            reward_threshold=0.7,
        ))
        result = curator.run()            # sync
        result = await curator.run_async() # async (faster for many LLM calls)
    """

    def __init__(self, config: CuratorConfig) -> None:
        self.config = config
        self._diagnostics = None
        self._reward_refiner = None

    def dry_run(self) -> list[dict[str, str]]:
        """
        Build the step list from CuratorConfig and print the plan without running.

        Returns the same plan list that Pipeline.dry_run() returns so callers
        can diff plans across config changes.
        """
        from curatorkit.pipeline import Pipeline

        steps = self._build_steps()
        output_dir = Path(self.config.output_dir)
        pipeline = Pipeline(steps, output_dir=output_dir, diagnostics=self._diagnostics)
        return pipeline.dry_run()

    def run(self) -> CuratorResult:
        """Synchronous pipeline execution.

        Automatically uses async execution (and therefore concurrent LLM calls)
        when a generation task is configured.  Callers never need to reach for
        run_async() themselves.
        """
        from curatorkit.pipeline import Pipeline

        splitting = bool(self.config.output_split)
        steps = self._build_steps(include_exporters=not splitting)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = Pipeline(steps, output_dir=output_dir, diagnostics=self._diagnostics)
        if self.config.generation_task:
            result = _run_async(pipeline.run_async())
        else:
            result = pipeline.run()

        if splitting:
            self._export_splits(result.passed, output_dir)

        if self._reward_refiner is not None:
            reward_rejects = [r for r in result.rejected if r.rejecting_step == "RewardGate"]
            if reward_rejects:
                recovered, still_rejected = self._reward_refiner.refine(reward_rejects)
                result.passed.extend(recovered)
                # Replace the original reward rejects with still-rejected ones
                result.rejected = [
                    r for r in result.rejected if r.rejecting_step != "RewardGate"
                ] + still_rejected

        self._write_provenance(result, output_dir)

        if result.diagnostics is not None:
            result.diagnostics.write_summary(output_dir / "diagnostic_summary.json")

        return CuratorResult(
            passed=result.passed,
            rejected=result.rejected,
            stage_counts=result.stage_counts,
            output_dir=output_dir,
            wall_clock_seconds=result.wall_clock_seconds,
            diagnostics=result.diagnostics,
        )

    async def run_async(self) -> CuratorResult:
        """Async pipeline execution — faster for generation-heavy pipelines."""
        from curatorkit.pipeline import Pipeline

        splitting = bool(self.config.output_split)
        steps = self._build_steps(include_exporters=not splitting)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = Pipeline(steps, output_dir=output_dir, diagnostics=self._diagnostics)
        result = await pipeline.run_async()

        if splitting:
            self._export_splits(result.passed, output_dir)

        self._write_provenance(result, output_dir)

        if result.diagnostics is not None:
            result.diagnostics.write_summary(output_dir / "diagnostic_summary.json")

        return CuratorResult(
            passed=result.passed,
            rejected=result.rejected,
            stage_counts=result.stage_counts,
            output_dir=output_dir,
            wall_clock_seconds=result.wall_clock_seconds,
            diagnostics=result.diagnostics,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Internal step builder
    # ═════════════════════════════════════════════════════════════════════════

    def _build_steps(self, include_exporters: bool = True) -> list:
        from curatorkit.exporters.alpaca import AlpacaExporter
        from curatorkit.exporters.corpus import CorpusExporter
        from curatorkit.exporters.dpo import DPOExporter
        from curatorkit.exporters.grpo import GRPOExporter
        from curatorkit.exporters.ppo import PPOExporter
        from curatorkit.exporters.sharegpt import ShareGPTExporter
        from curatorkit.gates.schema import SchemaGate
        from curatorkit.normalizers.clean import TextCleaner
        from curatorkit.normalizers.dedup import (
            ExactDeduplicator,
            MinHashDeduplicator,
        )

        cfg = self.config
        steps = []

        # ── Readers ─────────────────────────────────────────────────────────
        raw_datasets = cfg.dataset if isinstance(cfg.dataset, list) else [cfg.dataset]

        # Parse each entry: str → (name, None, None, None)
        # dict → (name, split, subset, max_samples)
        # max_samples per entry caps that reader independently before concatenation.
        datasets: list[str] = []
        per_ds_split: list[str | None] = []
        per_ds_subset: list[str | None] = []
        per_ds_max_samples: list[int | None] = []
        for entry in raw_datasets:
            if isinstance(entry, dict):
                datasets.append(entry["name"])
                per_ds_split.append(entry.get("split"))
                per_ds_subset.append(entry.get("subset"))
                per_ds_max_samples.append(entry.get("max_samples"))
            else:
                datasets.append(str(entry))
                per_ds_split.append(None)
                per_ds_subset.append(None)
                per_ds_max_samples.append(None)

        fns = cfg.preprocessing_fn
        if isinstance(fns, list):
            if len(fns) != len(datasets):
                raise ValueError(
                    f"preprocessing_fn list length ({len(fns)}) does not match "
                    f"dataset list length ({len(datasets)}). Pass either a single "
                    f"callable applied to all readers, or a list of the same length."
                )
        else:
            fns = [fns] * len(datasets)

        for ds, fn, ds_split, ds_subset, ds_max in zip(
            datasets, fns, per_ds_split, per_ds_subset, per_ds_max_samples
        ):
            reader = self._make_reader(
                ds,
                preprocessing_fn=fn,
                split_override=ds_split,
                subset_override=ds_subset,
            )
            # Per-reader cap: wrap the reader so it caps its own output
            # during read(), before samples are added to the pipeline list.
            # A post-reader MaxSamplesTruncator would cut the accumulated
            # list from all previous readers, not just this reader's output.
            if ds_max is not None:
                reader = _CappedReader(reader, ds_max)
            steps.append(reader)

        # ── Schema gate ─────────────────────────────────────────────────────
        if cfg.schema_gate:
            steps.append(
                SchemaGate(
                    required_fields=cfg.schema_required_fields or None,
                    min_tokens=cfg.min_tokens,
                    max_tokens=cfg.max_tokens,
                    use_tiktoken=cfg.use_tiktoken or cfg.schema_use_tiktoken,
                    enforce_task_types=cfg.schema_enforce_task_types or None,
                )
            )

        # ── Deduplication ───────────────────────────────────────────────────
        if cfg.dedup == "exact":
            steps.append(ExactDeduplicator())
        elif cfg.dedup == "minhash":
            steps.append(ExactDeduplicator())
            steps.append(
                MinHashDeduplicator(
                    threshold=cfg.minhash_threshold,
                    ngram=cfg.minhash_ngram,
                    num_perm=cfg.minhash_num_perm,
                    seed=cfg.minhash_seed,
                )
            )

        # ── Text cleaning ───────────────────────────────────────────────────
        if cfg.clean:
            steps.append(
                TextCleaner(
                    transforms=cfg.clean_transforms or None,
                    fields=cfg.clean_fields or None,
                )
            )

        # ── Data hygiene (pre-generation) ───────────────────────────────────
        # Runs before any LLM call so credentials and PII never reach the API,
        # and toxic source material is discarded before generating expensive
        # continuations.
        if cfg.secrets_gate:
            from curatorkit.hygiene.secrets import (
                _PLUGIN_KEYWORD,
                _PLUGINS_BASE,
                SecretsGate,
            )

            plugins = [
                {**p, "limit": cfg.secrets_hex_limit}
                if p["name"] == "HexHighEntropyString"
                else {**p, "limit": cfg.secrets_base64_limit}
                if p["name"] == "Base64HighEntropyString"
                else p
                for p in _PLUGINS_BASE
            ]
            if cfg.secrets_code_corpus_mode:
                plugins.append(_PLUGIN_KEYWORD)
            steps.append(
                SecretsGate(
                    fields=cfg.secrets_fields or None,
                    plugins=plugins,
                )
            )

        if cfg.pii_pseudonymize:
            from curatorkit.hygiene.pii import PIIPseudonymizer

            steps.append(
                PIIPseudonymizer(
                    entity_types=cfg.pii_entity_types or None,
                    fields=cfg.pii_fields or None,
                    score_threshold=cfg.pii_score_threshold,
                    spacy_model=cfg.pii_spacy_model,
                    faker_seed=cfg.pii_faker_seed,
                    language=cfg.pii_language,
                )
            )

        if cfg.toxicity_gate:
            from curatorkit.hygiene.toxicity import ToxicityGate

            _tox_llm = self._build_llm() if cfg.toxicity_llm_judge and cfg.llm_model else None
            steps.append(
                ToxicityGate(
                    classifier_pass_threshold=cfg.toxicity_classifier_pass_threshold,
                    classifier_reject_threshold=cfg.toxicity_classifier_reject_threshold,
                    detoxify_model=cfg.toxicity_detoxify_model,
                    llm=_tox_llm,
                    llm_reject_threshold=cfg.toxicity_llm_reject_threshold,
                    text_field=cfg.toxicity_text_field,
                )
            )

        # ═════════════════════════════════════════════════════════════════════
        # Generation task
        # ═════════════════════════════════════════════════════════════════════
        if cfg.generation_task and cfg.llm_model:
            llm = self._build_llm()
            gen_step = self._build_generator(llm)
            if gen_step is not None:
                steps.append(gen_step)

        # ═════════════════════════════════════════════════════════════════════
        # Quality gates (after generation)
        # ═════════════════════════════════════════════════════════════════════
        if cfg.hallucination_threshold is not None and cfg.llm_model:
            from curatorkit.gates.hallucination import HallucinationGate

            llm = self._build_judge_llm()
            _jconcurrency = cfg.judge_concurrency or cfg.llm_concurrency
            gate = HallucinationGate(
                llm=llm,
                threshold=cfg.hallucination_threshold,
                prompt_template=cfg.hallucination_prompt_template,
                concurrency=_jconcurrency,
            )

            if cfg.enable_diagnostic_probe:
                from curatorkit.diagnostic.probe import DiagnosticProbe

                probe_llm = self._build_probe_llm()
                if cfg.probe_generator_model and cfg.probe_generator_model != cfg.llm_model:
                    probe_llm.model = cfg.probe_generator_model
                gate.probe = DiagnosticProbe(
                    generator_llm=probe_llm,
                    gate=gate,
                    temperatures=cfg.probe_temperatures,
                    score_split=cfg.probe_score_split,
                    extra_templates=cfg.probe_extra_templates or None,
                )
                if PipelineDiagnostics is not None:
                    self._diagnostics = PipelineDiagnostics()

            steps.append(gate)

        if cfg.reward_threshold is not None and cfg.llm_model:
            from curatorkit.gates.reward import RewardGate

            llm = self._build_judge_llm()
            _reward_gate = RewardGate(
                llm=llm,
                threshold=cfg.reward_threshold,
                dimensions=cfg.reward_dimensions,
                prompt_template=cfg.reward_prompt_template,
                store_score_in_label=cfg.reward_store_score,
                concurrency=cfg.judge_concurrency or cfg.llm_concurrency,
            )
            # Attach DiagnosticProbe to RewardGate when probe is enabled
            if cfg.enable_diagnostic_probe:
                from curatorkit.diagnostic.probe import DiagnosticProbe

                probe_llm = self._build_probe_llm()
                _reward_gate.probe = DiagnosticProbe(
                    generator_llm=probe_llm,
                    gate=_reward_gate,
                    temperatures=cfg.probe_temperatures,
                    score_split=cfg.probe_score_split,
                    extra_templates=cfg.probe_extra_templates or None,
                )

            steps.append(_reward_gate)

            if cfg.enable_reward_refiner and cfg.llm_model:
                # Store on instance so run() can call refiner.refine(reward_rejects) post-pipeline
                from curatorkit.diagnostic.reward_refine import RewardRefiner

                refiner_llm = self._build_probe_llm()
                self._reward_refiner = RewardRefiner(
                    generator_llm=refiner_llm,
                    reward_gate=_reward_gate,
                    refine_prompt_template=cfg.reward_refine_prompt_template,
                    instruction_refine_template=cfg.reward_instruction_refine_template,
                )

        if cfg.diversity_threshold is not None:
            from curatorkit.gates.diversity import DiversityGate

            steps.append(
                DiversityGate(
                    embedding_model=cfg.diversity_embedding_model or cfg.embedding_model,
                    similarity_threshold=cfg.diversity_threshold,
                    device=cfg.embedding_device,
                    batch_size=cfg.embedding_batch_size,
                )
            )

        # ═════════════════════════════════════════════════════════════════════
        # Cross-run embedding dedup
        # ═════════════════════════════════════════════════════════════════════
        if cfg.embedding_dedup:
            from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator

            if cfg.embedding_reset_index:
                import shutil

                index_path = Path(cfg.embedding_index_dir)
                if index_path.exists():
                    shutil.rmtree(index_path)
            steps.append(
                EmbeddingDeduplicator(
                    index_dir=cfg.embedding_index_dir,
                    model=cfg.embedding_dedup_model or cfg.embedding_model,
                    threshold=cfg.embedding_dedup_threshold,
                    device=cfg.embedding_device,
                    batch_size=cfg.embedding_batch_size,
                )
            )

        # ── Resampling ──────────────────────────────────────────────────────
        if cfg.resample and cfg.target_distribution:
            from curatorkit.normalizers.sample import StratifiedSampler

            steps.append(
                StratifiedSampler(
                    category_field=cfg.resample_field,
                    target_distribution=cfg.target_distribution,
                    seed=cfg.resample_seed,
                )
            )

        # ── Global max_samples cap — runs after resampling so the distribution
        #    established by per-reader caps and StratifiedSampler is preserved.
        #    For single datasets this is simply the final size limit.
        #    To cap seeds before generation use the per-reader dict form:
        #      dataset={"name": "...", "max_samples": N}
        if cfg.max_samples is not None:
            from curatorkit.normalizers.truncate import MaxSamplesTruncator

            steps.append(MaxSamplesTruncator(cfg.max_samples))

        # ── Exporters (skipped when output_split is set — handled post-pipeline) ─
        if include_exporters:
            _exporter_map = {
                "alpaca": AlpacaExporter,
                "corpus": CorpusExporter,
                "sharegpt": ShareGPTExporter,
                "dpo": DPOExporter,
                "grpo": GRPOExporter,
                "ppo": PPOExporter,
            }
            for fmt in cfg.export_formats:
                cls = _exporter_map.get(fmt.lower())
                if cls:
                    steps.append(cls())
                else:
                    warnings.warn(f"Unknown export format '{fmt}' — skipped.")

        return steps

    # ─────────────────────────────────────────────────────────────────────────
    # LLM builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_llm(self):
        """Build the generator LLM backend from config."""
        cfg = self.config
        model = cfg.llm_model

        if model and (model.startswith("ollama/") or model.startswith("ollama_chat/")):
            from curatorkit.llm.ollama import OllamaBackend

            return OllamaBackend(
                model=model.split("/", 1)[1],
                base_url=cfg.llm_api_base or "http://localhost:11434",
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                timeout=cfg.llm_timeout,
            )

        from curatorkit.llm.litellm import LiteLLMBackend

        return LiteLLMBackend(
            model=model or "openai/gpt-4o-mini",
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            api_key=cfg.llm_api_key,
            api_base=cfg.llm_api_base,
            timeout=cfg.llm_timeout,
            max_retries=cfg.llm_max_retries,
            drop_params=cfg.llm_drop_params,
            extra_body=cfg.llm_extra_body or None,
        )

    def _build_llm_for_model(self, model: str):
        """Build an LLM backend for an arbitrary model string using global LLM config."""
        cfg = self.config
        if model.startswith("ollama/") or model.startswith("ollama_chat/"):
            from curatorkit.llm.ollama import OllamaBackend

            return OllamaBackend(
                model=model.split("/", 1)[1],
                base_url=cfg.llm_api_base or "http://localhost:11434",
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                timeout=cfg.llm_timeout,
            )
        from curatorkit.llm.litellm import LiteLLMBackend

        return LiteLLMBackend(
            model=model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            api_key=cfg.llm_api_key,
            api_base=cfg.llm_api_base,
            timeout=cfg.llm_timeout,
            max_retries=cfg.llm_max_retries,
            drop_params=cfg.llm_drop_params,
            extra_body=cfg.llm_extra_body or None,
        )

    def _build_probe_llm(self):
        """Build a generator LLM for probe/refiner use with thinking disabled.

        Probe and refiner calls must produce structured text (answers, questions)
        without reasoning preamble. Thinking mode is disabled at the API level
        regardless of the main generator's llm_extra_body setting — this is
        model-agnostic and more reliable than prompt-level tokens like /no_think.
        """
        cfg = self.config
        import copy

        probe_extra = copy.deepcopy(cfg.llm_extra_body or {})
        # Force thinking off for any model that supports this kwarg
        # (Qwen3, DeepSeek-R1, etc. honour chat_template_kwargs.enable_thinking)
        probe_extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

        model = cfg.llm_model
        from curatorkit.llm.litellm import LiteLLMBackend

        return LiteLLMBackend(
            model=model or "openai/gpt-4o-mini",
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            api_key=cfg.llm_api_key,
            api_base=cfg.llm_api_base,
            timeout=cfg.llm_timeout,
            max_retries=cfg.llm_max_retries,
            drop_params=cfg.llm_drop_params,
            extra_body=probe_extra,
        )

    def _build_judge_llm(self):
        """Build the judge LLM backend — uses judge_llm_model if set, else falls back to generator LLM."""
        cfg = self.config
        if not cfg.judge_llm_model:
            return self._build_llm()

        model = cfg.judge_llm_model
        if model.startswith("ollama/") or model.startswith("ollama_chat/"):
            from curatorkit.llm.ollama import OllamaBackend

            return OllamaBackend(
                model=model.split("/", 1)[1],
                base_url=cfg.judge_llm_api_base or "http://localhost:11434",
                temperature=cfg.judge_llm_temperature,
                max_tokens=cfg.judge_llm_max_tokens,
                timeout=cfg.judge_llm_timeout,
            )

        from curatorkit.llm.litellm import LiteLLMBackend

        return LiteLLMBackend(
            model=model,
            temperature=cfg.judge_llm_temperature,
            max_tokens=cfg.judge_llm_max_tokens,
            api_key=cfg.llm_api_key,
            api_base=cfg.judge_llm_api_base,
            timeout=cfg.judge_llm_timeout,
            max_retries=cfg.judge_llm_max_retries,
            drop_params=cfg.llm_drop_params,
            extra_body=cfg.judge_llm_extra_body or None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Generation task builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_generator(self, llm):
        """Build the generation task from config."""
        cfg = self.config
        task = cfg.generation_task
        concurrency = cfg.generation_concurrency or cfg.llm_concurrency

        if task == "qa":
            from curatorkit.generators.qa_generator import QAGenerationTask

            return QAGenerationTask(
                llm=llm,
                prompt_template=cfg.qa_prompt_template,
                table_prompt_template=cfg.qa_table_prompt_template,
                num_questions=cfg.num_questions,
                difficulty=cfg.difficulty,
                concurrency=concurrency,
            )
        elif task == "preference":
            from curatorkit.generators.preference_gen import PreferenceGenerationTask

            return PreferenceGenerationTask(
                llm=llm,
                prompt_template=cfg.preference_prompt_template,
                chosen_prompt=cfg.preference_chosen_prompt,
                rejected_prompt=cfg.preference_rejected_prompt,
                mode=cfg.preference_mode,
                concurrency=concurrency,
            )
        elif task == "grpo":
            from curatorkit.generators.grpo_rollout import GRPORolloutTask

            scoring_llm = None
            if cfg.grpo_scoring_llm_model:
                scoring_llm = self._build_llm_for_model(cfg.grpo_scoring_llm_model)
            return GRPORolloutTask(
                llm=llm,
                scoring_llm=scoring_llm,
                response_prompt=cfg.grpo_prompt_template,
                num_responses=cfg.num_responses,
                score_responses=cfg.score_responses,
                temperature_spread=cfg.grpo_temperature_spread,
                temperatures=cfg.grpo_temperatures,
                concurrency=concurrency,
            )
        elif task == "multiturn":
            from curatorkit.generators.multiturn_gen import MultiTurnTask

            return MultiTurnTask(
                llm=llm,
                prompt_template=cfg.multiturn_prompt_template,
                num_turns=cfg.num_turns,
                concurrency=concurrency,
            )
        elif task in ("evol", "evol_instruct"):
            from curatorkit.generators.evol_instruct import EvolInstructTask

            return EvolInstructTask(
                llm=llm,
                prompt_template=cfg.evol_prompt_template,
                answer_prompt_template=cfg.evol_answer_prompt_template,
                num_evolutions=cfg.num_evolutions,
                generate_answers=cfg.generate_answers,
                concurrency=concurrency,
            )
        elif task == "cot":
            from curatorkit.generators.cot_generator import ChainOfThoughtTask

            kwargs = {} if cfg.cot_marker is None else {"cot_marker": cfg.cot_marker}
            return ChainOfThoughtTask(
                llm=llm,
                prompt_template=cfg.cot_prompt_template,
                mode=cfg.cot_mode,
                concurrency=concurrency,
                **kwargs,
            )
        elif task == "adversarial_preference":
            from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

            return AdversarialPreferenceTask(
                llm=llm,
                num_questions=cfg.num_questions,
                injection_rate=cfg.injection_rate,
                injection_types=cfg.injection_types or None,
                seed=cfg.injection_seed,
                faithful_prompt_template=cfg.qa_prompt_template,
                adversarial_prompt_template=cfg.adversarial_prompt_template,
                concurrency=concurrency,
            )
        elif task == "adversarial_qa":
            from curatorkit.generators.adversarial_qa_generator import (
                AdversarialQAGenerationTask,
                InjectionType,
            )

            _inj_types = (
                [InjectionType(t) for t in cfg.injection_types] if cfg.injection_types else None
            )
            return AdversarialQAGenerationTask(
                llm=llm,
                num_questions=cfg.num_questions,
                injection_rate=cfg.injection_rate,
                injection_types=_inj_types,
                seed=cfg.injection_seed,
                difficulty=cfg.difficulty,
                high_temp=cfg.high_temp,
                concurrency=concurrency,
            )
        else:
            warnings.warn(f"Unknown generation_task '{task}' — skipping generation.")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Reader builder
    # ─────────────────────────────────────────────────────────────────────────

    def _make_reader(
        self,
        dataset: str,
        preprocessing_fn=None,
        split_override: str | None = None,
        subset_override: str | None = None,
    ):
        """
        Return the right reader for a dataset string.

        Rules (in order):
          1. Ends with .pdf            → PDFReader (with output_mode)
          2. Ends with .jsonl          → JSONLReader
          3. Ends with .json           → JSONReader
          4. Ends with .csv or .tsv    → CSVReader
          5. Ends with .parquet        → ParquetReader
          6. Looks like a file path    → JSONLReader (fallback)
          7. Anything else             → HuggingFaceReader
        """
        cfg = self.config
        p = Path(dataset)

        common: dict[str, Any] = dict(
            format=cfg.format,
            field_mapping=cfg.field_mapping or {},
            preprocessing_fn=preprocessing_fn,
            detection_sample_size=cfg.detection_sample_size,
        )

        suffix = p.suffix.lower()

        # ── PDF (with output_mode) ──────────────────────────────────────────
        if suffix == ".pdf":
            from curatorkit.connectors.pdf import PDFReader

            return PDFReader(
                path=p,
                chunk_strategy=cfg.pdf_chunk_strategy,
                chunk_max_tokens=cfg.pdf_chunk_max_tokens,
                chunk_overlap_tokens=cfg.pdf_chunk_overlap_tokens,
                extract_tables=cfg.pdf_extract_tables,
                ocr=cfg.pdf_ocr,
                min_section_tokens=cfg.pdf_min_section_tokens,
                output_mode=cfg.pdf_output_mode,
                llm_model=cfg.llm_model,
                llm_temperature=cfg.llm_temperature,
                llm_max_tokens=cfg.llm_max_tokens,
                llm_api_key=cfg.llm_api_key,
            )

        if suffix == ".jsonl":
            from curatorkit.connectors.jsonl import JSONLReader

            return JSONLReader(path=p, **common)

        if suffix == ".json":
            from curatorkit.connectors.json_reader import JSONReader

            return JSONReader(path=p, **common)

        if suffix in (".csv", ".tsv"):
            from curatorkit.connectors.csv_reader import CSVReader

            delim = "\t" if suffix == ".tsv" else None
            return CSVReader(path=p, delimiter=delim, **common)

        if suffix == ".parquet":
            from curatorkit.connectors.parquet_reader import ParquetReader

            return ParquetReader(path=p, **common)

        if p.exists():
            from curatorkit.connectors.jsonl import JSONLReader

            return JSONLReader(path=p, **common)

        # HuggingFace Hub dataset
        from curatorkit.connectors.huggingface import HuggingFaceReader

        return HuggingFaceReader(
            dataset_name=dataset,
            split=split_override or cfg.split,
            subset=subset_override or cfg.hf_subset or cfg.subset,
            streaming=cfg.streaming,
            token=cfg.hf_token,
            columns=cfg.hf_columns,
            **common,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Provenance
    # ─────────────────────────────────────────────────────────────────────────

    def _export_splits(self, samples: list, output_dir: Path) -> None:
        """Split accepted samples and export each split to suffixed files."""
        import math
        import random as _random

        from curatorkit.exporters.alpaca import AlpacaExporter
        from curatorkit.exporters.corpus import CorpusExporter
        from curatorkit.exporters.dpo import DPOExporter
        from curatorkit.exporters.grpo import GRPOExporter
        from curatorkit.exporters.ppo import PPOExporter
        from curatorkit.exporters.sharegpt import ShareGPTExporter

        split_def = self.config.output_split  # e.g. {"train": 0.8, "val": 0.1, "test": 0.1}
        total = sum(split_def.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"output_split fractions must sum to 1.0, got {total:.4f}. "
                f"Adjust your split ratios."
            )

        shuffled = list(samples)
        _random.Random(self.config.output_split_seed).shuffle(shuffled)
        n = len(shuffled)

        _exporter_map = {
            "alpaca": AlpacaExporter,
            "corpus": CorpusExporter,
            "sharegpt": ShareGPTExporter,
            "dpo": DPOExporter,
            "grpo": GRPOExporter,
            "ppo": PPOExporter,
        }

        start = 0
        split_items = list(split_def.items())
        for i, (split_name, fraction) in enumerate(split_items):
            # Last split gets all remaining samples to avoid rounding loss
            if i == len(split_items) - 1:
                end = n
            else:
                end = start + math.floor(n * fraction)

            split_samples = shuffled[start:end]
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for fmt in self.config.export_formats:
                cls = _exporter_map.get(fmt.lower())
                if cls:
                    cls().export(split_samples, split_dir)
                else:
                    warnings.warn(f"Unknown export format '{fmt}' — skipped.")

            start = end

    def _write_provenance(self, result, output_dir: Path) -> None:
        from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest

        cfg_hash = self._config_hash()
        manifest = ProvenanceManifest(result, cfg_hash, output_dir)
        manifest.write()
        manifest.write_rejected_sidecar()
        checksum_files = list(output_dir.glob("*.jsonl")) + list(output_dir.glob("*.json"))
        manifest.write_checksums(checksum_files)
        DatasetCardGenerator().generate(
            manifest.build(), output_dir, pipeline_name=self.config.name
        )

    def _config_hash(self) -> str:
        cfg = self.config
        payload = {
            "dataset": cfg.dataset,
            "split": cfg.split,
            "format": cfg.format,
            "min_tokens": cfg.min_tokens,
            "max_tokens": cfg.max_tokens,
            "dedup": cfg.dedup,
            "llm_model": cfg.llm_model,
            "generation_task": cfg.generation_task,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
