"""
CLI entry point: curatorkit

Commands:
  curatorkit run <pipeline_yaml> [--output-dir] [--dry-run] [--verbose] [--async] [--version]
  curatorkit setup-pdf [--check]

Features:
  - Sync and async pipeline execution (--async)
  - Generation tasks (qa, evol_instruct, preference, grpo, multiturn, cot)
  - Quality gates (hallucination, reward, diversity)
  - Exact, MinHash, and embedding-based deduplication
  - LLM backend construction from config
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

import typer

from curatorkit import __version__

app = typer.Typer(
    name="curatorkit",
    help="CuratorKIT data pipeline CLI — curate and export LLM training data.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"curatorkit {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version and exit"
        ),
    ] = None,
) -> None:
    """CuratorKIT data pipeline CLI."""


@app.command("run")
def run(
    pipeline_yaml: Annotated[Path, typer.Argument(help="Path to pipeline YAML config")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output directory")] = Path(
        "output"
    ),
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate config, print plan, then exit")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging")] = False,
    use_async: Annotated[
        bool, typer.Option("--async", help="Use async pipeline runner for generation tasks")
    ] = False,
    reset_index: Annotated[
        bool,
        typer.Option("--reset-index", help="Delete embedding-dedup index dir(s) before running"),
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """Run a CuratorKIT data pipeline from a YAML config file."""
    from curatorkit.config import PipelineConfig

    if not pipeline_yaml.exists():
        typer.echo(f"Error: config file not found: {pipeline_yaml}", err=True)
        raise typer.Exit(1)

    try:
        config = PipelineConfig.from_yaml(pipeline_yaml)
    except Exception as e:
        typer.echo(f"Error: invalid pipeline config: {e}", err=True)
        raise typer.Exit(1)

    config_hash = hashlib.sha256(pipeline_yaml.read_bytes()).hexdigest()[:16]

    if verbose:
        typer.echo(f"Pipeline: {config.name} v{config.version}")
        typer.echo(f"Config hash: {config_hash}")

    if dry_run:
        _print_dry_run_plan(config, output_dir)
        raise typer.Exit(0)

    # Optional reset of any embedding-dedup persistent indices before run
    if reset_index:
        import shutil

        for n in config.normalizers:
            if n.type == "embedding_dedup":
                idx = Path(n.embedding_index_dir)
                if idx.exists():
                    if verbose:
                        typer.echo(f"Resetting embedding index: {idx}")
                    shutil.rmtree(idx)

    splitting = bool(getattr(config, "output_split", None))
    steps = _build_steps(config, verbose, include_exporters=not splitting)
    output_dir.mkdir(parents=True, exist_ok=True)

    from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest
    from curatorkit.pipeline import Pipeline

    # Attach a PipelineDiagnostics accumulator if the diagnostic probe is enabled
    diagnostics = None
    if config.diagnostic is not None and config.diagnostic.enable_probe:
        try:
            from curatorkit.diagnostic.diagnostics import PipelineDiagnostics

            diagnostics = PipelineDiagnostics()
        except ImportError:
            diagnostics = None

    pipeline = Pipeline(steps, output_dir=output_dir, diagnostics=diagnostics)

    if verbose:
        mode = "async" if use_async else "sync"
        typer.echo(f"Running pipeline ({mode})...")

    if use_async:
        import asyncio

        result = asyncio.run(pipeline.run_async())
    else:
        result = pipeline.run()

    # Write diagnostic_summary.json when diagnostics were collected
    if result.diagnostics is not None:
        result.diagnostics.write_summary(output_dir / "diagnostic_summary.json")

    # Output split — shuffle accepted samples and export each split to a subdir
    if splitting:
        import math
        import random as _random

        from curatorkit.exporters.alpaca import AlpacaExporter
        from curatorkit.exporters.corpus import CorpusExporter
        from curatorkit.exporters.dpo import DPOExporter
        from curatorkit.exporters.grpo import GRPOExporter
        from curatorkit.exporters.ppo import PPOExporter
        from curatorkit.exporters.sharegpt import ShareGPTExporter

        split_def = config.output_split
        total_frac = sum(split_def.values())
        if abs(total_frac - 1.0) > 1e-6:
            typer.echo(
                f"Error: output_split fractions sum to {total_frac:.4f}, must be 1.0",
                err=True,
            )
            raise typer.Exit(1)

        _exporter_cls = {
            "alpaca": AlpacaExporter,
            "corpus": CorpusExporter,
            "sharegpt": ShareGPTExporter,
            "dpo": DPOExporter,
            "grpo": GRPOExporter,
            "ppo": PPOExporter,
        }
        shuffled = list(result.passed)
        _random.Random(getattr(config, "output_split_seed", 42)).shuffle(shuffled)
        n = len(shuffled)
        start = 0
        split_items = list(split_def.items())
        for i, (split_name, fraction) in enumerate(split_items):
            end = n if i == len(split_items) - 1 else start + math.floor(n * fraction)
            split_samples = shuffled[start:end]
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for e in config.exporters:
                cls = _exporter_cls.get(e.type)
                if cls:
                    cls().export(split_samples, split_dir)
            if verbose:
                typer.echo(f"  {split_name}: {len(split_samples):,} samples → {split_dir}/")
            start = end

    manifest_builder = ProvenanceManifest(
        result=result,
        pipeline_config_hash=config_hash,
        output_dir=output_dir,
    )
    manifest_path = manifest_builder.write()
    sidecar_path = manifest_builder.write_rejected_sidecar()
    output_files = [manifest_path, sidecar_path] + [
        output_dir / f
        for f in [
            "sft_alpaca.jsonl",
            "sft_sharegpt.jsonl",
            "grpo.jsonl",
            "ppo.jsonl",
            "dpo.jsonl",
        ]
        if (output_dir / f).exists()
    ]
    checksums_path = manifest_builder.write_checksums(output_files)

    manifest_data = manifest_builder.build()
    card_gen = DatasetCardGenerator()
    card_path = card_gen.generate(manifest_data, output_dir, pipeline_name=config.name)

    if verbose:
        typer.echo(f"Passed:   {len(result.passed)}")
        typer.echo(f"Rejected: {len(result.rejected)}")
        typer.echo(f"Wall clock: {result.wall_clock_seconds:.2f}s")

    typer.echo(f"Output written to: {output_dir}/")
    typer.echo(f"  manifest.json   — {manifest_path}")
    typer.echo(f"  rejected.jsonl  — {sidecar_path}")
    typer.echo(f"  dataset_card.md — {card_path}")
    typer.echo(f"  checksums.txt   — {checksums_path}")
    if result.diagnostics is not None:
        typer.echo(f"  diagnostic_summary.json — {output_dir / 'diagnostic_summary.json'}")


# ---------------------------------------------------------------------------
# Dry-run plan printer
# ---------------------------------------------------------------------------


def _print_dry_run_plan(config: object, output_dir: Path) -> None:
    from curatorkit.config import PipelineConfig

    assert isinstance(config, PipelineConfig)

    typer.echo("\n=== DRY RUN — execution plan ===\n")
    typer.echo(f"Pipeline: {config.name} v{config.version}")
    typer.echo(f"Output dir: {output_dir}\n")

    if config.llm:
        typer.echo(
            f"LLM: {config.llm.model} (temp={config.llm.temperature}, "
            f"concurrency={config.llm.concurrency})"
        )

    if config.diagnostic and config.diagnostic.enable_probe:
        typer.echo(
            f"Diagnostic probe: enabled "
            f"(temperatures={config.diagnostic.probe_temperatures}, "
            f"score_split={config.diagnostic.score_split})"
        )

    typer.echo("\nSteps (in order):")

    step_num = 1
    for r in config.readers:
        typer.echo(f"  {step_num}. Reader: {r.type} <- {r.path or '?'}")
        typer.echo(f"       format: {r.format}  detection_sample_size: {r.detection_sample_size}")
        if r.type == "pdf" and r.output_mode != "chunk":
            typer.echo(f"       output_mode: {r.output_mode} (LLM generation per chunk)")
        if r.field_mapping:
            typer.echo(f"       field_mapping: {r.field_mapping}")
        step_num += 1

    if config.max_samples is not None:
        typer.echo(f"  {step_num}. MaxSamplesTruncator(max_samples={config.max_samples})")
        step_num += 1

    for g in config.gates:
        if g.type == "schema":
            typer.echo(f"  {step_num}. Gate: schema (min={g.min_tokens}, max={g.max_tokens})")
        elif g.type == "hallucination":
            typer.echo(f"  {step_num}. Gate: hallucination (threshold={g.hallucination_threshold})")
        elif g.type == "reward":
            typer.echo(
                f"  {step_num}. Gate: reward (threshold={g.reward_threshold}, "
                f"dims={g.reward_dimensions})"
            )
        elif g.type == "diversity":
            typer.echo(
                f"  {step_num}. Gate: diversity (threshold={g.similarity_threshold}, "
                f"model={g.embedding_model})"
            )
        elif g.type == "secrets":
            typer.echo(
                f"  {step_num}. Gate: secrets (code_corpus_mode={g.secrets_code_corpus_mode})"
            )
        elif g.type == "toxicity":
            typer.echo(
                f"  {step_num}. Gate: toxicity "
                f"(pass={g.toxicity_classifier_pass_threshold}, "
                f"reject={g.toxicity_classifier_reject_threshold})"
            )
        step_num += 1

    for gen in config.generators:
        typer.echo(f"  {step_num}. Generator: {gen.type}")
        if gen.type == "qa":
            typer.echo(f"       num_questions={gen.num_questions}, difficulty={gen.difficulty}")
        elif gen.type == "grpo":
            typer.echo(f"       num_responses={gen.num_responses}, score={gen.score_responses}")
        step_num += 1

    for n in config.normalizers:
        typer.echo(f"  {step_num}. Normalizer: {n.type}")
        step_num += 1

    for e in config.exporters:
        typer.echo(f"  {step_num}. Exporter: {e.type}")
        step_num += 1

    if getattr(config, "output_split", None):
        typer.echo(f"\nOutput split: {config.output_split}  → subdirs per split")
    typer.echo("\nAlways emitted: manifest.json, dataset_card.md, rejected.jsonl, checksums.txt")
    typer.echo("\n=== END DRY RUN ===\n")


# ---------------------------------------------------------------------------
# LLM builder helper
# ---------------------------------------------------------------------------


def _build_llm(model: str | None, config: object) -> object:
    """Build an LLM backend from the global config or a model override."""
    from curatorkit.config import LLMConfig, PipelineConfig
    from curatorkit.llm.litellm import LiteLLMBackend

    assert isinstance(config, PipelineConfig)
    llm_cfg = config.llm or LLMConfig()

    effective_model = model or llm_cfg.model

    # Route Ollama models to the Ollama backend
    if effective_model.startswith("ollama/") or effective_model.startswith("ollama_chat/"):
        from curatorkit.llm.ollama import OllamaBackend

        return OllamaBackend(
            model=effective_model.split("/", 1)[1],
            base_url=llm_cfg.api_base or "http://localhost:11434",
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            timeout=llm_cfg.timeout,
            max_retries=llm_cfg.max_retries,
        )

    return LiteLLMBackend(
        model=effective_model,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        api_key=llm_cfg.api_key,
        api_base=llm_cfg.api_base,
        timeout=llm_cfg.timeout,
        max_retries=llm_cfg.max_retries,
        drop_params=llm_cfg.drop_params,
        extra_body=llm_cfg.extra_body or None,
    )


# ---------------------------------------------------------------------------
# Step builder — all reader/gate/normalizer/generator/exporter types wired
# ---------------------------------------------------------------------------


def _build_steps(config: object, verbose: bool, include_exporters: bool = True) -> list:
    from curatorkit.config import PipelineConfig
    from curatorkit.connectors.csv_reader import CSVReader
    from curatorkit.connectors.huggingface import HuggingFaceReader
    from curatorkit.connectors.json_reader import JSONReader
    from curatorkit.connectors.jsonl import JSONLReader
    from curatorkit.connectors.parquet_reader import ParquetReader
    from curatorkit.exporters.alpaca import AlpacaExporter
    from curatorkit.exporters.dpo import DPOExporter
    from curatorkit.exporters.grpo import GRPOExporter
    from curatorkit.exporters.ppo import PPOExporter
    from curatorkit.exporters.sharegpt import ShareGPTExporter
    from curatorkit.gates.schema import SchemaGate
    from curatorkit.normalizers.clean import TextCleaner
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.normalizers.sample import StratifiedSampler

    assert isinstance(config, PipelineConfig)
    steps = []

    # ---- Readers ----
    for r in config.readers:
        common = dict(
            field_mapping=r.field_mapping,
            format=r.format,
            source_uri=r.source_uri,
            preprocessing_fn=r.preprocessing_fn,
            detection_sample_size=r.detection_sample_size,
        )

        if r.type == "jsonl":
            steps.append(JSONLReader(path=r.path, **common))
        elif r.type == "json":
            steps.append(JSONReader(path=r.path, data_key=r.json_data_key, **common))
        elif r.type == "csv":
            steps.append(
                CSVReader(
                    path=r.path,
                    delimiter=r.csv_delimiter,
                    parse_json_cells=r.csv_parse_json_cells,
                    **common,
                )
            )
        elif r.type == "parquet":
            steps.append(
                ParquetReader(
                    path=r.path,
                    columns=r.parquet_columns,
                    batch_size=r.parquet_batch_size,
                    **common,
                )
            )
        elif r.type == "huggingface":
            hf_common = {k: v for k, v in common.items() if k != "source_uri"}
            steps.append(
                HuggingFaceReader(
                    dataset_name=str(r.path) if r.path else "",
                    split=r.hf_split,
                    subset=r.hf_subset,
                    streaming=r.hf_streaming,
                    token=r.hf_token,
                    columns=r.hf_columns,
                    source_uri=r.source_uri,
                    **hf_common,
                )
            )
        elif r.type == "pdf":
            from curatorkit.connectors.pdf import PDFReader

            llm_cfg = config.llm
            steps.append(
                PDFReader(
                    path=r.path,
                    chunk_strategy=r.chunk_strategy,
                    chunk_max_tokens=r.chunk_max_tokens,
                    chunk_overlap_tokens=r.chunk_overlap_tokens,
                    extract_tables=r.extract_tables,
                    ocr=r.ocr,
                    min_section_tokens=r.min_section_tokens,
                    output_mode=r.output_mode,
                    llm_model=r.llm_model or (llm_cfg.model if llm_cfg else None),
                    llm_temperature=llm_cfg.temperature if llm_cfg else 0.7,
                    llm_max_tokens=llm_cfg.max_tokens if llm_cfg else 1024,
                    llm_api_key=llm_cfg.api_key if llm_cfg else None,
                )
            )

    # ---- Max samples cap (immediately after readers) ----
    if config.max_samples is not None:
        from curatorkit.normalizers.truncate import MaxSamplesTruncator

        steps.append(MaxSamplesTruncator(config.max_samples))

    # ---- Gates (first pass — schema gate typically comes before generators) ----
    for g in config.gates:
        if g.type == "schema":
            steps.append(
                SchemaGate(
                    required_fields=g.required_fields,
                    min_tokens=g.min_tokens,
                    max_tokens=g.max_tokens,
                    use_tiktoken=g.use_tiktoken,
                    enforce_task_types=g.enforce_task_types or None,
                )
            )

    # ---- Cleaning normalizers (dedup + clean, before generation) ----
    for n in config.normalizers:
        if n.type == "exact_dedup":
            steps.append(ExactDeduplicator())
        elif n.type == "minhash_dedup":
            steps.append(
                MinHashDeduplicator(
                    threshold=n.minhash_threshold,
                    ngram=n.minhash_ngram,
                    num_perm=n.minhash_num_perm,
                    seed=n.minhash_seed,
                )
            )
        elif n.type == "text_cleaner":
            steps.append(
                TextCleaner(
                    transforms=n.transforms,
                    fields=n.clean_fields or None,
                )
            )
        elif n.type == "pii_pseudonymizer":
            from curatorkit.hygiene.pii import PIIPseudonymizer

            steps.append(
                PIIPseudonymizer(
                    entity_types=n.pii_entity_types or None,
                    fields=n.pii_fields or None,
                    score_threshold=n.pii_score_threshold,
                    faker_seed=n.pii_faker_seed,
                    language=n.pii_language,
                    spacy_model=n.pii_spacy_model,
                )
            )

    # ---- Data hygiene gates (SecretsGate, ToxicityGate — before generation) ----
    for g in config.gates:
        if g.type == "secrets":
            from curatorkit.hygiene.secrets import (
                _PLUGIN_KEYWORD,
                _PLUGINS_BASE,
                SecretsGate,
            )

            plugins = [
                {**p, "limit": g.secrets_hex_limit}
                if p["name"] == "HexHighEntropyString"
                else {**p, "limit": g.secrets_base64_limit}
                if p["name"] == "Base64HighEntropyString"
                else p
                for p in _PLUGINS_BASE
            ]
            if g.secrets_code_corpus_mode:
                plugins.append(_PLUGIN_KEYWORD)
            steps.append(
                SecretsGate(
                    fields=g.secrets_fields or None,
                    plugins=plugins,
                )
            )
        elif g.type == "toxicity":
            from curatorkit.hygiene.toxicity import ToxicityGate

            tox_llm = _build_llm(g.toxicity_llm_model, config) if g.toxicity_llm_model else None
            steps.append(
                ToxicityGate(
                    classifier_pass_threshold=g.toxicity_classifier_pass_threshold,
                    classifier_reject_threshold=g.toxicity_classifier_reject_threshold,
                    detoxify_model=g.toxicity_detoxify_model,
                    llm=tox_llm,
                    llm_reject_threshold=g.toxicity_llm_reject_threshold,
                    text_field=g.toxicity_text_field,
                )
            )

    # ---- Generation tasks ----
    for gen in config.generators:
        llm = _build_llm(gen.llm_model, config)
        concurrency = config.llm.concurrency if config.llm else 10

        if gen.type == "qa":
            from curatorkit.generators.qa_generator import QAGenerationTask

            steps.append(
                QAGenerationTask(
                    llm=llm,
                    prompt_template=gen.prompt_template,
                    table_prompt_template=gen.table_prompt_template,
                    num_questions=gen.num_questions,
                    difficulty=gen.difficulty,
                    concurrency=concurrency,
                )
            )
        elif gen.type == "evol_instruct":
            from curatorkit.generators.evol_instruct import EvolInstructTask

            steps.append(
                EvolInstructTask(
                    llm=llm,
                    prompt_template=gen.prompt_template,
                    answer_prompt_template=gen.answer_prompt_template,
                    num_evolutions=gen.num_evolutions,
                    strategies=gen.strategies,
                    generate_answers=gen.generate_answers,
                    concurrency=concurrency,
                )
            )
        elif gen.type == "preference":
            from curatorkit.generators.preference_gen import PreferenceGenerationTask

            steps.append(
                PreferenceGenerationTask(
                    llm=llm,
                    prompt_template=gen.prompt_template,
                    chosen_prompt=gen.chosen_prompt_template,
                    rejected_prompt=gen.rejected_prompt_template,
                    mode=gen.preference_mode,
                    concurrency=concurrency,
                )
            )
        elif gen.type == "grpo":
            from curatorkit.generators.grpo_rollout import GRPORolloutTask

            steps.append(
                GRPORolloutTask(
                    llm=llm,
                    response_prompt=gen.prompt_template,
                    num_responses=gen.num_responses,
                    score_responses=gen.score_responses,
                    temperature_spread=gen.temperature_spread,
                    concurrency=concurrency,
                )
            )
        elif gen.type == "adversarial_preference":
            from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

            steps.append(
                AdversarialPreferenceTask(
                    llm=llm,
                    num_questions=gen.num_questions,
                    injection_rate=gen.injection_rate,
                    injection_types=gen.injection_types or None,
                    seed=gen.injection_seed,
                    faithful_prompt_template=gen.prompt_template,
                    adversarial_prompt_template=gen.adversarial_prompt_template,
                    difficulty=gen.difficulty,
                    concurrency=concurrency,
                )
            )
        elif gen.type == "adversarial_qa":
            from curatorkit.generators.adversarial_qa_generator import (
                AdversarialQAGenerationTask,
                InjectionType,
            )

            _inj_types = (
                [InjectionType(t) for t in gen.injection_types] if gen.injection_types else None
            )
            steps.append(
                AdversarialQAGenerationTask(
                    llm=llm,
                    num_questions=gen.num_questions,
                    injection_rate=gen.injection_rate,
                    injection_types=_inj_types,
                    seed=gen.injection_seed,
                    difficulty=gen.difficulty,
                    high_temp=gen.high_temp,
                    concurrency=concurrency,
                )
            )
        elif gen.type == "multiturn":
            from curatorkit.generators.multiturn_gen import MultiTurnTask

            steps.append(
                MultiTurnTask(
                    llm=llm,
                    num_turns=gen.num_turns,
                    prompt_template=gen.prompt_template,
                    include_context=gen.include_context,
                    concurrency=concurrency,
                )
            )
        elif gen.type == "cot":
            from curatorkit.generators.cot_generator import ChainOfThoughtTask

            steps.append(
                ChainOfThoughtTask(
                    llm=llm,
                    mode=gen.cot_mode,
                    prompt_template=gen.prompt_template,
                    cot_marker=gen.cot_marker,
                    concurrency=concurrency,
                )
            )

    # ---- Quality gates (after generation) ----
    for g in config.gates:
        if g.type == "hallucination":
            from curatorkit.gates.hallucination import HallucinationGate

            llm = _build_llm(g.hallucination_llm_model, config)
            gate = HallucinationGate(
                llm=llm,
                threshold=g.hallucination_threshold,
                prompt_template=g.hallucination_prompt_template,
                skip_if_no_context=g.skip_if_no_context,
            )
            # ── Attach diagnostic probe if configured ───────────────────
            if config.diagnostic is not None and config.diagnostic.enable_probe:
                from curatorkit.diagnostic.probe import DiagnosticProbe

                probe_llm = _build_llm(
                    config.diagnostic.probe_generator_model or g.hallucination_llm_model,
                    config,
                )
                gate.probe = DiagnosticProbe(
                    generator_llm=probe_llm,
                    gate=gate,
                    temperatures=config.diagnostic.probe_temperatures,
                    score_split=config.diagnostic.score_split,
                    extra_templates=config.diagnostic.extra_templates or None,
                )
            steps.append(gate)
        elif g.type == "reward":
            from curatorkit.gates.reward import RewardGate

            llm = _build_llm(g.reward_llm_model, config)
            steps.append(
                RewardGate(
                    llm=llm,
                    threshold=g.reward_threshold,
                    dimensions=g.reward_dimensions,
                    prompt_template=g.reward_prompt_template,
                    store_score_in_label=g.store_score_in_label,
                )
            )
        elif g.type == "diversity":
            from curatorkit.gates.diversity import DiversityGate

            steps.append(
                DiversityGate(
                    embedding_model=g.embedding_model,
                    similarity_threshold=g.similarity_threshold,
                    text_field=g.diversity_text_field,
                    coverage_field=g.coverage_field,
                    device=g.embedding_device,
                    batch_size=g.embedding_batch_size,
                )
            )

    # ---- Embedding dedup (after quality gates) ----
    for n in config.normalizers:
        if n.type == "embedding_dedup":
            from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator

            steps.append(
                EmbeddingDeduplicator(
                    index_dir=n.embedding_index_dir,
                    model=n.embedding_model,
                    threshold=n.embedding_threshold,
                    text_field=n.embedding_text_field,
                    device=n.embedding_device,
                    batch_size=n.embedding_batch_size,
                )
            )

    # ---- Stratified sampler (always after generation and filtering) ----
    for n in config.normalizers:
        if n.type == "stratified_sampler":
            steps.append(
                StratifiedSampler(
                    category_field=n.category_field,
                    target_distribution=n.target_distribution,
                    seed=n.sampler_seed,
                )
            )

    # ---- Exporters (skipped when output_split is set — handled post-pipeline) ----
    if include_exporters:
        for e in config.exporters:
            if e.type == "alpaca":
                steps.append(AlpacaExporter())
            elif e.type == "corpus":
                from curatorkit.exporters.corpus import CorpusExporter as _CE

                steps.append(_CE())
            elif e.type == "sharegpt":
                steps.append(ShareGPTExporter())
            elif e.type == "grpo":
                steps.append(GRPOExporter())
            elif e.type == "ppo":
                steps.append(PPOExporter())
            elif e.type == "dpo":
                steps.append(DPOExporter())

    return steps


@app.command("setup-pdf")
def setup_pdf(
    check: Annotated[
        bool,
        typer.Option("--check", help="Verify mineru is installed and importable."),
    ] = False,
) -> None:
    """Verify MinerU is installed for PDF extraction.

    With mineru 3.x, model weights are downloaded automatically on first use.
    No manual setup is needed — just install the package.

    \b
    Examples:
      pip install "curatorkit[pdf]"   # PDF extraction support
      curatorkit setup-pdf --check    # Verify installation
    """
    from curatorkit.tools.setup_mineru import validate_setup

    ok, missing = validate_setup()
    if ok:
        typer.echo("MinerU is installed. Model weights download automatically on first PDF parse.")
    else:
        typer.echo("MinerU is not installed.", err=True)
        typer.echo("\nFix:", err=True)
        typer.echo('  pip install "curatorkit[pdf]"', err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
