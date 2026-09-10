"""
Task-type <-> export-format alignment.

Each exporter is designed for one specific DataSample.task_type shape (see
each exporter's own docstring for exactly which fields it reads — e.g.
DPOExporter reads chosen/rejected and explicitly skips anything that isn't
preference/implicit_preference; AlpacaExporter reads instruction/output,
which preference samples never populate). TASK_TYPE_EXPORT_FORMATS is the
single source of truth for this, used in three places that must all agree:

  1. Every exporter's own runtime gate: a sample whose task_type isn't
     listed as compatible with that exporter's format is skipped (counted
     in a summary warning), even when its fields happen to be non-empty —
     e.g. GRPOExporter no longer accepts instruction_following data just
     because `instruction` is populated; only "grpo" task_type is grpo.jsonl
     shaped. This also stops silent partial data loss: conversational
     samples are no longer accepted by alpaca (which would otherwise keep
     only the first turn and drop the rest with no indication anything was
     lost) — only sharegpt is compatible with conversational, so a
     conversational sample handed to alpaca is skipped and warned about
     instead of silently truncated.
  2. resolve_export_formats(): picks CuratorConfig.export_formats
     automatically when left as None, from whichever of these is known —
     generation_task "selects" the task_type ahead of time (before any
     reader runs); with no generation_task, the task_type must instead be
     "identified" from what the readers/connectors actually produced, once
     the pipeline has run.
  3. misaligned_formats(): powers a warning (not a filter) when a caller
     sets export_formats explicitly alongside a generation_task it doesn't
     suit — the config-level choice is still honored as given, but since
     (1) now gates at the sample level too, an explicit-but-wrong choice
     ends up producing an empty file with two warnings (one at build time,
     one from the exporter) rather than silently doing something unexpected.
"""

from __future__ import annotations

# task_type -> the export format(s) actually designed for it (README's own
# generation_task/task_type/export table, plus the two connector-only task
# types — unpaired_preference, prompt_only — that don't come from a
# generation task).
TASK_TYPE_EXPORT_FORMATS: dict[str, list[str]] = {
    "instruction_following": ["alpaca", "sharegpt"],
    "conversational": ["sharegpt"],
    "preference": ["dpo"],
    "implicit_preference": ["dpo"],
    "unpaired_preference": ["alpaca", "sharegpt"],
    "grpo": ["grpo"],
    "prompt_only": ["ppo"],
    "language_modeling": ["corpus"],
    # Never assigned by a built-in reader/generator today, but SchemaGate,
    # the PII/secrets field maps, and CorpusExporter's own docstring all
    # already special-case it — a documented extension point for a custom
    # preprocessing_fn that tags raw chunks this way instead of
    # "language_modeling". Treated identically to it here.
    "source_chunk": ["corpus"],
}

# generation_task -> the task_type it produces.
GENERATION_TASK_OUTPUT_TYPE: dict[str, str] = {
    "qa": "instruction_following",
    "preference": "preference",
    "grpo": "grpo",
    "multiturn": "conversational",
    "evol": "instruction_following",
    "cot": "instruction_following",
    "adversarial_preference": "preference",
    "adversarial_qa": "instruction_following",
}

# Fallback when neither generation_task nor any observed sample task_type is
# available yet (e.g. Curator.dry_run() printing a plan before any reader
# has actually run). Matches the pre-alignment default so dry_run's printed
# plan doesn't change for the common case.
DEFAULT_EXPORT_FORMATS: list[str] = ["alpaca", "sharegpt", "dpo"]


def compatible_formats_for(task_type: str) -> list[str]:
    """Export formats designed for one task_type. Empty list = not in the
    table at all (an unrecognized or custom task_type)."""
    return TASK_TYPE_EXPORT_FORMATS.get(task_type, [])


def is_compatible(task_type: str, export_format: str) -> bool:
    """The per-sample runtime gate every exporter calls: is this task_type
    one this export format is actually designed for? An unrecognized
    task_type is never compatible with anything (conservative default —
    silently accepting an unknown shape is how the original instruction_
    following-into-grpo/ppo/corpus bug happened)."""
    return export_format.lower() in TASK_TYPE_EXPORT_FORMATS.get(task_type, [])


def task_types_for(export_format: str) -> list[str]:
    """Task_types one export format accepts — the inverse of
    compatible_formats_for(), used to phrase each exporter's skip warning."""
    return sorted(
        task_type
        for task_type, formats in TASK_TYPE_EXPORT_FORMATS.items()
        if export_format.lower() in formats
    )


def resolve_export_formats(
    generation_task: str | None,
    observed_task_types: set[str] | None = None,
) -> list[str]:
    """Pick export formats when CuratorConfig.export_formats is left as None."""
    if generation_task is not None:
        task_type = GENERATION_TASK_OUTPUT_TYPE.get(generation_task)
        if task_type is not None:
            return compatible_formats_for(task_type)

    if observed_task_types:
        formats: list[str] = []
        for task_type in sorted(observed_task_types):
            for fmt in compatible_formats_for(task_type):
                if fmt not in formats:
                    formats.append(fmt)
        if formats:
            return formats

    return list(DEFAULT_EXPORT_FORMATS)


def misaligned_formats(generation_task: str, export_formats: list[str]) -> list[str]:
    """Formats in export_formats not designed for generation_task's output
    task_type. Used only to power a warning — callers still honor the
    explicit list as given."""
    task_type = GENERATION_TASK_OUTPUT_TYPE.get(generation_task)
    if task_type is None:
        return []
    compatible = set(compatible_formats_for(task_type))
    return [fmt for fmt in export_formats if fmt.lower() not in compatible]
