"""
Pipeline orchestration — synchronous and async runners.

run()       — synchronous execution: readers, gates, normalizers, exporters.
run_async() — async runner for generation tasks and LLM gates with
              concurrency. Generation tasks (BaseGenerationTask subclasses)
              are handled specially: their rejected samples are collected,
              and async execution is supported.

The sync runner remains under 100 lines. Async adds ~50 lines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from curatorkit.interfaces import BaseExporter, BaseGate, BaseNormalizer, BaseReader
from curatorkit.schema import DataSample, RejectedSample

if TYPE_CHECKING:
    pass

Step = BaseReader | BaseGate | BaseNormalizer | BaseExporter


def _is_generation_task(step: object) -> bool:
    """Check if a step is a generation task without hard import."""
    try:
        from curatorkit.generators.base import BaseGenerationTask

        return isinstance(step, BaseGenerationTask)
    except ImportError:
        return False


@dataclass
class PipelineResult:
    """Everything a Pipeline.run() / run_async() call produced.

    passed             — DataSample objects that cleared every step.
    rejected           — RejectedSample objects from all stages (readers,
                         gates, generation tasks), each with a structured
                         rejection reason.
    stage_counts       — per-step counters keyed by step display name, e.g.
                         {"input_count", "output_count", "rejected_count",
                          "probe_recovered", "exported_count"} depending on
                         the step kind.
    wall_clock_seconds — total run time.
    diagnostics        — PipelineDiagnostics accumulator when the diagnostic
                         probe is enabled, else None.
    """

    passed: list[DataSample]
    rejected: list[RejectedSample]
    stage_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    diagnostics: object = None


class Pipeline:
    """
    Pipeline runner supporting both synchronous and async execution.

    Steps are executed in order:
      Readers           → produce (samples, reader_rejections)
      Gates             → filter samples into (passed, gate_rejections)
      Normalizers       → transform samples (includes generation tasks)
      Exporters         → write samples to disk

    Generation tasks (BaseGenerationTask subclasses) are normalizers that
    call LLMs. In sync mode they run sequentially. In async mode they use
    concurrency-controlled parallel LLM calls.

    All rejections from every stage flow into PipelineResult.rejected.
    """

    def __init__(
        self,
        steps: list[Step],
        output_dir: Path | None = None,
        diagnostics: object = None,
    ) -> None:
        self.steps = steps
        self.output_dir = output_dir or Path("output")
        self._diagnostics = diagnostics

    def dry_run(self) -> list[dict[str, str]]:
        """
        Print and return the planned execution order without running anything.

        Each entry is {step_num, name, kind, config_hash} so callers can compare
        plans across runs. config_hash is the step's own _config_hash() if it
        defines one, else the empty string.
        """
        plan: list[dict[str, str]] = []
        print(f"\n=== Pipeline.dry_run — {len(self.steps)} step(s) ===")
        print(f"Output dir: {self.output_dir}\n")
        for idx, step in enumerate(self.steps, start=1):
            name = getattr(step, "_display_name", None) or type(step).__name__
            if isinstance(step, BaseReader):
                kind = "reader"
            elif isinstance(step, BaseGate):
                kind = "gate"
            elif isinstance(step, BaseExporter):
                kind = "exporter"
            elif isinstance(step, BaseNormalizer):
                kind = "generator" if _is_generation_task(step) else "normalizer"
            else:
                kind = "unknown"
            cfg_fn = getattr(step, "_config_hash", None)
            cfg_hash = cfg_fn() if callable(cfg_fn) else ""
            entry = {"step": idx, "name": name, "kind": kind, "config_hash": cfg_hash}
            plan.append(entry)
            line = f"  {idx:>2}. [{kind:<10}] {name}"
            if cfg_hash:
                line += f"  (cfg={cfg_hash})"
            print(line)
        print("\nAlways emitted: manifest.json, dataset_card.md, rejected.jsonl, checksums.txt")
        print("=== END Pipeline.dry_run ===\n")
        return plan

    def run(self) -> PipelineResult:
        """Synchronous pipeline execution."""
        t0 = time.monotonic()
        samples: list[DataSample] = []
        all_rejected: list[RejectedSample] = []
        stage_counts: dict[str, dict[str, int]] = {}

        for step in self.steps:
            name = getattr(step, "_display_name", None) or type(step).__name__

            if isinstance(step, BaseReader):
                new_samples, reader_rejected = step.read()
                all_rejected.extend(reader_rejected)
                samples.extend(new_samples)
                if name in stage_counts:
                    stage_counts[name]["output_count"] += len(new_samples)
                    stage_counts[name]["rejected_count"] += len(reader_rejected)
                else:
                    stage_counts[name] = {
                        "output_count": len(new_samples),
                        "rejected_count": len(reader_rejected),
                    }

            elif isinstance(step, BaseGate):
                input_count = len(samples)
                passed, rejected = step.run(samples)

                # Diagnostic probe — inline recovery.
                # Each rejected sample is diagnosed; if the probe found a passing
                # re-generation it is stored in diagnosis.recovered_sample and
                # re-enters the pipeline immediately (flows through subsequent gates).
                probe = getattr(step, "probe", None)
                probe_recovered: list = []
                if probe is not None and rejected:
                    diagnoses = probe.diagnose_batch(rejected)
                    for r, diag in zip(rejected, diagnoses):
                        r.diagnosis = diag
                        if self._diagnostics is not None:
                            self._diagnostics.record(r)
                        if diag.recovered_sample is not None:
                            probe_recovered.append(diag.recovered_sample)

                all_rejected.extend(rejected)
                samples = passed + probe_recovered
                stage_counts[name] = {
                    "input_count": input_count,
                    "output_count": len(passed),
                    "probe_recovered": len(probe_recovered),
                    "rejected_count": len(rejected),
                }

            elif isinstance(step, BaseNormalizer):
                input_count = len(samples)
                samples = step.run(samples)

                # Collect generation task rejections
                if _is_generation_task(step):
                    gen_rejected = step.flush_rejected()  # type: ignore[union-attr]
                    all_rejected.extend(gen_rejected)
                    stage_counts[name] = {
                        "input_count": input_count,
                        "output_count": len(samples),
                        "rejected_count": len(gen_rejected),
                    }
                else:
                    stage_counts[name] = {
                        "input_count": input_count,
                        "output_count": len(samples),
                    }

            elif isinstance(step, BaseExporter):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                step.export(samples, self.output_dir)
                stage_counts[name] = {"exported_count": len(samples)}

        return PipelineResult(
            passed=samples,
            rejected=all_rejected,
            stage_counts=stage_counts,
            wall_clock_seconds=time.monotonic() - t0,
            diagnostics=self._diagnostics,
        )

    async def run_async(self) -> PipelineResult:
        """
        Async pipeline execution.

        Steps with a run_async() method (generation tasks and LLM gates) use
        native async with semaphore-bounded concurrency. All other steps
        (readers, non-LLM normalizers, exporters) run synchronously.
        """
        t0 = time.monotonic()
        samples: list[DataSample] = []
        all_rejected: list[RejectedSample] = []
        stage_counts: dict[str, dict[str, int]] = {}

        for step in self.steps:
            name = getattr(step, "_display_name", None) or type(step).__name__

            if isinstance(step, BaseReader):
                new_samples, reader_rejected = step.read()
                all_rejected.extend(reader_rejected)
                samples.extend(new_samples)
                if name in stage_counts:
                    stage_counts[name]["output_count"] += len(new_samples)
                    stage_counts[name]["rejected_count"] += len(reader_rejected)
                else:
                    stage_counts[name] = {
                        "output_count": len(new_samples),
                        "rejected_count": len(reader_rejected),
                    }

            elif isinstance(step, BaseGate):
                input_count = len(samples)

                if hasattr(step, "run_async"):
                    passed, rejected = await step.run_async(samples)
                else:
                    passed, rejected = step.run(samples)

                # Diagnostic probe — inline recovery (same as sync path).
                probe = getattr(step, "probe", None)
                probe_recovered_async: list = []
                if probe is not None and rejected:
                    diagnoses = probe.diagnose_batch(rejected)
                    for r, diag in zip(rejected, diagnoses):
                        r.diagnosis = diag
                        if self._diagnostics is not None:
                            self._diagnostics.record(r)
                        if diag.recovered_sample is not None:
                            probe_recovered_async.append(diag.recovered_sample)

                all_rejected.extend(rejected)
                samples = passed + probe_recovered_async
                stage_counts[name] = {
                    "input_count": input_count,
                    "output_count": len(passed),
                    "probe_recovered": len(probe_recovered_async),
                    "rejected_count": len(rejected),
                }

            elif isinstance(step, BaseNormalizer):
                input_count = len(samples)

                # Use async for generation tasks
                if _is_generation_task(step) and hasattr(step, "run_async"):
                    samples = await step.run_async(samples)  # type: ignore[union-attr]
                    gen_rejected = step.flush_rejected()  # type: ignore[union-attr]
                    all_rejected.extend(gen_rejected)
                    stage_counts[name] = {
                        "input_count": input_count,
                        "output_count": len(samples),
                        "rejected_count": len(gen_rejected),
                    }
                else:
                    samples = step.run(samples)
                    if _is_generation_task(step):
                        gen_rejected = step.flush_rejected()  # type: ignore[union-attr]
                        all_rejected.extend(gen_rejected)
                        stage_counts[name] = {
                            "input_count": input_count,
                            "output_count": len(samples),
                            "rejected_count": len(gen_rejected),
                        }
                    else:
                        stage_counts[name] = {
                            "input_count": input_count,
                            "output_count": len(samples),
                        }

            elif isinstance(step, BaseExporter):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                step.export(samples, self.output_dir)
                stage_counts[name] = {"exported_count": len(samples)}

        return PipelineResult(
            passed=samples,
            rejected=all_rejected,
            stage_counts=stage_counts,
            wall_clock_seconds=time.monotonic() - t0,
            diagnostics=self._diagnostics,
        )
