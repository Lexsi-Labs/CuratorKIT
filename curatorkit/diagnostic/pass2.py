"""
AdaptivePass2Runner — offline batch re-generation utility for failure-mode analysis.

This class is NOT used by the main library pipeline. The DiagnosticProbe already
performs inline recovery and stores recovered samples in FailureDiagnosis.recovered_sample;
the pipeline routes those back into the accepted pool without a separate pass.

This runner exists for offline analysis that needs to apply a specific config
patch (force_patch) to ALL rejected samples regardless of inline probe results,
answering questions like "what if we lowered the temperature for every
parametric failure?". Pass2Result.summary reports per-mode conversion rates
for such patch sweeps.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curatorkit.diagnostic.failure_modes import FailureMode
from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

logger = logging.getLogger(__name__)


@dataclass
class Pass2Result:
    accepted: list[DataSample]
    rejected: list[RejectedSample]
    attempted: int = 0
    conversion_rate: float = 0.0
    wall_clock_seconds: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)


class AdaptivePass2Runner:
    """
    Parameters
    ----------
    generator : BaseGenerationTask
        The same generation task used in Pass 1. Its LLM temperature is overridden
        per-sample via the implied_patch.
    gate : BaseGate
        Gate to re-check regenerated samples against.
    base_config : CuratorConfig
        The original flat dataclass config. apply_patch() returns a copy.
    output_dir : Path
        Where to write rejected_pass2.jsonl and pass2_summary.json.
    """

    def __init__(self, generator, gate: BaseGate, base_config, output_dir: Path) -> None:
        self.generator = generator
        self.gate = gate
        self.base_config = base_config
        self.output_dir = Path(output_dir)

    def run(
        self,
        recoverable: list[RejectedSample],
        force_patch: dict[str, Any] | None = None,
    ) -> Pass2Result:
        """
        Re-run recoverable samples.

        force_patch : dict | None
            Config patch applied to every sample (e.g. {"llm_temperature": 0.3}).
            Required for re-generation — samples are skipped when None.
            Intended for offline patch-sweep analysis only; do not use
            force_patch in production pipelines.
        """
        t0 = time.monotonic()
        accepted: list[DataSample] = []
        rejected2: list[RejectedSample] = []
        mode_attempted: dict[str, int] = {}
        mode_converted: dict[str, int] = {}

        # Modes where regeneration is futile regardless of patch applied.
        _UNRECOVERABLE = {
            FailureMode.SOURCE_AMBIGUOUS,
            FailureMode.NEAR_DUPLICATE,
            FailureMode.UNKNOWN,
        }

        for sample in recoverable:
            diag = sample.diagnosis
            if diag is None:
                continue

            # Guard: skip modes where no patch can help.
            if diag.mode in _UNRECOVERABLE:
                rejected2.append(sample)
                continue

            # AdaptivePass2Runner is an offline analysis tool; it always
            # requires an explicit force_patch.
            # Inline probe recovery (FailureDiagnosis.recovered_sample) is the
            # production path and does not go through this runner.
            if force_patch is None:
                logger.debug(
                    "AdaptivePass2Runner: no force_patch for sample %s — skipping "
                    "(inline probe recovery is the production path).",
                    sample.id,
                )
                rejected2.append(sample)
                continue

            mode_key = diag.mode.value
            mode_attempted[mode_key] = mode_attempted.get(mode_key, 0) + 1

            patch = force_patch
            patched_config = self.base_config.apply_patch(patch)

            regen = self._regenerate_sample(sample, patched_config, patch)
            if regen is None:
                _mark_failed(sample, "regeneration_error")
                rejected2.append(sample)
                continue

            passed, failed = self.gate.run([regen])

            if passed:
                passed[0].append_provenance(
                    ProvenanceRecord(
                        step_name="AdaptivePass2Runner",
                        step_version="1.0.0",
                        config_hash="",
                        notes={
                            "original_id": sample.id,
                            "failure_mode": mode_key,
                            "patch": patch,
                            "forced": force_patch is not None,
                        },
                    )
                )
                accepted.append(passed[0])
                mode_converted[mode_key] = mode_converted.get(mode_key, 0) + 1
            else:
                failed_sample = failed[0] if failed else sample
                _mark_failed(failed_sample, "gate_failed_pass2")
                rejected2.append(failed_sample)

        attempted = len(recoverable)
        conversion_rate = round(len(accepted) / attempted, 4) if attempted else 0.0

        summary = {
            "attempted": attempted,
            "converted": len(accepted),
            "conversion_rate": conversion_rate,
            "mode_attempted": mode_attempted,
            "mode_converted": mode_converted,
            "mode_rates": {
                m: round(mode_converted.get(m, 0) / mode_attempted[m], 4) for m in mode_attempted
            },
            "forced_patch": force_patch,
        }

        self._write_outputs(rejected2, summary)

        return Pass2Result(
            accepted=accepted,
            rejected=rejected2,
            attempted=attempted,
            conversion_rate=conversion_rate,
            wall_clock_seconds=time.monotonic() - t0,
            summary=summary,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _regenerate_sample(
        self, sample: RejectedSample, patched_config, patch: dict
    ) -> DataSample | None:
        regenerate_field = patch.get("regenerate_field", "output")
        context_window = patch.get("context_window", "default")

        source_ctx = sample.input
        if context_window == "adjacent_chunks":
            adjacent = sample.metadata.get("adjacent_context", "")
            if adjacent:
                source_ctx = source_ctx + "\n\n" + adjacent

        seed = DataSample(
            source_uri=sample.source_uri,
            instruction=sample.instruction if regenerate_field != "instruction" else "",
            input=source_ctx,
            output="",
            task_type=sample.task_type,
            metadata=copy.deepcopy(sample.metadata),
            provenance_chain=list(sample.provenance_chain),
        )

        orig_temp = getattr(self.generator.llm, "temperature", None)
        try:
            if orig_temp is not None:
                self.generator.llm.temperature = patched_config.llm_temperature
            result = self.generator.run([seed])
            return result[0] if result else None
        except Exception as exc:
            logger.debug("Pass 2 generation error for %s: %s", sample.id, exc)
            return None
        finally:
            if orig_temp is not None:
                self.generator.llm.temperature = orig_temp

    def _write_outputs(self, rejected2: list[RejectedSample], summary: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "rejected_pass2.jsonl", "w") as f:
            for s in rejected2:
                f.write(s.model_dump_json() + "\n")
        (self.output_dir / "pass2_summary.json").write_text(json.dumps(summary, indent=2))


def _mark_failed(sample: RejectedSample, suffix: str) -> None:
    sample.rejection_reason = sample.rejection_reason + f":{suffix}"
