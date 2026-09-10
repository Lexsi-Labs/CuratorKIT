"""
GRPOExporter — serialize DataSamples to GRPO group rollout format.

Output: {output_dir}/grpo.jsonl
Each line: {"prompt": "...", "responses": [...], "rewards": [...]}

The prompt maps to DataSample.instruction; responses and rewards map to
DataSample.responses and DataSample.reward_scores, which GRPORolloutTask
populates during generation.

Only task_type "grpo" is written — every other task_type is skipped, even
one with a non-empty `instruction` (e.g. instruction_following data), since
that has nothing to do with GRPO rollouts and would otherwise silently
produce a record with permanently-empty responses/rewards. Within
task_type "grpo" itself, empty responses/rewards are expected before
GRPORolloutTask has run and never warn on their own.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.exporters.compatibility import is_compatible, task_types_for
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "grpo"


class GRPOExporter(BaseExporter):
    """Export to GRPO group rollout format.

    Uses DataSample.responses and DataSample.reward_scores if populated.
    Falls back to empty arrays when no rollouts have been generated yet.
    """

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "grpo.jsonl"
        skipped = 0
        exported = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if not is_compatible(sample.task_type, _FORMAT) or not sample.instruction.strip():
                    skipped += 1
                    continue

                record = {
                    "prompt": sample.instruction,
                    "responses": sample.responses,
                    "rewards": sample.reward_scores,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        if skipped:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"GRPOExporter skipped {skipped}/{len(samples)} samples (wrote {exported}) — "
                f"not task_type {task_types_for(_FORMAT)}, or empty prompt (sample task_type(s) "
                f"seen: {sorted(task_types)}). Empty responses/rewards on an otherwise-compatible "
                "sample are expected (rollouts not generated yet) and don't trigger this warning.",
                UserWarning,
                stacklevel=2,
            )
