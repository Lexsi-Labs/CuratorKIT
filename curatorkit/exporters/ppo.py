"""
PPOExporter — serialize DataSamples to PPO prompt format.

Output: {output_dir}/ppo.jsonl
Each line: {"prompt": "..."}

The prompt field maps directly to DataSample.instruction. PPO trainers
collect rollouts and rewards online during training, so only prompts are
emitted — reference responses, reward scores, and KL penalty settings are
configured in the trainer, not in the exported file.

Only task_type "prompt_only" is written — everything else is skipped, even
with a non-empty `instruction` (e.g. instruction_following data), since a
non-empty prompt field alone doesn't mean the sample was meant to seed a
PPO run; it just means some other task populated `instruction` too.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.exporters.compatibility import is_compatible, task_types_for
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "ppo"


class PPOExporter(BaseExporter):
    """Export prompts in PPO training format."""

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "ppo.jsonl"
        skipped = 0
        exported = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if not is_compatible(sample.task_type, _FORMAT) or not sample.instruction.strip():
                    skipped += 1
                    continue
                record = {"prompt": sample.instruction}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        if skipped:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"PPOExporter skipped {skipped}/{len(samples)} samples (wrote {exported}) — "
                f"not task_type {task_types_for(_FORMAT)}, or empty instruction (sample "
                f"task_type(s) seen: {sorted(task_types)}).",
                UserWarning,
                stacklevel=2,
            )
