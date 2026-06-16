"""
PPOExporter — serialize DataSamples to PPO prompt format.

Output: {output_dir}/ppo.jsonl
Each line: {"prompt": "..."}

The prompt field maps directly to DataSample.instruction. PPO trainers
collect rollouts and rewards online during training, so only prompts are
emitted — reference responses, reward scores, and KL penalty settings are
configured in the trainer, not in the exported file.
"""

from __future__ import annotations

import json
from pathlib import Path

from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample


class PPOExporter(BaseExporter):
    """Export prompts in PPO training format."""

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "ppo.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {"prompt": sample.instruction}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
