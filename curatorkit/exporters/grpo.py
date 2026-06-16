"""
GRPOExporter — serialize DataSamples to GRPO group rollout format.

Output: {output_dir}/grpo.jsonl
Each line: {"prompt": "...", "responses": [...], "rewards": [...]}

The prompt maps to DataSample.instruction; responses and rewards map to
DataSample.responses and DataSample.reward_scores, which GRPORolloutTask
populates during generation. Samples without rollouts (e.g. prompt-only
data) produce records with empty responses/rewards arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample


class GRPOExporter(BaseExporter):
    """Export to GRPO group rollout format.

    Uses DataSample.responses and DataSample.reward_scores if populated.
    Falls back to empty arrays when no rollouts have been generated.
    """

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "grpo.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "prompt": sample.instruction,
                    "responses": sample.responses,
                    "rewards": sample.reward_scores,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
