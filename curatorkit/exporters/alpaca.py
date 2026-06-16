"""
AlpacaExporter — serialize DataSamples to Alpaca JSONL format.

Output: {output_dir}/sft_alpaca.jsonl
Each line: {"instruction": "...", "input": "...", "output": "..."}

This is the primary SFT format. TRL's SFTTrainer accepts this directly.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample


class AlpacaExporter(BaseExporter):
    """Export to Alpaca instruction-following format."""

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "sft_alpaca.jsonl"
        empty = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "instruction": sample.instruction,
                    "input": sample.input if sample.input else "",
                    "output": sample.output if sample.output else "",
                }
                if not record["instruction"].strip() or not record["output"].strip():
                    empty += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if empty:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"AlpacaExporter wrote {empty}/{len(samples)} rows with an empty "
                f"instruction or output (sample task_type(s): {sorted(task_types)}). "
                "The data likely doesn't match the Alpaca SFT format — for preference "
                'data use export_formats=["dpo"].',
                UserWarning,
                stacklevel=2,
            )
