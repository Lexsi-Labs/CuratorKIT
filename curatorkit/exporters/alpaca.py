"""
AlpacaExporter — serialize DataSamples to Alpaca JSONL format.

Output: {output_dir}/sft_alpaca.jsonl
Each line: {"instruction": "...", "input": "...", "output": "..."}

This is the primary SFT format. TRL's SFTTrainer accepts this directly.

Only task_types curatorkit.exporters.compatibility lists as alpaca-compatible
(instruction_following, unpaired_preference) are written. Everything else is
skipped and counted in a warning — notably conversational: writing it here
would silently keep only the first turn and drop the rest, so it's skipped
instead (use ShareGPTExporter, which preserves the full turn list).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.exporters.compatibility import is_compatible, task_types_for
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "alpaca"


class AlpacaExporter(BaseExporter):
    """Export to Alpaca instruction-following format."""

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "sft_alpaca.jsonl"
        skipped = 0
        exported = 0
        empty = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if not is_compatible(sample.task_type, _FORMAT):
                    skipped += 1
                    continue

                record = {
                    "instruction": sample.instruction,
                    "input": sample.input if sample.input else "",
                    "output": sample.output if sample.output else "",
                }
                if not record["instruction"].strip() or not record["output"].strip():
                    empty += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        if skipped:
            skipped_task_types = {
                s.task_type for s in samples if not is_compatible(s.task_type, _FORMAT)
            }
            warnings.warn(
                f"AlpacaExporter skipped {skipped}/{len(samples)} samples (wrote {exported}) — "
                f"task_type not alpaca-compatible (skipped: {sorted(skipped_task_types)}; "
                f"alpaca accepts: {task_types_for(_FORMAT)}). Conversational data keeps only its "
                'first turn here and silently drops the rest if forced through — use '
                '"sharegpt" for multi-turn data instead.',
                UserWarning,
                stacklevel=2,
            )
        if empty:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"AlpacaExporter wrote {empty}/{exported} rows with an empty "
                f"instruction or output (sample task_type(s): {sorted(task_types)}).",
                UserWarning,
                stacklevel=2,
            )
