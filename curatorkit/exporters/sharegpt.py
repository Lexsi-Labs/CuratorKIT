"""
ShareGPTExporter — serialize DataSamples to ShareGPT conversation format.

Output: {output_dir}/sft_sharegpt.jsonl
Each line: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

If metadata['turns'] exists, serializes the full multi-turn list instead
of constructing a two-turn conversation from instruction/output.

Only task_types curatorkit.exporters.compatibility lists as sharegpt-compatible
(instruction_following, conversational, unpaired_preference) are written.
Everything else is skipped and counted in a warning.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.exporters.compatibility import is_compatible, task_types_for
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "sharegpt"


class ShareGPTExporter(BaseExporter):
    """Export to ShareGPT multi-turn conversation format."""

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "sft_sharegpt.jsonl"
        skipped = 0
        exported = 0
        empty = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if not is_compatible(sample.task_type, _FORMAT):
                    skipped += 1
                    continue

                turns = sample.metadata.get("turns")
                if turns and isinstance(turns, list) and len(turns) > 0:
                    # Full multi-turn: prepend the primary instruction/output pair
                    conversations = [
                        {"from": "human", "value": sample.instruction},
                        {"from": "gpt", "value": sample.output},
                        *turns,
                    ]
                else:
                    conversations = [
                        {"from": "human", "value": sample.instruction},
                        {"from": "gpt", "value": sample.output},
                    ]
                if not sample.instruction.strip() or not (sample.output or "").strip():
                    empty += 1
                record = {"conversations": conversations}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        if skipped:
            skipped_task_types = {
                s.task_type for s in samples if not is_compatible(s.task_type, _FORMAT)
            }
            warnings.warn(
                f"ShareGPTExporter skipped {skipped}/{len(samples)} samples (wrote {exported}) — "
                f"task_type not sharegpt-compatible (skipped: {sorted(skipped_task_types)}; "
                f"sharegpt accepts: {task_types_for(_FORMAT)}).",
                UserWarning,
                stacklevel=2,
            )
        if empty:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"ShareGPTExporter wrote {empty}/{exported} conversations with empty "
                f"turns (sample task_type(s): {sorted(task_types)}).",
                UserWarning,
                stacklevel=2,
            )
