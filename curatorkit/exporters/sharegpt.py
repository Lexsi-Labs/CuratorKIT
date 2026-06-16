"""
ShareGPTExporter — serialize DataSamples to ShareGPT conversation format.

Output: {output_dir}/sft_sharegpt.jsonl
Each line: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

If metadata['turns'] exists, serializes the full multi-turn list instead
of constructing a two-turn conversation from instruction/output.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample


class ShareGPTExporter(BaseExporter):
    """Export to ShareGPT multi-turn conversation format."""

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "sft_sharegpt.jsonl"
        empty = sum(1 for s in samples if not s.instruction.strip() or not (s.output or "").strip())
        if empty:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"ShareGPTExporter writing {empty}/{len(samples)} conversations with empty "
                f"turns (sample task_type(s): {sorted(task_types)}). The data likely doesn't "
                'match an SFT conversation format — for preference data use export_formats=["dpo"].',
                UserWarning,
                stacklevel=2,
            )
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
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
                record = {"conversations": conversations}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
