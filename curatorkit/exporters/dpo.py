"""
DPOExporter — serialize DataSamples to TRL DPOTrainer format.

Output: {output_dir}/dpo.jsonl

Each line: {"prompt": "...", "chosen": "...", "rejected": "..."}

For conversational chosen/rejected (stored as JSON-encoded turn lists),
the values are written as lists of role/content dicts — exactly what
TRL's DPOTrainer expects when processing conversational preference data.

Only task_types curatorkit.exporters.compatibility lists as dpo-compatible
(preference, implicit_preference) are written.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.exporters.compatibility import is_compatible, task_types_for
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "dpo"


def _try_parse_turns(value: str) -> list | str:
    """
    If value is a JSON-encoded list of turns, parse it.
    Return the original string if it is not JSON or not a list.
    """
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return value


class DPOExporter(BaseExporter):
    """
    Export preference data in TRL DPO format.

    Only samples with task_type "preference" or "implicit_preference"
    are written. All others are skipped (not rejected — skipping is
    intentional when exporting a multi-task pipeline subset).
    """

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "dpo.jsonl"
        exported = 0
        skipped = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if not is_compatible(sample.task_type, _FORMAT):
                    skipped += 1
                    continue

                if not sample.chosen or not sample.rejected:
                    skipped += 1
                    continue

                chosen_val = _try_parse_turns(sample.chosen)
                rejected_val = _try_parse_turns(sample.rejected)

                # Build instruction (prompt) value
                # For implicit preference, instruction may be a JSON list of turns
                prompt_val: str | list
                if sample.instruction.startswith("["):
                    prompt_val = _try_parse_turns(sample.instruction)
                else:
                    prompt_val = sample.instruction

                record = {
                    "prompt": prompt_val,
                    "chosen": chosen_val,
                    "rejected": rejected_val,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        if skipped:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"DPOExporter skipped {skipped}/{len(samples)} samples (wrote {exported}) — "
                f"not task_type {task_types_for(_FORMAT)}, or missing chosen/rejected "
                f"(sample task_type(s) seen: {sorted(task_types)}). If this run wasn't meant to "
                'produce preference data, drop "dpo" from export_formats.',
                UserWarning,
                stacklevel=2,
            )
