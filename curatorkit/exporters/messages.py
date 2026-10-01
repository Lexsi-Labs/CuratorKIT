"""
MessagesExporter — serialize DataSamples to chat `messages` format.

Output: {output_dir}/sft_messages.jsonl
Each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

This is the conversational layout TRL's SFTTrainer and tokenizer chat templates
take directly. metadata['system_prompt'] becomes a leading system turn, a
non-empty `input` is appended to the first user turn, and metadata['turns']
follow the first exchange (ShareGPT from/value turns are normalised to
role/content).

Only task_types curatorkit.exporters.compatibility lists as messages-compatible
(instruction_following, conversational, unpaired_preference) are written.
Everything else is skipped and counted in a warning.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.detection.normalizer import normalize_turn
from curatorkit.exporters.compatibility import is_compatible, task_types_for
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "messages"


def build_messages(sample: DataSample) -> list[dict[str, str]]:
    """The sample's conversation as role/content turns. Shared with
    ShareGPTExporter, which maps the same turns to from/value."""
    system = sample.metadata.get("system_prompt")
    user = f"{sample.instruction}\n\n{sample.input}" if sample.input else sample.instruction
    messages = [{"role": "system", "content": system}] if system else []
    messages += [
        {"role": "user", "content": user},
        {"role": "assistant", "content": sample.output},
    ]
    turns = sample.metadata.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            turn = normalize_turn(turn)[0]
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
    return messages


def write_conversations(
    samples: list[DataSample],
    output_path: Path,
    export_format: str,
    exporter_name: str,
    to_record,
) -> None:
    """Write one JSON line per compatible sample (`to_record(messages)`) and warn
    about skipped or empty samples. Shared by MessagesExporter and ShareGPTExporter."""
    skipped = 0
    exported = 0
    empty = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            if not is_compatible(sample.task_type, export_format):
                skipped += 1
                continue
            if not sample.instruction.strip() or not (sample.output or "").strip():
                empty += 1
            f.write(json.dumps(to_record(build_messages(sample)), ensure_ascii=False) + "\n")
            exported += 1

    if skipped:
        skipped_task_types = {
            s.task_type for s in samples if not is_compatible(s.task_type, export_format)
        }
        warnings.warn(
            f"{exporter_name} skipped {skipped}/{len(samples)} samples (wrote {exported}) — "
            f"task_type not {export_format}-compatible (skipped: {sorted(skipped_task_types)}; "
            f"{export_format} accepts: {task_types_for(export_format)}).",
            UserWarning,
            stacklevel=3,
        )
    if empty:
        task_types = {s.task_type for s in samples}
        warnings.warn(
            f"{exporter_name} wrote {empty}/{exported} conversations with empty "
            f"turns (sample task_type(s): {sorted(task_types)}).",
            UserWarning,
            stacklevel=3,
        )


class MessagesExporter(BaseExporter):
    """Export to multi-turn chat `messages` (role/content) format."""

    filename = "sft_messages.jsonl"
    columns = {"messages": "list of {role, content}; role is system, user or assistant"}

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        write_conversations(
            samples,
            output_dir / self.filename,
            _FORMAT,
            "MessagesExporter",
            lambda messages: {"messages": messages},
        )
