"""
ShareGPTExporter — serialize DataSamples to ShareGPT conversation format.

Output: {output_dir}/sft_sharegpt.jsonl
Each line: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

The turns are the same ones MessagesExporter writes, with roles mapped to
ShareGPT names (user -> human, assistant -> gpt; system and any other role
keep their name): metadata['system_prompt'] becomes a leading system turn, a
non-empty `input` is appended to the first human turn, and metadata['turns']
follow the first exchange. For TRL's SFTTrainer use MessagesExporter
(sft_messages.jsonl) instead.

Only task_types curatorkit.exporters.compatibility lists as sharegpt-compatible
(instruction_following, conversational, unpaired_preference) are written.
Everything else is skipped and counted in a warning.
"""

from __future__ import annotations

from pathlib import Path

from curatorkit.exporters.messages import write_conversations
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "sharegpt"
_SHAREGPT_ROLE = {"user": "human", "assistant": "gpt"}


def _to_sharegpt(messages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        "conversations": [
            {"from": _SHAREGPT_ROLE.get(m["role"], m["role"]), "value": m["content"]}
            for m in messages
        ]
    }


class ShareGPTExporter(BaseExporter):
    """Export to ShareGPT multi-turn conversation (from/value) format."""

    filename = "sft_sharegpt.jsonl"
    columns = {
        "conversations": "list of {from, value}; from is system, human or gpt",
    }

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        write_conversations(
            samples,
            output_dir / self.filename,
            _FORMAT,
            "ShareGPTExporter",
            _to_sharegpt,
        )
