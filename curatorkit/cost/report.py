"""
CostReport — per-stage cost attribution.

The plan calls for a cost report "per pipeline stage." Each CostGateway
tags every call with a `stage_name` (set by the generator / gate that
made the call); the report aggregates by stage.

Built from the gateway's call log + the run's TokenBudget. Rendered as
both a JSON-safe dict (for manifest.json) and a markdown table (for the
human report alongside dataset_card.md).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CallRecord:
    """One LLM call's accounting entry."""

    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_seconds": round(self.latency_seconds, 3),
        }


@dataclass
class CostReport:
    """
    Aggregates CallRecords into per-stage and per-model rollups.

    Built by `CostReport.from_records(records)` — records are emitted by
    CostGateway each time `generate()` returns. The report itself is a
    plain dataclass with serialization helpers so it slots into the
    manifest without circular imports.
    """

    records: list[CallRecord] = field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None

    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_records(cls, records: list[CallRecord]) -> CostReport:
        return cls(records=list(records))

    def add(self, record: CallRecord) -> None:
        self.records.append(record)

    def mark_halted(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    # ─────────────────────────────────────────────────────────────────────

    def totals(self) -> dict[str, Any]:
        return {
            "calls": len(self.records),
            "prompt_tokens": sum(r.prompt_tokens for r in self.records),
            "completion_tokens": sum(r.completion_tokens for r in self.records),
            "total_tokens": sum(r.total_tokens for r in self.records),
            "cost_usd": round(sum(r.cost_usd for r in self.records), 6),
            "latency_seconds": round(sum(r.latency_seconds for r in self.records), 3),
        }

    def per_stage(self) -> dict[str, dict[str, Any]]:
        agg: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_seconds": 0.0,
            }
        )
        for r in self.records:
            slot = agg[r.stage]
            slot["calls"] += 1
            slot["prompt_tokens"] += r.prompt_tokens
            slot["completion_tokens"] += r.completion_tokens
            slot["total_tokens"] += r.total_tokens
            slot["cost_usd"] += r.cost_usd
            slot["latency_seconds"] += r.latency_seconds
        out: dict[str, dict[str, Any]] = {}
        for stage, slot in agg.items():
            out[stage] = {
                "calls": int(slot["calls"]),
                "prompt_tokens": int(slot["prompt_tokens"]),
                "completion_tokens": int(slot["completion_tokens"]),
                "total_tokens": int(slot["total_tokens"]),
                "cost_usd": round(slot["cost_usd"], 6),
                "latency_seconds": round(slot["latency_seconds"], 3),
            }
        return out

    def per_model(self) -> dict[str, dict[str, Any]]:
        agg: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "total_tokens": 0, "cost_usd": 0.0}
        )
        for r in self.records:
            slot = agg[r.model]
            slot["calls"] += 1
            slot["total_tokens"] += r.total_tokens
            slot["cost_usd"] += r.cost_usd
        return {
            m: {
                "calls": int(s["calls"]),
                "total_tokens": int(s["total_tokens"]),
                "cost_usd": round(s["cost_usd"], 6),
            }
            for m, s in agg.items()
        }

    # ─────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": self.totals(),
            "per_stage": self.per_stage(),
            "per_model": self.per_model(),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    def render_markdown(self) -> str:
        t = self.totals()
        lines = ["# Cost Attribution Report\n"]
        lines.append(
            f"- **Total LLM calls**: {t['calls']}\n"
            f"- **Total tokens**: {t['total_tokens']:,} "
            f"(prompt {t['prompt_tokens']:,}, completion {t['completion_tokens']:,})\n"
            f"- **Total cost**: ${t['cost_usd']:.4f}\n"
            f"- **Total LLM latency**: {t['latency_seconds']:.1f}s\n"
        )

        if self.halted:
            lines.append(f"\n> **Pipeline halted by budget gate.** Reason: `{self.halt_reason}`\n")

        lines.append("\n## Per-stage Attribution\n")
        lines.append("| Stage | Calls | Total tokens | Cost (USD) |")
        lines.append("|-------|------:|-------------:|-----------:|")
        for stage, slot in sorted(self.per_stage().items(), key=lambda kv: -kv[1]["cost_usd"]):
            lines.append(
                f"| {stage} | {slot['calls']:,} | {slot['total_tokens']:,} "
                f"| ${slot['cost_usd']:.4f} |"
            )

        lines.append("\n## Per-model Attribution\n")
        lines.append("| Model | Calls | Total tokens | Cost (USD) |")
        lines.append("|-------|------:|-------------:|-----------:|")
        for model, slot in sorted(self.per_model().items(), key=lambda kv: -kv[1]["cost_usd"]):
            lines.append(
                f"| `{model}` | {slot['calls']:,} | {slot['total_tokens']:,} "
                f"| ${slot['cost_usd']:.4f} |"
            )

        return "\n".join(lines) + "\n"

    def write(self, output_dir: Path, filename: str = "cost_report.md") -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        path.write_text(self.render_markdown(), encoding="utf-8")
        return path
