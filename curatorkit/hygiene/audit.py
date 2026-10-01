"""
SafetyComplianceReport — aggregate safety-gate rejections into a report.

Rendered alongside the dataset card.

Aggregates per-gate counts, per-category breakdowns, redaction stats,
and backend (presidio / llama-guard / heuristic) info. Renders as
markdown alongside dataset_card.md when any safety gate is enabled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curatorkit.schema import DataSample, RejectedSample

_SAFETY_STEP_NAMES = ("PIIGate", "JailbreakGate", "OutputSafetyGate")


@dataclass
class SafetyComplianceReport:
    """
    Summary of every safety-gate decision in a run.

    Built from a PipelineResult's passed + rejected lists by reading
    ProvenanceRecord notes from each safety gate.

    Fields:
      pii_redactions       — {field: {entity: count}} from passed samples
      pii_rejections       — count of samples rejected by PIIGate
      jailbreak_rejections — count rejected by JailbreakGate
      output_rejections    — count rejected by OutputSafetyGate
      hazard_categories    — {S-code: count} from OutputSafetyGate rejections
      backends             — {gate: backend_used}
    """

    pii_redactions: dict[str, dict[str, int]] = field(default_factory=dict)
    pii_rejections: int = 0
    jailbreak_rejections: int = 0
    output_rejections: int = 0
    hazard_categories: dict[str, int] = field(default_factory=dict)
    backends: dict[str, str] = field(default_factory=dict)
    total_safety_rejections: int = 0
    total_samples_audited: int = 0

    @classmethod
    def from_pipeline(
        cls,
        passed: list[DataSample],
        rejected: list[RejectedSample],
    ) -> SafetyComplianceReport:
        rep = cls()
        rep.total_samples_audited = len(passed) + len(rejected)

        # Pass 1: pull redaction stats and backends from passed-through provenance.
        for s in passed:
            for rec in s.provenance_chain:
                if rec.step_name in _SAFETY_STEP_NAMES:
                    rep.backends[rec.step_name] = rec.notes.get("backend", "unknown")
                if rec.step_name == "PIIGate" and rec.notes.get("pii_detected"):
                    if rec.notes.get("mode") == "redact":
                        entity_summary = rec.notes.get("entities", {})
                        for field_name, ent_counts in entity_summary.items():
                            slot = rep.pii_redactions.setdefault(field_name, {})
                            for ent, c in ent_counts.items():
                                slot[ent] = slot.get(ent, 0) + int(c)

        # Pass 2: count rejections per gate + hazard categories.
        cat_counter: Counter[str] = Counter()
        for r in rejected:
            step = r.rejecting_step
            if step not in _SAFETY_STEP_NAMES:
                continue
            rep.total_safety_rejections += 1
            if step == "PIIGate":
                rep.pii_rejections += 1
            elif step == "JailbreakGate":
                rep.jailbreak_rejections += 1
            elif step == "OutputSafetyGate":
                rep.output_rejections += 1
                for rec in r.provenance_chain:
                    if rec.step_name == "OutputSafetyGate":
                        cat_counter.update(rec.notes.get("categories", []))
            # Record backend for the rejecting gate too
            for rec in r.provenance_chain:
                if rec.step_name == step:
                    rep.backends.setdefault(step, rec.notes.get("backend", "unknown"))

        rep.hazard_categories = dict(cat_counter)
        return rep

    # ─────────────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """True if no safety-gate provenance was found anywhere in the run."""
        return not self.backends and self.total_safety_rejections == 0 and not self.pii_redactions

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples_audited": self.total_samples_audited,
            "total_safety_rejections": self.total_safety_rejections,
            "pii_rejections": self.pii_rejections,
            "jailbreak_rejections": self.jailbreak_rejections,
            "output_rejections": self.output_rejections,
            "hazard_categories": self.hazard_categories,
            "pii_redactions": self.pii_redactions,
            "backends": self.backends,
        }

    def render_markdown(self) -> str:
        lines = ["# Safety Compliance Report\n"]
        lines.append(
            f"Samples audited: **{self.total_samples_audited}**  \n"
            f"Total safety rejections: **{self.total_safety_rejections}**\n"
        )

        if self.backends:
            lines.append("\n## Gates and Backends\n")
            for gate, backend in sorted(self.backends.items()):
                lines.append(f"- **{gate}** — backend: `{backend}`")
            lines.append("")

        lines.append("\n## Per-gate Rejection Counts\n")
        lines.append(f"- PIIGate:           {self.pii_rejections}")
        lines.append(f"- JailbreakGate:     {self.jailbreak_rejections}")
        lines.append(f"- OutputSafetyGate:  {self.output_rejections}")

        if self.hazard_categories:
            from curatorkit.hygiene.output_safety import _HAZARD_CATEGORIES

            lines.append("\n## OutputSafetyGate Hazard Categories\n")
            for code, count in sorted(self.hazard_categories.items(), key=lambda x: -x[1]):
                name = _HAZARD_CATEGORIES.get(code, "unknown")
                lines.append(f"- `{code}` ({name}): {count}")

        if self.pii_redactions:
            lines.append("\n## PII Redactions (samples passed through)\n")
            for field_name, ent_counts in sorted(self.pii_redactions.items()):
                lines.append(f"- **{field_name}**:")
                for ent, c in sorted(ent_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  - `{ent}`: {c}")

        lines.append(
            "\n---\n"
            "*Every safety rejection has a structured RejectedSample with a "
            "ProvenanceRecord — see rejected.jsonl for per-sample audit detail.*\n"
        )
        return "\n".join(lines)

    def write(self, output_dir: Path, filename: str = "safety_report.md") -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        path.write_text(self.render_markdown(), encoding="utf-8")
        return path
