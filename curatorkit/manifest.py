"""
ProvenanceManifest and DatasetCardGenerator.

manifest.json and dataset_card.md are ALWAYS emitted after a pipeline run.
They cannot be disabled via config. Provenance is a design constraint, not
an optional feature.

manifest.json top-level keys:
  pipeline_config_hash    SHA-256 of the full pipeline YAML config
  run_timestamp           ISO-8601 UTC
  source_files            list of {path, sha256}
  stage_counts            {step_name: {input_count, output_count, rejected_count}}
  rejected_breakdown      {rejection_reason: count}
  dedup_stats             extracted from MinHashDeduplicator provenance notes
  minhash_threshold       float | null
  wall_clock_seconds      float
  tool_versions           {curatorkit: "<package version>", python: "..."}
  diversity_stats         reserved; currently null
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from curatorkit import __version__ as _CURATORKIT_VERSION
from curatorkit.pipeline import PipelineResult
from curatorkit.schema import DataSample


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_dedup_stats(
    samples: list[DataSample],
) -> tuple[dict[str, object], float | None]:
    """Pull dedup stats from the most recent MinHashDeduplicator provenance record."""
    stats: dict[str, object] = {}
    threshold = None

    for sample in samples:
        for rec in reversed(sample.provenance_chain):
            if rec.step_name == "MinHashDeduplicator":
                notes = rec.notes
                stats = {
                    "near_duplicates_removed": notes.get("near_duplicates_removed"),
                    "surviving_count": notes.get("surviving_count"),
                }
                threshold = notes.get("minhash_threshold")
                return stats, threshold
            if rec.step_name == "ExactDeduplicator":
                notes = rec.notes
                stats["exact_duplicates_removed"] = notes.get("exact_duplicates_removed")

    return stats, threshold


def _collect_source_files(samples: list[DataSample]) -> list[dict[str, str]]:
    """Extract unique source URIs from provenance chains."""
    seen: set[str] = set()
    for sample in samples:
        for rec in sample.provenance_chain:
            if "source_file" in rec.notes:
                seen.add(rec.notes["source_file"])
    return [{"path": p} for p in sorted(seen)]


def _extract_sampler_stats(samples: list[DataSample]) -> dict[str, object] | None:
    """Pull StratifiedSampler stats (adjustments + shortfalls) from the most recent record."""
    for sample in samples:
        for rec in reversed(sample.provenance_chain):
            if rec.step_name == "StratifiedSampler":
                notes = rec.notes
                return {
                    "category_field": notes.get("category_field"),
                    "binding_total": notes.get("binding_total"),
                    "output_total": notes.get("output_total"),
                    "adjustments": notes.get("adjustments"),
                    "shortfalls": notes.get("shortfalls"),
                    "has_shortfall": notes.get("has_shortfall", False),
                }
    return None


def _extract_token_stats(samples: list[DataSample]) -> dict[str, int]:
    """Sum prompt/completion tokens across all generation task ProvenanceRecords."""
    prompt_total = completion_total = 0
    for sample in samples:
        for rec in sample.provenance_chain:
            notes = rec.notes
            if "prompt_tokens" in notes:
                prompt_total += int(notes.get("prompt_tokens", 0) or 0)
                completion_total += int(notes.get("completion_tokens", 0) or 0)
    return {
        "total_prompt_tokens": prompt_total,
        "total_completion_tokens": completion_total,
        "total_tokens": prompt_total + completion_total,
    }


class ProvenanceManifest:
    """Build and write the pipeline manifest after a run completes."""

    def __init__(
        self,
        result: PipelineResult,
        pipeline_config_hash: str = "unknown",
        output_dir: Path | None = None,
    ) -> None:
        self.result = result
        self.pipeline_config_hash = pipeline_config_hash
        self.output_dir = output_dir or Path("output")

    def build(self) -> dict[str, object]:
        all_samples = self.result.passed + list(self.result.rejected)

        rejected_breakdown: dict[str, int] = Counter(
            r.rejection_reason for r in self.result.rejected
        )

        dedup_stats, minhash_threshold = _extract_dedup_stats(self.result.passed)
        source_files = _collect_source_files(all_samples)
        sampler_stats = _extract_sampler_stats(self.result.passed)
        token_stats = _extract_token_stats(all_samples)

        # Diagnostic stats — populated when enable_diagnostic_probe=True
        diagnostics = getattr(self.result, "diagnostics", None)
        diagnostic_stats = diagnostics.to_dict() if diagnostics is not None else None

        return {
            "pipeline_config_hash": self.pipeline_config_hash,
            "run_timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z",
            "source_files": source_files,
            "stage_counts": self.result.stage_counts,
            "rejected_breakdown": dict(rejected_breakdown),
            "dedup_stats": dedup_stats,
            "minhash_threshold": minhash_threshold,
            "sampler_stats": sampler_stats,
            "token_stats": token_stats,
            "wall_clock_seconds": round(self.result.wall_clock_seconds, 3),
            "tool_versions": {
                "curatorkit": _CURATORKIT_VERSION,
                "python": sys.version.split()[0],
            },
            "diversity_stats": None,
            "diagnostic_stats": diagnostic_stats,
            "diagnostic_files": (
                ["diagnostic_summary.json"] if diagnostic_stats is not None else []
            ),
        }

    def write(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.build()
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        return manifest_path

    def write_rejected_sidecar(self) -> Path:
        """Write rejected.jsonl — always written, even when empty."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = self.output_dir / "rejected.jsonl"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            for sample in self.result.rejected:
                # Use model_dump() not model_dump_json() — RejectedSample.model_dump()
                # handles FailureDiagnosis (a dataclass) via to_dict(). Pydantic's
                # JSON encoder cannot serialize dataclasses and silently writes null.
                # provenance_chain is included so rejected samples can be audited
                # post-run (which step rejected, with what config, at what timestamp).
                d = sample.model_dump()
                f.write(json.dumps(d, default=str) + "\n")
        return sidecar_path

    def write_checksums(self, output_files: list[Path]) -> Path:
        """Write SHA-256 checksums for all output files."""
        checksums_path = self.output_dir / "checksums.txt"
        with open(checksums_path, "w", encoding="utf-8") as f:
            for path in output_files:
                if path.exists():
                    f.write(f"{_file_sha256(path)}  {path.name}\n")
        return checksums_path


class DatasetCardGenerator:
    """Generate a human-readable Markdown dataset card from a manifest."""

    def generate(
        self,
        manifest: dict[str, object],
        output_dir: Path,
        pipeline_name: str = "curatorkit_pipeline",
    ) -> Path:
        card_path = output_dir / "dataset_card.md"
        content = self._render(manifest, pipeline_name)
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(content)
        return card_path

    def _render(self, m: dict[str, object], name: str) -> str:
        stage_counts = m.get("stage_counts", {})
        rejected_breakdown = m.get("rejected_breakdown", {})
        dedup_stats = m.get("dedup_stats", {}) or {}
        minhash_threshold = m.get("minhash_threshold")
        tool_versions = m.get("tool_versions", {})
        source_files = m.get("source_files", [])

        # Gate pass rates
        gate_section = ""
        for step, counts in stage_counts.items():
            if "rejected_count" in counts:
                total_in = counts.get("input_count", 0)
                passed = counts.get("output_count", 0)
                rejected = counts.get("rejected_count", 0)
                rate = f"{100 * passed / total_in:.1f}%" if total_in else "N/A"
                gate_section += (
                    f"- **{step}**: {passed}/{total_in} passed ({rate}), {rejected} rejected\n"
                )

        rejection_detail = (
            "\n".join(
                f"  - `{reason}`: {count}"
                for reason, count in sorted(rejected_breakdown.items(), key=lambda x: -x[1])
            )
            or "  - None"
        )

        dedup_detail = (
            "\n".join(f"  - {k}: {v}" for k, v in dedup_stats.items())
            or "  - No deduplication step recorded"
        )

        source_list = (
            "\n".join(f"  - `{s.get('path', 'unknown')}`" for s in source_files) or "  - Unknown"
        )

        minhash_caveat = (
            f"MinHash threshold used: **{minhash_threshold}** — this is a starting point, "
            "not a universal answer. Adjust per domain (legal, medical, and other "
            "domain-specific corpora may need a lower threshold to avoid losing structurally "
            "similar but semantically valid examples)."
            if minhash_threshold is not None
            else "No MinHash deduplication applied."
        )

        return f"""# Dataset Card: {name}

## Summary

- **Run timestamp**: {m.get("run_timestamp", "unknown")}
- **Wall clock time**: {m.get("wall_clock_seconds", 0):.1f}s
- **Pipeline config hash**: `{m.get("pipeline_config_hash", "unknown")}`

## Source Documents

{source_list}

## Pipeline Stages

| Step | Input | Output | Rejected |
|------|-------|--------|----------|
{self._stage_table(stage_counts)}

## Gate Pass Rates and Rejection Breakdown

{gate_section or "No gates applied."}

**Rejection reasons:**
{rejection_detail}

## Deduplication

{dedup_detail}

## Known Limitations

- {minhash_caveat}
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

{chr(10).join(f"- **{k}**: {v}" for k, v in tool_versions.items())}
"""

    def _stage_table(self, stage_counts: dict[str, object]) -> str:
        rows = []
        for step, counts in stage_counts.items():
            inp = counts.get("input_count", counts.get("output_count", "—"))
            out = counts.get("output_count", counts.get("exported_count", "—"))
            rej = counts.get("rejected_count", "—")
            rows.append(f"| {step} | {inp} | {out} | {rej} |")
        return "\n".join(rows) if rows else "| — | — | — | — |"
