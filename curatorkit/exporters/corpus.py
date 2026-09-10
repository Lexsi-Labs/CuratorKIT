"""
CorpusExporter — export raw corpus chunks with full provenance metadata.

Designed for corpus building. Unlike the Alpaca exporter which wraps
chunks in an instruction-tuning schema (discarding page, heading, table_html),
CorpusExporter writes a flat JSONL where every field is useful for downstream
inspection, filtering, and generation task routing.

Output: corpus.jsonl
Each line is a JSON object. Text chunks:
    {
        "text":         "<chunk text>",
        "source_file":  "ICC-URDG-758.pdf",
        "source_uri":   "ICC-URDG-758.pdf",
        "page":         3,
        "heading":      "Article 5 — Extend or Pay",
        "chunk_index":  12,
        "content_type": "text",
        "task_type":    "language_modeling",
        "table_html":   "",
        "table_bbox":   []
    }

Table chunks (when extract_tables=True) additionally carry:
    {
        ...
        "content_type": "table",
        "task_type":    "language_modeling",
        "table_html":   "<table><tr><th>Bank</th>...</table>",
        "table_bbox":   [72, 200, 540, 380]
    }

The `text` field is populated from:
    sample.output  → language_modeling task_type (PDFReader default)
    sample.input   → source_chunk task_type
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from curatorkit.exporters.compatibility import is_compatible, task_types_for
from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample

_FORMAT = "corpus"


class CorpusExporter(BaseExporter):
    """Export corpus chunks to corpus.jsonl with full chunk metadata.

    Only task_type "language_modeling" or "source_chunk" is written —
    everything else is skipped, even with non-empty output/input (e.g. any
    SFT/preference/GRPO task_type also has *some* text in one of those
    fields; that doesn't make it a corpus chunk).
    """

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_path = output_dir / "corpus.jsonl"
        skipped = 0
        exported = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if not is_compatible(sample.task_type, _FORMAT):
                    skipped += 1
                    continue

                # Resolve chunk text regardless of which field it landed in.
                # PDFReader uses language_modeling (text in output).
                # source_chunk task_type puts text in input.
                text = sample.output or sample.input or ""
                if not text:
                    skipped += 1
                    continue

                meta = sample.metadata or {}
                record = {
                    "text": text,
                    "source_file": meta.get("source_file", ""),
                    "source_uri": sample.source_uri or "",
                    "page": meta.get("page"),
                    "heading": meta.get("parent_heading"),
                    "chunk_index": meta.get("chunk_index"),
                    "content_type": meta.get("content_type", "text"),
                    "task_type": sample.task_type,
                    # Table-specific fields — empty string / empty list for text chunks
                    # so downstream code can always access them without key checks.
                    "table_html": meta.get("table_html", ""),
                    "table_bbox": meta.get("table_bbox", []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        if skipped:
            task_types = {s.task_type for s in samples}
            warnings.warn(
                f"CorpusExporter skipped {skipped}/{len(samples)} samples (wrote {exported}) — "
                f"not task_type {task_types_for(_FORMAT)}, or neither output nor input has text "
                f"(sample task_type(s) seen: {sorted(task_types)}).",
                UserWarning,
                stacklevel=2,
            )
