"""
ArgillaExporter — push CuratorKIT samples to an Argilla workspace for review.

Argilla is an OSS data-review platform (https://argilla.io). Samples land
in a Dataset where annotators can rate, edit, or override LLM-generated
content. The output of human review can then be re-ingested into
CuratorKIT via the standard JSONLReader / huggingface connector.

This module is intentionally minimal — it does not depend on Argilla being
installed (lazy import) and falls back to a no-op with a clear warning if
the SDK is unavailable. The dataset schema mirrors the CuratorKIT
DataSample contract so a reviewer can see every relevant field:

  Fields:
    instruction, input, output, chosen, rejected
  Questions (for the reviewer):
    rating          — 1-5 Likert on overall quality
    correctness     — pass / fail
    comment         — free-text feedback
  Metadata properties:
    source_uri, task_type, generator, bundle_id, failure_mode (if present),
    diagnosis (compact JSON), rejection_reason (for rejected items)

Both passed and rejected samples can be pushed in one call so reviewers
see the full picture.

Compatibility: Argilla 2.x SDK (rg.Dataset / rg.Settings / dataset.records.log).
Argilla 1.x is not supported — the API surface changed substantively.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from curatorkit.interfaces import BaseExporter
from curatorkit.schema import DataSample, RejectedSample

STEP_VERSION = "1.0.0"


def _load_argilla() -> Any:
    try:
        import argilla as rg  # type: ignore

        return rg
    except ImportError:
        return None


# ───────────────────────────────────────────────────────────────────────────
# Record building
# ───────────────────────────────────────────────────────────────────────────

_TEXT_FIELDS = ("instruction", "input", "output", "chosen", "rejected")


def _sample_to_record_dict(
    sample: DataSample,
    record_type: str,
) -> dict[str, Any]:
    """
    Build the dict an Argilla Record consumes.

    Argilla 2.x records take:
      - fields: dict[str, str]
      - metadata: dict[str, Any]   (typed via dataset.settings.metadata)
      - id: str  (we use DataSample.id)
    """
    fields = {}
    for f in _TEXT_FIELDS:
        value = getattr(sample, f, "") or ""
        if value:
            fields[f] = str(value)

    # Always at least one non-empty field — Argilla requires it.
    if not fields:
        fields["instruction"] = "(empty sample)"

    metadata: dict[str, Any] = {
        "source_uri": str(sample.source_uri or "unknown"),
        "task_type": str(sample.task_type),
        "record_type": record_type,
    }

    # Carry common metadata keys forward
    for key in ("generator", "bundle_id", "role_in_bundle", "persona", "style"):
        if key in sample.metadata:
            metadata[key] = str(sample.metadata[key])[:200]

    # RejectedSample diagnostic fields (from the Adaptive Diagnostic Loop)
    if isinstance(sample, RejectedSample):
        metadata["rejection_reason"] = sample.rejection_reason
        metadata["rejecting_step"] = sample.rejecting_step
        if sample.diagnosis is not None:
            metadata["failure_mode"] = sample.diagnosis.mode.value
            # Compact JSON so Argilla's UI shows it cleanly
            metadata["diagnosis"] = json.dumps(sample.diagnosis.to_dict(), default=str)[:500]

    if sample.label is not None:
        metadata["label"] = float(sample.label)

    return {"id": sample.id, "fields": fields, "metadata": metadata}


# ───────────────────────────────────────────────────────────────────────────
# Settings builder
# ───────────────────────────────────────────────────────────────────────────


def _build_settings(rg: Any) -> Any:
    """
    Build the Argilla Dataset settings (fields, questions, metadata schema).

    Each text field is optional so a record carrying only `output` (pretrain)
    or only `chosen/rejected` (preference) is still valid.
    """
    fields = [rg.TextField(name=f, required=False, use_markdown=True) for f in _TEXT_FIELDS]

    questions = [
        rg.RatingQuestion(
            name="rating",
            values=[1, 2, 3, 4, 5],
            description="Overall quality on a 1-5 Likert scale.",
        ),
        rg.LabelQuestion(
            name="correctness",
            labels=["pass", "fail", "needs_edit"],
            description="Is this sample acceptable training data?",
        ),
        rg.TextQuestion(
            name="comment",
            description="Optional free-text feedback / reasoning.",
            required=False,
        ),
    ]

    metadata = [
        rg.TermsMetadataProperty(name="task_type"),
        rg.TermsMetadataProperty(name="record_type"),
        rg.TermsMetadataProperty(name="source_uri"),
        rg.TermsMetadataProperty(name="failure_mode"),
        rg.TermsMetadataProperty(name="rejection_reason"),
        rg.TermsMetadataProperty(name="generator"),
        rg.TermsMetadataProperty(name="role_in_bundle"),
        rg.FloatMetadataProperty(name="label"),
    ]

    return rg.Settings(
        guidelines=(
            "Review CuratorKIT-curated samples. Use rating + correctness to "
            "flag samples for inclusion / exclusion / rewrite. Comments are "
            "preserved in the audit trail."
        ),
        fields=fields,
        questions=questions,
        metadata=metadata,
        allow_extra_metadata=True,
    )


# ───────────────────────────────────────────────────────────────────────────
# ArgillaExporter
# ───────────────────────────────────────────────────────────────────────────


class ArgillaExporter(BaseExporter):
    """
    Push CuratorKIT samples to an Argilla 2.x workspace.

    Parameters
    ----------
    api_url : str
        Argilla server URL (e.g., "https://my-argilla.huggingface.cloud").
    api_key : str
        API token.
    workspace : str
        Argilla workspace name. Created if it doesn't exist.
    dataset_name : str
        Argilla dataset name. Created with the standard CuratorKIT schema
        if it doesn't exist.
    include_rejected : bool
        Push rejected samples alongside passed (default True).
        Reviewers can then confirm or override rejection decisions.
    rejected_samples : list[RejectedSample] | None
        Override the rejected list passed at construction. Useful when the
        Pipeline result's rejected list lives in a separate object.
    write_local_jsonl_fallback : bool
        If Argilla isn't installed or the connection fails, write the
        records to `{output_dir}/argilla_records.jsonl` so the human-review
        bundle isn't lost. Default True.
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        workspace: str = "argilla",
        dataset_name: str = "curatorkit_review",
        include_rejected: bool = True,
        rejected_samples: list[RejectedSample] | None = None,
        write_local_jsonl_fallback: bool = True,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.workspace = workspace
        self.dataset_name = dataset_name
        self.include_rejected = include_rejected
        self.rejected_samples = rejected_samples or []
        self.write_local_jsonl_fallback = write_local_jsonl_fallback

    # ─────────────────────────────────────────────────────────────────────

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        rg = _load_argilla()
        records = [_sample_to_record_dict(s, record_type="passed") for s in samples]
        if self.include_rejected and self.rejected_samples:
            records.extend(
                _sample_to_record_dict(s, record_type="rejected") for s in self.rejected_samples
            )

        # Local JSONL fallback — always write it so the bundle is preserved
        # even if the Argilla push succeeds (it acts as an audit log).
        if self.write_local_jsonl_fallback:
            self._write_local_jsonl(records, output_dir)

        if rg is None:
            warnings.warn(
                "[ArgillaExporter] argilla SDK not installed — wrote local "
                "argilla_records.jsonl only. Install `curatorkit[review]` "
                "to push to a live workspace.",
                stacklevel=2,
            )
            return

        if not self.api_url or not self.api_key:
            warnings.warn(
                "[ArgillaExporter] api_url or api_key missing — wrote local "
                "argilla_records.jsonl only.",
                stacklevel=2,
            )
            return

        try:
            self._push_to_argilla(rg, records)
        except Exception as e:
            warnings.warn(
                f"[ArgillaExporter] push failed ({type(e).__name__}: {e}). "
                "Local argilla_records.jsonl is intact for retry.",
                stacklevel=2,
            )

    # ─────────────────────────────────────────────────────────────────────

    def _write_local_jsonl(
        self,
        records: list[dict[str, Any]],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "argilla_records.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    def _push_to_argilla(self, rg: Any, records: list[dict[str, Any]]) -> None:
        """Connect, ensure workspace + dataset exist, log records."""
        client = rg.Argilla(api_url=self.api_url, api_key=self.api_key)

        # Workspace — create if missing
        ws = None
        try:
            ws = client.workspaces(self.workspace)
        except Exception:
            ws = None
        if ws is None:
            try:
                rg.Workspace(name=self.workspace, client=client).create()
            except Exception:
                # Workspace creation requires admin; if it fails, the dataset
                # creation below will surface a clearer error.
                pass

        # Dataset — create if missing, otherwise reuse
        dataset = None
        try:
            dataset = client.datasets(self.dataset_name, workspace=self.workspace)
        except Exception:
            dataset = None

        if dataset is None:
            dataset = rg.Dataset(
                name=self.dataset_name,
                settings=_build_settings(rg),
                workspace=self.workspace,
                client=client,
            )
            dataset.create()

        # Build Record objects
        rg_records = [
            rg.Record(
                id=r["id"],
                fields=r["fields"],
                metadata=r["metadata"],
            )
            for r in records
        ]
        dataset.records.log(rg_records)
