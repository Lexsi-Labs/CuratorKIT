"""
CheckpointManager — resume a mid-pipeline run from disk after a process restart.

Two checkpoint granularities:

  Stage-level (non-LLM steps):
    After each step completes, the full DataSample list is serialized atomically.
    On resume, completed steps are skipped and their output is loaded from disk.

  Batch-level (LLM generation tasks):
    During generation, results are appended after every checkpoint_batch_size
    samples. On failure, only the last incomplete batch needs to be re-processed.
    Worst-case re-work = one batch (default 256 samples).

A manifest.json in the checkpoint directory tracks progress. If the pipeline
config hash changes, the manifest is invalidated and the run starts fresh.

Usage (from Curator):
    mgr = CheckpointManager(Path("output/.checkpoints"), config_hash="abc123")

    # Stage-level (pipeline handles this automatically)
    mgr.save_stage("SchemaGate", samples)
    samples = mgr.load_stage("SchemaGate")

    # Batch-level (BaseGenerationTask handles this automatically)
    start = mgr.get_batch_resume_idx("QAGenerationTask")  # 0 on first run
    pre_passed, pre_rejected = mgr.load_batch_results("QAGenerationTask")
    # ... process batch ...
    mgr.append_batch("QAGenerationTask", batch_passed, batch_rejected, next_start=256)
    # ... when fully done ...
    mgr.finalize_batch_stage("QAGenerationTask", all_passed)
"""

from __future__ import annotations

import json
from pathlib import Path

from curatorkit.schema import DataSample, RejectedSample


def _slugify(name: str) -> str:
    """Convert a stage name to a filesystem-safe filename stem."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


class CheckpointManager:
    """Pipeline checkpoint tracker — stage-level and batch-level resume support."""

    def __init__(self, checkpoint_dir: Path, config_hash: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_hash = config_hash
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.checkpoint_dir / "manifest.json"
        self._manifest: dict = self._load_or_init_manifest()

    # ── Manifest helpers ─────────────────────────────────────────────────────

    def _load_or_init_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as fh:
                    data = json.load(fh)
                if data.get("config_hash") == self.config_hash:
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "config_hash": self.config_hash,
            "completed_stages": [],
            "batch_progress": {},
        }

    def _save_manifest(self) -> None:
        tmp = self._manifest_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(self._manifest, fh, indent=2)
        tmp.rename(self._manifest_path)

    # ── Stage-level API ──────────────────────────────────────────────────────

    def is_stage_complete(self, stage_name: str) -> bool:
        """Return True if this stage has a completed checkpoint on disk."""
        return stage_name in self._manifest["completed_stages"]

    def load_stage(self, stage_name: str) -> list[DataSample] | None:
        """Load the DataSample list saved after ``stage_name`` completed.

        Returns None if the checkpoint file does not exist.
        """
        path = self.checkpoint_dir / f"{_slugify(stage_name)}.jsonl"
        if not path.exists():
            return None
        samples: list[DataSample] = []
        with open(path) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    samples.append(DataSample.model_validate_json(stripped))
        return samples

    def save_stage(self, stage_name: str, samples: list[DataSample]) -> None:
        """Atomically serialize ``samples`` and mark ``stage_name`` as complete."""
        path = self.checkpoint_dir / f"{_slugify(stage_name)}.jsonl"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for s in samples:
                fh.write(s.model_dump_json() + "\n")
        tmp.rename(path)
        if stage_name not in self._manifest["completed_stages"]:
            self._manifest["completed_stages"].append(stage_name)
        self._save_manifest()

    # ── Batch-level API (LLM generation tasks) ───────────────────────────────

    def get_batch_resume_idx(self, stage_name: str) -> int:
        """Return the input-sample index to resume generation from.

        0 on the first run; the index of the first unprocessed sample on resume.
        """
        return self._manifest["batch_progress"].get(stage_name, {}).get("next_start", 0)

    def load_batch_results(
        self, stage_name: str
    ) -> tuple[list[DataSample], list[RejectedSample]]:
        """Return all DataSamples and RejectedSamples written so far for ``stage_name``."""
        batch_file = self.checkpoint_dir / f"{_slugify(stage_name)}_batches.jsonl"
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        if not batch_file.exists():
            return passed, rejected
        with open(batch_file) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                for s in record.get("passed", []):
                    passed.append(DataSample.model_validate(s))
                for r in record.get("rejected", []):
                    rejected.append(RejectedSample.model_validate(r))
        return passed, rejected

    def append_batch(
        self,
        stage_name: str,
        passed: list[DataSample],
        rejected: list[RejectedSample],
        next_start: int,
    ) -> None:
        """Append one completed batch to disk and commit the progress marker.

        ``next_start`` is the absolute index of the first sample that has NOT
        yet been processed (i.e. the next batch's starting index into the
        original input list). Updating the manifest AFTER the file write
        ensures that a crash between the two leaves the manifest pointing at
        the previous safe batch — the partial file write is harmless since we
        always re-read up to the committed ``next_start``.
        """
        batch_file = self.checkpoint_dir / f"{_slugify(stage_name)}_batches.jsonl"
        record = {
            "passed": [json.loads(s.model_dump_json()) for s in passed],
            "rejected": [json.loads(r.model_dump_json()) for r in rejected],
        }
        with open(batch_file, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if stage_name not in self._manifest["batch_progress"]:
            self._manifest["batch_progress"][stage_name] = {}
        self._manifest["batch_progress"][stage_name]["next_start"] = next_start
        self._save_manifest()

    def finalize_batch_stage(
        self, stage_name: str, total_passed: list[DataSample]
    ) -> None:
        """Mark a batch-mode stage fully complete.

        Saves a consolidated stage-level snapshot (for fast resume past this
        stage on future runs) and removes the incremental batch progress entry.
        """
        self.save_stage(stage_name, total_passed)
        self._manifest["batch_progress"].pop(stage_name, None)
        self._save_manifest()
