"""
BillOfMaterials — cryptographically verifiable run artifact.

The BoM extends the existing manifest + checksums.txt with:

  * SHA-256 of every output file the run produced.
  * SHA-256 of every source file the run consumed (from
    ProvenanceRecord.notes["source_file"] entries).
  * SHA-256 of the manifest itself.
  * A `bom_signature` = SHA-256 over the canonical-JSON BoM body.

`BillOfMaterials.verify()` re-hashes the listed files and confirms
none has been tampered with since the run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _curatorkit_version() -> str:
    from curatorkit import __version__

    return __version__


@dataclass
class FileEntry:
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass
class BillOfMaterials:
    """
    Run-scoped BoM. Construct via `BillOfMaterials.build(...)`.

    Fields:
      pipeline_config_hash — copied from the manifest for cross-reference.
      run_timestamp        — UTC ISO-8601.
      sources              — input files the run consumed.
      outputs              — files the run wrote (jsonl + manifest + reports).
      manifest_sha256      — SHA-256 of manifest.json.
      bom_signature        — SHA-256 over the canonical-JSON BoM body
                             (everything in this object except `bom_signature`
                             itself). Re-computing it and comparing is how
                             callers verify the BoM has not been edited.
    """

    pipeline_config_hash: str
    run_timestamp: str
    sources: list[FileEntry] = field(default_factory=list)
    outputs: list[FileEntry] = field(default_factory=list)
    manifest_sha256: str = ""
    bom_signature: str = ""
    tool_version: str = ""

    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        manifest_path: Path,
        output_dir: Path,
        source_files: list[str | Path],
        pipeline_config_hash: str,
        tool_version: str | None = None,
        extra_outputs: list[Path] | None = None,
    ) -> BillOfMaterials:
        """Hash everything and assemble the BoM."""
        bom = cls(
            pipeline_config_hash=pipeline_config_hash,
            run_timestamp=datetime.now(UTC).replace(tzinfo=None).isoformat(),
            tool_version=tool_version if tool_version is not None else _curatorkit_version(),
        )

        # Source files
        for s in source_files:
            p = Path(s)
            if p.exists() and p.is_file():
                bom.sources.append(
                    FileEntry(
                        path=str(p),
                        sha256=_sha256_file(p),
                        size_bytes=p.stat().st_size,
                    )
                )

        # Output files
        out_paths: list[Path] = list(output_dir.glob("*.jsonl"))
        out_paths += list(output_dir.glob("*.md"))
        out_paths += list(output_dir.glob("checksums.txt"))
        if extra_outputs:
            out_paths += [Path(p) for p in extra_outputs]
        # Deduplicate while preserving order
        seen: set[Path] = set()
        for p in out_paths:
            if p not in seen and p.exists() and p.is_file():
                seen.add(p)
                bom.outputs.append(
                    FileEntry(
                        path=str(p.relative_to(output_dir)),
                        sha256=_sha256_file(p),
                        size_bytes=p.stat().st_size,
                    )
                )

        if manifest_path.exists():
            bom.manifest_sha256 = _sha256_file(manifest_path)

        bom.bom_signature = bom._compute_signature()
        return bom

    # ─────────────────────────────────────────────────────────────────────

    def _body_for_signature(self) -> dict[str, Any]:
        """Canonical body — bom_signature excluded so it can sign itself."""
        return {
            "pipeline_config_hash": self.pipeline_config_hash,
            "run_timestamp": self.run_timestamp,
            "tool_version": self.tool_version,
            "manifest_sha256": self.manifest_sha256,
            "sources": [s.to_dict() for s in self.sources],
            "outputs": [o.to_dict() for o in self.outputs],
        }

    def _compute_signature(self) -> str:
        payload = json.dumps(self._body_for_signature(), sort_keys=True).encode()
        return _sha256_bytes(payload)

    # ─────────────────────────────────────────────────────────────────────
    # Serialization + verification
    # ─────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {**self._body_for_signature(), "bom_signature": self.bom_signature}

    def write(self, output_dir: Path, filename: str = "bom.json") -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, bom_path: Path) -> BillOfMaterials:
        data = json.loads(bom_path.read_text(encoding="utf-8"))
        bom = cls(
            pipeline_config_hash=data.get("pipeline_config_hash", ""),
            run_timestamp=data.get("run_timestamp", ""),
            manifest_sha256=data.get("manifest_sha256", ""),
            bom_signature=data.get("bom_signature", ""),
            tool_version=data.get("tool_version", ""),
        )
        bom.sources = [
            FileEntry(s["path"], s["sha256"], int(s.get("size_bytes", 0)))
            for s in data.get("sources", [])
        ]
        bom.outputs = [
            FileEntry(o["path"], o["sha256"], int(o.get("size_bytes", 0)))
            for o in data.get("outputs", [])
        ]
        return bom

    def verify(
        self,
        manifest_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Re-hash every file in the BoM and report any mismatches.

        Returns a dict:
          {
            "signature_ok": bool,             # the BoM itself wasn't edited
            "manifest_ok":  bool,             # manifest.json is unchanged
            "sources_ok":   {path: bool},     # per-source file integrity
            "outputs_ok":   {path: bool},     # per-output file integrity
            "all_ok":       bool,             # AND of all of the above
          }
        """
        result: dict[str, Any] = {
            "signature_ok": self.bom_signature == self._compute_signature(),
            "manifest_ok": True,
            "sources_ok": {},
            "outputs_ok": {},
        }

        if self.manifest_sha256 and manifest_path.exists():
            result["manifest_ok"] = self.manifest_sha256 == _sha256_file(manifest_path)
        elif self.manifest_sha256:
            result["manifest_ok"] = False

        for src in self.sources:
            p = Path(src.path)
            result["sources_ok"][src.path] = p.exists() and _sha256_file(p) == src.sha256

        for out in self.outputs:
            p = output_dir / out.path
            result["outputs_ok"][out.path] = p.exists() and _sha256_file(p) == out.sha256

        result["all_ok"] = (
            result["signature_ok"]
            and result["manifest_ok"]
            and all(result["sources_ok"].values())
            and all(result["outputs_ok"].values())
        )
        return result
