"""
EmbeddingDeduplicator — cross-run deduplication via persistent embedding index.

The exact and MinHash deduplicators are within-batch only. This normalizer
adds a persistent embedding index so samples from run 2 are deduplicated
against everything produced in run 1.

Both within-run and cross-run comparison use FAISS IndexFlatIP when available
(inner product over L2-normalised embeddings == cosine similarity). Falls back
to numpy brute-force when faiss-cpu is not installed.

On-disk layout of ``index_dir`` (``index_dir/<namespace>`` when a namespace is set)::

    index_meta.json      manifest: model, dim, count, storage, version, current generation
    gen-000007/          the committed index (vectors + metadata.json)
    gen-000006/          previous generation, kept for rollback
    config.json          ignore_for_dedup sidecar
    pending/             (default pending_dir) staged entries of defer_commit runs
    .index.lock          held while an update is written

Safety guarantees:
  - A save is atomic: vectors and metadata are written into a new generation
    directory and only become visible when index_meta.json is swapped in.
  - Every save re-reads the index under a lock and appends only entries it
    doesn't already hold, so concurrent runs never drop each other's entries
    and re-committing the same entries is a no-op.
  - An index that exists but cannot be read consistently raises
    EmbeddingIndexError instead of silently starting fresh (which would
    overwrite and lose it). on_unreadable_index="rebuild" opts into starting
    fresh, after moving the unreadable files aside.
  - The manifest records the embedding model and dimension; reusing the index
    with a different model raises EmbeddingIndexMismatchError instead of
    comparing incompatible vectors.
  - defer_commit=True stages accepted entries instead of writing the index;
    commit_pending() adds them once the caller's run has succeeded.

Install:
  pip install curatorkit[embedding]           # sentence-transformers
  pip install curatorkit[embedding] faiss-cpu  # + FAISS for fast ANN
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import warnings
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from curatorkit.interfaces import BaseNormalizer
from curatorkit.normalizers.dedup_utils import apply_ignore_patterns, compile_ignore_patterns
from curatorkit.schema import DataSample, ProvenanceRecord

STEP_VERSION = "1.2.0"
INDEX_FORMAT_VERSION = 1

_MANIFEST_FILE = "index_meta.json"
_METADATA_FILE = "metadata.json"
_FAISS_FILE = "index.faiss"
_NPY_FILE = "embeddings.npy"
_CONFIG_FILE = "config.json"
_LOCK_FILE = ".index.lock"
_GEN_PREFIX = "gen-"
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class EmbeddingIndexError(RuntimeError):
    """The persisted cross-run index exists but cannot be read consistently."""


class EmbeddingIndexMismatchError(EmbeddingIndexError):
    """The persisted index was built with a different embedding model or dimension."""


def _ensure_sentence_transformers() -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. Install with: "
            "pip install curatorkit[embedding]"
        ) from e


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError:
        return None


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def _file_lock(lock_path: Path, timeout: float, stale_after: float):
    """Cross-process lock: an O_EXCL lock file, broken once older than stale_after."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after:
                warnings.warn(
                    f"EmbeddingDeduplicator: removing stale index lock {lock_path} ({age:.0f}s old).",
                    stacklevel=3,
                )
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"EmbeddingDeduplicator: timed out after {timeout:.0f}s waiting for {lock_path}. "
                    "Another run is updating this index; if no run is active, delete the lock file."
                )
            time.sleep(0.2)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class _IndexState:
    """In-memory view of one committed index (or of a pending stage)."""

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        metadata: list[dict] | None = None,
        manifest: dict | None = None,
        legacy: bool = False,
    ) -> None:
        self.embeddings = embeddings  # (N, dim) float32, L2-normalised, or None when empty
        self.metadata = metadata or []
        self.manifest = manifest
        self.legacy = legacy  # read from the pre-manifest flat layout

    @property
    def count(self) -> int:
        return 0 if self.embeddings is None else len(self.embeddings)

    @property
    def dim(self) -> int | None:
        if self.embeddings is not None and self.embeddings.ndim == 2 and len(self.embeddings):
            return int(self.embeddings.shape[1])
        return (self.manifest or {}).get("dim")

    @property
    def sample_ids(self) -> set[str]:
        return {m.get("sample_id") for m in self.metadata}


class EmbeddingDeduplicator(BaseNormalizer):
    """
    Cross-run embedding-based deduplication with persistent index.

    Parameters
    ----------
    index_dir : str | Path
        Directory to store/load the persistent embedding index.
    model : str
        Sentence-transformers model name. Recorded in the index manifest; an
        index built with a different model raises EmbeddingIndexMismatchError.
    threshold : float
        Cosine similarity above this → duplicate.
    batch_size : int
        Encoding batch size.
    text_field : str
        Which DataSample field to embed. "auto" picks based on task_type.
    ignore_for_dedup : list[str] | None
        Regex patterns. Matches are masked out (replaced with a space) from
        each field's text *before it's embedded*, so a real but unimportant
        difference (a per-sample UUID, a timestamp) doesn't push two
        otherwise-identical samples below the similarity threshold. Because
        masking changes what gets embedded, changing this between runs makes
        a persisted cross-run index not directly comparable to new
        embeddings — run() warns if the index on disk was built with a
        different ignore_for_dedup than the current one.
    namespace : str | None
        Keeps a separate index under ``index_dir/<namespace>`` (letters,
        digits, ``.``, ``_``, ``-``), so unrelated datasets or tasks sharing
        one index_dir don't deduplicate against each other.
    defer_commit : bool
        False (default): run() adds accepted entries to the index immediately.
        True: run() stages them in ``pending_dir`` and they are only added by
        commit_pending() — call it once the surrounding run has succeeded, so
        a failed run never adds its samples to the shared index.
    pending_dir : str | Path | None
        Where defer_commit stages entries. Default ``<index>/pending``.
    on_unreadable_index : "raise" | "rebuild"
        What to do when an index exists but can't be read consistently.
        "raise" (default) fails the run, leaving the files untouched.
        "rebuild" moves the unreadable files to ``<index>/unreadable-<ts>/``
        and starts a fresh index.
    keep_generations : int
        Committed generations kept on disk (the current one plus older ones
        for rollback). Minimum 1.
    lock_timeout : float
        Seconds to wait for another run's index update to finish.
    """

    def __init__(
        self,
        index_dir: str | Path = "output/embedding_index",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.92,
        batch_size: int = 64,
        text_field: str = "auto",
        device: str | None = None,
        ignore_for_dedup: list[str] | None = None,
        namespace: str | None = None,
        defer_commit: bool = False,
        pending_dir: str | Path | None = None,
        on_unreadable_index: Literal["raise", "rebuild"] = "raise",
        keep_generations: int = 2,
        lock_timeout: float = 600.0,
        lock_stale_after: float = 3600.0,
    ) -> None:
        if namespace is not None and not _NAMESPACE_RE.match(namespace):
            raise ValueError(
                f"EmbeddingDeduplicator: invalid namespace {namespace!r} — use letters, "
                "digits, '.', '_' or '-', starting with a letter or digit."
            )
        if on_unreadable_index not in ("raise", "rebuild"):
            raise ValueError("on_unreadable_index must be 'raise' or 'rebuild'")

        self.base_index_dir = Path(index_dir)
        self.namespace = namespace
        self.index_dir = self.base_index_dir / namespace if namespace else self.base_index_dir
        self.model_name = model
        self.threshold = threshold
        self.batch_size = batch_size
        self.text_field = text_field
        self.device = device
        self.ignore_for_dedup = list(ignore_for_dedup) if ignore_for_dedup else []
        self._ignore_patterns = compile_ignore_patterns(self.ignore_for_dedup)
        self.defer_commit = defer_commit
        self.pending_dir = Path(pending_dir) if pending_dir else self.index_dir / "pending"
        self.on_unreadable_index = on_unreadable_index
        self.keep_generations = max(1, int(keep_generations))
        self.lock_timeout = lock_timeout
        self.lock_stale_after = lock_stale_after
        self._model: Any = None
        # Set at _load_index time
        self._state = _IndexState()
        self._faiss_index: Any = None  # FAISS search index over the committed entries
        self._has_faiss: bool = False

    # ── Model / config ───────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        if self._model is None:
            SentenceTransformer = _ensure_sentence_transformers()
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "threshold": self.threshold,
                "text_field": self.text_field,
                "ignore_for_dedup": self.ignore_for_dedup,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_text(self, sample: DataSample) -> str:
        if self.text_field != "auto":
            text = getattr(sample, self.text_field, "") or ""
            return apply_ignore_patterns(text, self._ignore_patterns)

        task = sample.task_type
        if task == "language_modeling":
            fields = [sample.output or ""]
        elif task in ("preference", "implicit_preference"):
            fields = [sample.instruction, sample.chosen]
        elif task == "grpo" and sample.responses:
            fields = [sample.instruction, sample.responses[0]]
        else:
            fields = [sample.instruction, sample.output]

        fields = [apply_ignore_patterns(f, self._ignore_patterns) for f in fields]
        return " ".join(fields).strip()

    def _check_ignore_for_dedup_mismatch(self) -> None:
        """Warn if the persisted index was built with a different
        ignore_for_dedup than this run's — masking changes what gets
        embedded, so the two sets of vectors aren't directly comparable."""
        config_path = self.index_dir / _CONFIG_FILE
        if not config_path.exists():
            return
        try:
            with open(config_path) as f:
                prior_cfg = json.load(f)
        except Exception:
            return  # a corrupt/missing sidecar shouldn't fail the run
        prior_patterns = prior_cfg.get("ignore_for_dedup", [])
        if prior_patterns != self.ignore_for_dedup:
            warnings.warn(
                f"EmbeddingDeduplicator: index at {self.index_dir} was built with "
                f"ignore_for_dedup={prior_patterns!r}, but this run uses "
                f"{self.ignore_for_dedup!r}. Cross-run comparisons may not be "
                "apples-to-apples since masking changes what gets embedded. "
                "Set embedding_reset_index=True to start a fresh index.",
                stacklevel=2,
            )

    def _save_ignore_for_dedup_config(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            self.index_dir / _CONFIG_FILE, {"ignore_for_dedup": self.ignore_for_dedup}
        )

    # ── Reading the committed index ──────────────────────────────────────────

    def _read_vectors(self, directory: Path, storage: str | None) -> tuple[np.ndarray, str]:
        faiss_path, npy_path = directory / _FAISS_FILE, directory / _NPY_FILE
        if storage is None:
            storage = "faiss" if faiss_path.exists() else "numpy"
        if storage == "faiss":
            faiss = _try_import_faiss()
            if faiss is None:
                raise EmbeddingIndexError(
                    f"EmbeddingDeduplicator: the index at {self.index_dir} is stored in FAISS "
                    "format but faiss-cpu is not installed. Install it with: pip install faiss-cpu"
                )
            idx = faiss.read_index(str(faiss_path))
            vectors = (
                idx.reconstruct_n(0, idx.ntotal) if idx.ntotal else np.zeros((0, idx.d), np.float32)
            )
        else:
            vectors = np.load(str(npy_path))
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise EmbeddingIndexError(
                f"EmbeddingDeduplicator: index vectors in {directory} are not a 2-D matrix"
            )
        return vectors, storage

    def _read_state(self) -> _IndexState:
        """Read the committed index from disk, validating that it is complete and consistent.

        Raises EmbeddingIndexError for an index that exists but can't be read
        consistently. A directory with no index files is a valid empty index.
        """
        d = self.index_dir
        manifest_path = d / _MANIFEST_FILE
        manifest: dict | None = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception as e:
                raise EmbeddingIndexError(
                    f"EmbeddingDeduplicator: unreadable manifest {manifest_path}: {e}"
                ) from e

        if manifest is not None:
            data_dir = d / manifest.get("generation", "")
            legacy = False
            if not manifest.get("generation") or not data_dir.is_dir():
                raise EmbeddingIndexError(
                    f"EmbeddingDeduplicator: manifest {manifest_path} points at generation "
                    f"{manifest.get('generation')!r}, which is missing."
                )
        else:
            data_dir = d
            legacy = True

        meta_path = data_dir / _METADATA_FILE
        has_vectors = (data_dir / _FAISS_FILE).exists() or (data_dir / _NPY_FILE).exists()
        if legacy and not has_vectors and not meta_path.exists():
            return _IndexState()  # no index yet
        if not has_vectors or not meta_path.exists():
            found = "vectors" if has_vectors else "metadata.json"
            raise EmbeddingIndexError(
                f"EmbeddingDeduplicator: index at {d} is incomplete — found {found} but not "
                f"{'metadata.json' if has_vectors else 'the vector file'}."
            )

        try:
            vectors, _storage = self._read_vectors(data_dir, (manifest or {}).get("storage"))
            metadata = json.loads(meta_path.read_text())
        except EmbeddingIndexError:
            raise
        except Exception as e:
            raise EmbeddingIndexError(
                f"EmbeddingDeduplicator: could not read index at {d}: {e}"
            ) from e

        if not isinstance(metadata, list) or len(metadata) != len(vectors):
            raise EmbeddingIndexError(
                f"EmbeddingDeduplicator: index at {d} is inconsistent — {len(vectors)} vectors "
                f"but {len(metadata) if isinstance(metadata, list) else 'unparseable'} metadata entries."
            )
        if manifest is not None and manifest.get("count") != len(vectors):
            raise EmbeddingIndexError(
                f"EmbeddingDeduplicator: index at {d} holds {len(vectors)} entries but its "
                f"manifest records {manifest.get('count')}."
            )
        return _IndexState(
            _normalise(vectors) if len(vectors) else None, metadata, manifest, legacy
        )

    def _quarantine_unreadable(self, error: Exception) -> None:
        """on_unreadable_index='rebuild': move the index files aside instead of overwriting them."""
        target = self.index_dir / f"unreadable-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        target.mkdir(parents=True, exist_ok=True)
        keep = {"pending", _LOCK_FILE, target.name}
        for entry in list(self.index_dir.iterdir()):
            if entry.name in keep or entry.name.startswith("unreadable-"):
                continue
            shutil.move(str(entry), str(target / entry.name))
        warnings.warn(
            f"EmbeddingDeduplicator: index at {self.index_dir} was unreadable ({error}); moved it to "
            f"{target} and started a fresh index (on_unreadable_index='rebuild').",
            stacklevel=3,
        )

    def _read_state_or_recover(self) -> _IndexState:
        try:
            return self._read_state()
        except EmbeddingIndexMismatchError:
            raise
        except EmbeddingIndexError as e:
            if self.on_unreadable_index != "rebuild":
                raise
            self._quarantine_unreadable(e)
            return _IndexState()

    def _check_model(self, state: _IndexState) -> None:
        recorded = (state.manifest or {}).get("model")
        if recorded and recorded != self.model_name:
            raise EmbeddingIndexMismatchError(
                f"EmbeddingDeduplicator: index at {self.index_dir} was built with model "
                f"{recorded!r}, but this run uses {self.model_name!r}. Embeddings from different "
                "models aren't comparable. Use the original model (embedding_dedup_model), a "
                "different namespace or index_dir, or reset the index."
            )

    def _check_dim(self, state: _IndexState, dim: int) -> None:
        if state.count and state.dim != dim:
            raise EmbeddingIndexMismatchError(
                f"EmbeddingDeduplicator: index at {self.index_dir} holds {state.dim}-dimensional "
                f"embeddings but model {self.model_name!r} produces {dim}-dimensional ones. "
                "Use the original model, a different namespace or index_dir, or reset the index."
            )

    def _load_index(self) -> None:
        """Load the persistent cross-run index from disk."""
        self._check_ignore_for_dedup_mismatch()
        faiss = _try_import_faiss()
        self._has_faiss = faiss is not None
        self._state = self._read_state_or_recover()
        self._check_model(self._state)
        if self._state.legacy and self._state.count:
            warnings.warn(
                f"EmbeddingDeduplicator: index at {self.index_dir} has no manifest (written by an "
                "older CuratorKIT), so the model that built it can't be verified. It will be "
                f"recorded as {self.model_name!r} on the next save.",
                stacklevel=2,
            )
        self._faiss_index = None
        if self._has_faiss and self._state.count:
            self._faiss_index = faiss.IndexFlatIP(self._state.dim)
            self._faiss_index.add(self._state.embeddings)

    # ── Writing the committed index ──────────────────────────────────────────

    def _write_generation(
        self, state: _IndexState, embeddings: np.ndarray, metadata: list[dict]
    ) -> int:
        """Write a new generation and swap the manifest to it. Caller holds the lock."""
        prev = state.manifest or {}
        version = int(prev.get("version", 0)) + 1
        generation = f"{_GEN_PREFIX}{version:06d}"
        gen_dir = self.index_dir / generation
        if gen_dir.exists():
            shutil.rmtree(gen_dir)  # leftover of a write that never committed
        gen_dir.mkdir(parents=True)

        faiss = _try_import_faiss()
        if faiss is not None:
            idx = faiss.IndexFlatIP(embeddings.shape[1])
            idx.add(embeddings)
            faiss.write_index(idx, str(gen_dir / _FAISS_FILE))
            storage = "faiss"
        else:
            np.save(str(gen_dir / _NPY_FILE), embeddings)
            storage = "numpy"
        _write_json_atomic(gen_dir / _METADATA_FILE, metadata)

        # The manifest swap is the commit point: until it lands, readers see the old generation.
        _write_json_atomic(
            self.index_dir / _MANIFEST_FILE,
            {
                "format_version": INDEX_FORMAT_VERSION,
                "generation": generation,
                "version": version,
                "model": self.model_name,
                "dim": int(embeddings.shape[1]),
                "count": len(embeddings),
                "storage": storage,
                "normalized": True,
                "ignore_for_dedup": self.ignore_for_dedup,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

        if state.legacy:  # migrated into gen-*: drop the flat pre-manifest files
            for name in (_FAISS_FILE, _NPY_FILE, _METADATA_FILE):
                try:
                    (self.index_dir / name).unlink()
                except FileNotFoundError:
                    pass
        generations = sorted(p for p in self.index_dir.glob(f"{_GEN_PREFIX}*") if p.is_dir())
        for old in generations[: -self.keep_generations]:
            shutil.rmtree(old, ignore_errors=True)
        return version

    def _save_index(self, new_embeddings: np.ndarray, new_metadata: list[dict]) -> int:
        """Merge entries into the persisted index; returns how many were actually added.

        Re-reads the index under the lock, so entries another run committed since
        this one loaded are kept, and entries already present (same sample_id)
        are skipped — committing the same entries twice is a no-op.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        new_embeddings = _normalise(np.asarray(new_embeddings, dtype=np.float32))
        with _file_lock(self.index_dir / _LOCK_FILE, self.lock_timeout, self.lock_stale_after):
            state = self._read_state_or_recover()
            self._check_model(state)
            self._check_dim(state, int(new_embeddings.shape[1]))

            known = state.sample_ids
            keep = [i for i, m in enumerate(new_metadata) if m.get("sample_id") not in known]
            if not keep:
                return 0
            to_add, meta_add = new_embeddings[keep], [new_metadata[i] for i in keep]
            combined = to_add if state.embeddings is None else np.vstack([state.embeddings, to_add])
            self._write_generation(state, combined, state.metadata + meta_add)
            return len(keep)

    # ── Pending (defer_commit) ───────────────────────────────────────────────

    def _read_pending(self) -> _IndexState:
        meta_path, npy_path = self.pending_dir / _METADATA_FILE, self.pending_dir / _NPY_FILE
        if not meta_path.exists() and not npy_path.exists():
            return _IndexState()
        try:
            metadata = json.loads(meta_path.read_text())
            vectors = np.asarray(np.load(str(npy_path)), dtype=np.float32)
        except Exception as e:
            raise EmbeddingIndexError(
                f"EmbeddingDeduplicator: unreadable pending entries in {self.pending_dir}: {e}"
            ) from e
        if len(metadata) != len(vectors):
            raise EmbeddingIndexError(
                f"EmbeddingDeduplicator: pending entries in {self.pending_dir} are inconsistent "
                f"({len(vectors)} vectors, {len(metadata)} metadata)."
            )
        return _IndexState(vectors if len(vectors) else None, metadata)

    def _stage_pending(self, new_embeddings: np.ndarray, new_metadata: list[dict]) -> None:
        pending = self._read_pending()
        combined = (
            new_embeddings
            if pending.embeddings is None
            else np.vstack([pending.embeddings, new_embeddings])
        )
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        tmp_npy = self.pending_dir / (_NPY_FILE + ".tmp.npy")
        np.save(str(tmp_npy), combined.astype(np.float32))
        # metadata first, vectors last: a crash in between leaves a mismatch that _read_pending reports
        _write_json_atomic(self.pending_dir / _METADATA_FILE, pending.metadata + new_metadata)
        os.replace(tmp_npy, self.pending_dir / _NPY_FILE)

    def pending_count(self) -> int:
        """Entries staged by defer_commit runs and not yet committed."""
        return self._read_pending().count

    def commit_pending(self) -> int:
        """Add staged entries to the persisted index; returns how many were added.

        Safe to call more than once (already-committed entries are skipped).
        """
        pending = self._read_pending()
        added = 0
        if pending.count:
            added = self._save_index(pending.embeddings, pending.metadata)
        self._save_ignore_for_dedup_config()
        self.discard_pending()
        return added

    def discard_pending(self) -> None:
        """Drop staged entries without committing them (e.g. after a failed run)."""
        if self.pending_dir.exists():
            shutil.rmtree(self.pending_dir, ignore_errors=True)

    # ── Maintenance ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Summary of the committed index (does not load the embedding model)."""
        state = self._read_state()
        manifest = state.manifest or {}
        return {
            "index_dir": str(self.index_dir),
            "namespace": self.namespace,
            "entries": state.count,
            "model": manifest.get("model"),
            "dim": state.dim,
            "storage": manifest.get("storage"),
            "version": manifest.get("version"),
            "updated_at": manifest.get("updated_at"),
            "legacy_layout": state.legacy and state.count > 0,
            "pending": self.pending_count(),
        }

    def reset(self) -> None:
        """Delete the committed index (all generations, the manifest and the sidecar)."""
        if not self.index_dir.exists():
            return
        with _file_lock(self.index_dir / _LOCK_FILE, self.lock_timeout, self.lock_stale_after):
            for entry in list(self.index_dir.iterdir()):
                if entry.name == _LOCK_FILE or entry.resolve() == self.pending_dir.resolve():
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)

    # ── Dedup ────────────────────────────────────────────────────────────────

    def _check_against_index(self, emb_norm: np.ndarray) -> tuple[bool, float]:
        """Check whether a single L2-normalised embedding duplicates the committed cross-run index."""
        if not self._state.count:
            return False, 0.0
        if self._faiss_index is not None:
            sims, _ = self._faiss_index.search(emb_norm, 1)
            max_sim = float(sims[0][0])
        else:
            max_sim = float(np.max(self._state.embeddings @ emb_norm[0]))
        return max_sim >= self.threshold, max_sim

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        if not samples:
            return []

        model = self._load_model()
        self._load_index()

        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)

        # Encode and L2-normalise all embeddings upfront
        texts = [self._get_text(s) for s in samples]
        raw_embs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        embeddings = _normalise(raw_embs)  # shape (N, dim)
        dim = embeddings.shape[1]
        self._check_dim(self._state, dim)

        # ── Within-run index, seeded with entries staged earlier in this run ─
        faiss = _try_import_faiss() if self._has_faiss else None
        pending = self._read_pending() if self.defer_commit else _IndexState()
        if pending.count and pending.dim != dim:
            raise EmbeddingIndexMismatchError(
                f"EmbeddingDeduplicator: pending entries in {self.pending_dir} are "
                f"{pending.dim}-dimensional but this run produces {dim}-dimensional embeddings."
            )
        if faiss is not None:
            within_index = faiss.IndexFlatIP(dim)
            if pending.count:
                within_index.add(pending.embeddings)
            use_within_faiss = True
        else:
            within_index = None
            use_within_faiss = False
            within_matrix: np.ndarray | None = pending.embeddings if pending.count else None

        passed: list[DataSample] = []
        new_embeddings: list[np.ndarray] = []
        new_metadata: list[dict[str, str]] = []
        cross_run_removed = 0
        within_run_removed = 0

        from tqdm import tqdm

        for i, sample in enumerate(tqdm(samples, desc="EmbeddingDeduplicator", unit="sample")):
            emb = embeddings[i : i + 1]  # shape (1, dim) — already normalised

            # Cross-run check
            is_cross_dup, _ = self._check_against_index(emb)
            if is_cross_dup:
                cross_run_removed += 1
                continue

            # Within-run check
            is_within_dup = False
            if use_within_faiss:
                if within_index.ntotal > 0:
                    sims, _ = within_index.search(emb, 1)
                    if float(sims[0][0]) >= self.threshold:
                        is_within_dup = True
            else:
                if within_matrix is not None:
                    sims = within_matrix @ emb[0]
                    if float(np.max(sims)) >= self.threshold:
                        is_within_dup = True

            if is_within_dup:
                within_run_removed += 1
                continue

            # Accept — add to within-run index
            if use_within_faiss:
                within_index.add(emb)
            else:
                within_matrix = emb if within_matrix is None else np.vstack([within_matrix, emb])

            new_embeddings.append(emb[0])
            new_metadata.append(
                {
                    "sample_id": sample.id,
                    "source_uri": sample.source_uri,
                }
            )

            sample.append_provenance(
                ProvenanceRecord(
                    step_name="EmbeddingDeduplicator",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "cross_run_removed": cross_run_removed,
                        "within_run_removed": within_run_removed,
                        "index_size_before": self._state.count,
                        "threshold": self.threshold,
                        "ann_backend": "faiss" if use_within_faiss else "numpy",
                        "index_namespace": self.namespace,
                        "commit": "deferred" if self.defer_commit else "immediate",
                    },
                )
            )
            passed.append(sample)

        if new_embeddings:
            batch = np.array(new_embeddings, dtype=np.float32)
            if self.defer_commit:
                self._stage_pending(batch, new_metadata)
            else:
                self._save_index(batch, new_metadata)
        if not self.defer_commit:
            # Always keep the sidecar in sync, even on a run that added nothing
            # new, so the next run's mismatch check reflects the latest config.
            self._save_ignore_for_dedup_config()

        return passed
