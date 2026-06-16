"""
EmbeddingDeduplicator — cross-run deduplication via persistent embedding index.

The exact and MinHash deduplicators are within-batch only. This normalizer
adds a persistent embedding index so samples from run 2 are deduplicated
against everything produced in run 1.

Both within-run and cross-run comparison use FAISS IndexFlatIP when available
(inner product over L2-normalised embeddings == cosine similarity). Falls back
to numpy brute-force when faiss-cpu is not installed.

Install:
  pip install curatorkit[embedding]           # sentence-transformers
  pip install curatorkit[embedding] faiss-cpu  # + FAISS for fast ANN
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from curatorkit.interfaces import BaseNormalizer
from curatorkit.schema import DataSample, ProvenanceRecord

STEP_VERSION = "1.1.0"


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


class EmbeddingDeduplicator(BaseNormalizer):
    """
    Cross-run embedding-based deduplication with persistent index.

    Parameters
    ----------
    index_dir : str | Path
        Directory to store/load the persistent embedding index.
    model : str
        Sentence-transformers model name.
    threshold : float
        Cosine similarity above this → duplicate.
    batch_size : int
        Encoding batch size.
    text_field : str
        Which DataSample field to embed. "auto" picks based on task_type.
    """

    def __init__(
        self,
        index_dir: str | Path = "output/embedding_index",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.92,
        batch_size: int = 64,
        text_field: str = "auto",
        device: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.model_name = model
        self.threshold = threshold
        self.batch_size = batch_size
        self.text_field = text_field
        self.device = device
        self._model: Any = None
        # Set at _load_index time
        self._faiss_index: Any = None  # FAISS cross-run index (if faiss available)
        self._index_embeddings: np.ndarray | None = None  # numpy fallback
        self._index_metadata: list[dict[str, str]] = []
        self._has_faiss: bool = False

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
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_text(self, sample: DataSample) -> str:
        if self.text_field != "auto":
            return getattr(sample, self.text_field, "") or ""

        task = sample.task_type
        if task == "language_modeling":
            return sample.output or ""
        if task in ("preference", "implicit_preference"):
            return f"{sample.instruction} {sample.chosen}".strip()
        if task == "grpo" and sample.responses:
            return f"{sample.instruction} {sample.responses[0]}".strip()
        return f"{sample.instruction} {sample.output}".strip()

    def _load_index(self) -> None:
        """Load the persistent cross-run index from disk."""
        faiss = _try_import_faiss()
        self._has_faiss = faiss is not None
        self._faiss_index = None
        self._index_embeddings = None
        self._index_metadata = []

        meta_path = self.index_dir / "metadata.json"

        if self._has_faiss:
            faiss_path = self.index_dir / "index.faiss"
            if faiss_path.exists() and meta_path.exists():
                try:
                    self._faiss_index = faiss.read_index(str(faiss_path))
                    with open(meta_path) as f:
                        self._index_metadata = json.load(f)
                except Exception as e:
                    warnings.warn(
                        f"EmbeddingDeduplicator: could not load FAISS index: {e}. Starting fresh.",
                        stacklevel=2,
                    )
                    self._faiss_index = None
                    self._index_metadata = []
        else:
            npy_path = self.index_dir / "embeddings.npy"
            if npy_path.exists() and meta_path.exists():
                try:
                    self._index_embeddings = np.load(str(npy_path))
                    with open(meta_path) as f:
                        self._index_metadata = json.load(f)
                except Exception as e:
                    warnings.warn(
                        f"EmbeddingDeduplicator: could not load numpy index: {e}. Starting fresh.",
                        stacklevel=2,
                    )
                    self._index_embeddings = None
                    self._index_metadata = []

    def _save_index(self, new_embeddings: np.ndarray, new_metadata: list[dict]) -> None:
        """Persist the updated cross-run index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        combined_meta = self._index_metadata + new_metadata

        if self._has_faiss:
            faiss = _try_import_faiss()
            faiss_path = self.index_dir / "index.faiss"
            if self._faiss_index is not None:
                idx = self._faiss_index
            else:
                idx = faiss.IndexFlatIP(new_embeddings.shape[1])
            idx.add(new_embeddings.astype(np.float32))
            faiss.write_index(idx, str(faiss_path))
        else:
            npy_path = self.index_dir / "embeddings.npy"
            if self._index_embeddings is not None and len(self._index_embeddings) > 0:
                combined = np.vstack([self._index_embeddings, new_embeddings])
            else:
                combined = new_embeddings
            np.save(str(npy_path), combined)

        with open(self.index_dir / "metadata.json", "w") as f:
            json.dump(combined_meta, f)

    def _check_against_index(self, emb_norm: np.ndarray) -> tuple[bool, float]:
        """Check whether a single L2-normalised embedding duplicates the cross-run index."""
        if self._has_faiss:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return False, 0.0
            sims, _ = self._faiss_index.search(emb_norm, 1)
            max_sim = float(sims[0][0])
        else:
            if self._index_embeddings is None or len(self._index_embeddings) == 0:
                return False, 0.0
            # index embeddings may not be normalised (legacy); normalise on lookup
            norms = np.linalg.norm(self._index_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            index_normed = self._index_embeddings / norms
            sims = index_normed @ emb_norm[0]
            max_sim = float(np.max(sims))

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
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = raw_embs / norms  # shape (N, dim)

        # ── Set up within-run index ──────────────────────────────────────────
        faiss = _try_import_faiss() if self._has_faiss else None
        dim = embeddings.shape[1]
        if faiss is not None:
            within_index = faiss.IndexFlatIP(dim)
            use_within_faiss = True
        else:
            within_index = None
            use_within_faiss = False
            within_matrix: np.ndarray | None = None

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
                        "index_size_before": (
                            self._faiss_index.ntotal
                            if self._faiss_index is not None
                            else (
                                len(self._index_embeddings)
                                if self._index_embeddings is not None
                                else 0
                            )
                        ),
                        "threshold": self.threshold,
                        "ann_backend": "faiss" if use_within_faiss else "numpy",
                    },
                )
            )
            passed.append(sample)

        if new_embeddings:
            self._save_index(np.array(new_embeddings, dtype=np.float32), new_metadata)

        return passed
