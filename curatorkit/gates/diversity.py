"""
DiversityGate — embedding-based semantic diversity filter.

Embeds all samples and rejects those that are semantically too similar
to existing samples in the batch. This catches paraphrased duplicates
that survive exact and MinHash dedup.

Uses sentence-transformers for embedding. Install with:
  pip install curatorkit[embedding]

Also checks coverage gaps against the manifest from a previous run
if a coverage_field is specified.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


def _ensure_sentence_transformers() -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. Install with: "
            "pip install curatorkit[embedding]"
        ) from e


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DiversityGate(BaseGate):
    """
    Reject samples that are semantically too similar to existing ones.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformers model name.
    similarity_threshold : float
        Cosine similarity above this → reject as near-duplicate.
    text_field : str
        Which DataSample field to embed. "auto" picks based on task_type.
    coverage_field : str
        Metadata or DataSample field to check for category coverage gaps.
    batch_size : int
        Encoding batch size for the embedding model.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        text_field: str = "auto",
        coverage_field: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.text_field = text_field
        self.coverage_field = coverage_field
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            SentenceTransformer = _ensure_sentence_transformers()
            self._model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._model

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "embedding_model": self.embedding_model_name,
                "similarity_threshold": self.similarity_threshold,
                "text_field": self.text_field,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_text(self, sample: DataSample) -> str:
        """Extract the text to embed from a sample."""
        if self.text_field != "auto":
            return getattr(sample, self.text_field, "") or ""

        # Auto-detect based on task_type
        task = sample.task_type
        if task == "language_modeling":
            return sample.output or ""
        if task in ("preference", "implicit_preference"):
            return f"{sample.instruction} {sample.chosen}".strip()
        if task == "grpo" and sample.responses:
            return f"{sample.instruction} {sample.responses[0]}".strip()
        # Default: instruction + output
        return f"{sample.instruction} {sample.output}".strip()

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        if not samples:
            return [], []

        model = self._load_model()
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)

        # Extract texts
        texts = [self._get_text(s) for s in samples]

        # Encode all at once — keep as numpy float32 (no .tolist() blowup)
        import numpy as np

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # L2-normalize so inner-product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings_normed = embeddings / norms
        dim = embeddings_normed.shape[1]

        # Try FAISS for ANN; fall back to numpy brute-force if not installed
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(dim)
            use_faiss = True
        except ImportError:
            index = None
            use_faiss = False
            accepted_matrix: np.ndarray | None = None

        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        removed = 0

        from tqdm import tqdm

        for i, sample in enumerate(tqdm(samples, desc="DiversityGate", unit="sample")):
            emb = embeddings_normed[i : i + 1]  # shape (1, dim)
            max_sim = 0.0

            if use_faiss:
                if index.ntotal > 0:
                    sims, _ = index.search(emb, 1)
                    max_sim = float(sims[0][0])
            else:
                if accepted_matrix is not None and len(accepted_matrix) > 0:
                    sims = accepted_matrix @ emb[0]
                    max_sim = float(np.max(sims))

            if max_sim >= self.similarity_threshold:
                removed += 1
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"diversity_gate:too_similar:{max_sim:.3f}",
                    rejecting_step="DiversityGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="DiversityGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "max_similarity": round(max_sim, 4),
                            "threshold": self.similarity_threshold,
                            "passed": False,
                            "ann_backend": "faiss" if use_faiss else "numpy",
                        },
                    )
                )
                rejected.append(rej)
            else:
                if use_faiss:
                    index.add(emb)
                else:
                    accepted_matrix = (
                        emb if accepted_matrix is None else np.vstack([accepted_matrix, emb])
                    )
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="DiversityGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "max_similarity": round(max_sim, 4) if max_sim > 0 else 0.0,
                            "threshold": self.similarity_threshold,
                            "passed": True,
                            "semantic_duplicates_removed_so_far": removed,
                            "ann_backend": "faiss" if use_faiss else "numpy",
                        },
                    )
                )
                passed.append(sample)

        # Coverage analysis
        if self.coverage_field and passed:
            self._check_coverage(passed, self.coverage_field)

        return passed, rejected

    def _check_coverage(self, samples: list[DataSample], coverage_field: str) -> None:
        """Analyze coverage distribution and warn about gaps."""
        from collections import Counter

        categories = []
        for s in samples:
            if coverage_field == "task_type":
                categories.append(s.task_type)
            else:
                categories.append(str(s.metadata.get(coverage_field, "unknown")))

        dist = Counter(categories)
        total = len(categories)

        # Warn about very sparse categories (< 5% of total)
        for cat, count in dist.items():
            if count / total < 0.05:
                warnings.warn(
                    f"DiversityGate: category '{cat}' has only "
                    f"{count}/{total} ({100 * count / total:.1f}%) samples. "
                    f"Consider generating more data for this category.",
                    stacklevel=2,
                )
