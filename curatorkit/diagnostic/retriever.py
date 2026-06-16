"""
PostHocRetriever — embedding-similarity chunk retriever for benchmarking.

Purpose: simulate a pipeline that does NOT track provenance and must retrieve
the source chunk post-hoc via similarity search. Use it as a comparison
baseline against CuratorKIT's exact provenance tracking: retrieval returns
the wrong chunk on a non-trivial fraction of samples, which lowers
faithfulness-judging accuracy relative to passing the judge the exact
chunk each sample was generated from.

NOT used in production pipelines — a standalone evaluation utility.
"""

from __future__ import annotations

import numpy as np


class PostHocRetriever:
    """
    Given a corpus of text chunks, retrieve the most similar chunk for a query.

    Usage:
        retriever = PostHocRetriever(all_chunks, model="BAAI/bge-base-en-v1.5")
        retrieved_chunk = retriever.retrieve(generated_answer)
        # Pass retrieved_chunk to the judge instead of the exact provenance chunk
    """

    def __init__(
        self,
        chunks: list[str],
        model: str = "BAAI/bge-base-en-v1.5",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for PostHocRetriever. "
                "Install with: pip install sentence-transformers"
            ) from e
        self.chunks = chunks
        self.model = SentenceTransformer(model)
        self.index = self.model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)

    def retrieve(self, query: str, top_k: int = 1) -> list[str]:
        """Return top_k most similar chunks to query."""
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores = (self.index @ q_emb.T).flatten()
        idxs = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in idxs]

    def retrieval_accuracy(
        self,
        queries: list[str],
        exact_chunks: list[str],
    ) -> float:
        """
        Compute fraction of queries where retrieved chunk == exact provenance chunk.
        Quantifies how often post-hoc retrieval returns the wrong chunk.
        """
        if not queries:
            return 0.0
        correct = sum(
            self.retrieve(q)[0].strip() == exact.strip() for q, exact in zip(queries, exact_chunks)
        )
        return correct / len(queries)
