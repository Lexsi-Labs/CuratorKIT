from curatorkit.normalizers.clean import TextCleaner
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.normalizers.sample import StratifiedSampler
from curatorkit.normalizers.truncate import MaxSamplesTruncator

__all__ = [
    "TextCleaner",
    "ExactDeduplicator",
    "MinHashDeduplicator",
    "StratifiedSampler",
    "MaxSamplesTruncator",
    "EmbeddingDeduplicator",
]


def __getattr__(name: str):
    """Lazy import for EmbeddingDeduplicator (requires the [embedding] extra)."""
    if name == "EmbeddingDeduplicator":
        from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator

        return EmbeddingDeduplicator
    raise AttributeError(f"module 'curatorkit.normalizers' has no attribute {name!r}")
