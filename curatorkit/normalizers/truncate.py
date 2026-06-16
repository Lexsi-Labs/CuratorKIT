"""MaxSamplesTruncator — cap the pipeline to the first N samples."""

from curatorkit.interfaces import BaseNormalizer
from curatorkit.schema import DataSample


class MaxSamplesTruncator(BaseNormalizer):
    """Keep only the first ``max_samples`` samples, dropping the rest.

    Useful for smoke tests and cost-bounded runs. Applied after readers
    (CLI pipelines) or after resampling (Curator).
    """

    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        """Return the first ``max_samples`` entries of ``samples``."""
        return samples[: self.max_samples]
