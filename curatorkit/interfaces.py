"""
Abstract base classes for all pipeline components.

BaseReader.read() returns the same tuple contract as BaseGate.run() —
(passed, rejected). This makes reader-level parse failures first-class
citizens that flow into rejected.jsonl rather than disappearing.

Every pipeline step implements one of these four interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from curatorkit.schema import DataSample, RejectedSample


class BaseReader(ABC):
    """
    Reads raw data from a source and produces DataSample objects.

    MUST return both lists. Parse failures that produce no DataSample
    must be returned as RejectedSample objects — never silently dropped.
    Every line/row that enters the reader must leave as either a
    DataSample or a RejectedSample.
    """

    @abstractmethod
    def read(self) -> tuple[list[DataSample], list[RejectedSample]]: ...


class BaseGate(ABC):
    """
    Validates samples against a contract.

    MUST return both lists. Silent drops are bugs. Every sample that does not
    pass goes into the rejected list as a RejectedSample with a reason string.
    """

    @abstractmethod
    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]: ...


class BaseNormalizer(ABC):
    """
    Transforms samples in-place (dedup, cleaning, sampling).

    Returns the transformed list. Samples removed by a normalizer (e.g. dedup)
    are NOT added to the rejected list — removal is intentional, not a contract
    violation. The normalizer must record removal counts in a ProvenanceRecord
    appended to surviving samples.
    """

    @abstractmethod
    def run(self, samples: list[DataSample]) -> list[DataSample]: ...


class BaseExporter(ABC):
    """Serialises DataSample objects to a training-ready format on disk."""

    @abstractmethod
    def export(self, samples: list[DataSample], output_dir: Path) -> None: ...
