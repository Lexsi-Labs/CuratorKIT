"""
HuggingFace datasets connector — loads datasets from the HF Hub or local
datasets cache via the `datasets` library.

    pip install curatorkit[hf]  or  pip install datasets

Supports streaming mode for large datasets (avoids full download).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from curatorkit.connectors.base import BaseConnector


class HuggingFaceReader(BaseConnector):
    """
    Load a HuggingFace dataset and produce DataSample objects.

    Args:
        dataset_name:       HF Hub dataset name or local path.
                            e.g. "tatsu-lab/alpaca" or "/path/to/dataset"
        split:              Dataset split: "train", "test", "validation", etc.
                            Defaults to "train".
        subset:             Dataset configuration / subset name if applicable.
                            e.g. "helpful-base" for Anthropic/hh-rlhf
        streaming:          If True, stream rows without full download.
                            Disables consistency check (no random access).
        token:              HF access token for gated datasets.
        columns:            Optional list of columns to load. If None, all columns.
        field_mapping:      Optional key renames applied before detection.
        format:             Force a format ('auto' by default).
        source_uri:         Override for provenance records.
                            Defaults to "{dataset_name}/{split}".
        preprocessing_fn:   Callable or dotted import path.
        detection_sample_size: Rows to inspect before committing detection.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        streaming: bool = False,
        token: str | None = None,
        columns: list[str] | None = None,
        field_mapping: dict[str, str] | None = None,
        format: str = "auto",
        source_uri: str | None = None,
        preprocessing_fn: Callable | str | None = None,
        detection_sample_size: int = 10,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.streaming = streaming
        self.token = token
        self.columns = columns
        super().__init__(
            source_uri=source_uri or f"{dataset_name}/{split}",
            field_mapping=field_mapping,
            format=format,
            preprocessing_fn=preprocessing_fn,
            detection_sample_size=detection_sample_size,
        )

    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "The `datasets` library is required for HuggingFace support. "
                "Install with: pip install curatorkit[hf]"
            ) from e

        kwargs: dict[str, Any] = {
            "streaming": self.streaming,
        }
        if self.token:
            kwargs["token"] = self.token

        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            **kwargs,
        )

        for line_no, row in enumerate(ds, start=1):
            record: dict[str, Any] = dict(row)

            # Filter to requested columns
            if self.columns:
                record = {k: record[k] for k in self.columns if k in record}

            # HF datasets may contain None values — replace with empty string
            # for primitive types; leave complex types (lists, dicts) as-is
            record = {
                k: ("" if v is None and not isinstance(v, (list, dict)) else v)
                for k, v in record.items()
            }

            yield line_no, record
