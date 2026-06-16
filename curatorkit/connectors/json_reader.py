"""
JSON connector — reads .json files.

Handles three common structures:
  1. Array at top level:           [{...}, {...}, ...]
  2. Dict with a "data" key:       {"data": [{...}, ...], "metadata": {...}}
  3. Dict with any list value:     {"examples": [{...}, ...]}  (heuristic scan)
  4. Single object:                {...}  (treated as one-record dataset)

All records that are not dicts produce RejectedSamples.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from curatorkit.connectors.base import BaseConnector

# Keys to check when scanning for the list payload in a wrapped dict
_COMMON_DATA_KEYS = ["data", "examples", "samples", "records", "items", "dataset", "train"]


class JSONReader(BaseConnector):
    """
    Read a .json file and produce DataSample objects.

    Args:
        path:               Path to the .json file.
        data_key:           If the JSON is a dict wrapping a list, use this
                            key to extract it. If None, auto-detect.
        field_mapping:      Optional key renames applied before detection.
        format:             Force a format ('auto' by default).
        source_uri:         Override for provenance records.
        preprocessing_fn:   Callable or dotted import path.
        detection_sample_size: Rows to inspect before committing detection.
    """

    def __init__(
        self,
        path: Path | str,
        data_key: str | None = None,
        field_mapping: dict[str, str] | None = None,
        format: str = "auto",
        source_uri: str | None = None,
        preprocessing_fn: Callable | str | None = None,
        detection_sample_size: int = 10,
    ) -> None:
        self.path = Path(path)
        self.data_key = data_key
        super().__init__(
            source_uri=source_uri or str(path),
            field_mapping=field_mapping,
            format=format,
            preprocessing_fn=preprocessing_fn,
            detection_sample_size=detection_sample_size,
        )

    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            yield 1, {"__json_error__": str(e), "__raw_line__": ""}
            return

        records = self._extract_records(data)

        for i, record in enumerate(records, start=1):
            if isinstance(record, dict):
                yield i, record
            else:
                yield (
                    i,
                    {
                        "__json_error__": f"record_is_{type(record).__name__}_not_dict",
                        "__raw_line__": str(record)[:200],
                    },
                )

    def _extract_records(self, data: Any) -> list[Any]:
        # Case 1: top-level list
        if isinstance(data, list):
            return data

        # Case 2: explicit data_key specified
        if isinstance(data, dict) and self.data_key:
            if self.data_key in data and isinstance(data[self.data_key], list):
                return data[self.data_key]

        # Case 3: dict wrapping a list — scan common keys first
        if isinstance(data, dict):
            for key in _COMMON_DATA_KEYS:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Fall back: find the first list-valued key
            for key, val in data.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return val

        # Case 4: single dict record
        if isinstance(data, dict):
            return [data]

        return []
