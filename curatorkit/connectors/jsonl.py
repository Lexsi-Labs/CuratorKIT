"""
JSONL connector — reads .jsonl files (one JSON object per line).

Every line that cannot be parsed becomes a RejectedSample.
Format detection is delegated to BaseConnector.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from curatorkit.connectors.base import BaseConnector


class JSONLReader(BaseConnector):
    """
    Read a .jsonl file and produce DataSample objects.

    Args:
        path:               Path to the .jsonl file.
        field_mapping:      Optional key renames applied before detection.
                            Supports dot notation for nested source keys.
        format:             Force a format ('auto' by default).
        source_uri:         Override for provenance records.
        preprocessing_fn:   Callable or dotted import path.
                            Signature: (dict) -> dict | DataSample | None
        detection_sample_size: Rows to inspect before committing detection.
    """

    def __init__(
        self,
        path: Path | str,
        field_mapping: dict[str, str] | None = None,
        format: str = "auto",
        source_uri: str | None = None,
        preprocessing_fn: Callable | str | None = None,
        detection_sample_size: int = 10,
    ) -> None:
        self.path = Path(path)
        super().__init__(
            source_uri=source_uri or str(path),
            field_mapping=field_mapping,
            format=format,
            preprocessing_fn=preprocessing_fn,
            detection_sample_size=detection_sample_size,
        )

    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                    if isinstance(obj, dict):
                        yield line_no, obj
                    else:
                        # Line is valid JSON but not an object (e.g. a list)
                        yield line_no, {"__raw__": obj}
                except json.JSONDecodeError as e:
                    # Yield a sentinel that triggers rejection in base class
                    # We can't call _make_rejected here directly (it's in base)
                    # so we yield a special dict that base's preprocessing
                    # step will convert to a RejectedSample.
                    # Since the base class handles this we re-raise via sentinel:
                    yield line_no, {"__json_error__": str(e), "__raw_line__": raw_line[:500]}
