"""
CSV connector — reads .csv and .tsv files.

Each row becomes a raw dict via csv.DictReader. All values are strings
initially; the detection layer handles type inspection on string values.

Note: Lists and dicts encoded as JSON strings inside CSV cells are
automatically parsed, which allows CSV files to carry conversational
data (e.g. a "messages" column containing a JSON-encoded list of turns).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from curatorkit.connectors.base import BaseConnector


def _try_parse_json(value: str) -> Any:
    """Try to parse a string as JSON. Return original string on failure."""
    stripped = value.strip()
    if stripped.startswith(("{", "[", '"')):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


class CSVReader(BaseConnector):
    """
    Read a CSV or TSV file and produce DataSample objects.

    Args:
        path:               Path to the .csv or .tsv file.
        delimiter:          Column delimiter. Auto-detected if None.
                            Use '\\t' for TSV files.
        parse_json_cells:   If True (default), attempt to parse cell values
                            that look like JSON (lists, dicts, strings).
        field_mapping:      Optional key renames applied before detection.
        format:             Force a format ('auto' by default).
        source_uri:         Override for provenance records.
        preprocessing_fn:   Callable or dotted import path.
        detection_sample_size: Rows to inspect before committing detection.
    """

    def __init__(
        self,
        path: Path | str,
        delimiter: str | None = None,
        parse_json_cells: bool = True,
        field_mapping: dict[str, str] | None = None,
        format: str = "auto",
        source_uri: str | None = None,
        preprocessing_fn: Callable | str | None = None,
        detection_sample_size: int = 10,
    ) -> None:
        self.path = Path(path)
        self.delimiter = delimiter
        self.parse_json_cells = parse_json_cells
        super().__init__(
            source_uri=source_uri or str(path),
            field_mapping=field_mapping,
            format=format,
            preprocessing_fn=preprocessing_fn,
            detection_sample_size=detection_sample_size,
        )

    def _detect_delimiter(self) -> str:
        """Sniff the delimiter from the first 4096 bytes of the file."""
        with open(self.path, encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
            return dialect.delimiter
        except csv.Error:
            # Default to comma
            return ","

    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        delimiter = self.delimiter or self._detect_delimiter()

        with open(self.path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for line_no, row in enumerate(reader, start=2):  # start=2: row 1 is header
                if row is None:
                    continue
                record: dict[str, Any] = {}
                for key, val in row.items():
                    if key is None:
                        continue
                    if val is None:
                        record[key] = ""
                    elif self.parse_json_cells and isinstance(val, str):
                        record[key] = _try_parse_json(val)
                    else:
                        record[key] = val
                yield line_no, record
