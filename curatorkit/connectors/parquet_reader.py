"""
Parquet connector — reads .parquet files via PyArrow.

PyArrow is an optional dependency:
    pip install curatorkit[parquet]  or  pip install pyarrow

Parquet files carry native type information (lists, structs) so JSON cell
parsing is not needed — PyArrow deserializes directly to Python objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from curatorkit.connectors.base import BaseConnector


class ParquetReader(BaseConnector):
    """
    Read a .parquet file and produce DataSample objects.

    Args:
        path:               Path to the .parquet file.
        columns:            Optional list of column names to load.
                            If None, all columns are loaded.
        batch_size:         Number of rows to read at a time. Default 1000.
        field_mapping:      Optional key renames applied before detection.
        format:             Force a format ('auto' by default).
        source_uri:         Override for provenance records.
        preprocessing_fn:   Callable or dotted import path.
        detection_sample_size: Rows to inspect before committing detection.
    """

    def __init__(
        self,
        path: Path | str,
        columns: list[str] | None = None,
        batch_size: int = 1000,
        field_mapping: dict[str, str] | None = None,
        format: str = "auto",
        source_uri: str | None = None,
        preprocessing_fn: Callable | str | None = None,
        detection_sample_size: int = 10,
    ) -> None:
        self.path = Path(path)
        self.columns = columns
        self.batch_size = max(1, batch_size)
        super().__init__(
            source_uri=source_uri or str(path),
            field_mapping=field_mapping,
            format=format,
            preprocessing_fn=preprocessing_fn,
            detection_sample_size=detection_sample_size,
        )

    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "PyArrow is required for Parquet support. "
                "Install with: pip install curatorkit[parquet]"
            ) from e

        pf = pq.ParquetFile(self.path)
        line_no = 1

        for batch in pf.iter_batches(batch_size=self.batch_size, columns=self.columns):
            table = batch.to_pydict()
            n_rows = len(next(iter(table.values()), []))
            for i in range(n_rows):
                row: dict[str, Any] = {col: table[col][i] for col in table}
                # PyArrow may yield None for missing cells — replace with empty string
                row = {k: ("" if v is None else v) for k, v in row.items()}
                yield line_no, row
                line_no += 1
