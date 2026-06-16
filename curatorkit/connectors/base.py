"""
BaseConnector — abstract base for all CuratorKIT data connectors.

Subclasses implement _iter_rows() which yields (line_no, raw_dict) tuples.
This base handles the full ingestion pipeline for every row:

  1. preprocessing_fn   (optional user callable — structural normalization)
  2. field_mapping      (flat key renames before detection)
  3. FormatDetector     (committed once from first N rows, cached for the file)
  4. DataSample construction  (format-specific, with layer 3 role normalization)
  5. RejectedSample emission  (for every failure — no silent drops)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

from curatorkit.detection.detector import (
    DataFormat,
    DetectionConfidence,
    DetectionResult,
    FormatDetector,
)
from curatorkit.detection.normalizer import (
    extract_implicit_prompt,
    extract_system_prompt,
    normalize_conversations,
)
from curatorkit.interfaces import BaseReader
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "0.2.0"
_detector = FormatDetector()


# ---------------------------------------------------------------------------
# Preprocessing function loading
# ---------------------------------------------------------------------------


def _load_fn(fn_spec: str | Callable | None) -> Callable | None:
    """
    Load a preprocessing function from a dotted path string or return
    the callable directly. Returns None if fn_spec is None or "identity".
    """
    if fn_spec is None or fn_spec == "identity":
        return None
    if callable(fn_spec):
        return fn_spec
    if isinstance(fn_spec, str):
        parts = fn_spec.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"preprocessing_fn must be a dotted import path like "
                f"'mymodule.my_fn', got: {fn_spec!r}"
            )
        module_path, fn_name = parts
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Could not import module '{module_path}' for preprocessing_fn: {e}"
            ) from e
        try:
            fn = getattr(module, fn_name)
        except AttributeError:
            raise ImportError(f"Module '{module_path}' has no attribute '{fn_name}'")
        if not callable(fn):
            raise TypeError(f"'{fn_spec}' is not callable")
        return fn
    raise TypeError(
        f"preprocessing_fn must be a callable or dotted import path string, "
        f"got {type(fn_spec).__name__}"
    )


# ---------------------------------------------------------------------------
# BaseConnector
# ---------------------------------------------------------------------------


class BaseConnector(BaseReader):
    """
    Abstract base connector.

    Parameters
    ----------
    source_uri : str
        Human-readable identifier for the data source. Used in provenance
        records and RejectedSample metadata. Defaults to str(path) if path
        is provided.
    field_mapping : dict[str, str]
        Pre-detection flat key renames: {source_key: canonical_key}.
        Applied before format detection runs.
        Supports nested source keys via dot notation: "meta.prompt" → "instruction"
    format : str
        Force a specific format instead of auto-detecting.
        One of: "auto", "alpaca", "sharegpt", "preference", "implicit_preference",
        "unpaired_preference", "grpo", "prompt_only", "pretrain"
    preprocessing_fn : callable or dotted-path string
        Optional user function applied to each raw dict before field_mapping
        and detection. Signature: (dict) -> dict | DataSample | None.
        Returning None causes the row to be emitted as a RejectedSample.
        Returning a DataSample bypasses all detection and construction.
    detection_sample_size : int
        Number of rows to inspect before committing to a format.
        Default 10. Higher values improve accuracy for heterogeneous files.
    """

    def __init__(
        self,
        source_uri: str | None = None,
        field_mapping: dict[str, str] | None = None,
        format: str = "auto",
        preprocessing_fn: Callable | str | None = None,
        detection_sample_size: int = 10,
    ) -> None:
        self.source_uri = source_uri or "unknown"
        self.field_mapping = field_mapping or {}
        self.format_override = format
        self.preprocessing_fn = _load_fn(preprocessing_fn)
        self.detection_sample_size = max(1, detection_sample_size)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """
        Yield (line_no, raw_dict) tuples.
        line_no is 1-based for error reporting.
        raw_dict is the parsed record — a Python dict.
        """
        ...

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self) -> tuple[list[DataSample], list[RejectedSample]]:
        samples: list[DataSample] = []
        rejected: list[RejectedSample] = []

        # Materialize enough rows to commit detection
        row_buffer: list[tuple[int, dict[str, Any]]] = []
        detection: DetectionResult | None = None
        cfg_hash: str | None = None

        for line_no, raw in self._iter_rows():
            # -- Handle decode errors from readers that use sentinel dicts --
            if "__json_error__" in raw:
                rejected.append(
                    self._make_rejected(
                        f"json_decode_error:{raw['__json_error__']}",
                        line_no,
                        {"__raw_line__": raw.get("__raw_line__", "")},
                        "unknown",
                    )
                )
                continue

            # -- Step 1: preprocessing_fn --
            result = self._apply_preprocessing(raw, line_no, rejected)
            if result is None:
                continue  # rejection already appended

            if isinstance(result, DataSample):
                samples.append(result)
                continue

            processed = result  # dict

            # -- Step 2: field_mapping --
            processed = self._apply_field_mapping(processed)

            # -- Step 3: collect sample rows, then commit detection --
            if detection is None:
                row_buffer.append((line_no, processed))
                if len(row_buffer) < self.detection_sample_size:
                    continue  # keep buffering
                # Enough rows collected — commit detection
                detection, cfg_hash = self._commit_detection(row_buffer, rejected)
                if detection is None:
                    # Unknown format — all buffered rows already rejected
                    continue
                # Now process buffered rows
                for buf_line_no, buf_row in row_buffer:
                    self._build_and_append(
                        buf_row,
                        buf_line_no,
                        detection,
                        cfg_hash,
                        samples,
                        rejected,
                    )
                row_buffer = []
                continue

            # Detection already committed
            self._build_and_append(
                processed,
                line_no,
                detection,
                cfg_hash,
                samples,
                rejected,
            )

        # Process any remaining buffered rows (file < detection_sample_size rows)
        if row_buffer and detection is None:
            detection, cfg_hash = self._commit_detection(row_buffer, rejected)
            if detection is not None:
                for buf_line_no, buf_row in row_buffer:
                    self._build_and_append(
                        buf_row,
                        buf_line_no,
                        detection,
                        cfg_hash,
                        samples,
                        rejected,
                    )

        return samples, rejected

    # ------------------------------------------------------------------
    # Detection commitment
    # ------------------------------------------------------------------

    def _commit_detection(
        self,
        row_buffer: list[tuple[int, dict[str, Any]]],
        rejected: list[RejectedSample],
    ) -> tuple[DetectionResult | None, str]:
        """
        Run the 3-stage detection on buffered rows.
        Returns (DetectionResult, cfg_hash) or (None, "") if unknown.
        Unknown rows are added to rejected here.
        """
        if self.format_override != "auto":
            # User forced a format — build a minimal DetectionResult
            fmt_str = self.format_override.lower()
            try:
                forced_fmt = DataFormat(fmt_str)
            except ValueError:
                forced_fmt = DataFormat.UNKNOWN
            if forced_fmt == DataFormat.UNKNOWN:
                for line_no, row in row_buffer:
                    rejected.append(
                        self._make_rejected(
                            f"invalid_forced_format:{self.format_override}",
                            line_no,
                            row,
                            "unknown",
                        )
                    )
                return None, ""
            # Let detector resolve column names against the actual keys,
            # but skip layer-2 validation (confidence is set by user)
            first_keys = set(row_buffer[0][1].keys()) if row_buffer else set()
            sample_rows = [r for _, r in row_buffer]
            result = _detector.detect(first_keys, sample_rows)
            # Override the format but keep the column_map
            result.format = forced_fmt
            result.confidence = DetectionConfidence.HIGH
        else:
            all_keys = [set(r.keys()) for _, r in row_buffer]
            sample_rows = [r for _, r in row_buffer]
            result, is_consistent = _detector.detect_with_consistency_check(all_keys, sample_rows)
            if not is_consistent:
                warnings.warn(
                    f"[CuratorKIT] Inconsistent column sets in {self.source_uri} — "
                    f"detection may be unreliable. Check rejected.jsonl for details.",
                    stacklevel=4,
                )

        if result.format == DataFormat.UNKNOWN:
            for line_no, row in row_buffer:
                rejected.append(
                    self._make_rejected(
                        "unrecognized_format:no_known_column_pattern",
                        line_no,
                        row,
                        "unknown",
                    )
                )
            return None, ""

        if result.confidence == DetectionConfidence.LOW:
            warnings.warn(
                f"[CuratorKIT] Low-confidence format detection ({result.format.value}) "
                f"for {self.source_uri}. Provide format: or field_mapping: in YAML to "
                f"suppress this warning.",
                stacklevel=4,
            )

        cfg_hash = self._make_config_hash(result)
        return result, cfg_hash

    # ------------------------------------------------------------------
    # Sample construction (format-specific)
    # ------------------------------------------------------------------

    def _build_and_append(
        self,
        row: dict[str, Any],
        line_no: int,
        detection: DetectionResult,
        cfg_hash: str,
        samples: list[DataSample],
        rejected: list[RejectedSample],
    ) -> None:
        sample, reason = self._build_sample(row, line_no, detection, cfg_hash)
        if sample is not None:
            samples.append(sample)
        else:
            rejected.append(
                self._make_rejected(
                    reason or "construction_error:unknown",
                    line_no,
                    row,
                    detection.format.value,
                )
            )

    def _build_sample(
        self,
        row: dict[str, Any],
        line_no: int,
        detection: DetectionResult,
        cfg_hash: str,
    ) -> tuple[DataSample | None, str | None]:
        """
        Construct a DataSample from a row using the committed DetectionResult.
        Returns (DataSample, None) on success, (None, reason_str) on failure.
        """
        fmt = detection.format
        ts = datetime.now(UTC).replace(tzinfo=None)

        try:
            if fmt == DataFormat.ALPACA:
                sample = self._build_alpaca(row, detection)

            elif fmt == DataFormat.SHAREGPT:
                sample = self._build_sharegpt(row, detection)

            elif fmt == DataFormat.PREFERENCE:
                sample = self._build_preference(row, detection)

            elif fmt == DataFormat.IMPLICIT_PREFERENCE:
                sample = self._build_implicit_preference(row, detection)

            elif fmt == DataFormat.UNPAIRED_PREFERENCE:
                sample = self._build_unpaired_preference(row, detection)

            elif fmt == DataFormat.GRPO:
                sample = self._build_grpo(row, detection)

            elif fmt == DataFormat.PROMPT_ONLY:
                sample = self._build_prompt_only(row, detection)

            elif fmt == DataFormat.PRETRAIN:
                sample = self._build_pretrain(row, detection)

            else:
                return None, f"unsupported_format:{fmt.value}"

        except Exception as e:
            return None, f"construction_error:{e}"

        if sample is None:
            return None, "construction_returned_none"

        # Preserve original id and source_uri if present in the raw row.
        # Builders generate a new UUID and use self.source_uri by default;
        # promote the row values so downstream ID-based lookups stay consistent.
        if "id" in row and row["id"]:
            sample.id = str(row["id"])
            sample.metadata.pop("id", None)
        if "source_uri" in row and row["source_uri"]:
            sample.source_uri = str(row["source_uri"])
            sample.metadata.pop("source_uri", None)

        # Append provenance
        notes: dict[str, Any] = {
            "source_uri": self.source_uri,
            "line_number": line_no,
            "detected_format": fmt.value,
            "detection_confidence": detection.confidence.value,
            "column_map": {k: v for k, v in detection.column_map.items() if v},
            "field_mapping_applied": self.field_mapping,
        }
        if detection.warnings:
            notes["detection_warnings"] = detection.warnings
        if detection.unrecognized_cols:
            notes["unrecognized_cols"] = detection.unrecognized_cols

        sample.append_provenance(
            ProvenanceRecord(
                step_name="Connector",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=cfg_hash,
                notes=notes,
            )
        )

        return sample, None

    # ------------------------------------------------------------------
    # Format-specific builders
    # ------------------------------------------------------------------

    def _build_alpaca(self, row: dict[str, Any], d: DetectionResult) -> DataSample | None:
        instr_col = d.get("instruction")
        output_col = d.get("output")
        context_col = d.get("context")
        system_col = d.get("system")

        if not instr_col or not output_col:
            return None

        instruction = str(row.get(instr_col, ""))
        output = str(row.get(output_col, ""))
        context = str(row.get(context_col, "")) if context_col else ""

        # Avoid treating the instruction column as context too
        if context_col == instr_col:
            context = ""

        metadata = self._collect_metadata(row, d)
        if system_col and system_col in row:
            metadata["system_prompt"] = str(row[system_col])

        return DataSample(
            source_uri=self.source_uri,
            instruction=instruction,
            input=context,
            output=output,
            task_type="instruction_following",
            metadata=metadata,
        )

    def _build_sharegpt(self, row: dict[str, Any], d: DetectionResult) -> DataSample | None:
        conv_col = d.get("conversation")
        system_col = d.get("system")

        if not conv_col:
            return None

        convs = row.get(conv_col)
        if not isinstance(convs, list):
            return None

        # Layer 3: role normalization
        convs, norm_warnings = normalize_conversations(convs)

        # Extract system prompt
        system_prompt, convs = extract_system_prompt(convs)
        if not system_prompt and system_col and system_col in row:
            system_prompt = str(row[system_col])

        # First human turn → instruction, first assistant turn → output
        instruction = next((t["content"] for t in convs if t.get("role") == "user"), "")
        output = next((t["content"] for t in convs if t.get("role") == "assistant"), "")

        # Remaining turns beyond the first exchange → metadata
        extra_turns = convs[2:] if len(convs) > 2 else []

        metadata = self._collect_metadata(row, d)
        if extra_turns:
            metadata["turns"] = extra_turns
        if system_prompt:
            metadata["system_prompt"] = system_prompt
        if norm_warnings:
            metadata["role_normalization_warnings"] = norm_warnings

        return DataSample(
            source_uri=self.source_uri,
            instruction=instruction,
            output=output,
            task_type="conversational",
            metadata=metadata,
        )

    def _build_preference(self, row: dict[str, Any], d: DetectionResult) -> DataSample | None:
        instr_col = d.get("instruction")
        chosen_col = d.get("chosen")
        rejected_col = d.get("rejected")
        system_col = d.get("system")

        if not chosen_col or not rejected_col:
            return None

        instruction = str(row.get(instr_col, "")) if instr_col else ""
        chosen_val = row.get(chosen_col, "")
        rejected_val = row.get(rejected_col, "")

        # Handle conversational chosen/rejected
        if isinstance(chosen_val, list):
            chosen_val, _ = normalize_conversations(chosen_val)
            chosen_str = json.dumps(chosen_val, ensure_ascii=False)
        else:
            chosen_str = str(chosen_val)

        if isinstance(rejected_val, list):
            rejected_val, _ = normalize_conversations(rejected_val)
            rejected_str = json.dumps(rejected_val, ensure_ascii=False)
        else:
            rejected_str = str(rejected_val)

        metadata = self._collect_metadata(row, d)
        if system_col and system_col in row:
            metadata["system_prompt"] = str(row[system_col])

        return DataSample(
            source_uri=self.source_uri,
            instruction=instruction,
            chosen=chosen_str,
            rejected=rejected_str,
            task_type="preference",
            metadata=metadata,
        )

    def _build_implicit_preference(
        self, row: dict[str, Any], d: DetectionResult
    ) -> DataSample | None:
        chosen_col = d.get("chosen")
        rejected_col = d.get("rejected")

        if not chosen_col or not rejected_col:
            return None

        chosen_val = row.get(chosen_col, [])
        rejected_val = row.get(rejected_col, [])

        if not isinstance(chosen_val, list) or not isinstance(rejected_val, list):
            # Flat string implicit preference — treat chosen as output, rejected as rejected
            instruction = ""
            chosen_str = str(chosen_val)
            rejected_str = str(rejected_val)
            metadata = self._collect_metadata(row, d)
            return DataSample(
                source_uri=self.source_uri,
                chosen=chosen_str,
                rejected=rejected_str,
                task_type="implicit_preference",
                metadata=metadata,
            )

        # Conversational: normalize then extract common prefix
        chosen_norm, _ = normalize_conversations(chosen_val)
        rejected_norm, _ = normalize_conversations(rejected_val)
        prompt_turns, c_remain, r_remain = extract_implicit_prompt(chosen_norm, rejected_norm)

        metadata = self._collect_metadata(row, d)
        if not prompt_turns:
            metadata["implicit_prompt_warning"] = "no_common_prefix_found"

        instruction = json.dumps(prompt_turns, ensure_ascii=False) if prompt_turns else ""
        chosen_str = json.dumps(c_remain, ensure_ascii=False)
        rejected_str = json.dumps(r_remain, ensure_ascii=False)

        return DataSample(
            source_uri=self.source_uri,
            instruction=instruction,
            chosen=chosen_str,
            rejected=rejected_str,
            task_type="implicit_preference",
            metadata=metadata,
        )

    def _build_unpaired_preference(
        self, row: dict[str, Any], d: DetectionResult
    ) -> DataSample | None:
        instr_col = d.get("instruction")
        output_col = d.get("output")
        label_col = d.get("label")
        context_col = d.get("context")

        if not instr_col or not output_col:
            return None

        instruction = str(row.get(instr_col, ""))
        output = str(row.get(output_col, ""))
        context = str(row.get(context_col, "")) if context_col else ""

        raw_label = row.get(label_col) if label_col else None
        label: float | None = None
        if raw_label is not None:
            try:
                label = float(raw_label)
            except (TypeError, ValueError):
                # Boolean True/False → 1.0/0.0
                if isinstance(raw_label, bool):
                    label = 1.0 if raw_label else 0.0

        metadata = self._collect_metadata(row, d)

        return DataSample(
            source_uri=self.source_uri,
            instruction=instruction,
            input=context,
            output=output,
            label=label,
            task_type="unpaired_preference",
            metadata=metadata,
        )

    def _build_grpo(self, row: dict[str, Any], d: DetectionResult) -> DataSample | None:
        instr_col = d.get("instruction")
        resp_col = d.get("responses")
        rewards_col = d.get("rewards")
        system_col = d.get("system")

        if not instr_col or not resp_col:
            return None

        instruction = str(row.get(instr_col, ""))
        responses = row.get(resp_col, [])
        if not isinstance(responses, list):
            responses = [str(responses)]
        responses = [str(r) for r in responses]

        raw_rewards = row.get(rewards_col, []) if rewards_col else []
        rewards: list[float] = []
        if isinstance(raw_rewards, list):
            for r in raw_rewards:
                try:
                    rewards.append(float(r))
                except (TypeError, ValueError):
                    rewards.append(0.0)

        metadata = self._collect_metadata(row, d)
        if system_col and system_col in row:
            metadata["system_prompt"] = str(row[system_col])

        return DataSample(
            source_uri=self.source_uri,
            instruction=instruction,
            responses=responses,
            reward_scores=rewards,
            task_type="grpo",
            metadata=metadata,
        )

    def _build_prompt_only(self, row: dict[str, Any], d: DetectionResult) -> DataSample | None:
        instr_col = d.get("instruction")
        if not instr_col:
            return None
        return DataSample(
            source_uri=self.source_uri,
            instruction=str(row.get(instr_col, "")),
            task_type="prompt_only",
            metadata=self._collect_metadata(row, d),
        )

    def _build_pretrain(self, row: dict[str, Any], d: DetectionResult) -> DataSample | None:
        text_col = d.get("text")
        if not text_col:
            return None
        return DataSample(
            source_uri=self.source_uri,
            output=str(row.get(text_col, "")),
            task_type="language_modeling",
            metadata=self._collect_metadata(row, d),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_preprocessing(
        self,
        raw: dict[str, Any],
        line_no: int,
        rejected: list[RejectedSample],
    ) -> dict[str, Any] | DataSample | None:
        """Apply preprocessing_fn. Returns normalized dict, DataSample, or None."""
        if self.preprocessing_fn is None:
            return raw

        try:
            result = self.preprocessing_fn(raw)
        except Exception as e:
            rejected.append(
                self._make_rejected(
                    f"preprocessing_fn_error:{e}",
                    line_no,
                    raw,
                    "unknown",
                )
            )
            return None

        if result is None:
            rejected.append(
                self._make_rejected(
                    "preprocessing_fn_returned_none",
                    line_no,
                    raw,
                    "unknown",
                )
            )
            return None

        return result

    def _apply_field_mapping(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Apply field_mapping (flat key renames).
        Supports nested source paths via dot notation: "meta.prompt" → "instruction".
        """
        if not self.field_mapping:
            return row

        result = dict(row)
        for src_path, dst_key in self.field_mapping.items():
            parts = src_path.split(".")
            if len(parts) == 1:
                # Simple rename
                if src_path in result:
                    result[dst_key] = result.pop(src_path)
            else:
                # Nested access
                obj: Any = row
                for part in parts:
                    if isinstance(obj, dict) and part in obj:
                        obj = obj[part]
                    else:
                        obj = None
                        break
                if obj is not None:
                    result[dst_key] = obj

        return result

    def _collect_metadata(self, row: dict[str, Any], d: DetectionResult) -> dict[str, Any]:
        """Collect all columns not mapped to semantic slots into metadata."""
        mapped = {v for v in d.column_map.values() if v is not None}
        return {k: v for k, v in row.items() if k not in mapped}

    def _make_rejected(
        self,
        reason: str,
        line_no: int,
        raw: dict[str, Any],
        detected_format: str,
    ) -> RejectedSample:
        ts = datetime.now(UTC).replace(tzinfo=None)
        cfg = self._make_config_hash_simple()

        sample = RejectedSample(
            source_uri=self.source_uri,
            instruction="",
            rejection_reason=reason,
            rejecting_step="Connector",
            metadata={
                "raw_preview": {k: str(v)[:200] for k, v in list(raw.items())[:5]},
                "line_number": line_no,
            },
        )
        sample.append_provenance(
            ProvenanceRecord(
                step_name="Connector",
                step_version=STEP_VERSION,
                timestamp=ts,
                config_hash=cfg,
                notes={
                    "source_uri": self.source_uri,
                    "line_number": line_no,
                    "rejection_reason": reason,
                    "detected_format": detected_format,
                },
            )
        )
        return sample

    def _make_config_hash(self, detection: DetectionResult) -> str:
        payload = json.dumps(
            {
                "source_uri": self.source_uri,
                "format": detection.format.value,
                "column_map": detection.column_map,
                "field_mapping": self.field_mapping,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _make_config_hash_simple(self) -> str:
        payload = json.dumps(
            {
                "source_uri": self.source_uri,
                "field_mapping": self.field_mapping,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
