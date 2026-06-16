"""
Core data models for CuratorKIT.

Supports:
  - DPO preference data  (chosen, rejected)
  - Unpaired preference  (label)
  - Extended task_type vocabulary covering all post-training paradigms

provenance_chain is append-only — no step may modify or remove existing records.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

# FailureDiagnosis is a plain dataclass with no CuratorKIT deps — safe to import directly.
# Pydantic treats it as an arbitrary type (no circular import).
try:
    from curatorkit.diagnostic.failure_modes import FailureDiagnosis
except ImportError:
    FailureDiagnosis = None  # type: ignore


class ProvenanceRecord(BaseModel):
    """Immutable record appended by each pipeline step."""

    step_name: str
    step_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None))
    config_hash: str
    notes: dict[str, Any] = Field(default_factory=dict)


class DataSample(BaseModel):
    """
    The canonical unit of data moving through the pipeline.

    task_type vocabulary (use these strings in YAML and code):
      instruction_following  — single-turn SFT (Alpaca family)
      conversational         — multi-turn SFT (ShareGPT / ChatML)
      preference             — DPO with explicit chosen/rejected + optional prompt
      implicit_preference    — DPO where prompt is embedded inside chosen/rejected turns
      unpaired_preference    — single completion with scalar quality label
      grpo                   — group rollouts with reward scores
      prompt_only            — PPO-style: prompt only, response generated at runtime
      language_modeling      — continued pre-training, full text sequences

    Field usage by task_type:
      instruction_following  → instruction, input (optional), output
      conversational         → instruction (first human turn), output (first assistant turn),
                               metadata["turns"] for subsequent turns,
                               metadata["system_prompt"] if present
      preference             → instruction (prompt), chosen, rejected
      implicit_preference    → chosen, rejected (instruction extracted from common prefix)
      unpaired_preference    → instruction (prompt), output (completion), label
      grpo                   → instruction (prompt), responses, reward_scores
      prompt_only            → instruction (prompt)
      language_modeling      → output (full text sequence)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_uri: str
    instruction: str = ""
    input: str = ""
    output: str = ""

    # --- DPO / preference fields ---
    chosen: str = ""
    rejected: str = ""

    # --- Unpaired preference / reward annotation ---
    label: float | None = None

    # --- GRPO group rollouts ---
    responses: list[str] = Field(default_factory=list)
    reward_scores: list[float] = Field(default_factory=list)

    # --- Classification ---
    task_type: str = "instruction_following"

    # --- Passthrough ---
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Append-only provenance chain ---
    provenance_chain: list[ProvenanceRecord] = Field(default_factory=list)

    def append_provenance(self, record: ProvenanceRecord) -> None:
        self.provenance_chain.append(record)


class RejectedSample(DataSample):
    """
    A DataSample that could not be parsed or failed a gate check.

    rejection_reason is a structured string:
      "missing_field:{field}"
      "json_decode_error:{msg}"
      "format_mismatch:{detail}"
      "unrecognized_format:{detail}"
      "preprocessing_fn_error:{msg}"
      "low_confidence_format:{candidate}"
      "encoding_error:{detail}"
      "below_min_tokens:{count}"
      "above_max_tokens:{count}"
    """

    rejection_reason: str
    rejecting_step: str
    # Optional[...] (not `| None`): when the diagnostic extra is missing,
    # FailureDiagnosis is None at runtime and `None | None` cannot be evaluated.
    diagnosis: Optional[FailureDiagnosis] = None  # noqa: UP045

    def model_dump(self, **kwargs) -> dict:
        d = super().model_dump(**kwargs)
        if self.diagnosis is not None:
            d["diagnosis"] = self.diagnosis.to_dict()
        return d


# Resolve the FailureDiagnosis forward reference now that it is imported above.
# Required by Pydantic v2 when `from __future__ import annotations` is active,
# which makes all annotations lazy strings that Pydantic cannot resolve at class
# definition time.
if FailureDiagnosis is not None:
    RejectedSample.model_rebuild()
