"""
SchemaGate — validates DataSamples against field, token, and encoding constraints.

Validation is task-type-aware. Different task types have different
meaningful fields:
  instruction_following  → instruction + output must be non-empty
  conversational         → instruction + output must be non-empty
  preference             → chosen + rejected must be non-empty
  implicit_preference    → chosen + rejected must be non-empty
  unpaired_preference    → instruction + output must be non-empty; label is optional
  grpo                   → instruction must be non-empty; responses must be a non-empty list
  prompt_only            → instruction must be non-empty; output may be empty
  language_modeling      → output must be non-empty

Token counting is also task-type-aware:
  SFT:                instruction + output
  Preference:         instruction + chosen  (chosen is typically the longer of the two)
  Unpaired pref:      instruction + output
  GRPO:               instruction + longest response
  Pretrain:           output
  Prompt-only:        instruction

Every failed sample becomes a RejectedSample — no silent drops.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from tqdm import tqdm

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample
from curatorkit.utils.tokens import count_tokens

STEP_VERSION = "0.2.0"

# task types where the primary text is in chosen/rejected, not instruction/output
_PREFERENCE_TYPES = {"preference", "implicit_preference"}

# task types that only require instruction (no output expected yet)
_PROMPT_ONLY_TYPES = {"prompt_only"}

# task types where output holds the full sequence
_PRETRAIN_TYPES = {"language_modeling"}

# task types for raw source chunks fed into LLM generation tasks
# Only `input` (the chunk text) is required; instruction/output are generated later
_SOURCE_CHUNK_TYPES = {"source_chunk"}

# task types using group rollouts
_GRPO_TYPES = {"grpo"}


class SchemaGate(BaseGate):
    """
    Validate samples against field, token-length, and encoding constraints.

    Args:
        required_fields:
            Fields that must be non-empty. Default behaviour is auto-derived
            from the sample's task_type. Explicitly setting this overrides
            the automatic per-task-type check entirely.
        min_tokens:     Minimum token count for the primary text fields.
        max_tokens:     Maximum token count for the primary text fields.
        use_tiktoken:   Use tiktoken cl100k_base instead of whitespace tokenizer.
        enforce_task_types:
            If non-empty, only samples with these task_type values pass.
            Useful for single-paradigm pipelines (e.g. pure DPO).
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        min_tokens: int = 10,
        max_tokens: int = 2048,
        use_tiktoken: bool = False,
        enforce_task_types: list[str] | None = None,
    ) -> None:
        self.required_fields = required_fields  # None = use per-task-type defaults
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.use_tiktoken = use_tiktoken
        self.enforce_task_types = set(enforce_task_types) if enforce_task_types else set()

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "required_fields": sorted(self.required_fields or []),
                "min_tokens": self.min_tokens,
                "max_tokens": self.max_tokens,
                "use_tiktoken": self.use_tiktoken,
                "enforce_task_types": sorted(self.enforce_task_types),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC).replace(tzinfo=None)

        for sample in tqdm(samples, desc="SchemaGate", unit="sample"):
            reason = self._validate(sample)
            if reason is None:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="SchemaGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"passed": True, "task_type": sample.task_type},
                    )
                )
                passed.append(sample)
            else:
                rejected_sample = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=reason,
                    rejecting_step="SchemaGate",
                )
                rejected_sample.append_provenance(
                    ProvenanceRecord(
                        step_name="SchemaGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": False,
                            "rejection_reason": reason,
                            "task_type": sample.task_type,
                        },
                    )
                )
                rejected.append(rejected_sample)

        return passed, rejected

    # ------------------------------------------------------------------
    # Validation logic
    # ------------------------------------------------------------------

    def _validate(self, sample: DataSample) -> str | None:
        """Return structured rejection reason string, or None if sample passes."""
        task = sample.task_type

        # --- Task-type whitelist check ---
        if self.enforce_task_types and task not in self.enforce_task_types:
            return f"task_type_not_allowed:{task}"

        # --- Field presence check ---
        if self.required_fields is not None:
            # Explicit override — use as-is
            for f in self.required_fields:
                if not self._field_present(sample, f):
                    return f"missing_field:{f}"
        else:
            # Auto-derive from task_type
            reason = self._check_task_fields(sample, task)
            if reason:
                return reason

        # --- Encoding sanity (null bytes in text fields) ---
        for field_name in ("instruction", "input", "output", "chosen", "rejected"):
            val = getattr(sample, field_name, "") or ""
            if "\x00" in val:
                return f"encoding_error:null_byte_in_{field_name}"

        # --- Token count ---
        text = self._primary_text(sample, task)
        if not text and task not in _PROMPT_ONLY_TYPES:
            # Empty text is caught by field checks above; skip token count
            return None

        if text:
            try:
                token_count = count_tokens(text.strip(), use_tiktoken=self.use_tiktoken)
            except Exception as e:
                return f"encoding_error:{e}"

            if token_count < self.min_tokens:
                return f"below_min_tokens:{token_count}"
            if token_count > self.max_tokens:
                return f"above_max_tokens:{token_count}"

        return None

    def _field_present(self, sample: DataSample, field_name: str) -> bool:
        """Return True if the field is present and non-empty."""
        val = getattr(sample, field_name, None)
        if val is None:
            return False
        if isinstance(val, str):
            return bool(val.strip())
        if isinstance(val, list):
            return len(val) > 0
        if isinstance(val, float):
            return True  # label=0.0 is a valid label
        return bool(val)

    def _check_task_fields(self, sample: DataSample, task: str) -> str | None:
        """Auto-derive required field checks from task_type."""

        if task in _PREFERENCE_TYPES:
            if not self._field_present(sample, "chosen"):
                return "missing_field:chosen"
            if not self._field_present(sample, "rejected"):
                return "missing_field:rejected"

        elif task in _GRPO_TYPES:
            if not self._field_present(sample, "instruction"):
                return "missing_field:instruction"
            if not self._field_present(sample, "responses"):
                return "missing_field:responses"

        elif task in _PRETRAIN_TYPES:
            if not self._field_present(sample, "output"):
                return "missing_field:output"

        elif task in _PROMPT_ONLY_TYPES:
            if not self._field_present(sample, "instruction"):
                return "missing_field:instruction"
            # output is intentionally absent — that's fine

        elif task in _SOURCE_CHUNK_TYPES:
            if not self._field_present(sample, "input"):
                return "missing_field:input"
            # instruction and output are generated later — not required here

        else:
            # Default: SFT family (instruction_following, conversational,
            # unpaired_preference, and any unknown task_type)
            if not self._field_present(sample, "instruction"):
                return "missing_field:instruction"
            if not self._field_present(sample, "output"):
                return "missing_field:output"

        return None

    def _primary_text(self, sample: DataSample, task: str) -> str:
        """
        Build the text to token-count for this sample's task type.
        Returns empty string if no meaningful text is available.
        """
        if task in _PREFERENCE_TYPES:
            # Count the prompt + the longer completion
            chosen_len = len((sample.chosen or "").split())
            rejected_len = len((sample.rejected or "").split())
            longer = sample.chosen if chosen_len >= rejected_len else sample.rejected
            return f"{sample.instruction} {longer}".strip()

        elif task in _GRPO_TYPES:
            if sample.responses:
                longest_resp = max(sample.responses, key=len)
                return f"{sample.instruction} {longest_resp}".strip()
            return sample.instruction or ""

        elif task in _PRETRAIN_TYPES:
            return sample.output or ""

        elif task in _PROMPT_ONLY_TYPES:
            return sample.instruction or ""

        elif task in _SOURCE_CHUNK_TYPES:
            return sample.input or ""

        else:
            # SFT family
            return f"{sample.instruction or ''} {sample.output or ''}".strip()
