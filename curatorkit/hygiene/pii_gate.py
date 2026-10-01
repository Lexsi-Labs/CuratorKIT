"""
PIIGate — PII detection at ingest using Microsoft Presidio.

Two modes:
  mode="reject"  → samples with detected PII go to rejected.jsonl with
                   reason "safety_pii:{entity_type}:{count}"
  mode="redact"  → PII spans replaced with "<ENTITY_TYPE>" placeholders;
                   sample passes through with audit provenance attached

Presidio is an opt-in extra (`pip install curatorkit[hygiene]`). When
unavailable, the gate falls back to a regex heuristic covering email,
phone, IPv4, credit card, US-SSN. The heuristic is *strictly* worse and
intended for local development / CI without GPU. Production runs MUST
have Presidio installed — the gate logs a one-time warning if it falls
back.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"

_VALID_MODES = ("reject", "redact")

# Regex fallback patterns. Kept conservative — false positives are better
# than false negatives for a safety gate.
_HEURISTIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL_ADDRESS": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE_NUMBER": re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

_FALLBACK_WARNED: list[bool] = [False]  # mutable global, fires once per process


@dataclass
class PIIMatch:
    entity_type: str
    start: int
    end: int
    text: str
    score: float = 1.0


def _load_presidio() -> Any:
    """Return AnalyzerEngine if available, else None."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore

        return AnalyzerEngine
    except ImportError:
        return None


class PIIGate(BaseGate):
    """
    Detect and either reject or redact PII in DataSample text fields.

    Parameters
    ----------
    mode : "reject" | "redact"
        Disposition for samples containing PII.
    fields : list[str]
        DataSample text fields to scan. Default: instruction, input, output,
        chosen, rejected. Excludes label and metadata.
    entities : list[str] | None
        Specific entity types to detect. None → Presidio default set.
    min_score : float
        Minimum Presidio confidence to count as a match. Default 0.5.
    language : str
        Language hint passed to Presidio. Default "en".
    """

    def __init__(
        self,
        mode: str = "reject",
        fields: list[str] | None = None,
        entities: list[str] | None = None,
        min_score: float = 0.5,
        language: str = "en",
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.fields = fields or ["instruction", "input", "output", "chosen", "rejected"]
        self.entities = entities
        self.min_score = float(min_score)
        self.language = language

        analyzer_cls = _load_presidio()
        if analyzer_cls is None:
            if not _FALLBACK_WARNED[0]:
                warnings.warn(
                    "[PIIGate] Presidio not installed — falling back to regex "
                    "heuristic (email/phone/ip/credit-card/SSN only). "
                    "Install `curatorkit[hygiene]` for production runs.",
                    stacklevel=2,
                )
                _FALLBACK_WARNED[0] = True
            self._analyzer = None
        else:
            self._analyzer = analyzer_cls()

    # ─────────────────────────────────────────────────────────────────────

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "mode": self.mode,
                "fields": sorted(self.fields),
                "entities": sorted(self.entities) if self.entities else None,
                "min_score": self.min_score,
                "language": self.language,
                "backend": "presidio" if self._analyzer else "heuristic",
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _scan_text(self, text: str) -> list[PIIMatch]:
        if not text:
            return []

        if self._analyzer is not None:
            # Presidio path
            try:
                results = self._analyzer.analyze(
                    text=text,
                    entities=self.entities,
                    language=self.language,
                )
            except Exception:
                return []
            return [
                PIIMatch(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    text=text[r.start : r.end],
                    score=float(r.score),
                )
                for r in results
                if float(r.score) >= self.min_score
            ]

        # Heuristic fallback
        matches: list[PIIMatch] = []
        for entity_type, pattern in _HEURISTIC_PATTERNS.items():
            if self.entities and entity_type not in self.entities:
                continue
            for m in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        entity_type=entity_type,
                        start=m.start(),
                        end=m.end(),
                        text=m.group(0),
                        score=1.0,
                    )
                )
        return matches

    def _redact_text(self, text: str, matches: list[PIIMatch]) -> str:
        if not matches:
            return text
        # Sort descending so index math doesn't shift under us
        for m in sorted(matches, key=lambda x: -x.start):
            text = text[: m.start] + f"<{m.entity_type}>" + text[m.end :]
        return text

    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC).replace(tzinfo=None)

        for sample in samples:
            per_field_matches: dict[str, list[PIIMatch]] = {}
            for field in self.fields:
                text = getattr(sample, field, "") or ""
                hits = self._scan_text(text)
                if hits:
                    per_field_matches[field] = hits

            if not per_field_matches:
                # No PII — pass with a clean audit record
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="PIIGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"pii_detected": False, "mode": self.mode},
                    )
                )
                passed.append(sample)
                continue

            # PII present — choose disposition
            entity_summary = _summarize_matches(per_field_matches)

            if self.mode == "reject":
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"safety_pii:{_top_entity(per_field_matches)}",
                    rejecting_step="PIIGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="PIIGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "pii_detected": True,
                            "mode": self.mode,
                            "entities": entity_summary,
                            "backend": "presidio" if self._analyzer else "heuristic",
                        },
                    )
                )
                rejected.append(rej)
                continue

            # mode == "redact" — rewrite fields, attach audit, pass through
            for field, hits in per_field_matches.items():
                original = getattr(sample, field, "") or ""
                setattr(sample, field, self._redact_text(original, hits))

            sample.append_provenance(
                ProvenanceRecord(
                    step_name="PIIGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "pii_detected": True,
                        "mode": self.mode,
                        "entities": entity_summary,
                        "redacted_fields": list(per_field_matches.keys()),
                        "backend": "presidio" if self._analyzer else "heuristic",
                    },
                )
            )
            passed.append(sample)

        return passed, rejected


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _summarize_matches(per_field: dict[str, list[PIIMatch]]) -> dict[str, dict[str, int]]:
    """{field: {entity_type: count}} for audit logging — no raw text."""
    out: dict[str, dict[str, int]] = {}
    for field, matches in per_field.items():
        counts: dict[str, int] = {}
        for m in matches:
            counts[m.entity_type] = counts.get(m.entity_type, 0) + 1
        out[field] = counts
    return out


def _top_entity(per_field: dict[str, list[PIIMatch]]) -> str:
    """Pick the most common entity type across all fields for the reason code."""
    counts: dict[str, int] = {}
    for matches in per_field.values():
        for m in matches:
            counts[m.entity_type] = counts.get(m.entity_type, 0) + 1
    if not counts:
        return "UNKNOWN"
    top = max(counts.items(), key=lambda x: x[1])
    return f"{top[0]}:{top[1]}"
