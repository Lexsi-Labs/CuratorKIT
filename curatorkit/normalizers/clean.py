"""
TextCleaner — composable text normalisation chain.

Each transform is a separate private method, independently toggleable via the
transforms config dict. No DataTrove or other framework — written from scratch.

Transforms (all enabled by default):
  strip_html            — regex-based HTML tag removal
  normalise_unicode     — NFC normalisation via unicodedata
  fix_encoding_artifacts — Windows-1252 mojibake pattern replacement
  collapse_whitespace   — collapse runs of whitespace to single space
  remove_control_chars  — strip bytes < 0x20 except tab (0x09) and newline (0x0A)
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime

from curatorkit.interfaces import BaseNormalizer
from curatorkit.schema import DataSample, ProvenanceRecord

STEP_VERSION = "0.1.0"

# Common Windows-1252 / latin-1 mojibake sequences and their UTF-8 equivalents
_MOJIBAKE_MAP: list[tuple[str, str]] = [
    ("\u00e2\u0080\u0099", "'"),  # â€™ → '
    ("\u00e2\u0080\u0098", "'"),  # â€˜ → '
    ("\u00e2\u0080\u009c", '"'),  # â€œ → "
    ("\u00e2\u0080\u009d", '"'),  # â€ → "
    ("\u00e2\u0080\u0093", "–"),  # â€" → –
    ("\u00e2\u0080\u0094", "—"),  # â€" → —
    ("\u00e2\u0080\u00a6", "…"),  # â€¦ → …
    ("\u00c3\u00a9", "é"),
    ("\u00c3\u00a8", "è"),
    ("\u00c3\u00aa", "ê"),
    ("\u00c3\u00ab", "ë"),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TextCleaner(BaseNormalizer):
    """Apply a configurable chain of text-cleaning transforms to each DataSample.

    Args:
        transforms: Dict of transform name -> bool. Missing keys default to True.
            Keys: strip_html, normalise_unicode, fix_encoding_artifacts,
                  collapse_whitespace, remove_control_chars.
        fields: DataSample fields to clean (default: instruction, input, output).
    """

    DEFAULT_TRANSFORMS = {
        "strip_html": True,
        "normalise_unicode": True,
        "fix_encoding_artifacts": True,
        "collapse_whitespace": True,
        "remove_control_chars": True,
    }

    def __init__(
        self,
        transforms: dict[str, bool] | None = None,
        fields: list[str] | None = None,
    ) -> None:
        self.transforms = {**self.DEFAULT_TRANSFORMS, **(transforms or {})}
        self.fields = fields or ["instruction", "input", "output"]

    def _config_hash(self) -> str:
        payload = json.dumps(
            {"transforms": self.transforms, "fields": sorted(self.fields)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC).replace(tzinfo=None)
        applied = [k for k, v in self.transforms.items() if v]

        from tqdm import tqdm

        for sample in tqdm(samples, desc="TextCleaner", unit="sample"):
            for field in self.fields:
                value = getattr(sample, field, None)
                if isinstance(value, str):
                    setattr(sample, field, self._clean(value))

            sample.append_provenance(
                ProvenanceRecord(
                    step_name="TextCleaner",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={"transforms_applied": applied, "fields_cleaned": self.fields},
                )
            )

        return samples

    def _clean(self, text: str) -> str:
        if self.transforms.get("strip_html", True):
            text = self._strip_html(text)
        if self.transforms.get("normalise_unicode", True):
            text = self._normalise_unicode(text)
        if self.transforms.get("fix_encoding_artifacts", True):
            text = self._fix_encoding_artifacts(text)
        if self.transforms.get("remove_control_chars", True):
            text = self._remove_control_chars(text)
        if self.transforms.get("collapse_whitespace", True):
            text = self._collapse_whitespace(text)
        return text.strip()

    def _strip_html(self, text: str) -> str:
        return _HTML_TAG_RE.sub("", text)

    def _normalise_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _fix_encoding_artifacts(self, text: str) -> str:
        for pattern, replacement in _MOJIBAKE_MAP:
            text = text.replace(pattern, replacement)
        return text

    def _collapse_whitespace(self, text: str) -> str:
        # Preserve newlines; collapse other whitespace runs
        lines = text.split("\n")
        lines = [_WHITESPACE_RE.sub(" ", line) for line in lines]
        return "\n".join(lines)

    def _remove_control_chars(self, text: str) -> str:
        return _CONTROL_CHAR_RE.sub("", text)
