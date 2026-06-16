"""
SecretsGate — reject training samples that contain credentials or API keys.

Uses yelp/detect-secrets for detection: a battery of deterministic regex
plugins (AWS keys, GitHub tokens, PEM private keys, JWTs, Stripe/Slack/
SendGrid tokens, etc.) plus Shannon-entropy analysis for unknown formats.

Design: reject, do not redact.
Redacting the secret value while keeping the surrounding context still
teaches the fine-tuned model the structural pattern of credential exposure
("API keys appear in this slot relative to this code"). Removing the whole
sample is the correct default for non-code corpora.

KeywordDetector (entropy + keyword proximity) is OFF by default because it
generates false positives in prose corpora: any sentence mentioning
"encryption key", "password policy", "secret sauce" etc. will trigger it.
Enable it with code_corpus_mode=True for datasets that include source code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"

_PLUGINS_BASE = [
    {"name": "AWSKeyDetector"},
    {"name": "ArtifactoryDetector"},
    {"name": "BasicAuthDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "HexHighEntropyString", "limit": 3.0},
    {"name": "Base64HighEntropyString", "limit": 4.5},
    {"name": "JwtTokenDetector"},
    {"name": "MailchimpDetector"},
    {"name": "NpmDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "SendGridDetector"},
    {"name": "SlackDetector"},
    {"name": "StripeDetector"},
    {"name": "TwilioKeyDetector"},
]

_PLUGIN_KEYWORD = {"name": "KeywordDetector"}


def _ensure_detect_secrets() -> None:
    try:
        import detect_secrets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "detect-secrets is not installed. Install with: pip install 'curatorkit[hygiene]'"
        ) from e


def _scan_text(text: str, ds_config: dict) -> list[dict]:
    """
    Scan text for secrets line by line. Returns list of {type, line_number}.

    Uses detect_secrets.core.scan.scan_line which reads from the active
    transient_settings context — no temp files required.
    """
    from detect_secrets.core.scan import scan_line
    from detect_secrets.settings import transient_settings

    findings: list[dict] = []
    with transient_settings(ds_config):
        for line_num, line in enumerate(text.split("\n"), start=1):
            for secret in scan_line(line):
                findings.append({"type": secret.type, "line_number": line_num})
    return findings


_TASK_FIELDS: dict[str, list[str]] = {
    "language_modeling": ["output"],
    "source_chunk": ["output"],
    "prompt_only": ["instruction", "input"],
    "preference": ["instruction", "input", "chosen", "rejected"],
    "implicit_preference": ["instruction", "input", "chosen", "rejected"],
    "unpaired_preference": ["instruction", "input", "output"],
    "grpo": ["instruction", "input", "responses"],
    "conversational": ["instruction", "input", "output"],
    "instruction_following": ["instruction", "input", "output"],
}


class SecretsGate(BaseGate):
    """
    Reject samples containing credentials, API keys, or high-entropy secrets.

    Task-aware: selects fields based on sample.task_type so preference pairs
    (chosen/rejected) and GRPO rollouts (responses) are always scanned when
    relevant.

    Parameters
    ----------
    fields : list[str] | None
        DataSample fields to scan. Defaults to all task-relevant fields.
        Explicitly setting this overrides the task-aware selection entirely.
    code_corpus_mode : bool
        If True, enables KeywordDetector (entropy + keyword proximity scanning).
        Disable for prose corpora (legal, medical, academic) to avoid false
        positives on legitimate mentions of "key", "password", "secret", etc.
    plugins : list[dict] | None
        Full detect-secrets plugin config. Overrides code_corpus_mode if set.
    """

    def __init__(
        self,
        fields: list[str] | None = None,
        code_corpus_mode: bool = False,
        plugins: list[dict] | None = None,
    ) -> None:
        _ensure_detect_secrets()
        self._fields_override = fields  # None = use task-aware selection
        self.fields = fields or [
            "instruction",
            "input",
            "output",
            "chosen",
            "rejected",  # DPO preference pairs
            "responses",  # GRPO rollouts (list field — handled below)
        ]
        self.code_corpus_mode = code_corpus_mode

        if plugins is not None:
            self._plugins = plugins
        elif code_corpus_mode:
            self._plugins = _PLUGINS_BASE + [_PLUGIN_KEYWORD]
        else:
            self._plugins = _PLUGINS_BASE

        self._ds_config = {
            "plugins_used": self._plugins,
            "filters_used": [],
        }

    def _fields_for_sample(self, sample: DataSample) -> list[str]:
        """Return fields to scan, selected by task_type when no explicit override."""
        if self._fields_override is not None:
            return self._fields_override
        tt = getattr(sample, "task_type", None) or ""
        candidates = _TASK_FIELDS.get(tt, self.fields)
        return [f for f in candidates if f in self.fields]

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "fields": sorted(self.fields),
                "plugins": self._plugins,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _scan_sample(self, sample: DataSample) -> tuple[list[dict], list[str]]:
        """Scan a sample for secrets. Returns (findings, fields_actually_scanned)."""
        active_fields = self._fields_for_sample(sample)
        findings: list[dict] = []
        for field in active_fields:
            val = getattr(sample, field, None)
            if val is None:
                continue
            if isinstance(val, list):
                # GRPO responses — scan each completion independently
                for i, text in enumerate(val):
                    if isinstance(text, str) and text.strip():
                        for finding in _scan_text(text, self._ds_config):
                            findings.append({**finding, "field": f"{field}[{i}]"})
            elif isinstance(val, str) and val.strip():
                for finding in _scan_text(val, self._ds_config):
                    findings.append({**finding, "field": field})
        return findings, active_fields

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        if not samples:
            return [], []

        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []

        from tqdm import tqdm

        for sample in tqdm(samples, desc="SecretsGate", unit="sample"):
            findings, scanned_fields = self._scan_sample(sample)

            if findings:
                type_counts: dict[str, int] = {}
                for f in findings:
                    type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1
                types_str = ",".join(sorted(type_counts.keys()))

                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"secret_detected:{types_str}",
                    rejecting_step="SecretsGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="SecretsGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": False,
                            "secret_type_counts": type_counts,
                            "fields_scanned": scanned_fields,
                            "total_findings": len(findings),
                        },
                    )
                )
                rejected.append(rej)
            else:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="SecretsGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": True,
                            "fields_scanned": scanned_fields,
                        },
                    )
                )
                passed.append(sample)

        return passed, rejected
