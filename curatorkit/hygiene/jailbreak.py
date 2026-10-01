"""
JailbreakGate — detect prompt-injection / jailbreak in instruction fields.

Backed by Meta's PromptGuard 2 (Llama-Prompt-Guard-2-86M) when available
via huggingface transformers. Falls back to a keyword heuristic when
the model isn't installed.

PromptGuard returns three logits per input: BENIGN / INJECTION / JAILBREAK.
We treat a sample as flagged if max(injection_score, jailbreak_score)
exceeds the threshold.

Runs on the prompt fields of every sample so a jailbroken seed cannot
propagate into generation as a hidden prompt.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


# Heuristic patterns — present in nearly every published prompt-injection
# corpus. Used only when PromptGuard isn't available. Case-insensitive
# substring match.
_JAILBREAK_HEURISTICS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard the instructions",
    "you are now",
    "act as if",
    "pretend you are",
    "developer mode",
    "do anything now",
    "dan mode",
    "jailbreak",
    "system prompt:",
    "system:",
    "<|im_start|>",
    "<|im_end|>",
    "override your",
    "bypass your",
)

_FALLBACK_WARNED: list[bool] = [False]


def _load_promptguard() -> Any:
    """Return a (tokenizer, model) pair for PromptGuard, or None."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        # Lazy: caller decides whether to load weights
        return AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        return None


class JailbreakGate(BaseGate):
    """
    Detect prompt-injection / jailbreak attempts.

    Parameters
    ----------
    threshold : float
        Reject samples with max(injection, jailbreak) score above this.
        Default 0.85.
    fields : list[str]
        Fields to inspect. Default ["instruction", "input"].
    model_name : str
        PromptGuard model id. Defaults to Meta's Llama-Prompt-Guard-2-86M.
    device : str
        "cpu" or "cuda". Default "cpu".
    use_model : bool
        If False, force heuristic mode regardless of availability. Useful
        for tests and CPU-only quickstart runs.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        fields: list[str] | None = None,
        model_name: str = "meta-llama/Llama-Prompt-Guard-2-86M",
        device: str = "cpu",
        use_model: bool = True,
    ) -> None:
        self.threshold = float(threshold)
        self.fields = fields or ["instruction", "input"]
        self.model_name = model_name
        self.device = device
        self.use_model = use_model

        self._tokenizer = None
        self._model = None
        if self.use_model:
            self._maybe_load_model()
        elif not _FALLBACK_WARNED[0]:
            _FALLBACK_WARNED[0] = True

    def _maybe_load_model(self) -> None:
        loaders = _load_promptguard()
        if loaders is None:
            if not _FALLBACK_WARNED[0]:
                warnings.warn(
                    "[JailbreakGate] transformers not installed — falling back "
                    "to keyword heuristic. Install `curatorkit[safety]` for "
                    "PromptGuard-backed detection.",
                    stacklevel=3,
                )
                _FALLBACK_WARNED[0] = True
            return

        AutoTokenizer, AutoModel = loaders
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
        except Exception as e:
            warnings.warn(
                f"[JailbreakGate] could not load {self.model_name}: {e}. "
                "Falling back to keyword heuristic.",
                stacklevel=3,
            )
            self._tokenizer = None
            self._model = None

    # ─────────────────────────────────────────────────────────────────────

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "threshold": self.threshold,
                "fields": sorted(self.fields),
                "model_name": self.model_name if self._model else "heuristic",
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _score(self, text: str) -> tuple[float, str]:
        """Return (max_unsafe_score, verdict)."""
        if not text:
            return 0.0, "empty"

        if self._model is not None and self._tokenizer is not None:
            return self._score_model(text)
        return self._score_heuristic(text)

    def _score_model(self, text: str) -> tuple[float, str]:
        import torch  # type: ignore

        with torch.no_grad():
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        # PromptGuard 2 layout: [BENIGN, INJECTION, JAILBREAK]
        if len(probs) < 3:
            return float(max(probs[1:])) if len(probs) > 1 else 0.0, "model"
        injection, jailbreak = probs[1], probs[2]
        worst = max(injection, jailbreak)
        verdict = "jailbreak" if jailbreak >= injection else "injection"
        return float(worst), verdict

    def _score_heuristic(self, text: str) -> tuple[float, str]:
        low = text.lower()
        hits = [p for p in _JAILBREAK_HEURISTICS if p in low]
        if not hits:
            return 0.0, "benign"
        # Treat any heuristic hit as high-confidence (regex matches are
        # binary). The threshold check below will still apply.
        return 1.0, f"heuristic:{hits[0]}"

    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC).replace(tzinfo=None)
        backend = "promptguard" if self._model else "heuristic"

        for sample in samples:
            worst_score = 0.0
            worst_field = ""
            worst_verdict = "benign"
            for field in self.fields:
                text = getattr(sample, field, "") or ""
                score, verdict = self._score(text)
                if score > worst_score:
                    worst_score = score
                    worst_field = field
                    worst_verdict = verdict

            if worst_score < self.threshold:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="JailbreakGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "max_unsafe_score": round(worst_score, 4),
                            "verdict": worst_verdict,
                            "threshold": self.threshold,
                            "backend": backend,
                            "passed": True,
                        },
                    )
                )
                passed.append(sample)
                continue

            rej = RejectedSample(
                **sample.model_dump(),
                rejection_reason=f"safety_jailbreak:{worst_verdict}:{worst_score:.2f}",
                rejecting_step="JailbreakGate",
            )
            rej.append_provenance(
                ProvenanceRecord(
                    step_name="JailbreakGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "max_unsafe_score": round(worst_score, 4),
                        "verdict": worst_verdict,
                        "worst_field": worst_field,
                        "threshold": self.threshold,
                        "backend": backend,
                        "passed": False,
                    },
                )
            )
            rejected.append(rej)

        return passed, rejected
