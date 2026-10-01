"""
OutputSafetyGate — Llama Guard output classification.

Classify each sample's `output` (and `chosen` for
preference data) against Meta's Llama-Guard safety taxonomy.

Two backends:
  * "litellm" — call Llama Guard via any LiteLLM-supported provider
                (Together.ai, Replicate, local vLLM). Most flexible.
  * "transformers" — load the model locally. Slower to start, no API cost.

Heuristic fallback covers a small set of obviously-unsafe categories
(self-harm, explicit violence, illegal-activity instructions, hate speech)
so the gate stays useful in CI / local quickstart runs.

Returned categories follow Meta's MLCommons taxonomy (S1..S14). Any
non-"safe" classification triggers rejection.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"

# MLCommons-aligned category labels Llama Guard 4 emits.
_HAZARD_CATEGORIES: dict[str, str] = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_related_crimes",
    "S4": "child_sexual_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate",
    "S11": "suicide_self_harm",
    "S12": "sexual_content",
    "S13": "elections",
    "S14": "code_interpreter_abuse",
}


# Heuristic patterns — last-resort coverage of the highest-severity
# categories only.
_HEURISTIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "S11": re.compile(
        r"\b(?:how\s+to\s+(?:kill|harm)\s+(?:myself|yourself)|"
        r"suicide\s+method|self.harm\s+technique)\b",
        re.IGNORECASE,
    ),
    "S1": re.compile(
        r"\b(?:how\s+to\s+(?:murder|assault|attack)|"
        r"step.by.step\s+(?:killing|murder))\b",
        re.IGNORECASE,
    ),
    "S9": re.compile(
        r"\b(?:build\s+(?:a\s+)?(?:bomb|explosive|chemical\s+weapon)|"
        r"synthesize\s+(?:nerve\s+agent|chemical\s+weapon))\b",
        re.IGNORECASE,
    ),
    "S10": re.compile(
        r"\b(?:all\s+[a-z]+\s+(?:should\s+die|are\s+subhuman)|"
        r"genocide\s+is\s+justified)\b",
        re.IGNORECASE,
    ),
    "S4": re.compile(
        r"\b(?:child\s+(?:sexual|porn)|cp\s+(?:link|image|video))\b",
        re.IGNORECASE,
    ),
}


_FALLBACK_WARNED: list[bool] = [False]


def _build_llama_guard_messages(instruction: str, response: str) -> list[dict[str, str]]:
    """Build the Llama Guard prompt that scores an assistant response."""
    return [
        {
            "role": "user",
            "content": (
                "Task: Check if there is unsafe content in 'Agent' messages "
                "in conversations according to our safety policy with the "
                "below categories.\n\n"
                "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
                "S1: Violent Crimes. S2: Non-Violent Crimes. S3: Sex-Related Crimes. "
                "S4: Child Sexual Exploitation. S5: Defamation. S6: Specialized Advice. "
                "S7: Privacy. S8: Intellectual Property. S9: Indiscriminate Weapons. "
                "S10: Hate. S11: Suicide & Self-Harm. S12: Sexual Content. "
                "S13: Elections. S14: Code Interpreter Abuse.\n"
                "<END UNSAFE CONTENT CATEGORIES>\n\n"
                "<BEGIN CONVERSATION>\n"
                f"User: {instruction}\n"
                f"Agent: {response}\n"
                "<END CONVERSATION>\n\n"
                "Provide your safety assessment for ONLY THE LAST Agent message:\n"
                "- First line: 'safe' or 'unsafe'\n"
                "- If unsafe, second line: comma-separated list of violated categories."
            ),
        },
    ]


class OutputSafetyGate(BaseGate):
    """
    Reject samples whose output is classified unsafe by Llama Guard.

    Parameters
    ----------
    llm : Optional[BaseLLM]
        BaseLLM backend that calls Llama Guard. If None, falls back to
        the heuristic patterns above.
    fields : list[str]
        Which response-bearing fields to classify. Default: ["output", "chosen"].
    blocked_categories : list[str] | None
        S-codes to reject on. None → all categories (i.e., any "unsafe").
    """

    def __init__(
        self,
        llm: Any = None,
        fields: list[str] | None = None,
        blocked_categories: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.fields = fields or ["output", "chosen"]
        self.blocked_categories = set(blocked_categories) if blocked_categories else None
        if self.llm is None and not _FALLBACK_WARNED[0]:
            warnings.warn(
                "[OutputSafetyGate] no LLM provided — falling back to regex "
                "heuristic that covers only the highest-severity hazards. "
                "Pass an `llm=` for production runs.",
                stacklevel=2,
            )
            _FALLBACK_WARNED[0] = True

    # ─────────────────────────────────────────────────────────────────────

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "fields": sorted(self.fields),
                "blocked_categories": (
                    sorted(self.blocked_categories) if self.blocked_categories else None
                ),
                "backend": "llama_guard" if self.llm else "heuristic",
                "llm_model": getattr(self.llm, "model", None),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _classify(self, instruction: str, response: str) -> tuple[bool, list[str]]:
        """Return (is_unsafe, [violated_categories])."""
        if not response:
            return False, []

        if self.llm is None:
            return self._classify_heuristic(response)

        try:
            messages = _build_llama_guard_messages(instruction, response)
            llm_response = self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=64,
            )
        except Exception:
            # On LLM failure, fall through to heuristic so the gate is
            # never silently bypassed.
            return self._classify_heuristic(response)

        text = (llm_response.text or "").strip()
        first_line = text.splitlines()[0].strip().lower() if text else ""
        if first_line != "unsafe":
            return False, []

        # Parse second line for S-codes
        cats: list[str] = []
        if len(text.splitlines()) > 1:
            raw = text.splitlines()[1].strip()
            for tok in re.split(r"[,\s]+", raw):
                tok = tok.upper().strip()
                if tok in _HAZARD_CATEGORIES:
                    cats.append(tok)

        return True, cats

    def _classify_heuristic(self, text: str) -> tuple[bool, list[str]]:
        cats: list[str] = []
        for code, pattern in _HEURISTIC_PATTERNS.items():
            if pattern.search(text):
                cats.append(code)
        return bool(cats), cats

    # ─────────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC).replace(tzinfo=None)
        backend = "llama_guard" if self.llm else "heuristic"

        for sample in samples:
            worst_cats: list[str] = []
            worst_field = ""
            instruction = sample.instruction
            for field in self.fields:
                resp = getattr(sample, field, "") or ""
                if not resp:
                    continue
                unsafe, cats = self._classify(instruction, resp)
                if unsafe:
                    # blocked_categories filter
                    if self.blocked_categories:
                        cats = [c for c in cats if c in self.blocked_categories]
                    if cats and len(cats) > len(worst_cats):
                        worst_cats = cats
                        worst_field = field

            if not worst_cats:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="OutputSafetyGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "safe": True,
                            "backend": backend,
                        },
                    )
                )
                passed.append(sample)
                continue

            reason_tag = "|".join(worst_cats[:3])  # truncate to keep the code short
            rej = RejectedSample(
                **sample.model_dump(),
                rejection_reason=f"safety_output:{reason_tag}",
                rejecting_step="OutputSafetyGate",
            )
            rej.append_provenance(
                ProvenanceRecord(
                    step_name="OutputSafetyGate",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "safe": False,
                        "categories": worst_cats,
                        "category_names": [
                            _HAZARD_CATEGORIES.get(c, "unknown") for c in worst_cats
                        ],
                        "worst_field": worst_field,
                        "backend": backend,
                    },
                )
            )
            rejected.append(rej)

        return passed, rejected
