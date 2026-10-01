"""
FActScoreJudge — atomic-fact decomposition + per-fact verification.

Reference: Min et al., "FActScore: Fine-grained Atomic Evaluation of
Factual Precision in Long Form Text Generation" (EMNLP 2023)
  Paper: https://arxiv.org/abs/2305.14251
  Repo:  https://github.com/shmsw25/FActScore

The canonical FActScore pipeline:

  1. Decompose a generated response into a list of "atomic facts" — short,
     standalone propositions, one per claim.
  2. For each atomic fact, verify it against a knowledge source (in the
     paper: Wikipedia; in our pipeline: the source chunk attached to the
     DataSample). Each fact gets a binary supported/unsupported label.
  3. FActScore = supported_facts / total_facts. A response is "factually
     precise" when this fraction crosses a threshold (paper uses 0.5; this
     class defaults to 0.7 for stricter SFT filtering).

Why this is stronger than the existing HallucinationGate:
  HallucinationGate returns one aggregate grounding score per response.
  FActScore decomposes first — a long response containing 9 correct facts
  and 1 hallucinated fact gets a clean 0.9 score, AND the unsupported
  fact is itemised in the provenance. This is exactly the granularity a
  reviewer needs to audit *which* claim failed.

Two callable surfaces:
  FActScoreJudge.judge(instruction, response, source) -> FActScoreResult
  FActScoreGate(judge, threshold) -> BaseGate (drop-in for HallucinationGate)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseGate
from curatorkit.llm.base import BaseLLM
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


# ───────────────────────────────────────────────────────────────────────────
# Prompts
# ───────────────────────────────────────────────────────────────────────────

# Following the paper's atomic-fact extraction prompt (Appendix A.2):
# "Please break down the following sentence into independent facts."
_DECOMPOSE_PROMPT = """Please break the following statement into independent
atomic facts. An atomic fact is a single, standalone claim that is either
true or false on its own.

Statement: "{statement}"

Output ONE atomic fact per line, with no numbering or bullets. If the
statement contains no factual claims (e.g., it is a refusal or an
acknowledgment), output the single line: NO_FACTS"""


# Per-fact verification prompt — answers VERIFIED / UNVERIFIED based on whether
# the fact is entailed by the source.
_VERIFY_PROMPT = """You are a strict fact-checker. Given a source text and a
single atomic fact, decide whether the fact is fully supported by the source.

Source text:
\"\"\"
{source}
\"\"\"

Atomic fact: "{fact}"

Answer with EXACTLY one word on the first line:
  - SUPPORTED  - the source clearly entails the fact
  - UNSUPPORTED  - the source does not contain or entails this fact
  - CONTRADICTED  - the source contains a claim that contradicts this fact

Then on a second line, give a brief one-sentence justification."""


_NO_FACTS_TOKEN = "NO_FACTS"

_VERIFY_LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")


# ───────────────────────────────────────────────────────────────────────────
# Result
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class AtomicFact:
    text: str
    label: str  # SUPPORTED | UNSUPPORTED | CONTRADICTED
    justification: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "label": self.label, "justification": self.justification}


@dataclass
class FActScoreResult:
    facts: list[AtomicFact] = field(default_factory=list)
    n_supported: int = 0
    n_unsupported: int = 0
    n_contradicted: int = 0

    @property
    def factscore(self) -> float:
        """Precision of supported facts. 1.0 if no facts (vacuous truth)."""
        total = len(self.facts)
        if total == 0:
            return 1.0
        return self.n_supported / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "factscore": round(self.factscore, 4),
            "n_facts": len(self.facts),
            "n_supported": self.n_supported,
            "n_unsupported": self.n_unsupported,
            "n_contradicted": self.n_contradicted,
            "facts": [f.to_dict() for f in self.facts],
        }


# ───────────────────────────────────────────────────────────────────────────
# Judge
# ───────────────────────────────────────────────────────────────────────────


class FActScoreJudge:
    """
    Decompose and verify atomic facts. One judge instance is reusable
    across many samples — it keeps no per-sample state.

    Parameters
    ----------
    llm : BaseLLM
        Backend for both decomposition and verification. The paper uses
        GPT-4 / ChatGPT for both; smaller models work but noisier facts.
    decompose_temperature : float
        Temperature for decomposition. Keep low (default 0.1) for stability.
    verify_temperature : float
        Temperature for verification. Keep at 0 for hard binary judgments.
    max_facts : int
        Cap on facts per response (avoid runaway costs on very long outputs).
        Default 20.
    """

    def __init__(
        self,
        llm: BaseLLM,
        decompose_temperature: float = 0.1,
        verify_temperature: float = 0.0,
        max_facts: int = 20,
    ) -> None:
        self.llm = llm
        self.decompose_temperature = float(decompose_temperature)
        self.verify_temperature = float(verify_temperature)
        self.max_facts = max(1, int(max_facts))

    # ─────────────────────────────────────────────────────────────────────

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "llm_model": self.llm.model,
                "decompose_temperature": self.decompose_temperature,
                "verify_temperature": self.verify_temperature,
                "max_facts": self.max_facts,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _decompose(self, statement: str) -> list[str]:
        try:
            r = self.llm.generate(
                [{"role": "user", "content": _DECOMPOSE_PROMPT.format(statement=statement)}],
                temperature=self.decompose_temperature,
                max_tokens=512,
            )
        except Exception:
            return []
        text = (r.text or "").strip()
        if _NO_FACTS_TOKEN in text:
            return []
        facts: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            # Strip leading "1." / "-" / "*"
            line = re.sub(r"^\s*(?:\d+[\.\)]|\-|\*)\s*", "", line)
            if line and line.upper() != _NO_FACTS_TOKEN:
                facts.append(line)
            if len(facts) >= self.max_facts:
                break
        return facts

    def _verify(self, source: str, fact: str) -> tuple[str, str]:
        try:
            r = self.llm.generate(
                [{"role": "user", "content": _VERIFY_PROMPT.format(source=source, fact=fact)}],
                temperature=self.verify_temperature,
                max_tokens=200,
            )
        except Exception:
            return "UNSUPPORTED", "verification_error"
        text = (r.text or "").strip()
        lines = text.splitlines()
        first = lines[0].strip().upper() if lines else ""
        label = "UNSUPPORTED"
        for token in _VERIFY_LABELS:
            if first.startswith(token):
                label = token
                break
        justification = lines[1].strip() if len(lines) > 1 else ""
        return label, justification

    # ─────────────────────────────────────────────────────────────────────

    def judge(
        self,
        instruction: str,
        response: str,
        source: str,
    ) -> FActScoreResult:
        """Decompose `response`, verify each fact against `source`."""
        result = FActScoreResult()
        if not response.strip():
            return result

        fact_texts = self._decompose(response)
        for text in fact_texts:
            label, justification = self._verify(source, text)
            af = AtomicFact(text=text, label=label, justification=justification)
            result.facts.append(af)
            if label == "SUPPORTED":
                result.n_supported += 1
            elif label == "UNSUPPORTED":
                result.n_unsupported += 1
            elif label == "CONTRADICTED":
                result.n_contradicted += 1
        return result


# ───────────────────────────────────────────────────────────────────────────
# Gate
# ───────────────────────────────────────────────────────────────────────────


class FActScoreGate(BaseGate):
    """
    Drop-in replacement for HallucinationGate using atomic-fact verification.

    Parameters
    ----------
    judge : FActScoreJudge
        The decomposer + verifier.
    threshold : float
        Reject if factscore < threshold. Default 0.7. Set lower for
        looser filtering, higher for stricter.
    context_field : str
        Where the source / context lives. "input" (default) or
        "metadata.source_chunk".
    response_field : str
        Which response field to decompose. Default "output".
    contradicted_is_hard_fail : bool
        If True (default), a single CONTRADICTED fact rejects the sample
        regardless of the overall factscore — a contradiction is strictly
        worse than an unsupported claim.
    """

    def __init__(
        self,
        judge: FActScoreJudge,
        threshold: float = 0.7,
        context_field: str = "input",
        response_field: str = "output",
        contradicted_is_hard_fail: bool = True,
    ) -> None:
        self.judge = judge
        self.threshold = float(threshold)
        self.context_field = context_field
        self.response_field = response_field
        self.contradicted_is_hard_fail = bool(contradicted_is_hard_fail)

    # ─────────────────────────────────────────────────────────────────────

    def _resolve_source(self, sample: DataSample) -> str:
        if self.context_field.startswith("metadata."):
            key = self.context_field.split(".", 1)[1]
            return str(sample.metadata.get(key, ""))
        return str(getattr(sample, self.context_field, "") or "")

    def _resolve_response(self, sample: DataSample) -> str:
        return str(getattr(sample, self.response_field, "") or "")

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        ts = datetime.now(UTC).replace(tzinfo=None)
        cfg_hash = self.judge.config_hash()

        for sample in samples:
            source = self._resolve_source(sample)
            response = self._resolve_response(sample)
            # No source means we can't verify — pass through with skip audit.
            if not source or not response:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="FActScoreGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"skipped": True, "reason": "missing_source_or_response"},
                    )
                )
                passed.append(sample)
                continue

            try:
                result = self.judge.judge(sample.instruction, response, source)
            except Exception as e:
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="FActScoreGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={"error": str(e), "passed_on_error": True},
                    )
                )
                passed.append(sample)
                continue

            score = result.factscore
            hard_fail = self.contradicted_is_hard_fail and result.n_contradicted > 0

            if score >= self.threshold and not hard_fail:
                sample.label = score
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="FActScoreGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": True,
                            "factscore": score,
                            "threshold": self.threshold,
                            **result.to_dict(),
                        },
                    )
                )
                passed.append(sample)
            else:
                reason_suffix = "contradicted" if hard_fail else f"{score:.2f}"
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"factscore_below_threshold:{reason_suffix}",
                    rejecting_step="FActScoreGate",
                )
                rej.append_provenance(
                    ProvenanceRecord(
                        step_name="FActScoreGate",
                        step_version=STEP_VERSION,
                        timestamp=ts,
                        config_hash=cfg_hash,
                        notes={
                            "passed": False,
                            "factscore": score,
                            "threshold": self.threshold,
                            "hard_fail_contradicted": hard_fail,
                            **result.to_dict(),
                        },
                    )
                )
                rejected.append(rej)

        return passed, rejected
