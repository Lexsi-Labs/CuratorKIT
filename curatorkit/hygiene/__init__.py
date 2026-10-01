"""
CuratorKIT hygiene — data safety components.

Lightweight components that scrub training data before LLM generation
begins, plus output-side safety checks:

  ToxicityGate       — two-stage filter: Detoxify classifier for clear
                        cases, LLM judge for borderline samples.
  SecretsGate        — rejects samples containing credentials, API keys,
                        or high-entropy secrets (yelp/detect-secrets).
  PIIPseudonymizer   — replaces PII entities with consistent fake values
                        per document (clinical NLP de-identification style).
  PIIGate            — rejects samples containing PII, or redacts spans to
                        <ENTITY_TYPE> placeholders (Presidio, regex fallback).
  JailbreakGate      — rejects prompt-injection / jailbreak prompts
                        (Meta PromptGuard 2, keyword fallback).
  OutputSafetyGate   — rejects unsafe outputs (Llama Guard hazard taxonomy
                        via any judge LLM, regex fallback).
  SafetyComplianceReport — aggregates every safety rejection into an audit
                        report.

All plug into existing extension points (BaseGate / BaseNormalizer)
and write structured notes to the provenance chain.

Install dependencies:
  pip install "curatorkit[hygiene]"   # Presidio / Detoxify / detect-secrets
  pip install "curatorkit[safety]"    # PromptGuard (transformers + torch)
"""

from curatorkit.hygiene.audit import SafetyComplianceReport
from curatorkit.hygiene.jailbreak import JailbreakGate
from curatorkit.hygiene.output_safety import OutputSafetyGate
from curatorkit.hygiene.pii import PIIPseudonymizer
from curatorkit.hygiene.pii_gate import PIIGate
from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.hygiene.toxicity import ToxicityGate

__all__ = [
    "ToxicityGate",
    "SecretsGate",
    "PIIPseudonymizer",
    "PIIGate",
    "JailbreakGate",
    "OutputSafetyGate",
    "SafetyComplianceReport",
]
