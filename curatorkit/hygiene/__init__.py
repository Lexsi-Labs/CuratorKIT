"""
CuratorKIT hygiene — data safety components.

Three lightweight, infrastructure-reusing components that scrub training
data before LLM generation begins:

  ToxicityGate       — two-stage filter: Detoxify classifier for clear
                        cases, LLM judge for borderline samples.
  GuardModelGate     — Hugging Face guard model (Qwen3Guard-Gen, Llama Guard);
                        reads languages Detoxify cannot. Needs [guard] extras.
  SecretsGate        — rejects samples containing credentials, API keys,
                        or high-entropy secrets (yelp/detect-secrets).
  PIIPseudonymizer   — replaces PII entities with consistent fake values
                        per document (clinical NLP de-identification style).

All three plug into existing extension points (BaseGate / BaseNormalizer)
and write structured notes to the provenance chain.

Install dependencies:
  pip install "curatorkit[hygiene]"
"""

from curatorkit.hygiene.guard import GuardModelGate
from curatorkit.hygiene.pii import PIIPseudonymizer
from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.hygiene.toxicity import ToxicityGate

__all__ = [
    "ToxicityGate",
    "GuardModelGate",
    "SecretsGate",
    "PIIPseudonymizer",
]
