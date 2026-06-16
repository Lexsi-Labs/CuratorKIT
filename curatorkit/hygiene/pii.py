"""
PIIPseudonymizer — replace PII entities with consistent fake values.

Implements the clinical NLP de-identification approach (i2b2, MIMIC-III):
pseudonymize, do not mask. Replacing "John Smith" with "[PERSON]" corrupts
chunk structure and degrades QA generation quality. Replacing it with
"Alice Johnson" (a consistent, realistic fake name) preserves the document's
semantic coherence for all downstream generation and verification tasks.

Entity map is per-sample:
  - Same original string → same fake value within one DataSample (coreference
    preserved across instruction / input / output).
  - Independent map per sample — no cross-sample linkage risk.

Detection: Microsoft Presidio (spaCy NER + regex recognizers).
Replacement: Faker library (realistic, syntactically valid values per type).

Runs as a BaseNormalizer during ingestion/cleaning, before any LLM calls. HallucinationGate
verification remains consistent because the source chunk and the generated
answer both reference the same pseudonymized entities.

Default entity types (conservative — no DATE_TIME to avoid over-redacting
contract metadata):
  PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, IP_ADDRESS,
  US_BANK_NUMBER, IBAN_CODE

Opt-in clinical preset adds: DATE_TIME, MEDICAL_LICENSE, NRP, LOCATION.

spaCy model: en_core_web_lg by default (Presidio default, ~800 MB).
Use spacy_model="en_core_web_sm" for dev/CI environments (~12 MB).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from curatorkit.interfaces import BaseNormalizer
from curatorkit.schema import DataSample, ProvenanceRecord

STEP_VERSION = "1.0.0"

ENTITY_TYPES_DEFAULT: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "US_BANK_NUMBER",
    "IBAN_CODE",
]

ENTITY_TYPES_CLINICAL: list[str] = ENTITY_TYPES_DEFAULT + [
    "DATE_TIME",
    "MEDICAL_LICENSE",
    "NRP",
    "LOCATION",
]


def _ensure_presidio() -> tuple[Any, Any]:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        return AnalyzerEngine, AnonymizerEngine
    except ImportError as e:
        raise ImportError(
            "presidio-analyzer and presidio-anonymizer are not installed. "
            "Install with: pip install 'curatorkit[hygiene]'"
        ) from e


def _ensure_faker() -> Any:
    try:
        from faker import Faker

        return Faker
    except ImportError as e:
        raise ImportError(
            "faker is not installed. Install with: pip install 'curatorkit[hygiene]'"
        ) from e


def _make_faker(seed: int) -> Any:
    Faker = _ensure_faker()
    f = Faker()
    Faker.seed(seed)
    return f


def _fake_for_type(entity_type: str, faker: Any) -> str:
    """Generate a realistic replacement value for the given Presidio entity type."""
    dispatch: dict[str, Any] = {
        "PERSON": faker.name,
        "EMAIL_ADDRESS": faker.email,
        "PHONE_NUMBER": faker.phone_number,
        "US_SSN": faker.ssn,
        "CREDIT_CARD": lambda: faker.credit_card_number(card_type=None),
        "IP_ADDRESS": faker.ipv4_private,
        "URL": faker.url,
        "US_BANK_NUMBER": lambda: str(faker.random_number(digits=10, fix_len=True)),
        "IBAN_CODE": faker.iban,
        "DATE_TIME": lambda: faker.date_this_decade().isoformat(),
        "MEDICAL_LICENSE": lambda: f"ML-{faker.random_number(digits=7, fix_len=True)}",
        "NRP": lambda: faker.numerify("###-##-####"),
        "LOCATION": faker.city,
    }
    fn = dispatch.get(entity_type)
    if fn is not None:
        return fn()
    return f"[{entity_type}]"


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


def _build_analyzer(spacy_model: str) -> Any:
    AnalyzerEngine, _ = _ensure_presidio()
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine)


class PIIPseudonymizer(BaseNormalizer):
    """
    Replace PII entities with consistent fake values (per-sample scope).

    Task-aware: selects fields based on sample.task_type so preference pairs
    (chosen/rejected) and GRPO rollouts (responses) are always pseudonymized
    when relevant.

    Parameters
    ----------
    entity_types : list[str] | None
        Presidio entity types to detect and replace. Use ENTITY_TYPES_CLINICAL
        for clinical corpora that need DATE_TIME and location replacement.
    fields : list[str] | None
        DataSample fields to process. Defaults to all task-relevant fields.
        Explicitly setting this overrides the task-aware selection entirely.
    score_threshold : float
        Presidio confidence threshold. Lower = more aggressive detection.
    faker_seed : int
        Seed for Faker (reproducible replacements across runs with same seed).
    language : str
        Presidio analysis language.
    spacy_model : str
        spaCy model name. "en_core_web_lg" (default, ~800 MB, highest accuracy)
        or "en_core_web_sm" (~12 MB, adequate for standard PII types).
    """

    def __init__(
        self,
        entity_types: list[str] | None = None,
        fields: list[str] | None = None,
        score_threshold: float = 0.7,
        faker_seed: int = 42,
        language: str = "en",
        spacy_model: str = "en_core_web_lg",
    ) -> None:
        self.entity_types = entity_types or ENTITY_TYPES_DEFAULT
        self._fields_override = fields  # None = use task-aware selection
        self.fields = fields or [
            "instruction",
            "input",
            "output",
            "chosen",
            "rejected",  # DPO preference pairs
            "responses",  # GRPO rollouts (list field — handled below)
        ]
        self.score_threshold = score_threshold
        self.faker_seed = faker_seed
        self.language = language
        self.spacy_model = spacy_model
        self._analyzer: Any = None
        self._faker: Any = None

    def _fields_for_sample(self, sample: DataSample) -> list[str]:
        """Return fields to pseudonymize, selected by task_type when no explicit override."""
        if self._fields_override is not None:
            return self._fields_override
        tt = getattr(sample, "task_type", None) or ""
        candidates = _TASK_FIELDS.get(tt, self.fields)
        return [f for f in candidates if f in self.fields]

    def _load(self) -> tuple[Any, Any]:
        if self._analyzer is None:
            self._analyzer = _build_analyzer(self.spacy_model)
        if self._faker is None:
            self._faker = _make_faker(self.faker_seed)
        return self._analyzer, self._faker

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "entity_types": sorted(self.entity_types),
                "fields": sorted(self.fields),
                "score_threshold": self.score_threshold,
                "language": self.language,
                "spacy_model": self.spacy_model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _pseudonymize(
        self,
        text: str,
        entity_map: dict[str, str],
        analyzer: Any,
        faker: Any,
    ) -> tuple[str, dict[str, int]]:
        """
        Replace PII in text using entity_map for consistency.

        Processes Presidio results in reverse position order so that earlier
        replacements (different length) don't shift offsets for later ones.

        Returns (pseudonymized_text, entity_type_counts).
        entity_map is mutated in-place for cross-field consistency within sample.
        """
        if not text.strip():
            return text, {}

        results = analyzer.analyze(
            text=text,
            entities=self.entity_types,
            language=self.language,
            score_threshold=self.score_threshold,
        )
        if not results:
            return text, {}

        results_desc = sorted(results, key=lambda r: r.start, reverse=True)
        chars = list(text)
        type_counts: dict[str, int] = {}

        for result in results_desc:
            original = text[result.start : result.end]
            cache_key = f"{result.entity_type}:{original}"
            if cache_key not in entity_map:
                entity_map[cache_key] = _fake_for_type(result.entity_type, faker)
            replacement = entity_map[cache_key]
            chars[result.start : result.end] = list(replacement)
            type_counts[result.entity_type] = type_counts.get(result.entity_type, 0) + 1

        return "".join(chars), type_counts

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)

        from tqdm import tqdm

        for sample in tqdm(samples, desc="PIIPseudonymizer", unit="sample"):
            analyzer, faker = self._load()
            entity_map: dict[str, str] = {}  # per-sample scope
            total_counts: dict[str, int] = {}
            active_fields = self._fields_for_sample(sample)

            for field in active_fields:
                val = getattr(sample, field, None)
                if val is None:
                    continue
                if isinstance(val, list):
                    # GRPO responses — pseudonymize each completion,
                    # maintain entity_map so the same entity gets the same
                    # fake value across all responses in this sample.
                    new_list = []
                    for item in val:
                        if isinstance(item, str) and item:
                            pseudonymized, counts = self._pseudonymize(
                                item, entity_map, analyzer, faker
                            )
                            for etype, count in counts.items():
                                total_counts[etype] = total_counts.get(etype, 0) + count
                            new_list.append(pseudonymized)
                        else:
                            new_list.append(item)
                    setattr(sample, field, new_list)
                elif isinstance(val, str) and val:
                    pseudonymized, counts = self._pseudonymize(val, entity_map, analyzer, faker)
                    setattr(sample, field, pseudonymized)
                    for etype, count in counts.items():
                        total_counts[etype] = total_counts.get(etype, 0) + count

            sample.append_provenance(
                ProvenanceRecord(
                    step_name="PIIPseudonymizer",
                    step_version=STEP_VERSION,
                    timestamp=ts,
                    config_hash=cfg_hash,
                    notes={
                        "entities_replaced": total_counts,
                        "fields_processed": active_fields,
                        "total_replacements": sum(total_counts.values()),
                    },
                )
            )

        return samples
