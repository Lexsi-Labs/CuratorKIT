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

NER backend (nlp_engine):
  "spacy" (default) — spacy_model, e.g. "en_core_web_lg" (~800 MB, Presidio
      default) or "en_core_web_sm" (~12 MB, dev/CI). Fast; weaker recall on
      domain-specific names (clinical, legal) than the options below.
      Requires: curatorkit[hygiene].
  "transformers" — transformer_model, an entity-tagging model from the Hub
      (see RECOMMENDED_NER_MODELS below), paired with spacy_model as a
      lightweight tokenizer/sentence-splitter only (NER itself comes from
      the transformer). Best recall for clinical/legal PII; heavier and
      slower than spaCy. Requires: curatorkit[hygiene,hygiene-transformers].
  "stanza" — stanza_model, a language code (default "en"). Stanford's
      general-purpose multilingual pipeline; useful for languages spaCy
      doesn't cover well. Requires: curatorkit[hygiene,hygiene-stanza].

RECOMMENDED_NER_MODELS documents which transformer/stanza model to pick for
which domain — surface it to users as a dropdown rather than a free-text
field, since the exact HF model id is not something most users will know
to type correctly.
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

# Curated NER model catalog for the "transformers" and "stanza" engines.
# Intended to back a frontend dropdown — most users won't know a correct
# HF model id to type by hand. Keys are the model_name value to pass as
# `transformer_model` / `stanza_model`; values are (domain, note).
RECOMMENDED_NER_MODELS: dict[str, dict[str, tuple[str, str]]] = {
    "transformers": {
        "StanfordAIMI/stanford-deidentifier-base": (
            "clinical/medical",
            "Presidio's own default pick for the transformers engine. RoBERTa "
            "fine-tuned on i2b2 clinical notes for PHI de-identification "
            "(patient names, hospitals, dates, IDs). Best default for medical text.",
        ),
        "obi/deid_roberta_i2b2": (
            "clinical/medical",
            "Alternative i2b2-trained clinical de-identification model. Comparable "
            "to stanford-deidentifier-base; use as a second opinion / ensemble check "
            "on high-stakes medical corpora rather than a first choice.",
        ),
        "dslim/bert-base-NER": (
            "legal/general",
            "General-purpose NER (CoNLL-2003 labels: person/org/location/misc). "
            "No clinical PHI awareness, but solid recall on names/orgs/locations in "
            "contracts, filings, and other legal text with no medical content.",
        ),
        "Jean-Baptiste/roberta-large-ner-english": (
            "legal/general",
            "Higher-accuracy general NER than dslim/bert-base-NER, at roughly 3x "
            "the parameter count/latency. Use when dslim is missing entities on a "
            "legal corpus and the extra latency is acceptable.",
        ),
    },
    "stanza": {
        "en": (
            "multilingual/general",
            "Stanford's general-purpose NLP pipeline. Reasonable general-domain "
            "English NER, but no clinical/legal specialization — pick it when you "
            "need a non-English language spaCy doesn't have a good model for, not "
            "as an English clinical/legal upgrade over spaCy.",
        ),
    },
}

# Default HF label -> Presidio entity_type mapping for the transformers engine.
# Covers both clinical models above (PATIENT/STAFF/HOSP/... labels) and generic
# CoNLL-style models (PER/LOC/ORG/MISC) with one shared config, so all four
# RECOMMENDED_NER_MODELS entries work out of the box without per-model tuning.
# Copied from Presidio's own conf/transformers.yaml default.
_DEFAULT_TRANSFORMER_NER_CONFIG: dict[str, Any] = {
    "labels_to_ignore": ["O"],
    "aggregation_strategy": "max",
    "stride": 16,
    "alignment_mode": "expand",
    "model_to_presidio_entity_mapping": {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "GPE": "LOCATION",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "NORP": "NRP",
        "AGE": "AGE",
        "ID": "ID",
        "EMAIL": "EMAIL_ADDRESS",
        "PATIENT": "PERSON",
        "STAFF": "PERSON",
        "HOSP": "ORGANIZATION",
        "PATORG": "ORGANIZATION",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
        "PHONE": "PHONE_NUMBER",
        "HCW": "PERSON",
        "HOSPITAL": "LOCATION",
        "FACILITY": "LOCATION",
        "VENDOR": "ORGANIZATION",
    },
    "low_confidence_score_multiplier": 0.4,
    "low_score_entity_names": ["ID"],
}


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


def _ensure_engine_extra(nlp_engine: str) -> None:
    if nlp_engine == "transformers":
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'transformers' NER engine needs the transformers/torch stack. "
                "Install with: pip install 'curatorkit[hygiene,hygiene-transformers]'"
            ) from e
    elif nlp_engine == "stanza":
        try:
            import stanza  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'stanza' NER engine needs the stanza package. "
                "Install with: pip install 'curatorkit[hygiene,hygiene-stanza]'"
            ) from e


def _build_analyzer(
    nlp_engine: str,
    spacy_model: str,
    transformer_model: str | None,
    stanza_model: str,
    language: str,
    ner_model_configuration: dict | None,
) -> Any:
    AnalyzerEngine, _ = _ensure_presidio()
    _ensure_engine_extra(nlp_engine)
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    if nlp_engine == "spacy":
        configuration: dict[str, Any] = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": spacy_model}],
        }
    elif nlp_engine == "stanza":
        configuration = {
            "nlp_engine_name": "stanza",
            "models": [{"lang_code": language, "model_name": stanza_model}],
        }
    elif nlp_engine == "transformers":
        if not transformer_model:
            raise ValueError(
                "transformer_model is required when nlp_engine='transformers'. "
                "See curatorkit.hygiene.pii.RECOMMENDED_NER_MODELS for options."
            )
        configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": language,
                    "model_name": {"spacy": spacy_model, "transformers": transformer_model},
                }
            ],
            "ner_model_configuration": ner_model_configuration or _DEFAULT_TRANSFORMER_NER_CONFIG,
        }
    else:
        raise ValueError(
            f"Unknown nlp_engine: {nlp_engine!r}. Use 'spacy', 'stanza', or 'transformers'."
        )

    provider = NlpEngineProvider(nlp_configuration=configuration)
    engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=engine)


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
    nlp_engine : str
        NER backend: "spacy" (default), "transformers", or "stanza". See the
        module docstring and RECOMMENDED_NER_MODELS for when to use each.
    spacy_model : str
        spaCy model name. Used directly when nlp_engine="spacy" ("en_core_web_lg"
        default, ~800 MB, highest accuracy, or "en_core_web_sm", ~12 MB). When
        nlp_engine="transformers", this is only the tokenizer/sentence-splitter —
        NER itself comes from transformer_model, so "en_core_web_sm" is enough.
    transformer_model : str | None
        HuggingFace model id for nlp_engine="transformers", e.g.
        "StanfordAIMI/stanford-deidentifier-base". Required when
        nlp_engine="transformers". See RECOMMENDED_NER_MODELS.
    stanza_model : str
        Language code for nlp_engine="stanza" (default "en").
    ner_model_configuration : dict | None
        Advanced override for the transformers engine's label mapping /
        aggregation strategy. None = use the built-in default mapping, which
        covers all models in RECOMMENDED_NER_MODELS["transformers"].
    """

    def __init__(
        self,
        entity_types: list[str] | None = None,
        fields: list[str] | None = None,
        score_threshold: float = 0.7,
        faker_seed: int = 42,
        language: str = "en",
        nlp_engine: str = "spacy",
        spacy_model: str = "en_core_web_lg",
        transformer_model: str | None = None,
        stanza_model: str = "en",
        ner_model_configuration: dict | None = None,
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
        self.nlp_engine = nlp_engine
        self.spacy_model = spacy_model
        self.transformer_model = transformer_model
        self.stanza_model = stanza_model
        self.ner_model_configuration = ner_model_configuration
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
            self._analyzer = _build_analyzer(
                self.nlp_engine,
                self.spacy_model,
                self.transformer_model,
                self.stanza_model,
                self.language,
                self.ner_model_configuration,
            )
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
                "nlp_engine": self.nlp_engine,
                "spacy_model": self.spacy_model,
                "transformer_model": self.transformer_model,
                "stanza_model": self.stanza_model,
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
