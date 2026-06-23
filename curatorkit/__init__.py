"""CuratorKIT — data curation and synthesis for LLM post-training."""

__version__ = "0.1.0"

from curatorkit.connectors.csv_reader import CSVReader
from curatorkit.connectors.huggingface import HuggingFaceReader
from curatorkit.connectors.json_reader import JSONReader
from curatorkit.connectors.jsonl import JSONLReader
from curatorkit.connectors.parquet_reader import ParquetReader
from curatorkit.curator import Curator, CuratorConfig, CuratorResult
from curatorkit.schema import DataSample, RejectedSample

__all__ = [
    "Curator",
    "CuratorConfig",
    "CuratorResult",
    "DataSample",
    "RejectedSample",
    "JSONLReader",
    "JSONReader",
    "CSVReader",
    "ParquetReader",
    "HuggingFaceReader",
]


# Optional imports — available only when extras are installed
def __getattr__(name: str):
    """Lazy imports for optional components."""
    _lazy_map = {
        "BaseLLM": "curatorkit.llm.base",
        "LLMResponse": "curatorkit.llm.base",
        "LiteLLMBackend": "curatorkit.llm.litellm",
        "OllamaBackend": "curatorkit.llm.ollama",
        "BaseGenerationTask": "curatorkit.generators.base",
        "QAGenerationTask": "curatorkit.generators.qa_generator",
        "EvolInstructTask": "curatorkit.generators.evol_instruct",
        "PreferenceGenerationTask": "curatorkit.generators.preference_gen",
        "GRPORolloutTask": "curatorkit.generators.grpo_rollout",
        "MultiTurnTask": "curatorkit.generators.multiturn_gen",
        "ChainOfThoughtTask": "curatorkit.generators.cot_generator",
        "HallucinationGate": "curatorkit.gates.hallucination",
        "RewardGate": "curatorkit.gates.reward",
        "DiversityGate": "curatorkit.gates.diversity",
        "EmbeddingDeduplicator": "curatorkit.normalizers.embedding_dedup",
    }
    if name in _lazy_map:
        import importlib

        module = importlib.import_module(_lazy_map[name])
        return getattr(module, name)

    _diagnostic_names = {
        "DiagnosticProbe",
        "PipelineDiagnostics",
        "FailureDiagnosis",
        "FailureMode",
    }
    if name in _diagnostic_names:
        try:
            from curatorkit import diagnostic

            return getattr(diagnostic, name)
        except ImportError as e:
            raise ImportError(
                "CuratorKIT diagnostic loop requires [generation] extras. "
                "Install with: pip install 'curatorkit[generation]'"
            ) from e

    _hygiene_map = {
        "ToxicityGate": "curatorkit.hygiene.toxicity",
        "SecretsGate": "curatorkit.hygiene.secrets",
        "PIIPseudonymizer": "curatorkit.hygiene.pii",
    }
    if name in _hygiene_map:
        try:
            import importlib

            module = importlib.import_module(_hygiene_map[name])
            return getattr(module, name)
        except ImportError as e:
            raise ImportError(
                "CuratorKIT hygiene components require [hygiene] extras. "
                "Install with: pip install 'curatorkit[hygiene]'"
            ) from e

    raise AttributeError(f"module 'curatorkit' has no attribute {name!r}")
