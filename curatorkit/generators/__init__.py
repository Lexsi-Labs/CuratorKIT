"""LLM-powered generation tasks: QA pairs, preference pairs, GRPO rollouts,
multi-turn dialogues, chain-of-thought traces, and adversarial variants."""

from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask
from curatorkit.generators.adversarial_qa_generator import AdversarialQAGenerationTask
from curatorkit.generators.base import BaseGenerationTask
from curatorkit.generators.cot_generator import ChainOfThoughtTask
from curatorkit.generators.evol_instruct import EvolInstructTask
from curatorkit.generators.grpo_rollout import GRPORolloutTask
from curatorkit.generators.injector import BadSampleInjector
from curatorkit.generators.multiturn_gen import MultiTurnTask
from curatorkit.generators.preference_gen import PreferenceGenerationTask
from curatorkit.generators.qa_generator import QAGenerationTask

__all__ = [
    "BaseGenerationTask",
    "QAGenerationTask",
    "EvolInstructTask",
    "PreferenceGenerationTask",
    "GRPORolloutTask",
    "MultiTurnTask",
    "ChainOfThoughtTask",
    "AdversarialQAGenerationTask",
    "AdversarialPreferenceTask",
    "BadSampleInjector",
]
