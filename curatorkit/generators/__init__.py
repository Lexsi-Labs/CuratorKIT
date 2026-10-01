"""LLM-powered generation tasks: QA pairs, preference pairs, GRPO rollouts,
multi-turn dialogues, chain-of-thought traces, adversarial variants, and
Magpie / Persona Hub / Self-Rewarding / Constitutional AI / Best-of-N /
joint SFT+DPO+GRPO bundles."""

from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask
from curatorkit.generators.adversarial_qa_generator import AdversarialQAGenerationTask
from curatorkit.generators.base import BaseGenerationTask
from curatorkit.generators.best_of_n import BestOfNTask
from curatorkit.generators.constitutional import (
    ConstitutionalPreferenceTask,
    ConstitutionalRevisionTask,
)
from curatorkit.generators.cot_generator import ChainOfThoughtTask
from curatorkit.generators.evol_instruct import EvolInstructTask
from curatorkit.generators.grpo_rollout import GRPORolloutTask
from curatorkit.generators.injector import BadSampleInjector
from curatorkit.generators.joint_bundle import JointBundleTask
from curatorkit.generators.magpie import MagpieTask
from curatorkit.generators.multiturn_gen import MultiTurnTask
from curatorkit.generators.persona import PersonaDrivenTask
from curatorkit.generators.preference_gen import PreferenceGenerationTask
from curatorkit.generators.qa_generator import QAGenerationTask
from curatorkit.generators.self_rewarding import SelfRewardingTask

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
    "MagpieTask",
    "PersonaDrivenTask",
    "JointBundleTask",
    "SelfRewardingTask",
    "ConstitutionalRevisionTask",
    "ConstitutionalPreferenceTask",
    "BestOfNTask",
]
