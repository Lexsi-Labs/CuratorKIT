"""
Regression tests for YAML/CLI per-role LLM override parity with CuratorConfig.

Before this change, the YAML/CLI pipeline surface (curatorkit/config.py +
curatorkit/cli.py) had only one global `llm:` bucket and bare `*_llm_model`
model-string overrides per role — no judging/generation bucket split, no
per-role temperature/max_tokens/concurrency, and `grpo_scoring`/`refiner`
weren't exposed at all. These tests cover the new `LLMOverrideConfig` nested
blocks, the `generator_llm:`/`judge_llm:` mid-tier buckets, the 3-tier
cascade shared with the Python API's `_ROLE_DEFAULT_LLM_PARAMS`, backward
compatibility with the legacy bare-string fields, and the two newly-exposed
features (`grpo_scoring_llm`, `enable_reward_refiner`/`refiner_llm`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install curatorkit[generation]")

from curatorkit.cli import _build_steps
from curatorkit.config import (
    DiagnosticConfig,
    GateConfig,
    GenerationConfig,
    LLMConfig,
    PipelineConfig,
    ReaderConfig,
)


def _base_config(**overrides) -> PipelineConfig:
    defaults = {
        "readers": [ReaderConfig(type="jsonl", path="dummy.jsonl")],
        "llm": LLMConfig(model="openai/gpt-4o-mini"),
    }
    return PipelineConfig(**{**defaults, **overrides})


class TestLegacyBackwardCompat:
    def test_bare_model_string_still_works_for_hallucination(self):
        cfg = _base_config(
            gates=[GateConfig(type="hallucination", hallucination_threshold=0.7, hallucination_llm_model="openai/gpt-4o")]
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.gates.hallucination import HallucinationGate

        gate = next(s for s in steps if isinstance(s, HallucinationGate))
        assert gate.llm.model == "openai/gpt-4o"

    def test_bare_model_string_still_works_for_generator(self):
        cfg = _base_config(generators=[GenerationConfig(type="qa", llm_model="openai/gpt-4o")])
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.generators.qa_generator import QAGenerationTask

        gen = next(s for s in steps if isinstance(s, QAGenerationTask))
        assert gen.llm.model == "openai/gpt-4o"


class TestNestedOverrideBlocks:
    def test_hallucination_llm_nested_block_sets_sampling_params(self):
        cfg = _base_config(
            gates=[
                GateConfig(
                    type="hallucination",
                    hallucination_threshold=0.7,
                    hallucination_llm={"model": "openai/gpt-4o-mini", "temperature": 0.0, "max_tokens": 300},
                )
            ]
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.gates.hallucination import HallucinationGate

        gate = next(s for s in steps if isinstance(s, HallucinationGate))
        assert gate.llm.temperature == 0.0
        assert gate.llm.max_tokens == 300

    def test_hallucination_default_matches_historical_literal(self):
        cfg = _base_config(gates=[GateConfig(type="hallucination", hallucination_threshold=0.7)])
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.gates.hallucination import HallucinationGate

        gate = next(s for s in steps if isinstance(s, HallucinationGate))
        assert gate.llm.temperature == 0.1
        assert gate.llm.max_tokens == 512

    def test_toxicity_default_matches_historical_literal(self):
        cfg = _base_config(
            gates=[GateConfig(type="toxicity", toxicity_llm={"model": "openai/gpt-4o-mini"})]
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.hygiene.toxicity import ToxicityGate

        gate = next(s for s in steps if isinstance(s, ToxicityGate))
        assert gate.llm.temperature == 0.1
        assert gate.llm.max_tokens == 200


class TestCascade:
    def test_judge_llm_mid_tier_beats_global_bucket(self):
        cfg = _base_config(
            judge_llm={"temperature": 0.55},
            gates=[GateConfig(type="reward", reward_threshold=0.7)],
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.gates.reward import RewardGate

        gate = next(s for s in steps if isinstance(s, RewardGate))
        assert gate.llm.temperature == 0.55

    def test_role_override_beats_judge_llm_mid_tier(self):
        cfg = _base_config(
            judge_llm={"temperature": 0.55},
            gates=[GateConfig(type="reward", reward_threshold=0.7, reward_llm={"temperature": 0.11})],
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.gates.reward import RewardGate

        gate = next(s for s in steps if isinstance(s, RewardGate))
        assert gate.llm.temperature == 0.11

    def test_generator_llm_mid_tier_beats_global_bucket(self):
        cfg = _base_config(
            generator_llm={"temperature": 0.25},
            generators=[GenerationConfig(type="qa")],
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.generators.qa_generator import QAGenerationTask

        gen = next(s for s in steps if isinstance(s, QAGenerationTask))
        assert gen.llm.temperature == 0.25


class TestConcurrencyWiring:
    def test_hallucination_gate_now_receives_concurrency(self):
        """Previously HallucinationGate() in cli.py was never passed a concurrency=
        argument at all — it silently used the class default (16)."""
        cfg = _base_config(
            gates=[
                GateConfig(
                    type="hallucination",
                    hallucination_threshold=0.7,
                    hallucination_llm={"concurrency": 3},
                )
            ]
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.gates.hallucination import HallucinationGate

        gate = next(s for s in steps if isinstance(s, HallucinationGate))
        assert gate.concurrency == 3

    def test_diagnostic_probe_receives_concurrency(self):
        cfg = _base_config(
            gates=[GateConfig(type="hallucination", hallucination_threshold=0.7)],
            diagnostic=DiagnosticConfig(enable_probe=True, probe_llm={"concurrency": 9}),
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.gates.hallucination import HallucinationGate

        gate = next(s for s in steps if isinstance(s, HallucinationGate))
        assert gate.probe is not None
        assert gate.probe.concurrency == 9


class TestNewlyExposedFeatures:
    def test_grpo_scoring_llm_now_exposed(self):
        """CuratorConfig has always had grpo_scoring_llm; the YAML/CLI path had no
        equivalent at all — the rollout LLM was reused for scoring too."""
        cfg = _base_config(
            generators=[
                GenerationConfig(
                    type="grpo",
                    llm_model="openai/gpt-4o-mini",
                    grpo_scoring_llm={"model": "openai/gpt-4o", "temperature": 0.05},
                )
            ]
        )
        steps, _ = _build_steps(cfg, verbose=False)
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        gen = next(s for s in steps if isinstance(s, GRPORolloutTask))
        assert gen.scoring_llm.model == "openai/gpt-4o"
        assert gen.scoring_llm.temperature == 0.05
        assert gen.scoring_llm is not gen.llm

    def test_enable_reward_refiner_now_exposed(self):
        """RewardRefiner had no YAML/CLI wiring at all before this change."""
        cfg = _base_config(
            gates=[
                GateConfig(
                    type="reward",
                    reward_threshold=0.7,
                    enable_reward_refiner=True,
                    refiner_llm={"temperature": 0.9, "concurrency": 5},
                )
            ]
        )
        steps, reward_refiner = _build_steps(cfg, verbose=False)
        from curatorkit.diagnostic.reward_refine import RewardRefiner

        assert isinstance(reward_refiner, RewardRefiner)
        assert reward_refiner.generator_llm.temperature == 0.9
        assert reward_refiner.concurrency == 5

    def test_no_refiner_when_flag_unset(self):
        cfg = _base_config(gates=[GateConfig(type="reward", reward_threshold=0.7)])
        _, reward_refiner = _build_steps(cfg, verbose=False)
        assert reward_refiner is None
