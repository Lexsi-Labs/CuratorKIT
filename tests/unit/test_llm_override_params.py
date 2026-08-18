"""
Regression tests for per-role LLM sampling-param overrides.

Historically, LLMOverride only carried model/api_base/api_key — sampling
params (temperature, max_tokens, timeout, max_retries, extra_body,
drop_params, concurrency) were configurable only for generator/judge via flat
CuratorConfig fields, and the other six roles (hallucination, reward,
toxicity, grpo_scoring, probe, refiner) hardcoded literal temperature/
max_tokens at their .generate()/.agenerate() call sites — so even editing
e.g. toxicity_llm={"temperature": ...} had no effect. These tests cover the
new 3-tier cascade (task/role override -> mid-tier bucket -> role default or
global bucket) added to close that gap, plus the default-preservation
regression guard (no override set -> resolves to the exact pre-existing
hardcoded literal).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install curatorkit[generation]")

from curatorkit.curator import _ROLE_DEFAULT_LLM_PARAMS, Curator, CuratorConfig, LLMOverride


def _config(**overrides) -> CuratorConfig:
    defaults = {"dataset": "dummy.jsonl", "llm_model": "openai/gpt-4o-mini"}
    return CuratorConfig(**{**defaults, **overrides})


JUDGING_ROLES = ["hallucination", "reward", "toxicity", "grpo_scoring"]
_BUILDER_METHOD = {
    "hallucination": "_build_hallucination_backend",
    "reward": "_build_reward_backend",
    "toxicity": "_build_toxicity_backend",
    "grpo_scoring": "_build_grpo_scoring_backend",
}


class TestLLMOverrideBackwardCompat:
    def test_model_only_dataclass_construction_still_works(self):
        ov = LLMOverride(model="openai/gpt-4o")
        assert ov.model == "openai/gpt-4o"
        assert ov.temperature is None
        assert ov.concurrency is None

    def test_dict_coercion_accepts_new_fields(self):
        cfg = _config(
            hallucination_llm={"model": "openai/gpt-4o-mini", "temperature": 0.3, "max_tokens": 999}
        )
        assert isinstance(cfg.hallucination_llm, LLMOverride)
        assert cfg.hallucination_llm.temperature == 0.3
        assert cfg.hallucination_llm.max_tokens == 999


class TestJudgingRoleDefaults:
    """No override set -> backend matches the pre-existing hardcoded literal."""

    @pytest.mark.parametrize("role", JUDGING_ROLES)
    def test_default_matches_historical_literal(self, role):
        cfg = _config()
        curator = Curator(cfg)
        builder = getattr(curator, _BUILDER_METHOD[role])
        backend = builder(getattr(cfg, f"{role}_llm"))
        defaults = _ROLE_DEFAULT_LLM_PARAMS[role]
        assert backend.temperature == defaults["temperature"]
        assert backend.max_tokens == defaults["max_tokens"]


class TestJudgingRoleOverrides:
    @pytest.mark.parametrize("role", JUDGING_ROLES)
    def test_role_override_takes_effect(self, role):
        cfg = _config(**{f"{role}_llm": {"model": "openai/gpt-4o-mini", "temperature": 0.33, "max_tokens": 777}})
        curator = Curator(cfg)
        builder = getattr(curator, _BUILDER_METHOD[role])
        backend = builder(getattr(cfg, f"{role}_llm"))
        assert backend.temperature == 0.33
        assert backend.max_tokens == 777

    @pytest.mark.parametrize("role", JUDGING_ROLES)
    def test_cascade_role_beats_judge_llm_beats_role_default(self, role):
        cfg = _config(judge_llm={"temperature": 0.55})
        curator = Curator(cfg)
        builder = getattr(curator, _BUILDER_METHOD[role])
        backend = builder(getattr(cfg, f"{role}_llm"))
        # judge_llm mid-tier beats the role-specific historical default
        assert backend.temperature == 0.55

        cfg2 = _config(
            judge_llm={"temperature": 0.55},
            **{f"{role}_llm": {"temperature": 0.11}},
        )
        curator2 = Curator(cfg2)
        builder2 = getattr(curator2, _BUILDER_METHOD[role])
        backend2 = builder2(getattr(cfg2, f"{role}_llm"))
        # role-specific override beats judge_llm mid-tier
        assert backend2.temperature == 0.11


class TestGeneratorJudgeOverrides:
    def test_generator_llm_override_now_respected(self):
        cfg = _config(generator_llm={"temperature": 0.25, "max_tokens": 333})
        backend = Curator(cfg)._build_gen_llm(LLMOverride())
        assert backend.temperature == 0.25
        assert backend.max_tokens == 333

    def test_generator_llm_unset_falls_back_to_global_bucket(self):
        cfg = _config(llm_temperature=0.42, llm_max_tokens=111)
        backend = Curator(cfg)._build_gen_llm(LLMOverride())
        assert backend.temperature == 0.42
        assert backend.max_tokens == 111

    def test_judge_llm_override_now_respected(self):
        cfg = _config(judge_llm={"temperature": 0.05, "max_tokens": 42})
        backend = Curator(cfg)._build_judge_backend(LLMOverride())
        assert backend.temperature == 0.05
        assert backend.max_tokens == 42


class TestProbeRefinerDefaults:
    def test_probe_default_max_tokens_unified(self):
        cfg = _config()
        backend = Curator(cfg)._build_probe_backend()
        assert backend.max_tokens == _ROLE_DEFAULT_LLM_PARAMS["probe"]["max_tokens"]

    def test_refiner_default_matches_historical_literal(self):
        cfg = _config()
        backend = Curator(cfg)._build_refiner_backend()
        defaults = _ROLE_DEFAULT_LLM_PARAMS["refiner"]
        assert backend.temperature == defaults["temperature"]
        assert backend.max_tokens == defaults["max_tokens"]

    def test_probe_override_takes_effect(self):
        cfg = _config(probe_llm={"max_tokens": 64})
        backend = Curator(cfg)._build_probe_backend()
        assert backend.max_tokens == 64

    def test_refiner_override_takes_effect(self):
        cfg = _config(refiner_llm={"temperature": 0.9, "max_tokens": 64})
        backend = Curator(cfg)._build_refiner_backend()
        assert backend.temperature == 0.9
        assert backend.max_tokens == 64


class TestEnableThinkingForceOverride:
    def test_default_forces_thinking_off(self):
        cfg = _config()
        backend = Curator(cfg)._build_probe_backend()
        assert backend.extra_body["chat_template_kwargs"]["enable_thinking"] is False

    def test_explicit_override_is_respected(self):
        cfg = _config(
            probe_llm={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
        )
        backend = Curator(cfg)._build_probe_backend()
        assert backend.extra_body["chat_template_kwargs"]["enable_thinking"] is True

    def test_refiner_default_forces_thinking_off(self):
        cfg = _config()
        backend = Curator(cfg)._build_refiner_backend()
        assert backend.extra_body["chat_template_kwargs"]["enable_thinking"] is False


class TestConcurrencyOverride:
    def test_toxicity_backend_has_no_concurrency_knob_by_design(self):
        # ToxicityGate.run() is a sequential loop with no executor — there is
        # nothing to configure here yet; this documents that limitation.
        cfg = _config()
        curator = Curator(cfg)
        assert curator._resolve_role_concurrency(cfg.toxicity_llm, cfg.judge_llm, 99) == 99

    def test_probe_concurrency_default_is_32(self):
        cfg = _config()
        curator = Curator(cfg)
        assert curator._resolve_role_concurrency(cfg.probe_llm, cfg.generator_llm, 32) == 32

    def test_probe_concurrency_role_override_wins(self):
        cfg = _config(probe_llm={"concurrency": 4})
        curator = Curator(cfg)
        assert curator._resolve_role_concurrency(cfg.probe_llm, cfg.generator_llm, 32) == 4

    def test_probe_concurrency_falls_back_to_generator_llm_mid_tier(self):
        cfg = _config(generator_llm={"concurrency": 7})
        curator = Curator(cfg)
        assert curator._resolve_role_concurrency(cfg.probe_llm, cfg.generator_llm, 32) == 7

    def test_hallucination_gate_receives_resolved_concurrency(self):
        cfg = _config(
            hallucination_threshold=0.7,
            hallucination_llm={"concurrency": 3},
        )
        curator = Curator(cfg)
        steps = curator._build_steps()
        from curatorkit.gates.hallucination import HallucinationGate

        gate = next(s for s in steps if isinstance(s, HallucinationGate))
        assert gate.concurrency == 3

    def test_diagnostic_probe_receives_resolved_concurrency(self):
        cfg = _config(
            hallucination_threshold=0.7,
            enable_diagnostic_probe=True,
            probe_llm={"concurrency": 9},
        )
        curator = Curator(cfg)
        steps = curator._build_steps()
        from curatorkit.gates.hallucination import HallucinationGate

        gate = next(s for s in steps if isinstance(s, HallucinationGate))
        assert gate.probe is not None
        assert gate.probe.concurrency == 9

    def test_refiner_receives_resolved_concurrency(self):
        cfg = _config(
            reward_threshold=0.7,
            enable_reward_refiner=True,
            refiner_llm={"concurrency": 5},
        )
        curator = Curator(cfg)
        curator._build_steps()
        assert curator._reward_refiner is not None
        assert curator._reward_refiner.concurrency == 5


def _sentinel_llm(**overrides):
    llm = MagicMock()
    llm.temperature = 0.9191
    llm.max_tokens = 9191
    for k, v in overrides.items():
        setattr(llm, k, v)
    mock_response = MagicMock()
    mock_response.text = '{"grounding_score": 1.0, "verdict": "supported", "overall_score": 1.0}'
    llm.generate.return_value = mock_response
    return llm


class TestCallSitesReadFromBackendNotLiterals:
    """Each of these call sites used to hardcode temperature=/max_tokens= directly
    in the .generate() call, ignoring whatever the backend was configured with."""

    def test_hallucination_gate_evaluate_grounding(self):
        from curatorkit.gates.hallucination import HallucinationGate

        gate = HallucinationGate(llm=_sentinel_llm())
        gate._evaluate_grounding("source", "question", "answer")
        _, kwargs = gate.llm.generate.call_args
        assert kwargs["temperature"] == 0.9191
        assert kwargs["max_tokens"] == 9191

    def test_reward_gate_evaluate_quality(self):
        from curatorkit.gates.reward import RewardGate

        gate = RewardGate(llm=_sentinel_llm())
        gate._evaluate_quality("instruction", "response")
        _, kwargs = gate.llm.generate.call_args
        assert kwargs["temperature"] == 0.9191
        assert kwargs["max_tokens"] == 9191

    def test_grpo_scoring_llm_score_single(self):
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        scoring_llm = _sentinel_llm()
        scoring_llm.generate.return_value.text = '{"score": 0.8, "reasoning": "ok"}'
        task = GRPORolloutTask(llm=MagicMock(), scoring_llm=scoring_llm)
        task._score_single("instruction", "response")
        _, kwargs = scoring_llm.generate.call_args
        assert kwargs["temperature"] == 0.9191
        assert kwargs["max_tokens"] == 9191

    def test_probe_regenerate_max_tokens(self):
        from curatorkit.diagnostic.probe import DiagnosticProbe
        from curatorkit.schema import RejectedSample

        generator_llm = _sentinel_llm()
        generator_llm.generate.return_value.text = "regenerated answer"
        probe = DiagnosticProbe(generator_llm=generator_llm, gate=MagicMock())
        original = RejectedSample(
            source_uri="test", instruction="q", rejection_reason="x", rejecting_step="x"
        )
        probe._regenerate(original, source_context="some source", temperature=0.42)
        _, kwargs = generator_llm.generate.call_args
        assert kwargs["temperature"] == 0.42  # temperature stays parametrized by caller
        assert kwargs["max_tokens"] == 9191  # max_tokens now read from the backend

    def test_probe_regenerate_instruction_max_tokens(self):
        from curatorkit.diagnostic.probe import DiagnosticProbe
        from curatorkit.schema import RejectedSample

        generator_llm = _sentinel_llm()
        generator_llm.generate.return_value.text = "regenerated question?"
        probe = DiagnosticProbe(generator_llm=generator_llm, gate=MagicMock())
        original = RejectedSample(
            source_uri="test", instruction="q", rejection_reason="x", rejecting_step="x"
        )
        probe._regenerate_instruction(original, source_context="some source")
        _, kwargs = generator_llm.generate.call_args
        assert kwargs["max_tokens"] == 9191

    def test_refiner_refine_answer(self):
        from curatorkit.diagnostic.reward_refine import RewardRefiner
        from curatorkit.schema import RejectedSample

        generator_llm = _sentinel_llm()
        generator_llm.generate.return_value.text = "refined answer"
        refiner = RewardRefiner(generator_llm=generator_llm, reward_gate=MagicMock())
        sample = RejectedSample(
            source_uri="test",
            instruction="q",
            output="a",
            rejection_reason="x",
            rejecting_step="x",
        )
        refiner._refine_answer(sample, axis="helpfulness", weakness="too short")
        _, kwargs = generator_llm.generate.call_args
        assert kwargs["temperature"] == 0.9191
        assert kwargs["max_tokens"] == 9191

    def test_refiner_refine_instruction(self):
        from curatorkit.diagnostic.reward_refine import RewardRefiner
        from curatorkit.schema import RejectedSample

        generator_llm = _sentinel_llm()
        generator_llm.generate.return_value.text = "refined instruction?"
        refiner = RewardRefiner(generator_llm=generator_llm, reward_gate=MagicMock())
        sample = RejectedSample(
            source_uri="test",
            instruction="q",
            output="a",
            rejection_reason="x",
            rejecting_step="x",
        )
        refiner._refine_instruction(sample, weakness="unclear")
        _, kwargs = generator_llm.generate.call_args
        assert kwargs["temperature"] == 0.9191
        assert kwargs["max_tokens"] == 9191
