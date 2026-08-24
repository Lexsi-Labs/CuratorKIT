"""
Deterministic, no-LLM regression tests for AdversarialQAGenerationTask and
AdversarialPreferenceTask.

These check the injection-plan sampling logic (rate -> count, seed
reproducibility, type coverage), the injection_types=None defaulting, the
curator.py-vs-cli.py `difficulty` wiring discrepancy for
AdversarialPreferenceTask (curator.py's _build_generator() never passes
cfg.difficulty to this task, while cli.py's YAML dispatch does), and the
forced-failure/parse-failure paths for both tasks using a trivial in-memory
fake LLM double (no network, no litellm/vllm dependency for the double
itself — only the dispatch-path tests import litellm-gated modules).
"""

from __future__ import annotations

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install curatorkit[generation]")

from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask
from curatorkit.generators.adversarial_qa_generator import (
    ALL_INJECTION_TYPES,
    AdversarialQAGenerationTask,
)
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, RejectedSample


class _FakeLLM(BaseLLM):
    """In-memory BaseLLM double — no network. Configurable canned response."""

    def __init__(self, text: str = "", raise_exc: Exception | None = None):
        super().__init__(model="fake/test-model", max_retries=1)  # avoid retry backoff sleep
        self._text = text
        self._raise_exc = raise_exc

    def _call(self, messages, **kwargs) -> LLMResponse:
        if self._raise_exc is not None:
            raise self._raise_exc
        return LLMResponse(text=self._text, model=self.model)


def _seeds(n: int, domain: str = "astronomy") -> list[DataSample]:
    return [
        DataSample(
            source_uri="test://seed",
            output=f"Passage {i} about {domain}: fact {i}.",
            task_type="language_modeling",
            metadata={"domain": domain},
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Injection-plan sampling
# ─────────────────────────────────────────────────────────────────────────────


class TestInjectionPlanCounts:
    @pytest.mark.parametrize("rate,expected", [(0.0, 0), (0.25, 5), (0.5, 10), (0.75, 15), (1.0, 20)])
    def test_qa_plan_count_matches_rate(self, rate, expected):
        task = AdversarialQAGenerationTask(llm=None, injection_rate=rate, seed=42)
        plan = task._make_plan([s.id for s in _seeds(20)])
        assert sum(1 for v in plan.values() if v is not None) == expected

    def test_qa_plan_reproducible_with_same_seed(self):
        ids = [s.id for s in _seeds(20)]
        plan_a = AdversarialQAGenerationTask(llm=None, injection_rate=0.5, seed=42)._make_plan(ids)
        plan_b = AdversarialQAGenerationTask(llm=None, injection_rate=0.5, seed=42)._make_plan(ids)
        assert plan_a == plan_b

    def test_qa_plan_differs_with_different_seed(self):
        ids = [s.id for s in _seeds(20)]
        plan_a = AdversarialQAGenerationTask(llm=None, injection_rate=0.5, seed=42)._make_plan(ids)
        plan_b = AdversarialQAGenerationTask(llm=None, injection_rate=0.5, seed=7)._make_plan(ids)
        assert plan_a != plan_b

    def test_qa_plan_at_full_rate_covers_every_injection_type(self):
        ids = [s.id for s in _seeds(40)]
        plan = AdversarialQAGenerationTask(llm=None, injection_rate=1.0, seed=42)._make_plan(ids)
        assert set(plan.values()) == set(ALL_INJECTION_TYPES)

    def test_qa_injection_types_none_defaults_to_all(self):
        task = AdversarialQAGenerationTask(llm=None, injection_types=None)
        assert task.injection_types == ALL_INJECTION_TYPES


class TestPreferenceInjectionAssignment:
    """AdversarialPreferenceTask decides adversarial-vs-naive and the type
    inline in _parse_response via self.rng, not a precomputed plan dict —
    exercise that RNG directly with the same seeded-reproducibility shape."""

    def test_injection_types_none_defaults_to_all_four(self):
        task = AdversarialPreferenceTask(llm=None)
        assert set(task.injection_types) == {
            "contradicts_source",
            "parametric_drift",
            "domain_mismatch",
            "instruction_quality",
        }
        assert "high_temperature_drift" not in task.injection_types

    def test_rng_decisions_reproducible_with_same_seed(self):
        task_a = AdversarialPreferenceTask(llm=None, injection_rate=0.5, seed=42)
        task_b = AdversarialPreferenceTask(llm=None, injection_rate=0.5, seed=42)
        decisions_a = [task_a.rng.random() < task_a.injection_rate for _ in range(20)]
        decisions_b = [task_b.rng.random() < task_b.injection_rate for _ in range(20)]
        assert decisions_a == decisions_b

    def test_rng_decisions_differ_with_different_seed(self):
        task_a = AdversarialPreferenceTask(llm=None, injection_rate=0.5, seed=42)
        task_b = AdversarialPreferenceTask(llm=None, injection_rate=0.5, seed=7)
        decisions_a = [task_a.rng.random() < task_a.injection_rate for _ in range(20)]
        decisions_b = [task_b.rng.random() < task_b.injection_rate for _ in range(20)]
        assert decisions_a != decisions_b


# ─────────────────────────────────────────────────────────────────────────────
# curator.py vs cli.py `difficulty` wiring discrepancy for adversarial_preference
# ─────────────────────────────────────────────────────────────────────────────


class TestDifficultyWiringDiscrepancy:
    """curator.py's Curator._build_generator() never passes cfg.difficulty when
    building AdversarialPreferenceTask, while cli.py's YAML dispatch does pass
    gen.difficulty for the same task. These tests document/pin that exact
    discrepancy so a future fix must consciously update them."""

    def test_preference_curator_py_drops_difficulty(self):
        from curatorkit.curator import Curator, CuratorConfig

        cfg = CuratorConfig(
            dataset="dummy.jsonl",
            llm_model="openai/gpt-4o-mini",
            generation_task="adversarial_preference",
            difficulty="hard",
        )
        gen = Curator(cfg)._build_generator()
        assert gen.difficulty != "hard", (
            "curator.py now passes difficulty to AdversarialPreferenceTask — "
            "the curator.py/cli.py discrepancy is fixed; update/remove this test "
            "and the corresponding CHANGELOG/docs note."
        )
        assert gen.difficulty == "medium"  # class default, confirms silent drop

    def test_preference_cli_py_keeps_difficulty(self):
        from curatorkit.cli import _build_steps
        from curatorkit.config import GenerationConfig, LLMConfig, PipelineConfig, ReaderConfig

        cfg = PipelineConfig(
            readers=[ReaderConfig(type="jsonl", path="dummy.jsonl")],
            llm=LLMConfig(model="openai/gpt-4o-mini"),
            generators=[GenerationConfig(type="adversarial_preference", difficulty="hard")],
        )
        steps, _ = _build_steps(cfg, verbose=False)
        gen = next(s for s in steps if isinstance(s, AdversarialPreferenceTask))
        assert gen.difficulty == "hard"

    def test_qa_negative_control_both_paths_agree(self):
        """adversarial_qa passes difficulty on both paths — proves the
        discrepancy above is preference-specific, not a general pattern."""
        from curatorkit.cli import _build_steps
        from curatorkit.config import GenerationConfig, LLMConfig, PipelineConfig, ReaderConfig
        from curatorkit.curator import Curator, CuratorConfig

        curator_cfg = CuratorConfig(
            dataset="dummy.jsonl",
            llm_model="openai/gpt-4o-mini",
            generation_task="adversarial_qa",
            difficulty="hard",
        )
        curator_gen = Curator(curator_cfg)._build_generator()

        cli_cfg = PipelineConfig(
            readers=[ReaderConfig(type="jsonl", path="dummy.jsonl")],
            llm=LLMConfig(model="openai/gpt-4o-mini"),
            generators=[GenerationConfig(type="adversarial_qa", difficulty="hard")],
        )
        steps, _ = _build_steps(cli_cfg, verbose=False)
        cli_gen = next(s for s in steps if isinstance(s, AdversarialQAGenerationTask))

        assert curator_gen.difficulty == cli_gen.difficulty == "hard"


# ─────────────────────────────────────────────────────────────────────────────
# Forced-failure / parse-failure paths (fake LLM, no network)
# ─────────────────────────────────────────────────────────────────────────────


class TestQAForcedFailurePaths:
    def test_run_emits_rejected_sample_on_llm_exception(self):
        task = AdversarialQAGenerationTask(
            llm=_FakeLLM(raise_exc=RuntimeError("boom")), injection_rate=0.0
        )
        results = task.run(_seeds(1))
        assert results == []
        assert len(task._rejected) == 1
        assert isinstance(task._rejected[0], RejectedSample)
        assert "generation_failed" in task._rejected[0].rejection_reason

    def test_run_multi_passage_emits_rejected_sample_on_llm_exception(self):
        task = AdversarialQAGenerationTask(
            llm=_FakeLLM(raise_exc=RuntimeError("boom")), injection_rate=0.0
        )
        seeds = _seeds(2)
        results = task.run_multi_passage([(seeds[0], seeds[1])])
        assert results == []
        assert len(task._rejected) == 1
        assert "multi_passage_generation_failed" in task._rejected[0].rejection_reason

    def test_run_malformed_json_yields_no_samples_no_rejection(self):
        """Malformed JSON with no fallback-extractable pairs produces neither a
        DataSample nor a RejectedSample from _generate_one directly — _build_samples
        just returns an empty list for a response it can't parse into pairs."""
        task = AdversarialQAGenerationTask(llm=_FakeLLM(text="not json"), injection_rate=0.0)
        results = task.run(_seeds(1))
        assert results == []


class TestPreferenceForcedFailurePaths:
    def test_malformed_json_silently_dropped_no_rejection(self):
        """AdversarialPreferenceTask._parse_response returns [] on JSONDecodeError
        with no RejectedSample emitted — documented asymmetry vs. QA's explicit
        rejection path. BaseGenerationTask.run() then records a
        generation_parse_failed RejectedSample after exhausting retries, since
        _parse_response itself yields nothing to work with."""
        task = AdversarialPreferenceTask(llm=_FakeLLM(text="not json"))
        results = task.run(_seeds(1))
        assert results == []
        assert len(task._rejected) == 1
        assert "generation_parse_failed" in task._rejected[0].rejection_reason

    def test_valid_faithful_json_with_failing_rejected_call_drops_pair(self):
        """If the faithful call parses fine but the inline rejected-response
        call raises, rejected_text stays empty and the pair is dropped."""

        class _TwoCallFakeLLM(BaseLLM):
            def __init__(self):
                super().__init__(model="fake/two-call", max_retries=1)  # avoid retry backoff sleep
                self.calls = 0

            def _call(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        text='[{"question": "Q1?", "answer": "A1."}]', model=self.model
                    )
                raise RuntimeError("rejected-generation failure")

        llm = _TwoCallFakeLLM()
        task = AdversarialPreferenceTask(llm=llm, injection_rate=1.0, injection_types=["contradicts_source"])
        results = task.run(_seeds(1))
        assert results == []
        # First call succeeds (faithful); every subsequent call raises. The
        # rejected-generation call retries internally (BaseLLM.generate's own
        # max_retries) before _parse_response gives up and drops the pair, and
        # BaseGenerationTask.run() retries the whole sample once more on an
        # empty parse — so call count is >1, not a fixed number.
        assert llm.calls > 1
