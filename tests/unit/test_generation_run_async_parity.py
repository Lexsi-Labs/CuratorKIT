"""
Regression tests: GRPORolloutTask, EvolInstructTask, MultiTurnTask, and
AdversarialQAGenerationTask each override the synchronous run() with real
logic (multi-rollout generation, strategy-cycling expansion, turn-by-turn
conversation, adversarial injection planning) but never overrode
run_async() — since Pipeline.run_async() picks run_async() whenever
hasattr(step, "run_async") is true (which it always was, via inheritance
from BaseGenerationTask), and Curator.run() always executes asynchronously
whenever any generation_task is configured, every one of these tasks
silently used the wrong, generic single-response async path instead of
their own real logic on every real invocation through Curator.

For GRPORolloutTask this was total and loud: _parse_response() is an
intentional no-op stub ("run() handles multi-response generation"), so
100% of samples were rejected with generation_parse_failed and zero
rollouts were ever produced. For the other three it was silent and
partial: EvolInstructTask silently skipped num_evolutions expansion and
the generate_answers pass; MultiTurnTask always behaved like single_call
regardless of multiturn_mode/its turn_by_turn default;
AdversarialQAGenerationTask silently produced zero adversarial samples
regardless of injection_rate, since its injection plan is only built
inside run().
"""

from __future__ import annotations

import asyncio
import json

from curatorkit.interfaces import BaseReader
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.pipeline import Pipeline
from curatorkit.schema import DataSample


class _FakeReader(BaseReader):
    """Pipeline always starts with an empty sample list — only readers
    populate it — so these end-to-end tests need one ahead of the
    generation task under test."""

    def __init__(self, samples: list[DataSample]):
        self._samples = samples

    def read(self):
        return self._samples, []


class _FakeLLM(BaseLLM):
    def __init__(self, text: str = "a response", **kw):
        super().__init__(**kw)
        self._text = text

    def _call(self, messages, **kw):
        return LLMResponse(text=self._text, model="fake", prompt_tokens=1, completion_tokens=1)

    async def _acall(self, messages, **kw):
        return LLMResponse(text=self._text, model="fake", prompt_tokens=1, completion_tokens=1)


def _llm(text: str = "a response") -> _FakeLLM:
    return _FakeLLM(text=text, model="fake", temperature=0.7, max_tokens=100)


class TestGRPORolloutTaskAsyncParity:
    def test_run_async_actually_produces_rollouts(self):
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        task = GRPORolloutTask(llm=_llm(), num_responses=2, score_responses=False)
        samples = [
            DataSample(source_uri="t://", instruction=f"q{i}", task_type="instruction_following")
            for i in range(4)
        ]
        results = asyncio.run(task.run_async(samples))

        assert len(task._rejected) == 0
        assert len(results) == 4
        assert all(len(r.responses) == 2 for r in results)
        assert all(r.task_type == "grpo" for r in results)

    def test_via_pipeline_run_async_does_not_reject_everything(self):
        """End-to-end: Pipeline.run_async() must dispatch to GRPORolloutTask's
        own run_async(), not the inherited generic one."""
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        task = GRPORolloutTask(llm=_llm(), num_responses=2, score_responses=False)
        samples = [
            DataSample(source_uri="t://", instruction=f"q{i}", task_type="instruction_following")
            for i in range(3)
        ]
        result = asyncio.run(Pipeline([_FakeReader(samples), task]).run_async())

        assert len(result.passed) == 3
        assert not any(
            r.rejection_reason.startswith("generation_parse_failed") for r in result.rejected
        )


class TestEvolInstructTaskAsyncParity:
    def test_run_async_expands_and_answers(self):
        from curatorkit.generators.evol_instruct import EvolInstructTask

        llm = _llm(
            json.dumps(
                {
                    "evolved_instruction": "harder version",
                    "strategy_applied": "x",
                    "complexity_notes": "n",
                }
            )
        )
        task = EvolInstructTask(llm=llm, num_evolutions=3, generate_answers=True)
        samples = [
            DataSample(source_uri="t://", instruction="q", task_type="instruction_following")
        ]
        results = asyncio.run(task.run_async(samples))

        assert len(results) == 3  # num_evolutions honored
        assert all(r.output for r in results)  # generate_answers honored

    def test_via_pipeline_num_evolutions_not_silently_dropped(self):
        from curatorkit.generators.evol_instruct import EvolInstructTask

        llm = _llm(json.dumps({"evolved_instruction": "x", "strategy_applied": "y"}))
        task = EvolInstructTask(llm=llm, num_evolutions=3, generate_answers=False)
        samples = [
            DataSample(source_uri="t://", instruction="q", task_type="instruction_following")
        ]
        result = asyncio.run(Pipeline([_FakeReader(samples), task]).run_async())

        assert len(result.passed) == 3


class TestMultiTurnTaskAsyncParity:
    def test_run_async_turn_by_turn_default_mode_produces_multiple_turns(self):
        from curatorkit.generators.multiturn_gen import MultiTurnTask

        task = MultiTurnTask(llm=_llm("a follow-up or answer"), num_turns=2)
        samples = [
            DataSample(source_uri="t://", instruction="q", task_type="instruction_following")
        ]
        results = asyncio.run(task.run_async(samples))

        assert len(task._rejected) == 0
        assert results[0].metadata["generation_mode"] == "turn_by_turn"
        assert results[0].metadata["total_turns"] > 2  # real multi-call conversation

    def test_via_pipeline_mode_is_not_silently_forced_to_single_call(self):
        from curatorkit.generators.multiturn_gen import MultiTurnTask

        task = MultiTurnTask(llm=_llm("a follow-up or answer"), num_turns=2)
        samples = [
            DataSample(source_uri="t://", instruction="q", task_type="instruction_following")
        ]
        result = asyncio.run(Pipeline([_FakeReader(samples), task]).run_async())

        assert len(result.passed) == 1
        assert result.passed[0].metadata["generation_mode"] == "turn_by_turn"


class TestAdversarialQAGenerationTaskAsyncParity:
    def test_run_async_actually_applies_injection_plan(self):
        from curatorkit.generators.adversarial_qa_generator import AdversarialQAGenerationTask

        llm = _llm(json.dumps([{"question": "q", "answer": "a"}]))
        task = AdversarialQAGenerationTask(
            llm=llm, injection_rate=0.5, seed=42, num_questions=1
        )
        samples = [
            DataSample(source_uri="t://", input=f"passage {i}", task_type="language_modeling")
            for i in range(10)
        ]
        results = asyncio.run(task.run_async(samples))

        injected = sum(1 for r in results if r.metadata.get("injected_failure"))
        assert injected == 5  # matches injection_rate=0.5 with this seed, not 0

    def test_via_pipeline_injection_rate_not_silently_ignored(self):
        from curatorkit.generators.adversarial_qa_generator import AdversarialQAGenerationTask

        llm = _llm(json.dumps([{"question": "q", "answer": "a"}]))
        task = AdversarialQAGenerationTask(
            llm=llm, injection_rate=1.0, seed=42, num_questions=1
        )
        samples = [
            DataSample(source_uri="t://", input=f"passage {i}", task_type="language_modeling")
            for i in range(5)
        ]
        result = asyncio.run(Pipeline([_FakeReader(samples), task]).run_async())

        assert all(s.metadata.get("injected_failure") for s in result.passed)
