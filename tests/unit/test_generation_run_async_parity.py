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


class _RecordingLLM(BaseLLM):
    """Returns the faithful-pass JSON on the first call, then a fixed
    "rejected" string on every later call — while recording every call's
    messages/temperature so tests can assert on what was actually sent
    (e.g. which prompt template and temperature the naive-rejected path used).
    """

    def __init__(self, faithful_json: str, **kw):
        super().__init__(**kw)
        self._faithful_json = faithful_json
        self.calls: list[dict] = []

    def _record(self, messages, kwargs):
        self.calls.append({"content": messages[0]["content"], "temperature": kwargs.get("temperature")})

    def _call(self, messages, **kwargs):
        self._record(messages, kwargs)
        if len(self.calls) == 1:
            return LLMResponse(text=self._faithful_json, model="fake")
        return LLMResponse(text="a rejected answer", model="fake")

    async def _acall(self, messages, **kwargs):
        return self._call(messages, **kwargs)


class TestAdversarialPreferenceTaskAsyncParity:
    def test_run_async_produces_chosen_rejected_pairs(self):
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        llm = _llm(json.dumps([{"question": "q1", "answer": "a1"}]))
        task = AdversarialPreferenceTask(llm=llm, num_questions=1, injection_rate=0.5, seed=42)
        samples = [
            DataSample(source_uri="t://", input=f"passage {i}", task_type="language_modeling")
            for i in range(6)
        ]
        results = asyncio.run(task.run_async(samples))

        assert len(task._rejected) == 0
        assert len(results) == 6
        assert all(r.task_type == "preference" for r in results)
        assert all(r.chosen == "a1" and r.rejected for r in results)

    def test_via_pipeline_does_not_reject_everything(self):
        """End-to-end: Pipeline.run_async() must dispatch to this task's own
        run_async(), not the inherited generic single-response one."""
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        llm = _llm(json.dumps([{"question": "q1", "answer": "a1"}]))
        task = AdversarialPreferenceTask(llm=llm, num_questions=1, injection_rate=0.5, seed=42)
        samples = [
            DataSample(source_uri="t://", input=f"passage {i}", task_type="language_modeling")
            for i in range(4)
        ]
        result = asyncio.run(Pipeline([_FakeReader(samples), task]).run_async())

        assert len(result.passed) == 4
        assert not any(
            r.rejection_reason.startswith("generation_parse_failed") for r in result.rejected
        )

    def test_naive_rejected_path_uses_explicit_degradation_prompt_and_temp_0_9(self):
        """The naive (non-adversarial) rejected branch must use the explicit
        degradation-instruction prompt, not a bare re-ask of the faithful
        prompt, and must sample at temperature=0.9 (not the old 1.1)."""
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        llm = _RecordingLLM(
            faithful_json=json.dumps([{"question": "q1", "answer": "a1"}]),
            model="fake",
            temperature=0.7,
            max_tokens=100,
        )
        # injection_rate=0.0 forces every pair down the naive path.
        task = AdversarialPreferenceTask(llm=llm, num_questions=1, injection_rate=0.0, seed=42)
        samples = [DataSample(source_uri="t://", input="passage", task_type="language_modeling")]

        results = asyncio.run(task.run_async(samples))

        assert len(results) == 1
        assert results[0].metadata["adversarial_rejected"] is False
        naive_call = llm.calls[1]
        assert "deliberately make the answer worse" in naive_call["content"]
        assert naive_call["temperature"] == 0.9

    def test_run_and_run_async_agree_on_naive_rejected_temperature(self):
        """Sync run() must use the same temperature=0.9 for the naive path
        as run_async() — verifies the fix was applied consistently to both
        the sync and async code paths."""
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        llm = _RecordingLLM(
            faithful_json=json.dumps([{"question": "q1", "answer": "a1"}]),
            model="fake",
            temperature=0.7,
            max_tokens=100,
        )
        task = AdversarialPreferenceTask(llm=llm, num_questions=1, injection_rate=0.0, seed=42)
        samples = [DataSample(source_uri="t://", input="passage", task_type="language_modeling")]

        results = task.run(samples)

        assert len(results) == 1
        naive_call = llm.calls[1]
        assert naive_call["temperature"] == 0.9
