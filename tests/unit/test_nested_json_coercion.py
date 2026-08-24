"""
Regression tests for the F-17 class of bugs: parsing code that guards on
emptiness ("if not value") rather than type, so a model that nests a string
field into an object (e.g. `{"chosen": {"poem": "..."}}` instead of
`{"chosen": "..."}`) survives the guard and only fails later — either as an
uncaught Pydantic ValidationError at DataSample construction (preference_gen,
adversarial_preference, cot_generator, evol_instruct) or an AttributeError
from calling a str method on a dict (adversarial_qa_generator, qa_generator,
multiturn_gen). Either way the sample is silently discarded with an
unhelpful reason instead of the recoverable text being coerced.

Also covers the related bug: PreferenceGenerationTask never overrode
run_async(), so Pipeline (which always prefers run_async when the method
exists — inherited or not) silently ignored `preference_mode: "two_pass"`
under async invocation, always falling through to the vulnerable
single-call _parse_response path regardless of configured mode.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install curatorkit[generation]")

from curatorkit.generators.base import coerce_text
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class TestCoerceText:
    def test_passes_plain_strings_through(self):
        assert coerce_text("hello") == "hello"

    def test_unwraps_single_key_dict_with_string_value(self):
        assert coerce_text({"poem": "Roses are red..."}) == "Roses are red..."

    def test_dumps_multi_key_dict_as_json(self):
        result = coerce_text({"a": "1", "b": "2"})
        assert result == '{"a": "1", "b": "2"}'

    def test_dumps_list_as_json(self):
        assert coerce_text(["a", "b"]) == '["a", "b"]'

    def test_none_becomes_empty_string(self):
        assert coerce_text(None) == ""

    def test_non_string_scalar_stringified(self):
        assert coerce_text(42) == "42"


def _fake_llm(text: str, max_retries: int = 1) -> BaseLLM:
    class _FakeLLM(BaseLLM):
        def __init__(self):
            super().__init__(model="fake/test-model", max_retries=max_retries)

        def _call(self, messages, **kwargs) -> LLMResponse:
            return LLMResponse(text=text, model=self.model)

    return _FakeLLM()


def _seed(instruction: str = "Write a short poem about the sea.") -> DataSample:
    return DataSample(source_uri="test://seed", instruction=instruction, task_type="instruction_following")


class TestPreferenceGenNestedJsonCoercion:
    def test_single_call_nested_chosen_rejected_survives(self):
        from curatorkit.generators.preference_gen import PreferenceGenerationTask

        nested_json = (
            '{"chosen": {"poem": "Waves crash softly."}, '
            '"rejected": {"poem": "Sea exists."}}'
        )
        task = PreferenceGenerationTask(llm=_fake_llm(nested_json), mode="single_call")
        out = task.run([_seed()])
        assert len(out) == 1
        assert out[0].chosen == "Waves crash softly."
        assert out[0].rejected == "Sea exists."
        assert len(task._rejected) == 0

    def test_two_pass_corpus_mode_nested_fields_survive(self):
        from curatorkit.generators.preference_gen import PreferenceGenerationTask

        nested_json = (
            '{"question": {"text": "What happens at the coast?"}, '
            '"chosen": {"poem": "Waves crash softly."}, '
            '"rejected": {"poem": "Sea exists."}}'
        )
        corpus_seed = DataSample(
            source_uri="test://seed", output="The tide rises twice a day.", task_type="language_modeling"
        )
        task = PreferenceGenerationTask(llm=_fake_llm(nested_json), mode="two_pass")
        out = task.run([corpus_seed])
        assert len(out) == 1
        assert out[0].instruction == "What happens at the coast?"
        assert out[0].chosen == "Waves crash softly."
        assert out[0].rejected == "Sea exists."


class TestPreferenceGenRunAsyncRespectsMode:
    def test_run_async_two_pass_makes_two_separate_calls(self):
        """Before the fix, run_async always used the inherited single-call
        path regardless of self.mode — this would make exactly ONE llm call
        per sample (via _build_messages/_parse_response) instead of two
        (chosen_prompt + rejected_prompt)."""
        from curatorkit.generators.preference_gen import PreferenceGenerationTask

        class _CountingFakeLLM(BaseLLM):
            def __init__(self):
                super().__init__(model="fake/counting", max_retries=1)
                self.call_count = 0
                self.prompts_seen = []

            def _call(self, messages, **kwargs):
                self.call_count += 1
                self.prompts_seen.append(messages[0]["content"])
                if "deliberately include" in messages[0]["content"]:
                    return LLMResponse(text="A vague, low-quality answer.", model=self.model)
                return LLMResponse(text="A thorough, high-quality answer.", model=self.model)

        llm = _CountingFakeLLM()
        task = PreferenceGenerationTask(llm=llm, mode="two_pass", concurrency=2)
        out = asyncio.run(task.run_async([_seed()]))

        assert llm.call_count == 2, "two_pass must issue a separate chosen call and rejected call"
        assert len(out) == 1
        assert out[0].chosen == "A thorough, high-quality answer."
        assert out[0].rejected == "A vague, low-quality answer."
        assert out[0].metadata["generation_mode"] == "two_pass"

    def test_run_async_single_call_mode_still_uses_inherited_path(self):
        from curatorkit.generators.preference_gen import PreferenceGenerationTask

        class _CountingFakeLLM(BaseLLM):
            def __init__(self):
                super().__init__(model="fake/counting", max_retries=1)
                self.call_count = 0

            def _call(self, messages, **kwargs):
                self.call_count += 1
                return LLMResponse(
                    text='{"chosen": "good answer", "rejected": "bad answer"}', model=self.model
                )

        llm = _CountingFakeLLM()
        task = PreferenceGenerationTask(llm=llm, mode="single_call")
        out = asyncio.run(task.run_async([_seed()]))

        assert llm.call_count == 1, "single_call must issue exactly one combined call"
        assert len(out) == 1
        assert out[0].metadata["generation_mode"] == "single_call"


class TestAdversarialQANestedJsonCoercion:
    def test_nested_question_answer_survives(self):
        from curatorkit.generators.adversarial_qa_generator import AdversarialQAGenerationTask

        nested_json = '[{"question": {"text": "What is the sea?"}, "answer": {"text": "A body of water."}}]'
        seed = DataSample(
            source_uri="test://seed", output="The sea covers most of Earth.", task_type="language_modeling"
        )
        task = AdversarialQAGenerationTask(llm=_fake_llm(nested_json), injection_rate=0.0)
        out = task.run([seed])
        assert len(out) == 1
        assert out[0].instruction == "What is the sea?"
        assert out[0].output == "A body of water."
        assert len(task._rejected) == 0


class TestAdversarialPreferenceNestedJsonCoercion:
    def test_nested_question_survives_and_reaches_dpo_pair(self):
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        class _TwoCallFakeLLM(BaseLLM):
            def __init__(self):
                super().__init__(model="fake/two-call", max_retries=1)
                self.calls = 0

            def _call(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        text='[{"question": {"text": "What is the sea?"}, "answer": "A body of water."}]',
                        model=self.model,
                    )
                return LLMResponse(text="An incorrect answer.", model=self.model)

        seed = DataSample(
            source_uri="test://seed", output="The sea covers most of Earth.", task_type="language_modeling"
        )
        task = AdversarialPreferenceTask(
            llm=_TwoCallFakeLLM(), injection_rate=1.0, injection_types=["contradicts_source"]
        )
        out = task.run([seed])
        assert len(out) == 1
        assert out[0].instruction == "What is the sea?"
        assert out[0].chosen == "A body of water."
        assert out[0].rejected == "An incorrect answer."


class TestCotGeneratorNestedJsonCoercion:
    def test_nested_reasoning_answer_survives(self):
        from curatorkit.generators.cot_generator import ChainOfThoughtTask

        nested_json = (
            '{"reasoning": {"text": "Step 1: consider the tide."}, '
            '"answer": {"text": "The tide rises twice daily."}}'
        )
        seed = DataSample(
            source_uri="test://seed",
            instruction="Explain tides.",
            output="baseline answer",
            task_type="instruction_following",
        )
        task = ChainOfThoughtTask(llm=_fake_llm(nested_json))
        out = task.run([seed])
        assert len(out) == 1
        assert "Step 1: consider the tide." in out[0].output
        assert "The tide rises twice daily." in out[0].output


class TestEvolInstructNestedJsonCoercion:
    def test_nested_evolved_instruction_survives(self):
        from curatorkit.generators.evol_instruct import EvolInstructTask

        nested_json = (
            '{"evolved_instruction": {"text": "Explain tides in more depth."}, '
            '"strategy_applied": "deepen", "complexity_notes": {"note": "added depth"}}'
        )
        seed = DataSample(
            source_uri="test://seed", instruction="Explain tides.", task_type="instruction_following"
        )
        task = EvolInstructTask(llm=_fake_llm(nested_json))
        out = task.run([seed])
        assert len(out) == 1
        assert out[0].instruction == "Explain tides in more depth."


class TestMultiturnGenNestedJsonCoercion:
    def test_format_conversation_handles_nested_role_content(self):
        from curatorkit.generators.multiturn_gen import MultiTurnTask

        turns = [
            {"role": {"name": "user"}, "content": {"text": "Hi"}},
            {"role": "assistant", "content": "Hello!"},
        ]
        # _format_conversation must not raise AttributeError on a nested role/content.
        result = MultiTurnTask._format_conversation(turns)
        assert "Hi" in result
        assert "Hello!" in result
