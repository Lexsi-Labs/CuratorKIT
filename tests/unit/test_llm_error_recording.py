"""
An LLM call that fails (rate limit, timeout, 5xx) is a generation failure, not a parse failure,
and it must never turn into data or vanish. Each generator below used to record the error under
the wrong label or drop it without a trace:

- PreferenceGenerationTask two_pass (run and run_async) labelled every error
  `generation_parse_failed`, so a rate-limited run looked like bad model output.
- AdversarialPreferenceTask swallowed an error from the rejected-answer call: run() mislabelled
  it `generation_parse_failed` after re-running the sample, run_async() recorded nothing at all.
- EvolInstructTask with generate_answers=True kept the sample with an empty answer when the
  answer call failed.
"""

from __future__ import annotations

import asyncio

import pytest

from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask
from curatorkit.generators.evol_instruct import EvolInstructTask
from curatorkit.generators.preference_gen import PreferenceGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class RateLimitError(Exception):
    pass


class _ScriptedLLM(BaseLLM):
    """Answers with `ok_text` until `fail_after` calls have been made, then raises a rate limit."""

    def __init__(self, ok_text: str = "an answer", fail_after: int = 0):
        super().__init__(model="fake/err", max_retries=0)
        self.ok_text = ok_text
        self.fail_after = fail_after
        self.calls = 0

    def _call(self, messages, **kwargs):
        self.calls += 1
        if self.calls > self.fail_after:
            raise RateLimitError("Trial key, which is limited to 20 API calls / minute")
        return LLMResponse(text=self.ok_text, model=self.model)


def _instr() -> DataSample:
    return DataSample(source_uri="t://", instruction="q", task_type="instruction_following")


def _corpus() -> DataSample:
    return DataSample(source_uri="t://", output="A passage.", task_type="language_modeling")


def _assert_api_error(rejected, n=1):
    assert len(rejected) == n
    for r in rejected:
        # One manifest bucket per error type: the message is in metadata, not in the reason.
        assert r.rejection_reason == "generation_failed:RateLimitError", r.rejection_reason
        assert "limited to 20 API calls" in r.metadata["error"]


# ── PreferenceGenerationTask two_pass ────────────────────────────────────────


@pytest.mark.parametrize("make", [_instr, _corpus], ids=["instruction", "corpus"])
def test_two_pass_run_records_api_error_as_generation_failed(make):
    task = PreferenceGenerationTask(llm=_ScriptedLLM(fail_after=0), mode="two_pass")
    assert task.run([make()]) == []
    _assert_api_error(task._rejected)


@pytest.mark.parametrize("make", [_instr, _corpus], ids=["instruction", "corpus"])
def test_two_pass_run_async_records_api_error_as_generation_failed(make):
    task = PreferenceGenerationTask(llm=_ScriptedLLM(fail_after=0), mode="two_pass")
    assert asyncio.run(task.run_async([make()])) == []
    _assert_api_error(task._rejected)


def test_two_pass_error_on_rejected_call_after_chosen_call_succeeds():
    task = PreferenceGenerationTask(llm=_ScriptedLLM(fail_after=1), mode="two_pass")
    assert task.run([_instr()]) == []
    _assert_api_error(task._rejected)


def test_two_pass_empty_answer_and_unparseable_corpus_reply_stay_parse_failures():
    task = PreferenceGenerationTask(llm=_ScriptedLLM(ok_text="", fail_after=99), mode="two_pass")
    task.run([_instr()])
    for text in ("not json", "[1, 2]"):  # prose, and JSON that is not an object
        corpus_task = PreferenceGenerationTask(
            llm=_ScriptedLLM(ok_text=text, fail_after=99), mode="two_pass"
        )
        corpus_task.run([_corpus()])
        task._rejected += corpus_task._rejected
    assert len(task._rejected) == 3
    assert all(r.rejection_reason.startswith("generation_parse_failed") for r in task._rejected)


# ── AdversarialPreferenceTask rejected-answer call ───────────────────────────

_FAITHFUL = '[{"question": "Q1?", "answer": "A1."}]'


def _adv_task(fail_after: int) -> AdversarialPreferenceTask:
    return AdversarialPreferenceTask(
        llm=_ScriptedLLM(ok_text=_FAITHFUL, fail_after=fail_after),
        injection_rate=1.0,
        injection_types=["contradicts_source"],
    )


def _passage() -> DataSample:
    return DataSample(source_uri="t://", output="Passage: fact.", task_type="language_modeling")


def test_adversarial_preference_run_records_error_from_rejected_call():
    task = _adv_task(fail_after=1)
    assert task.run([_passage()]) == []
    _assert_api_error(task._rejected)


def test_adversarial_preference_run_async_records_error_from_rejected_call():
    task = _adv_task(fail_after=1)
    assert asyncio.run(task.run_async([_passage()])) == []
    _assert_api_error(task._rejected)


def test_adversarial_preference_error_on_a_later_pair_rejects_the_sample_without_partial_output():
    """Pair 2 of 3 hits the error: the whole source sample is rejected, no pair is kept."""
    faithful = (
        '[{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."},'
        ' {"question": "Q3?", "answer": "A3."}]'
    )
    task = AdversarialPreferenceTask(
        llm=_ScriptedLLM(ok_text=faithful, fail_after=2),  # faithful call + pair 1 succeed
        injection_rate=1.0,
        injection_types=["contradicts_source"],
    )
    assert task.run([_passage()]) == []
    _assert_api_error(task._rejected)
    task = AdversarialPreferenceTask(
        llm=_ScriptedLLM(ok_text=faithful, fail_after=2),
        injection_rate=1.0,
        injection_types=["contradicts_source"],
    )
    assert asyncio.run(task.run_async([_passage()])) == []
    _assert_api_error(task._rejected)


# ── EvolInstructTask answer pass ─────────────────────────────────────────────

_EVOLVED = '{"evolved_instruction": "Explain X in depth", "strategy_applied": "s"}'


class _EvolLLM(_ScriptedLLM):
    """Evolves fine; every answer call fails."""

    def _call(self, messages, **kwargs):
        if "evolved_instruction" in messages[0]["content"]:
            return LLMResponse(text=_EVOLVED, model=self.model)
        raise RateLimitError("Trial key, which is limited to 20 API calls / minute")


def test_evol_instruct_answer_error_is_rejected_not_kept_without_an_answer():
    task = EvolInstructTask(llm=_EvolLLM(), num_evolutions=1, generate_answers=True)
    assert task.run([_instr()]) == []
    _assert_api_error(task._rejected)


def test_evol_instruct_async_answer_error_is_rejected_not_kept_without_an_answer():
    task = EvolInstructTask(llm=_EvolLLM(), num_evolutions=1, generate_answers=True)
    assert asyncio.run(task.run_async([_instr()])) == []
    _assert_api_error(task._rejected)
