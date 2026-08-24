"""
Regression tests for the reward/hallucination judge score-parsing fallback.

Before this fix, a custom `reward_prompt_template`/`hallucination_prompt_template`
whose expected output JSON omitted the exact key the gate parses
(`overall_score`/`grounding_score`) caused every sample to silently score 0.0
and get rejected — even when the judge returned perfectly good scores in a
different shape (flat dimension keys, or prose). These tests cover:
  1. the shared extract_score() fallback ladder in isolation,
  2. the early, pre-run static warning when a custom template omits the key,
  3. the end-of-run aggregated warning when fallback parsing actually fired,
  4. that the gate's pass/fail decision uses the recovered score, not 0.0.
"""

from __future__ import annotations

import warnings

import pytest

from curatorkit.gates._score_parsing import extract_score, template_mentions_key
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class TestExtractScore:
    def test_primary_key_present_used_directly(self):
        score, used_fallback = extract_score({"overall_score": 0.8}, "{}", "overall_score")
        assert score == 0.8
        assert used_fallback is False

    def test_primary_key_missing_falls_back_to_dimension_average(self):
        parsed = {"truthfulness": 0.9, "creativity": 0.7}
        score, used_fallback = extract_score(
            parsed, "{}", "overall_score", dimension_keys=("truthfulness", "creativity")
        )
        assert score == pytest.approx(0.8)
        assert used_fallback is True

    def test_no_json_falls_back_to_raw_text_number(self):
        score, used_fallback = extract_score(None, "I'd rate this an 8/10 overall.", "overall_score")
        assert score == pytest.approx(0.8)
        assert used_fallback is True

    def test_nothing_found_defaults_to_neutral_half(self):
        score, used_fallback = extract_score(None, "This response is quite good.", "overall_score")
        assert score == 0.5
        assert used_fallback is True

    def test_score_clamped_to_0_1_range(self):
        score, _ = extract_score({"overall_score": 1.7}, "{}", "overall_score")
        assert score == 1.0


class TestTemplateMentionsKey:
    def test_none_template_always_true(self):
        assert template_mentions_key(None, "overall_score") is True

    def test_custom_template_with_key_is_true(self):
        assert template_mentions_key('Return {"overall_score": 0.XX}', "overall_score") is True

    def test_custom_template_without_key_is_false(self):
        assert template_mentions_key('Return {"truthfulness": 0.XX}', "overall_score") is False


def _fake_llm(text: str) -> BaseLLM:
    class _FakeLLM(BaseLLM):
        def __init__(self):
            super().__init__(model="fake/judge", max_retries=1)

        def _call(self, messages, **kwargs):
            return LLMResponse(text=text, model=self.model)

    return _FakeLLM()


class TestRewardGateEarlyWarning:
    def test_custom_template_missing_overall_score_warns_at_construction(self):
        from curatorkit.gates.reward import RewardGate

        with pytest.warns(UserWarning, match="does not mention 'overall_score'"):
            RewardGate(
                llm=_fake_llm("{}"),
                dimensions=["truthfulness", "creativity"],
                prompt_template='Rate: {instruction} {response}. Return {{"truthfulness": 0.XX}}',
            )

    def test_default_template_does_not_warn(self):
        from curatorkit.gates.reward import RewardGate

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RewardGate(llm=_fake_llm("{}"))  # no prompt_template — must not raise/warn

    def test_custom_template_with_key_does_not_warn(self):
        from curatorkit.gates.reward import RewardGate

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RewardGate(
                llm=_fake_llm("{}"),
                prompt_template='Rate: {instruction} {response}. Return {{"overall_score": 0.XX}}',
            )


class TestRewardGateFallbackRecovery:
    def test_flat_dimension_json_recovers_real_score_instead_of_zero(self):
        """Jenish's exact reported scenario: custom template asks for flat
        dimension keys, no overall_score — the sample must NOT be silently
        scored 0.0 and rejected when the underlying scores are good."""
        from curatorkit.gates.reward import RewardGate

        sample = DataSample(
            source_uri="test://s", instruction="Explain tides.", output="The tide rises twice daily."
        )
        with pytest.warns(UserWarning, match="does not mention 'overall_score'"):
            gate = RewardGate(
                llm=_fake_llm('{"truthfulness": 0.9, "creativity": 0.7}'),
                threshold=0.5,
                dimensions=["truthfulness", "creativity"],
                prompt_template="Rate {instruction} / {response}",
            )
        with pytest.warns(UserWarning, match="RewardGate: 1/1"):
            passed, rejected = gate.run([sample])
        assert len(passed) == 1, "average(0.9, 0.7)=0.8 >= threshold=0.5 should PASS, not be rejected"
        assert len(rejected) == 0

    def test_no_fallback_used_emits_no_end_of_run_warning(self):
        from curatorkit.gates.reward import RewardGate

        sample = DataSample(source_uri="test://s", instruction="Q", output="A")
        gate = RewardGate(llm=_fake_llm('{"overall_score": 0.9}'), threshold=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gate.run([sample])


class TestHallucinationGateFallbackRecovery:
    def test_prose_response_recovers_score_instead_of_zero(self):
        from curatorkit.gates.hallucination import HallucinationGate

        sample = DataSample(
            source_uri="test://s",
            instruction="What rises twice daily?",
            input="The tide rises twice a day according to the passage.",
            output="The tide.",
        )
        with pytest.warns(UserWarning, match="does not mention 'grounding_score'"):
            gate = HallucinationGate(
                llm=_fake_llm("I'd say this is well grounded, about a 9 out of 10."),
                threshold=0.5,
                prompt_template="Source: {source_text}\nQ: {question}\nA: {answer}",
            )
        with pytest.warns(UserWarning, match="HallucinationGate: 1/1"):
            passed, rejected = gate.run([sample])
        assert len(passed) == 1, "0.9 >= threshold=0.5 should PASS"
        assert len(rejected) == 0
