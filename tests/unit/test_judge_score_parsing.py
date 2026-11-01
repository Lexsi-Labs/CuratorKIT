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

import json
import warnings

import pytest

from curatorkit.gates._score_parsing import (
    JudgeParseError,
    extract_score,
    parse_judge_reply,
    template_mentions_key,
)
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

    def test_nothing_found_raises_instead_of_neutral_half(self):
        # Was a neutral 0.5, which passed every sample at threshold <= 0.5.
        with pytest.raises(JudgeParseError):
            extract_score(None, "This response is quite good.", "overall_score")

    # Every reply here fails closed: a gate rejects rather than passes it.
    @pytest.mark.parametrize(
        "text",
        [
            # JSON: a requested key that is not a finite number from 0 to 1
            '{"overall_score": NaN, "reason": "failed at step 9"}',
            '{"overall_score": Infinity, "reason": "failed at step 9"}',
            '{"overall_score": 1e999}',
            '{"overall_score": 1' + "0" * 400 + "}",
            '{"overall_score": "NaN"}',
            '{"overall_score": "N/A", "reason": "failed at step 9"}',
            '{"overall_score": "0.9"}',
            '{"overall_score": null, "reason": "failed at step 9"}',
            '{"overall_score": true}',
            '{"overall_score": [0.9]}',
            '{"overall_score": 1.7}',
            '{"overall_score": 85}',
            '{"overall_score": -0.1}',
            '{"overall_score": NaN, "truthfulness": 0.2}',
            '{"overall_score": "N/A", "truthfulness": 0.2}',
            '{"overall_score": 0.9, "truthfulness": NaN}',
            '{"truthfulness": NaN, "creativity": 0.9}',
            '{"truthfulness": Infinity, "creativity": 0.9}',
            '{"truthfulness": 1e999, "creativity": 0.9}',
            '{"truthfulness": "N/A", "creativity": 0.9}',
            '{"truthfulness": 7, "creativity": 0.9}',
            # JSON: string values and other shapes are never scanned
            '{"reason": "failed at step 9"}',
            '{"rating": "8/10"}',
            '{"reason": "score: 9"}',
            '{"result": {"overall_score": 0.9}}',
            "[0.9]",
            '"8/10"',
            "1e999",
            "0.8",
            "8",
            "{}",
            # anything that is not JSON, whatever it states
            '{"overall_score": NaN} failed at step 9',
            '{"overall_score": 0.8, "reason": "cut off',
            '{"reason": "score: 9',
            "overall_score: 0.9",
            "8/10",
            "I'd rate this an 8/10 overall.",
            "9/10 claims",
            "about a 9 out of 10.",
            "score: 0.9",
            "score: 2, 9/10",
            "final score: 8/10",
            "failed at step 9",
            "Response 2 of 4 is weak.",
            "85%",
            'Here it is: {"overall_score": 0.9}',
            '{"overall_score": 0.9} I hope that helps',
            # nothing at all
            "This response is quite good.",
            "",
        ],
    )
    def test_refused(self, text):
        cleaned, parsed = parse_judge_reply(text)
        with pytest.raises(JudgeParseError):
            extract_score(parsed, cleaned, "overall_score", ("truthfulness", "creativity"))

    @pytest.mark.parametrize(
        "parsed, expected, used_fallback",
        [
            ({"overall_score": 0}, 0.0, False),
            ({"overall_score": 1}, 1.0, False),
            ({"overall_score": 0.9, "reason": "failed at step 2"}, 0.9, False),
            ({"overall_score": 0.9, "truthfulness": 0.1}, 0.9, False),
            ({"truthfulness": 0.9, "creativity": 0.7}, 0.8, True),
            ({"truthfulness": 0.2, "reason": "rated 9/10"}, 0.2, True),
        ],
    )
    def test_json_is_read_from_its_structure(self, parsed, expected, used_fallback):
        score, fallback = extract_score(
            parsed, json.dumps(parsed), "overall_score", ("truthfulness", "creativity")
        )
        assert score == pytest.approx(expected)
        assert fallback is used_fallback

    @pytest.mark.parametrize(
        "reply",
        [
            '{"overall_score": 0.9}',
            '  {"overall_score": 0.9}\n',
            '```json\n{"overall_score": 0.9}\n```',
            '```\n{"overall_score": 0.9}\n```',
            '```json {"overall_score": 0.9} ```',
        ],
    )
    def test_one_fence_around_the_whole_reply_is_stripped(self, reply):
        assert parse_judge_reply(reply)[1] == {"overall_score": 0.9}

    @pytest.mark.parametrize(
        "reply",
        [
            'Here it is:\n```json\n{"overall_score": 0.9}\n```',
            '```json\n{"overall_score": 0.9}\n```\nI hope that helps',
            '```json\n{"overall_score": 0.9}\n```\n```json\n{"overall_score": 0.1}\n```',
            '```json\n{"overall_score": 0.9}',
            '{"overall_score": ```0.9```}',
        ],
    )
    def test_any_other_fence_is_not_json(self, reply):
        assert parse_judge_reply(reply)[1] is None


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
