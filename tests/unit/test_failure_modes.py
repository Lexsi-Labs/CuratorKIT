"""Tests for curatorkit/diagnostic/failure_modes.py"""

import json

import pytest

from curatorkit.diagnostic.failure_modes import (
    PROMPT_TEMPLATES,
    FailureDiagnosis,
    FailureMode,
)
from curatorkit.schema import DataSample


def _make_sample() -> DataSample:
    return DataSample(
        source_uri="test/doc",
        instruction="What is X?",
        input="X is a thing that exists in many forms.",
        output="X is a thing.",
    )


class TestFailureModeCompleteness:
    def test_mode_values_are_unique_strings(self):
        values = [m.value for m in FailureMode]
        assert len(values) == len(set(values))
        assert all(isinstance(v, str) for v in values)

    def test_expected_modes_present(self):
        expected = {
            "source_ambiguous",
            "generator_temperature",
            "generator_parametric",
            "threshold_marginal",
            "instruction_quality",
            "response_quality",
            "domain_mismatch",
            "near_duplicate",
            "unknown",
        }
        assert {m.value for m in FailureMode} == expected

    def test_prompt_templates_has_required_keys(self):
        for key in ("default", "strict_grounding", "domain_specific", "generate_question"):
            assert key in PROMPT_TEMPLATES, f"PROMPT_TEMPLATES missing {key!r}"

    def test_prompt_templates_have_format_placeholders(self):
        # All templates need {source}. Answer templates also need {question}.
        # generate_question is a source-only template — it produces the question.
        source_only = {"generate_question"}
        for key, tmpl in PROMPT_TEMPLATES.items():
            assert "{source}" in tmpl, f"Template {key!r} missing {{source}}"
            if key not in source_only:
                assert "{question}" in tmpl, f"Template {key!r} missing {{question}}"

    def test_generate_question_template_has_no_question_placeholder(self):
        assert "{question}" not in PROMPT_TEMPLATES["generate_question"]


class TestFailureDiagnosisFromMode:
    @pytest.mark.parametrize("mode", list(FailureMode))
    def test_from_mode_all_modes(self, mode):
        diag = FailureDiagnosis.from_mode(mode, evidence=[True, False], probe_calls=3)
        assert diag.mode is mode
        assert diag.evidence == [True, False]
        assert diag.probe_calls == 3
        assert diag.recovered_sample is None
        assert diag.was_recovered is False

    def test_from_mode_with_recovered_sample(self):
        sample = _make_sample()
        diag = FailureDiagnosis.from_mode(
            FailureMode.GENERATOR_TEMPERATURE,
            evidence=[True, False],
            probe_calls=2,
            recovered_sample=sample,
        )
        assert diag.recovered_sample is sample
        assert diag.was_recovered is True

    def test_from_mode_notes_defaults_to_empty_dict(self):
        diag = FailureDiagnosis.from_mode(FailureMode.UNKNOWN, [], 0)
        assert diag.notes == {}

    def test_notes_not_shared_between_instances(self):
        diag1 = FailureDiagnosis.from_mode(FailureMode.GENERATOR_TEMPERATURE, [], 0)
        diag2 = FailureDiagnosis.from_mode(FailureMode.GENERATOR_TEMPERATURE, [], 0)
        diag1.notes["extra"] = "mutated"
        assert "extra" not in diag2.notes


class TestWasRecovered:
    def test_was_recovered_false_without_sample(self):
        diag = FailureDiagnosis(mode=FailureMode.UNKNOWN)
        assert diag.was_recovered is False

    def test_was_recovered_true_with_sample(self):
        diag = FailureDiagnosis(
            mode=FailureMode.THRESHOLD_MARGINAL,
            recovered_sample=_make_sample(),
        )
        assert diag.was_recovered is True


class TestFailureDiagnosisToDict:
    def test_to_dict_structure(self):
        diag = FailureDiagnosis.from_mode(
            FailureMode.THRESHOLD_MARGINAL,
            evidence=[True, True, False],
            probe_calls=3,
            notes={"pattern": "mixed"},
        )
        d = diag.to_dict()
        assert d["mode"] == "threshold_marginal"
        assert d["was_recovered"] is False
        assert d["evidence"] == [True, True, False]
        assert d["probe_calls"] == 3
        assert d["notes"]["pattern"] == "mixed"

    def test_to_dict_round_trips_json(self):
        diag = FailureDiagnosis.from_mode(FailureMode.DOMAIN_MISMATCH, [], 2)
        d = diag.to_dict()
        serialised = json.dumps(d)
        recovered = json.loads(serialised)
        assert recovered["mode"] == "domain_mismatch"

    def test_to_dict_reports_recovery_without_embedding_sample(self):
        diag = FailureDiagnosis.from_mode(
            FailureMode.GENERATOR_PARAMETRIC,
            evidence=[],
            probe_calls=3,
            recovered_sample=_make_sample(),
        )
        d = diag.to_dict()
        assert d["was_recovered"] is True
        assert "recovered_sample" not in d
        json.dumps(d)  # stays JSON-serialisable for rejected.jsonl
