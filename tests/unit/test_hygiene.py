"""
Unit tests for the data hygiene components.

All external dependencies (detoxify, detect_secrets, presidio, faker) are
mocked so these tests run without the actual packages installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from curatorkit.schema import DataSample


def make_sample(**kwargs) -> DataSample:
    defaults = {
        "source_uri": "test://hygiene",
        "instruction": "Explain the contract clause.",
        "output": "The clause requires timely delivery of goods.",
    }
    return DataSample(**{**defaults, **kwargs})


# ──────────────────────────────────────────────────────────────────────────────
# ToxicityGate
# ──────────────────────────────────────────────────────────────────────────────


class TestToxicityGateClassifierPhase:
    """Tests that exercise only the Detoxify classifier path (no LLM)."""

    def _make_gate(self, max_score: float):
        from curatorkit.hygiene.toxicity import ToxicityGate

        gate = ToxicityGate(
            classifier_pass_threshold=0.1,
            classifier_reject_threshold=0.5,
            llm=None,
        )
        scores = {
            "toxicity": max_score,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "identity_attack": 0.0,
            "insult": 0.0,
            "threat": 0.0,
            "sexual_explicit": 0.0,
        }
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = scores
        gate._classifier = mock_classifier
        return gate

    def test_clearly_safe_passes(self):
        gate = self._make_gate(max_score=0.02)
        passed, rejected = gate.run([make_sample()])
        assert len(passed) == 1
        assert len(rejected) == 0

    def test_clearly_toxic_rejected(self):
        gate = self._make_gate(max_score=0.8)
        passed, rejected = gate.run([make_sample()])
        assert len(passed) == 0
        assert len(rejected) == 1
        assert rejected[0].rejecting_step == "ToxicityGate"
        assert "toxic_content:classifier" in rejected[0].rejection_reason

    def test_borderline_rejected_without_llm(self):
        gate = self._make_gate(max_score=0.25)
        passed, rejected = gate.run([make_sample()])
        assert len(rejected) == 1
        assert "classifier" in rejected[0].rejection_reason

    def test_empty_text_passes_without_scoring(self):
        gate = self._make_gate(max_score=0.99)
        passed, rejected = gate.run([make_sample(instruction="", output="")])
        assert len(passed) == 1

    def test_rejection_reason_contains_score(self):
        gate = self._make_gate(max_score=0.75)
        _, rejected = gate.run([make_sample()])
        assert "0.750" in rejected[0].rejection_reason

    def test_provenance_written_on_pass(self):
        gate = self._make_gate(max_score=0.05)
        passed, _ = gate.run([make_sample()])
        steps = [r.step_name for r in passed[0].provenance_chain]
        assert "ToxicityGate" in steps

    def test_provenance_notes_contain_scores(self):
        gate = self._make_gate(max_score=0.05)
        passed, _ = gate.run([make_sample()])
        notes = passed[0].provenance_chain[-1].notes
        assert "max_toxicity_score" in notes
        assert "field_scores" in notes


class TestToxicityGatePreferenceData:
    """DPO preference pairs: chosen and rejected are scored independently."""

    def _make_gate_with_per_field_scores(self, chosen_score: float, rejected_score: float):
        from curatorkit.hygiene.toxicity import ToxicityGate

        gate = ToxicityGate(
            classifier_pass_threshold=0.1,
            classifier_reject_threshold=0.5,
            llm=None,
        )

        def mock_predict(text):
            if "chosen_text" in text:
                return {
                    "toxicity": chosen_score,
                    "severe_toxicity": 0.0,
                    "obscene": 0.0,
                    "identity_attack": 0.0,
                    "insult": 0.0,
                    "threat": 0.0,
                    "sexual_explicit": 0.0,
                }
            return {
                "toxicity": rejected_score,
                "severe_toxicity": 0.0,
                "obscene": 0.0,
                "identity_attack": 0.0,
                "insult": 0.0,
                "threat": 0.0,
                "sexual_explicit": 0.0,
            }

        mock_classifier = MagicMock()
        mock_classifier.predict.side_effect = mock_predict
        gate._classifier = mock_classifier
        return gate

    def test_toxic_rejected_completion_rejects_sample(self):
        gate = self._make_gate_with_per_field_scores(chosen_score=0.02, rejected_score=0.8)
        sample = make_sample(
            task_type="preference",
            instruction="Explain this.",
            chosen="chosen_text: safe and helpful answer",
            rejected="rejected_text: clearly harmful content here",
        )
        passed, rejected = gate.run([sample])
        assert len(rejected) == 1

    def test_safe_pair_passes(self):
        gate = self._make_gate_with_per_field_scores(chosen_score=0.02, rejected_score=0.03)
        sample = make_sample(
            task_type="preference",
            instruction="Explain this.",
            chosen="chosen_text: helpful answer",
            rejected="rejected_text: a worse but not toxic answer",
        )
        passed, rejected = gate.run([sample])
        assert len(passed) == 1


class TestToxicityGateLLMPhase:
    """Tests that the LLM judge is called only for borderline samples."""

    def _make_gate_with_llm(self, classifier_score: float, llm_score: float):
        from curatorkit.hygiene.toxicity import ToxicityGate

        mock_llm = MagicMock()
        mock_llm.config_hash.return_value = "abc123"
        mock_response = MagicMock()
        mock_response.text = (
            f'{{"toxicity_score": {llm_score}, "categories": ["toxicity"], '
            f'"reasoning": "test reasoning"}}'
        )
        mock_llm.generate.return_value = mock_response

        gate = ToxicityGate(
            classifier_pass_threshold=0.1,
            classifier_reject_threshold=0.5,
            llm=mock_llm,
            llm_reject_threshold=0.5,
        )
        scores = {
            "toxicity": classifier_score,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "identity_attack": 0.0,
            "insult": 0.0,
            "threat": 0.0,
            "sexual_explicit": 0.0,
        }
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = scores
        gate._classifier = mock_classifier
        return gate

    def test_borderline_passes_when_llm_says_safe(self):
        gate = self._make_gate_with_llm(classifier_score=0.25, llm_score=0.2)
        passed, rejected = gate.run([make_sample()])
        assert len(passed) == 1
        notes = passed[0].provenance_chain[-1].notes
        assert notes["phase"] == "llm_judge"

    def test_borderline_rejected_when_llm_says_toxic(self):
        gate = self._make_gate_with_llm(classifier_score=0.25, llm_score=0.7)
        passed, rejected = gate.run([make_sample()])
        assert len(rejected) == 1
        assert "llm_judge" in rejected[0].rejection_reason

    def test_llm_not_called_for_clearly_safe(self):
        gate = self._make_gate_with_llm(classifier_score=0.02, llm_score=0.0)
        gate.run([make_sample()])
        assert gate.llm is not None
        gate.llm.generate.assert_not_called()

    def test_llm_not_called_for_clearly_toxic(self):
        gate = self._make_gate_with_llm(classifier_score=0.8, llm_score=0.0)
        gate.run([make_sample()])
        assert gate.llm is not None
        gate.llm.generate.assert_not_called()

    def test_threshold_validation(self):
        from curatorkit.hygiene.toxicity import ToxicityGate

        with pytest.raises(ValueError):
            ToxicityGate(classifier_pass_threshold=0.5, classifier_reject_threshold=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# SecretsGate
# ──────────────────────────────────────────────────────────────────────────────


class TestSecretsGate:
    def _make_gate(self, findings_by_field: dict[str, list[dict]]):
        """findings_by_field: {field_name: [{"type": ..., "line_number": ...}, ...]}"""
        from curatorkit.hygiene.secrets import SecretsGate

        gate = SecretsGate.__new__(SecretsGate)
        gate.fields = ["instruction", "input", "output"]
        gate._fields_override = None
        gate.code_corpus_mode = False
        gate._plugins = []
        gate._ds_config = {}

        def mock_scan_sample(sample):
            # _scan_sample returns (findings, fields_actually_scanned)
            results = []
            for field, items in findings_by_field.items():
                for item in items:
                    results.append({**item, "field": field})
            return results, gate.fields

        gate._scan_sample = mock_scan_sample
        return gate

    def test_clean_sample_passes(self):
        gate = self._make_gate({})
        passed, rejected = gate.run([make_sample()])
        assert len(passed) == 1
        assert len(rejected) == 0

    def test_sample_with_aws_key_rejected(self):
        gate = self._make_gate({"output": [{"type": "AWS Access Key", "line_number": 3}]})
        passed, rejected = gate.run([make_sample()])
        assert len(rejected) == 1
        assert "secret_detected" in rejected[0].rejection_reason
        assert rejected[0].rejecting_step == "SecretsGate"

    def test_rejection_reason_includes_type(self):
        gate = self._make_gate({"output": [{"type": "Private Key", "line_number": 1}]})
        _, rejected = gate.run([make_sample()])
        assert "Private Key" in rejected[0].rejection_reason

    def test_multiple_secret_types_all_listed(self):
        gate = self._make_gate(
            {
                "instruction": [{"type": "AWSKeyDetector", "line_number": 1}],
                "output": [{"type": "JwtTokenDetector", "line_number": 2}],
            }
        )
        _, rejected = gate.run([make_sample()])
        assert "AWSKeyDetector" in rejected[0].rejection_reason
        assert "JwtTokenDetector" in rejected[0].rejection_reason

    def test_provenance_written_on_pass(self):
        gate = self._make_gate({})
        passed, _ = gate.run([make_sample()])
        steps = [r.step_name for r in passed[0].provenance_chain]
        assert "SecretsGate" in steps

    def test_provenance_notes_on_rejection(self):
        gate = self._make_gate({"output": [{"type": "GitHubTokenDetector", "line_number": 1}]})
        _, rejected = gate.run([make_sample()])
        notes = rejected[0].provenance_chain[-1].notes
        assert notes["passed"] is False
        assert notes["total_findings"] == 1
        assert "GitHubTokenDetector" in notes["secret_type_counts"]

    def test_empty_batch(self):
        gate = self._make_gate({})
        passed, rejected = gate.run([])
        assert passed == []
        assert rejected == []

    def test_code_corpus_mode_includes_keyword_detector(self):
        with patch("curatorkit.hygiene.secrets._ensure_detect_secrets"):
            from curatorkit.hygiene.secrets import _PLUGIN_KEYWORD, SecretsGate

            gate = SecretsGate(code_corpus_mode=True)
            assert _PLUGIN_KEYWORD in gate._plugins

    def test_prose_mode_excludes_keyword_detector(self):
        with patch("curatorkit.hygiene.secrets._ensure_detect_secrets"):
            from curatorkit.hygiene.secrets import _PLUGIN_KEYWORD, SecretsGate

            gate = SecretsGate(code_corpus_mode=False)
            assert _PLUGIN_KEYWORD not in gate._plugins

    def test_default_fields_include_chosen_rejected(self):
        with patch("curatorkit.hygiene.secrets._ensure_detect_secrets"):
            from curatorkit.hygiene.secrets import SecretsGate

            gate = SecretsGate()
            assert "chosen" in gate.fields
            assert "rejected" in gate.fields

    def test_secret_in_chosen_field_rejected(self):
        gate = self._make_gate({"chosen": [{"type": "AWS Access Key", "line_number": 1}]})
        sample = make_sample(
            task_type="preference",
            instruction="Explain auth.",
            chosen="Use AKIAIOSFODNN7EXAMPLE as your key",
            rejected="Use a secrets manager instead",
        )
        passed, rejected = gate.run([sample])
        assert len(rejected) == 1
        assert "chosen" in rejected[0].provenance_chain[-1].notes["secret_type_counts"] or rejected[
            0
        ].rejection_reason.startswith("secret_detected")

    def test_secret_in_grpo_response_rejected(self):
        gate = self._make_gate({"responses[1]": [{"type": "Private Key", "line_number": 2}]})

        def mock_scan_sample_grpo(sample):
            # simulate finding in responses[1]; returns (findings, fields_scanned)
            findings = [{"type": "Private Key", "line_number": 2, "field": "responses[1]"}]
            return findings, ["instruction", "input", "responses"]

        gate._scan_sample = mock_scan_sample_grpo
        sample = make_sample(
            task_type="grpo",
            instruction="Write code.",
            responses=["safe response", "response with -----BEGIN PRIVATE KEY-----"],
        )
        passed, rejected = gate.run([sample])
        assert len(rejected) == 1


# ──────────────────────────────────────────────────────────────────────────────
# PIIPseudonymizer
# ──────────────────────────────────────────────────────────────────────────────


class _MockPresidioResult:
    def __init__(self, entity_type: str, start: int, end: int, score: float = 0.9):
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


class TestPIIPseudonymizer:
    def _make_pseudonymizer(
        self,
        analyzer_results_by_text: dict[str, list[_MockPresidioResult]],
        faker_values: dict[str, str] | None = None,
    ):
        """
        Build a PIIPseudonymizer with mocked Presidio + Faker.
        analyzer_results_by_text maps text → list of mock RecognizerResult.
        faker_values maps entity_type → fake_value (deterministic for testing).
        """
        from curatorkit.hygiene.pii import PIIPseudonymizer

        pseudonymizer = PIIPseudonymizer.__new__(PIIPseudonymizer)
        pseudonymizer.entity_types = ["PERSON", "EMAIL_ADDRESS", "US_SSN"]
        pseudonymizer._fields_override = None  # use task-aware field selection
        pseudonymizer.fields = [
            "instruction",
            "input",
            "output",
            "chosen",
            "rejected",
            "responses",
        ]
        pseudonymizer.score_threshold = 0.7
        pseudonymizer.faker_seed = 42
        pseudonymizer.language = "en"
        pseudonymizer.spacy_model = "en_core_web_lg"

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = lambda text, entities, language, score_threshold: (
            analyzer_results_by_text.get(text, [])
        )

        _faker_vals = faker_values or {
            "PERSON": "Alice Johnson",
            "EMAIL_ADDRESS": "alice@example.com",
            "US_SSN": "987-65-4321",
        }
        mock_faker = MagicMock()
        mock_faker.name.return_value = _faker_vals.get("PERSON", "Fake Name")
        mock_faker.email.return_value = _faker_vals.get("EMAIL_ADDRESS", "fake@example.com")
        mock_faker.ssn.return_value = _faker_vals.get("US_SSN", "000-00-0000")

        pseudonymizer._analyzer = mock_analyzer
        pseudonymizer._faker = mock_faker
        return pseudonymizer

    def test_no_pii_unchanged(self):
        p = self._make_pseudonymizer({})
        sample = make_sample(output="The contract requires timely delivery.")
        result = p.run([sample])
        assert result[0].output == "The contract requires timely delivery."

    def test_person_replaced_in_output(self):
        text = "John Smith signed the contract."
        results = [_MockPresidioResult("PERSON", 0, 10)]
        p = self._make_pseudonymizer({text: results})
        sample = make_sample(output=text)
        result = p.run([sample])
        assert "John Smith" not in result[0].output
        assert "Alice Johnson" in result[0].output

    def test_cross_field_consistency(self):
        """Same entity in instruction and output gets the same fake value."""
        instruction_text = "John Smith asked a question."
        output_text = "John Smith received an answer."
        results = [_MockPresidioResult("PERSON", 0, 10)]
        p = self._make_pseudonymizer(
            {
                instruction_text: results,
                output_text: results,
            }
        )
        sample = make_sample(instruction=instruction_text, output=output_text)
        result = p.run([sample])
        # Both fields should have the same replacement
        assert "Alice Johnson" in result[0].instruction
        assert "Alice Johnson" in result[0].output

    def test_provenance_records_counts_not_values(self):
        text = "John Smith, SSN 123-45-6789."
        results = [
            _MockPresidioResult("PERSON", 0, 10),
            _MockPresidioResult("US_SSN", 12, 23),
        ]
        p = self._make_pseudonymizer({text: results})
        sample = make_sample(output=text)
        result = p.run([sample])
        notes = result[0].provenance_chain[-1].notes
        assert notes["entities_replaced"]["PERSON"] == 1
        assert notes["entities_replaced"]["US_SSN"] == 1
        # Confirm no actual PII values are in the provenance notes
        assert "John Smith" not in str(notes)
        assert "123-45-6789" not in str(notes)

    def test_provenance_step_name(self):
        p = self._make_pseudonymizer({})
        sample = make_sample()
        result = p.run([sample])
        steps = [r.step_name for r in result[0].provenance_chain]
        assert "PIIPseudonymizer" in steps

    def test_multiple_entity_types(self):
        text = "Contact john@acme.com or SSN 123-45-6789."
        results = [
            _MockPresidioResult("EMAIL_ADDRESS", 8, 20),
            _MockPresidioResult("US_SSN", 24, 35),
        ]
        p = self._make_pseudonymizer({text: results})
        sample = make_sample(output=text)
        result = p.run([sample])
        assert "john@acme.com" not in result[0].output
        assert "123-45-6789" not in result[0].output

    def test_empty_text_field_skipped(self):
        p = self._make_pseudonymizer({})
        sample = make_sample(instruction="", input="", output="")
        result = p.run([sample])
        notes = result[0].provenance_chain[-1].notes
        assert notes["total_replacements"] == 0

    def test_empty_batch(self):
        p = self._make_pseudonymizer({})
        result = p.run([])
        assert result == []

    def test_entity_types_clinical_has_date_time(self):
        from curatorkit.hygiene.pii import ENTITY_TYPES_CLINICAL

        assert "DATE_TIME" in ENTITY_TYPES_CLINICAL

    def test_entity_types_default_no_date_time(self):
        from curatorkit.hygiene.pii import ENTITY_TYPES_DEFAULT

        assert "DATE_TIME" not in ENTITY_TYPES_DEFAULT

    def test_default_fields_include_chosen_rejected(self):
        from curatorkit.hygiene.pii import PIIPseudonymizer

        p = PIIPseudonymizer.__new__(PIIPseudonymizer)
        p.entity_types = []
        p.fields = None  # type: ignore
        # re-run __init__ defaults check via a real instance
        p2 = PIIPseudonymizer()
        assert "chosen" in p2.fields
        assert "rejected" in p2.fields
        assert "responses" in p2.fields

    def test_pii_replaced_in_chosen_field(self):
        chosen_text = "John Smith agreed to the terms."
        results = [_MockPresidioResult("PERSON", 0, 10)]
        p = self._make_pseudonymizer({chosen_text: results})
        sample = make_sample(
            task_type="preference",
            instruction="Summarise the agreement.",
            chosen=chosen_text,
            rejected="The other party declined.",
        )
        result = p.run([sample])
        assert "John Smith" not in result[0].chosen
        assert "Alice Johnson" in result[0].chosen

    def test_grpo_responses_list_pseudonymized(self):
        resp0 = "John Smith answered first."
        resp1 = "John Smith gave a second answer."
        results_map = {
            resp0: [_MockPresidioResult("PERSON", 0, 10)],
            resp1: [_MockPresidioResult("PERSON", 0, 10)],
        }
        p = self._make_pseudonymizer(results_map)
        sample = make_sample(
            task_type="grpo",
            instruction="Answer the question.",
            responses=[resp0, resp1],
        )
        result = p.run([sample])
        assert isinstance(result[0].responses, list)
        assert len(result[0].responses) == 2
        assert "John Smith" not in result[0].responses[0]
        assert "John Smith" not in result[0].responses[1]
        # Same entity → same fake name across both responses
        assert result[0].responses[0].split()[0] == result[0].responses[1].split()[0]


# ──────────────────────────────────────────────────────────────────────────────
# Integration: pipeline position contracts
# ──────────────────────────────────────────────────────────────────────────────


class TestHygieneContracts:
    """Verify the gate/normalizer contracts are respected."""

    def test_toxicity_gate_is_basegate(self):
        from curatorkit.hygiene.toxicity import ToxicityGate
        from curatorkit.interfaces import BaseGate

        assert issubclass(ToxicityGate, BaseGate)

    def test_secrets_gate_is_basegate(self):
        from curatorkit.hygiene.secrets import SecretsGate
        from curatorkit.interfaces import BaseGate

        assert issubclass(SecretsGate, BaseGate)

    def test_pii_pseudonymizer_is_basenormalizer(self):
        from curatorkit.hygiene.pii import PIIPseudonymizer
        from curatorkit.interfaces import BaseNormalizer

        assert issubclass(PIIPseudonymizer, BaseNormalizer)

    def test_toxicity_gate_returns_tuple(self):
        from curatorkit.hygiene.toxicity import ToxicityGate

        gate = ToxicityGate.__new__(ToxicityGate)
        gate.classifier_pass_threshold = 0.1
        gate.classifier_reject_threshold = 0.5
        gate.llm = None
        gate.llm_reject_threshold = 0.5
        gate.detoxify_model = "unbiased"
        gate.text_field = "auto"

        scores = {
            "toxicity": 0.01,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "identity_attack": 0.0,
            "insult": 0.0,
            "threat": 0.0,
            "sexual_explicit": 0.0,
        }
        mock_clf = MagicMock()
        mock_clf.predict.return_value = scores
        gate._classifier = mock_clf

        result = gate.run([make_sample()])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pii_pseudonymizer_returns_list(self):
        from curatorkit.hygiene.pii import PIIPseudonymizer

        p = PIIPseudonymizer.__new__(PIIPseudonymizer)
        p.entity_types = ["PERSON"]
        p._fields_override = None
        p.fields = ["output"]
        p.score_threshold = 0.7
        p.faker_seed = 42
        p.language = "en"
        p.spacy_model = "en_core_web_lg"
        p._analyzer = MagicMock()
        p._analyzer.analyze.return_value = []
        p._faker = MagicMock()

        result = p.run([make_sample()])
        assert isinstance(result, list)
