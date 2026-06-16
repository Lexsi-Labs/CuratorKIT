"""Tests for curatorkit/diagnostic/probe.py"""

from unittest.mock import MagicMock

from curatorkit.diagnostic.failure_modes import FailureMode
from curatorkit.diagnostic.probe import DiagnosticProbe
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample


def _make_rejected(
    instruction: str = "What is X?",
    source: str = "X is a thing that exists in many forms.",
    provenance_notes: dict | None = None,
) -> RejectedSample:
    sample = RejectedSample(
        source_uri="test/doc",
        instruction=instruction,
        input=source,
        output="X is a concept.",
        rejection_reason="below_hallucination_threshold:0.3",
        rejecting_step="HallucinationGate",
    )
    if provenance_notes:
        sample.append_provenance(
            ProvenanceRecord(
                step_name="HallucinationGate",
                step_version="1.0.0",
                config_hash="abc",
                notes=provenance_notes,
            )
        )
    return sample


def _make_passing_sample() -> DataSample:
    return DataSample(
        source_uri="test/doc",
        instruction="What is X?",
        input="X is a thing that exists in many forms.",
        output="A regenerated passing answer.",
    )


def _make_probe(
    gate_always_pass: bool = True,
    generate_returns: str | None = "A generated answer.",
) -> DiagnosticProbe:
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = generate_returns or ""
    if generate_returns is None:
        mock_llm.generate.side_effect = RuntimeError("LLM error")
    else:
        mock_llm.generate.return_value = mock_resp

    mock_gate = MagicMock()
    if gate_always_pass:
        mock_gate.run.return_value = ([_make_passing_sample()], [])
    else:
        mock_gate.run.return_value = ([], [MagicMock()])

    return DiagnosticProbe(
        generator_llm=mock_llm,
        gate=mock_gate,
        temperatures=[0.3, 0.7, 1.1],
    )


class TestClassifyProbe1:
    def test_all_fail_returns_none(self):
        probe = _make_probe()
        result = probe._classify_probe1([False, False, False], [None, None, None], 3)
        assert result is None

    def test_all_pass_returns_threshold_marginal(self):
        probe = _make_probe()
        samples = [_make_passing_sample() for _ in range(3)]
        result = probe._classify_probe1([True, True, True], samples, 3)
        assert result is not None
        assert result.mode is FailureMode.THRESHOLD_MARGINAL
        assert result.notes["pattern"] == "all_pass"
        assert result.recovered_sample is samples[0]
        assert result.probe_calls == 3

    def test_low_t_pass_high_t_fail_returns_generator_temperature(self):
        probe = _make_probe()
        samples = [_make_passing_sample(), None, None]
        result = probe._classify_probe1([True, False, False], samples, 3)
        assert result is not None
        assert result.mode is FailureMode.GENERATOR_TEMPERATURE
        assert result.notes["pattern"] == "low_t_pass"
        assert result.recovered_sample is samples[0]

    def test_mixed_other_returns_threshold_marginal(self):
        probe = _make_probe()
        samples = [None, _make_passing_sample(), None]
        result = probe._classify_probe1([False, True, False], samples, 3)
        assert result is not None
        assert result.mode is FailureMode.THRESHOLD_MARGINAL
        assert result.notes["pattern"] == "mixed_unstable"
        assert result.recovered_sample is samples[1]

    def test_empty_returns_none(self):
        probe = _make_probe()
        result = probe._classify_probe1([], [], 0)
        assert result is None


class TestClassifySourceFailure:
    def test_no_provenance_notes_returns_unknown(self):
        probe = _make_probe()
        rejected = _make_rejected()  # no provenance records at all
        mode = probe._classify_source_failure(rejected)
        assert mode is FailureMode.UNKNOWN

    def test_low_score_with_claims_returns_source_ambiguous(self):
        probe = _make_probe()
        rejected = _make_rejected(
            provenance_notes={"grounding_score": 0.1, "unsupported_claims": ["claim1"]}
        )
        mode = probe._classify_source_failure(rejected)
        assert mode is FailureMode.SOURCE_AMBIGUOUS

    def test_moderate_score_with_claims_returns_source_ambiguous(self):
        probe = _make_probe()
        rejected = _make_rejected(
            provenance_notes={"grounding_score": 0.35, "unsupported_claims": ["claim1", "claim2"]}
        )
        mode = probe._classify_source_failure(rejected)
        assert mode is FailureMode.SOURCE_AMBIGUOUS


class TestDiagnose:
    def test_diagnose_never_raises_on_error(self):
        probe = _make_probe(generate_returns=None)
        rejected = _make_rejected(source="")  # empty source triggers UNKNOWN path
        diag = probe.diagnose(rejected)
        assert diag is not None
        assert diag.mode is FailureMode.UNKNOWN

    def test_empty_source_returns_unknown(self):
        probe = _make_probe()
        rejected = _make_rejected(instruction="Q?", source="")
        diag = probe.diagnose(rejected)
        assert diag.mode is FailureMode.UNKNOWN

    def test_all_temp_pass_returns_threshold_marginal(self):
        # Near-boundary route (grounding_score >= score_split): temperature
        # sweep runs first, all probes pass → THRESHOLD_MARGINAL + recovery.
        probe = _make_probe(gate_always_pass=True)
        rejected = _make_rejected(provenance_notes={"grounding_score": 0.6})
        diag = probe.diagnose(rejected)
        assert diag.mode is FailureMode.THRESHOLD_MARGINAL
        assert diag.was_recovered is True

    def test_low_grounding_strict_pass_returns_generator_parametric(self):
        # Low-grounding route (grounding_score < score_split): strict-grounding
        # probe runs first; if it passes → GENERATOR_PARAMETRIC + recovery.
        probe = _make_probe(gate_always_pass=True)
        rejected = _make_rejected(provenance_notes={"grounding_score": 0.1})
        diag = probe.diagnose(rejected)
        assert diag.mode is FailureMode.GENERATOR_PARAMETRIC
        assert diag.was_recovered is True
        assert diag.notes["passing_probe"] == "strict_grounding"

    def test_all_probes_fail_falls_back_to_source_classifier(self):
        # Gate always fails → every probe path is exhausted → fallback classifier.
        # No HallucinationGate provenance notes → UNKNOWN, no recovery.
        probe = _make_probe(gate_always_pass=False)
        rejected = _make_rejected()
        diag = probe.diagnose(rejected)
        assert diag.mode is FailureMode.UNKNOWN
        assert diag.was_recovered is False

    def test_all_probes_fail_with_claims_returns_source_ambiguous(self):
        probe = _make_probe(gate_always_pass=False)
        rejected = _make_rejected(
            provenance_notes={"grounding_score": 0.6, "unsupported_claims": ["claim1"]}
        )
        diag = probe.diagnose(rejected)
        assert diag.mode is FailureMode.SOURCE_AMBIGUOUS
        assert diag.was_recovered is False

    def test_diagnose_batch_returns_one_diagnosis_per_sample(self):
        probe = _make_probe(gate_always_pass=True)
        rejected = [
            _make_rejected(provenance_notes={"grounding_score": 0.6}),
            _make_rejected(provenance_notes={"grounding_score": 0.1}),
        ]
        diagnoses = probe.diagnose_batch(rejected)
        assert len(diagnoses) == 2
        assert all(d is not None for d in diagnoses)


class TestInstructionRegenProbe:
    def test_instruction_quality_diagnosed_when_regen_instr_passes(self):
        """Instruction-regen probe: gate fails on every answer probe but passes
        once the instruction itself is regenerated."""
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "What is X?"
        mock_llm.generate.return_value = mock_resp

        call_count = {"n": 0}

        def gate_side_effect(samples):
            call_count["n"] += 1
            # Low-grounding route (no grounding_score in provenance):
            # strict grounding (call 1), temperature sweep (calls 2-4),
            # domain-specific (call 5) all fail on answer probes.
            # Instruction regeneration (call 6) passes.
            if call_count["n"] < 6:
                return ([], [MagicMock()])
            return ([samples[0]], [])

        mock_gate = MagicMock()
        mock_gate.run.side_effect = gate_side_effect

        probe = DiagnosticProbe(
            generator_llm=mock_llm, gate=mock_gate, temperatures=[0.3, 0.7, 1.1]
        )
        rejected = _make_rejected()
        diag = probe.diagnose(rejected)
        assert diag.mode is FailureMode.INSTRUCTION_QUALITY
        assert diag.notes["passing_probe"] == "regenerated_instruction"
        assert diag.was_recovered is True

    def test_regenerate_instruction_empty_source_returns_none(self):
        probe = _make_probe()
        rejected = _make_rejected()
        result = probe._regenerate_instruction(rejected, source_context="")
        assert result is None

    def test_regenerate_instruction_returns_sample_with_new_instruction(self):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "What does X do?"
        mock_llm.generate.return_value = mock_resp
        mock_gate = MagicMock()
        mock_gate.run.return_value = ([MagicMock()], [])

        probe = DiagnosticProbe(generator_llm=mock_llm, gate=mock_gate)
        rejected = _make_rejected(instruction="original question?")
        result = probe._regenerate_instruction(rejected, source_context="X does things.")
        assert isinstance(result, DataSample)
        assert result.instruction == "What does X do?"
        assert result.output == rejected.output  # original output kept

    def test_regenerate_instruction_llm_error_returns_none(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        mock_gate = MagicMock()
        probe = DiagnosticProbe(generator_llm=mock_llm, gate=mock_gate)
        rejected = _make_rejected()
        result = probe._regenerate_instruction(rejected, source_context="some text")
        assert result is None


class TestRegenerate:
    def test_regenerate_empty_source_returns_none(self):
        probe = _make_probe()
        rejected = _make_rejected()
        result = probe._regenerate(rejected, source_context="", temperature=0.5)
        assert result is None

    def test_regenerate_llm_error_returns_none(self):
        probe = _make_probe(generate_returns=None)
        rejected = _make_rejected()
        result = probe._regenerate(rejected, source_context="some text", temperature=0.5)
        assert result is None

    def test_regenerate_returns_data_sample(self):
        probe = _make_probe()
        rejected = _make_rejected()
        result = probe._regenerate(rejected, source_context="some text", temperature=0.5)
        assert isinstance(result, DataSample)
        assert result.output == "A generated answer."
