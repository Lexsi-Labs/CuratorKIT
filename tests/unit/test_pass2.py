"""Tests for curatorkit/diagnostic/pass2.py"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from curatorkit.diagnostic.failure_modes import FailureDiagnosis, FailureMode
from curatorkit.diagnostic.pass2 import AdaptivePass2Runner, _mark_failed
from curatorkit.schema import DataSample, RejectedSample


def _make_config(temperature: float = 0.7):
    """Build a minimal CuratorConfig-like object."""
    from curatorkit.curator import CuratorConfig

    return CuratorConfig(llm_model="openai/gpt-4o-mini", llm_temperature=temperature)


def _make_recoverable(mode: FailureMode = FailureMode.GENERATOR_TEMPERATURE) -> RejectedSample:
    s = RejectedSample(
        source_uri="test/doc",
        instruction="What is X?",
        input="X is a concept in science.",
        output="X is something.",
        rejection_reason="below_threshold:0.4",
        rejecting_step="HallucinationGate",
    )
    s.diagnosis = FailureDiagnosis.from_mode(mode, evidence=[False, False, False], probe_calls=3)
    return s


def _make_runner(gate_passes: bool = True, output_dir: Path | None = None):
    mock_gen = MagicMock()
    regen = DataSample(
        source_uri="test/doc",
        instruction="What is X?",
        output="Regen answer.",
    )
    mock_gen.run.return_value = [regen]

    mock_gate = MagicMock()
    if gate_passes:
        mock_gate.run.return_value = ([regen], [])
    else:
        rejected_item = RejectedSample(
            source_uri="test/doc",
            instruction="What is X?",
            output="Bad answer.",
            rejection_reason="gate_failed_pass2",
            rejecting_step="MockGate",
        )
        mock_gate.run.return_value = ([], [rejected_item])

    with tempfile.TemporaryDirectory() as tmpdir:
        out = output_dir or Path(tmpdir)
        runner = AdaptivePass2Runner(
            generator=mock_gen,
            gate=mock_gate,
            base_config=_make_config(),
            output_dir=out,
        )
        return runner, mock_gen, mock_gate, out


class TestAdaptivePass2Runner:
    def test_conversion_rate_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, _, _, _ = _make_runner(gate_passes=True, output_dir=Path(tmpdir))
            samples = [_make_recoverable() for _ in range(4)]
            result = runner.run(samples, force_patch={"llm_temperature": 0.3})
            assert result.attempted == 4
            assert result.conversion_rate == 1.0

    def test_gate_fail_produces_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, _, _, _ = _make_runner(gate_passes=False, output_dir=Path(tmpdir))
            samples = [_make_recoverable()]
            result = runner.run(samples, force_patch={"llm_temperature": 0.3})
            assert result.conversion_rate == 0.0
            assert len(result.rejected) == 1

    def test_no_force_patch_skips_regeneration(self):
        # The runner is a research utility: without an explicit force_patch it
        # never regenerates (inline probe recovery is the production path).
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, mock_gen, _, _ = _make_runner(gate_passes=True, output_dir=Path(tmpdir))
            result = runner.run([_make_recoverable()])
            assert result.accepted == []
            assert len(result.rejected) == 1
            mock_gen.run.assert_not_called()

    def test_unrecoverable_mode_skipped_even_with_force_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, mock_gen, _, _ = _make_runner(gate_passes=True, output_dir=Path(tmpdir))
            sample = _make_recoverable(FailureMode.NEAR_DUPLICATE)
            result = runner.run([sample], force_patch={"llm_temperature": 0.3})
            assert result.accepted == []
            assert len(result.rejected) == 1
            mock_gen.run.assert_not_called()

    def test_force_patch_recorded_in_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, _, _, _ = _make_runner(output_dir=Path(tmpdir))
            sample = _make_recoverable(FailureMode.GENERATOR_PARAMETRIC)
            # force_patch is applied to every attempted sample and echoed in the summary
            result = runner.run([sample], force_patch={"llm_temperature": 0.3})
            assert result.summary["forced_patch"] == {"llm_temperature": 0.3}

    def test_apply_patch_does_not_mutate_base_config(self):
        cfg = _make_config(temperature=0.7)
        patched = cfg.apply_patch({"llm_temperature": 0.3})
        assert cfg.llm_temperature == 0.7, "Original config was mutated"
        assert patched.llm_temperature == 0.3

    def test_apply_patch_prompt_template(self):
        cfg = _make_config()
        patched = cfg.apply_patch({"prompt_template": "strict_grounding"})
        assert patched.llm_prompt_template == "strict_grounding"
        assert cfg.llm_prompt_template is None

    def test_sample_without_diagnosis_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, _, _, _ = _make_runner(output_dir=Path(tmpdir))
            s = RejectedSample(
                source_uri="test/doc",
                instruction="Q?",
                rejection_reason="reason",
                rejecting_step="gate",
            )
            result = runner.run([s])
            assert result.attempted == 1
            assert len(result.accepted) == 0

    def test_mode_conversion_rates_in_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, _, _, _ = _make_runner(gate_passes=True, output_dir=Path(tmpdir))
            samples = [_make_recoverable(FailureMode.GENERATOR_TEMPERATURE)] * 3
            result = runner.run(samples, force_patch={"llm_temperature": 0.3})
            assert "generator_temperature" in result.summary["mode_rates"]

    def test_write_outputs_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            runner, _, _, _ = _make_runner(gate_passes=True, output_dir=out)
            runner.run([_make_recoverable()])
            assert (out / "pass2_summary.json").exists()
            assert (out / "rejected_pass2.jsonl").exists()

    def test_llm_temperature_restored_after_run(self):
        """_regenerate_sample must not permanently mutate generator.llm.temperature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, mock_gen, _, _ = _make_runner(gate_passes=True, output_dir=Path(tmpdir))
            mock_gen.llm = MagicMock()
            mock_gen.llm.temperature = 0.9
            sample = _make_recoverable(FailureMode.GENERATOR_TEMPERATURE)
            # force_patch changes the LLM temperature to 0.3 during regeneration
            runner.run([sample], force_patch={"llm_temperature": 0.3})
            assert mock_gen.llm.temperature == 0.9, "LLM temperature was not restored"


class TestMarkFailed:
    def test_appends_suffix(self):
        s = RejectedSample(source_uri="t/d", rejection_reason="original", rejecting_step="gate")
        _mark_failed(s, "extra_fail")
        assert s.rejection_reason == "original:extra_fail"
