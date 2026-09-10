"""
Curator-level tests for export_formats auto-alignment.

CuratorConfig.export_formats defaults to None ("auto"): Curator resolves the
actual list from generation_task (known before the pipeline runs) or, with
no generation_task, from the task_type(s) the readers actually produce
(known only after). An explicit export_formats is always honored as given;
combining it with a generation_task it doesn't suit only warns.
"""

from __future__ import annotations

import json

import pytest

from curatorkit.curator import Curator, CuratorConfig
from curatorkit.interfaces import BaseReader
from curatorkit.schema import DataSample


def _config(**overrides) -> CuratorConfig:
    return CuratorConfig(dataset="unused", output_dir="unused", **overrides)


class TestExportFormatsNeedData:
    def test_true_when_auto_and_no_generation_task(self):
        curator = Curator(_config())
        assert curator._export_formats_need_data() is True

    def test_false_when_generation_task_set(self):
        curator = Curator(_config(generation_task="qa"))
        assert curator._export_formats_need_data() is False

    def test_false_when_export_formats_explicit(self):
        curator = Curator(_config(export_formats=["alpaca"]))
        assert curator._export_formats_need_data() is False


class TestResolveExportFormats:
    def test_explicit_config_value_always_wins(self):
        curator = Curator(_config(export_formats=["ppo"], generation_task="qa"))
        assert curator._resolve_export_formats() == ["ppo"]

    def test_resolves_from_generation_task(self):
        curator = Curator(_config(generation_task="grpo"))
        assert curator._resolve_export_formats() == ["grpo"]

    def test_resolves_from_observed_task_types_with_no_generation_task(self):
        curator = Curator(_config())
        assert curator._resolve_export_formats({"prompt_only"}) == ["ppo"]

    def test_falls_back_to_default_with_nothing_known(self):
        from curatorkit.exporters.compatibility import DEFAULT_EXPORT_FORMATS

        curator = Curator(_config())
        assert curator._resolve_export_formats() == DEFAULT_EXPORT_FORMATS


class TestWarnIfExportFormatsMisaligned:
    def test_no_warning_when_aligned(self, recwarn):
        curator = Curator(_config(generation_task="qa", export_formats=["alpaca", "sharegpt"]))
        curator._warn_if_export_formats_misaligned()
        assert len(recwarn) == 0

    def test_warns_when_misaligned(self):
        curator = Curator(_config(generation_task="qa", export_formats=["dpo"]))
        with pytest.warns(UserWarning, match=r"dpo.*not designed for generation_task='qa'"):
            curator._warn_if_export_formats_misaligned()

    def test_no_warning_when_export_formats_left_auto(self, recwarn):
        curator = Curator(_config(generation_task="qa"))
        curator._warn_if_export_formats_misaligned()
        assert len(recwarn) == 0

    def test_no_warning_when_no_generation_task(self, recwarn):
        curator = Curator(_config(export_formats=["dpo"]))
        curator._warn_if_export_formats_misaligned()
        assert len(recwarn) == 0


class _FakeReader(BaseReader):
    def __init__(self, samples):
        self._samples = samples

    def read(self):
        return self._samples, []


class TestAutoResolutionEndToEnd:
    """Drives Curator.run() with _build_steps swapped for a minimal fake
    reader (no gates/generation — the point here is exercising export
    deferral + resolution, not the rest of the pipeline), mirroring the
    approach in test_reward_refiner_export_order.py.
    """

    def _rig(self, tmp_path, samples, **cfg_overrides):
        """Fake reader, but the exporter section mirrors _build_steps' real
        logic exactly (same condition, same registry) rather than skipping
        exporters outright — otherwise a case where export_formats is
        resolvable up front (explicit, or via generation_task) would never
        get exported at all here, since only the post-pipeline deferred
        path (_export_after_pipeline) would ever run anything.
        """
        cfg = CuratorConfig(dataset="unused", output_dir=str(tmp_path), **cfg_overrides)
        curator = Curator(cfg)

        def _fake_build_steps(self, include_exporters=True):
            from curatorkit.curator import _exporter_classes

            steps = [_FakeReader(samples)]
            if include_exporters and self._reward_refiner is None and not self._export_formats_need_data():
                exporter_map = _exporter_classes()
                for fmt in self._resolve_export_formats():
                    cls = exporter_map.get(fmt.lower())
                    if cls:
                        steps.append(cls())
            return steps

        curator._build_steps = _fake_build_steps.__get__(curator, Curator)
        return curator

    def test_preference_only_data_exports_only_dpo(self, tmp_path):
        samples = [
            DataSample(
                source_uri="t://", instruction="q", chosen="good", rejected="bad", task_type="preference"
            )
        ]
        curator = self._rig(tmp_path, samples)
        curator.run()

        assert (tmp_path / "dpo.jsonl").exists()
        assert not (tmp_path / "sft_alpaca.jsonl").exists()
        assert not (tmp_path / "sft_sharegpt.jsonl").exists()

        lines = (tmp_path / "dpo.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["chosen"] == "good"

    def test_instruction_following_data_exports_alpaca_and_sharegpt(self, tmp_path):
        samples = [
            DataSample(
                source_uri="t://", instruction="q", output="a", task_type="instruction_following"
            )
        ]
        curator = self._rig(tmp_path, samples)
        curator.run()

        assert (tmp_path / "sft_alpaca.jsonl").exists()
        assert (tmp_path / "sft_sharegpt.jsonl").exists()
        assert not (tmp_path / "dpo.jsonl").exists()

    def test_explicit_export_formats_still_respected(self, tmp_path):
        """Auto-alignment must never override an explicit choice, even one
        that looks wrong for the data actually produced."""
        samples = [
            DataSample(
                source_uri="t://", instruction="q", chosen="good", rejected="bad", task_type="preference"
            )
        ]
        curator = self._rig(tmp_path, samples, export_formats=["alpaca"])
        curator.run()

        assert (tmp_path / "sft_alpaca.jsonl").exists()
        assert not (tmp_path / "dpo.jsonl").exists()
