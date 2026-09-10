"""Tests for curatorkit/exporters/compatibility.py — the task_type <->
export_format alignment used to auto-resolve CuratorConfig.export_formats
when left as None.
"""

from __future__ import annotations

from curatorkit.exporters.compatibility import (
    DEFAULT_EXPORT_FORMATS,
    GENERATION_TASK_OUTPUT_TYPE,
    TASK_TYPE_EXPORT_FORMATS,
    compatible_formats_for,
    misaligned_formats,
    resolve_export_formats,
)


class TestCompatibleFormatsFor:
    def test_known_task_type(self):
        assert compatible_formats_for("preference") == ["dpo"]

    def test_unknown_task_type_returns_empty(self):
        assert compatible_formats_for("some_custom_type") == []


class TestResolveExportFormats:
    def test_generation_task_resolves_before_reading_data(self):
        # qa -> instruction_following -> alpaca/sharegpt, regardless of
        # what observed_task_types says (generation_task takes priority).
        assert resolve_export_formats("qa", observed_task_types={"grpo"}) == [
            "alpaca",
            "sharegpt",
        ]

    def test_grpo_generation_task(self):
        assert resolve_export_formats("grpo") == ["grpo"]

    def test_preference_generation_task(self):
        assert resolve_export_formats("preference") == ["dpo"]
        assert resolve_export_formats("adversarial_preference") == ["dpo"]

    def test_no_generation_task_uses_observed_task_types(self):
        assert resolve_export_formats(None, observed_task_types={"prompt_only"}) == ["ppo"]

    def test_no_generation_task_multiple_observed_task_types_unioned(self):
        formats = resolve_export_formats(
            None, observed_task_types={"instruction_following", "preference"}
        )
        assert set(formats) == {"alpaca", "sharegpt", "dpo"}

    def test_falls_back_to_default_when_nothing_known(self):
        assert resolve_export_formats(None, observed_task_types=None) == DEFAULT_EXPORT_FORMATS
        assert resolve_export_formats(None, observed_task_types=set()) == DEFAULT_EXPORT_FORMATS

    def test_unknown_generation_task_falls_through_to_observed_or_default(self):
        # Defensive: a generation_task string not in GENERATION_TASK_OUTPUT_TYPE
        # (shouldn't happen via CuratorConfig, but resolve_export_formats
        # itself must not raise) falls back rather than erroring.
        assert resolve_export_formats("not_a_real_task") == DEFAULT_EXPORT_FORMATS


class TestMisalignedFormats:
    def test_no_mismatch_returns_empty(self):
        assert misaligned_formats("qa", ["alpaca", "sharegpt"]) == []

    def test_flags_incompatible_format(self):
        assert misaligned_formats("qa", ["dpo"]) == ["dpo"]

    def test_partial_mismatch_flags_only_the_bad_ones(self):
        assert misaligned_formats("grpo", ["grpo", "dpo"]) == ["dpo"]

    def test_unknown_generation_task_never_flags_anything(self):
        assert misaligned_formats("not_a_real_task", ["dpo", "ppo"]) == []

    def test_case_insensitive(self):
        assert misaligned_formats("qa", ["DPO"]) == ["DPO"]


class TestTablesStayInSync:
    """Every generation_task's output type must itself have a compatibility
    entry -- otherwise resolve_export_formats silently falls back to the
    generic default for that task, defeating the point of the alignment."""

    def test_every_generation_task_output_type_is_in_the_compatibility_table(self):
        for task, task_type in GENERATION_TASK_OUTPUT_TYPE.items():
            assert task_type in TASK_TYPE_EXPORT_FORMATS, (
                f"generation_task={task!r} produces task_type={task_type!r}, "
                "which has no entry in TASK_TYPE_EXPORT_FORMATS"
            )
