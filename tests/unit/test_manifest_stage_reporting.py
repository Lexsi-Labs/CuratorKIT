"""
Tests for the stage-wise reporting added to manifest.json / dataset_card.md.

Before this, a gate's `output_count` never included samples its inline
probe recovered, and the post-pipeline RewardRefiner had no stage_counts
entry at all — so neither manifest.json nor dataset_card.md could show how
much a stage's recovery mechanism actually contributed to the final yield,
only the raw aggregate pass/reject totals.
"""

from __future__ import annotations

from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest
from curatorkit.pipeline import PipelineResult
from curatorkit.schema import DataSample


def _result(stage_counts: dict) -> PipelineResult:
    return PipelineResult(
        passed=[DataSample(source_uri="t://", instruction="q", output="a")],
        rejected=[],
        stage_counts=stage_counts,
        wall_clock_seconds=1.0,
    )


class TestStageTableRecoveredAndForwardColumns:
    def test_gate_with_probe_recovery_shows_recovered_and_forward(self):
        result = _result(
            {
                "HallucinationGate": {
                    "input_count": 20,
                    "output_count": 15,
                    "probe_recovered": 3,
                    "rejected_count": 5,
                }
            }
        )
        manifest = ProvenanceManifest(result).build()
        table = DatasetCardGenerator()._stage_table(manifest["stage_counts"])

        assert "| HallucinationGate | 20 | 15 | 3 | 5 | 18 |" == table

    def test_gate_without_probe_shows_dash_for_recovered_and_passed_for_forward(self):
        result = _result(
            {
                "SchemaGate": {
                    "input_count": 20,
                    "output_count": 18,
                    "rejected_count": 2,
                }
            }
        )
        manifest = ProvenanceManifest(result).build()
        table = DatasetCardGenerator()._stage_table(manifest["stage_counts"])

        assert "| SchemaGate | 20 | 18 | — | 2 | 18 |" == table

    def test_reward_refiner_stage_shows_zero_passed_full_recovered_as_forward(self):
        """RewardRefiner's own stage entry: output_count=0 (no "clean pass"
        concept — every sample here already failed elsewhere), so Forward
        should equal probe_recovered alone.
        """
        result = _result(
            {
                "RewardRefiner": {
                    "input_count": 10,
                    "output_count": 0,
                    "probe_recovered": 5,
                    "rejected_count": 10,
                }
            }
        )
        manifest = ProvenanceManifest(result).build()
        table = DatasetCardGenerator()._stage_table(manifest["stage_counts"])

        assert "| RewardRefiner | 10 | 0 | 5 | 10 | 5 |" == table

    def test_empty_stage_counts_renders_six_dash_columns(self):
        table = DatasetCardGenerator()._stage_table({})
        assert table == "| — | — | — | — | — | — |"


class TestDatasetCardGateSection:
    def test_recovered_count_appears_in_gate_pass_rate_line(self):
        result = _result(
            {
                "HallucinationGate": {
                    "input_count": 20,
                    "output_count": 15,
                    "probe_recovered": 3,
                    "rejected_count": 5,
                }
            }
        )
        manifest = ProvenanceManifest(result).build()
        card = DatasetCardGenerator()._render(manifest, "test_pipeline")

        assert "**HallucinationGate**: 15/20 passed (75.0%), 3 recovered, 5 rejected" in card

    def test_no_recovered_clause_when_stage_has_no_probe(self):
        result = _result(
            {
                "SchemaGate": {
                    "input_count": 20,
                    "output_count": 18,
                    "rejected_count": 2,
                }
            }
        )
        manifest = ProvenanceManifest(result).build()
        card = DatasetCardGenerator()._render(manifest, "test_pipeline")

        assert "- **SchemaGate**: 18/20 passed (90.0%), 2 rejected\n" in card

    def test_reward_refiner_row_present_in_full_card(self):
        """End-to-end sanity check for the reported bug's own numbers: 18
        into RewardGate, 8 pass cleanly, 10 rejected, 5 recovered by the
        refiner -> Forward column shows the true final yield of 13 (8 + 5).
        """
        result = _result(
            {
                "RewardGate": {
                    "input_count": 18,
                    "output_count": 8,
                    "rejected_count": 10,
                },
                "RewardRefiner": {
                    "input_count": 10,
                    "output_count": 0,
                    "probe_recovered": 5,
                    "rejected_count": 10,
                },
            }
        )
        manifest = ProvenanceManifest(result).build()
        card = DatasetCardGenerator()._render(manifest, "test_pipeline")

        assert "| RewardGate | 18 | 8 | — | 10 | 8 |" in card
        assert "| RewardRefiner | 10 | 0 | 5 | 10 | 5 |" in card
