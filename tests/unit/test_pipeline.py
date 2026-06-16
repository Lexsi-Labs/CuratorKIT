"""Unit tests for the Pipeline runner."""

from __future__ import annotations

from curatorkit.pipeline import Pipeline, PipelineResult
from curatorkit.schema import DataSample


class TestEmptyPipeline:
    def test_empty_pipeline_returns_empty_result(self):
        result = Pipeline([]).run()
        assert isinstance(result, PipelineResult)
        assert result.passed == []
        assert result.rejected == []
        assert result.stage_counts == {}


class TestPipelineWithReader:
    def test_reader_produces_samples(self):
        from curatorkit.interfaces import BaseReader

        class FakeReader(BaseReader):
            def read(self):
                return [
                    DataSample(source_uri="test://", instruction="Hello", output="World"),
                    DataSample(source_uri="test://", instruction="Foo", output="Bar"),
                ], []

        result = Pipeline([FakeReader()]).run()
        assert len(result.passed) == 2


class TestPipelineWithGate:
    def test_gate_separates_passed_and_rejected(self):
        from curatorkit.interfaces import BaseReader

        class FakeReader(BaseReader):
            def read(self):
                return [
                    DataSample(
                        source_uri="t://", instruction="Good instruction", output="Good output"
                    ),
                    DataSample(source_uri="t://", instruction="", output=""),
                ], []

        from curatorkit.gates.schema import SchemaGate

        result = Pipeline([FakeReader(), SchemaGate(min_tokens=3)]).run()
        assert len(result.passed) == 1
        assert len(result.rejected) == 1

    def test_rejected_sidecar_always_has_content(self):
        """Rejected list is populated even when gate passes everything."""
        from curatorkit.interfaces import BaseGate, BaseReader

        class AllPassGate(BaseGate):
            def run(self, samples):
                return samples, []

        class FakeReader(BaseReader):
            def read(self):
                return [DataSample(source_uri="t://", instruction="Hi", output="There")], []

        result = Pipeline([FakeReader(), AllPassGate()]).run()
        assert result.rejected == []
        assert len(result.passed) == 1


class TestPipelineStageCountsTracking:
    def test_stage_counts_populated(self):
        from curatorkit.gates.schema import SchemaGate
        from curatorkit.interfaces import BaseReader

        class FakeReader(BaseReader):
            def read(self):
                return [
                    DataSample(
                        source_uri="t://",
                        instruction="Hello world Python programming",
                        output="A good answer here",
                    ),
                ], []

        result = Pipeline([FakeReader(), SchemaGate()]).run()
        assert "FakeReader" in result.stage_counts
        assert "SchemaGate" in result.stage_counts
