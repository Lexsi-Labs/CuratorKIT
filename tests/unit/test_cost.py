"""
Tests for cost controls and the bill of materials.

Covers:
  TokenBudget         — record / check / soft & hard caps / reset
  BudgetExceeded      — carries spent / cap context, marks __no_retry__
  CostGateway         — meter tokens, propagate halts past BaseLLM retry
  CostReport          — per-stage / per-model aggregation, markdown render
  BillOfMaterials     — build / write / load / verify clean + tampered
"""

from __future__ import annotations

import json
import warnings

import pytest

from curatorkit.cost.bom import BillOfMaterials
from curatorkit.cost.budget import BudgetExceeded, TokenBudget
from curatorkit.cost.gateway import CostGateway
from curatorkit.cost.report import CallRecord, CostReport
from curatorkit.llm.base import BaseLLM, LLMResponse

# ───────────────────────────────────────────────────────────────────────────
# Test fixtures
# ───────────────────────────────────────────────────────────────────────────


class StubLLM(BaseLLM):
    def _call(self, messages, **kwargs):
        return LLMResponse(
            text="ok",
            model=self.model,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )


# ───────────────────────────────────────────────────────────────────────────
# TokenBudget
# ───────────────────────────────────────────────────────────────────────────


class TestTokenBudget:
    def test_record_increments(self):
        b = TokenBudget()
        b.record(100, 0.05)
        b.record(50, 0.01)
        assert b.tokens_spent == 150
        assert b.usd_spent == pytest.approx(0.06)
        assert b.calls_made == 2

    def test_check_no_cap_never_raises(self):
        b = TokenBudget()
        b.record(1_000_000, 1000.0)
        b.check()  # no caps → no-op

    def test_hard_token_cap(self):
        b = TokenBudget(hard_token_cap=100)
        b.record(99, 0.0)
        b.check()  # still under
        b.record(2, 0.0)
        with pytest.raises(BudgetExceeded) as e:
            b.check()
        assert e.value.spent_tokens == 101
        assert e.value.token_cap == 100

    def test_hard_usd_cap(self):
        b = TokenBudget(hard_usd_cap=1.0)
        b.record(0, 0.5)
        b.check()
        b.record(0, 0.51)
        with pytest.raises(BudgetExceeded):
            b.check()

    def test_soft_cap_warns_once(self):
        b = TokenBudget(hard_token_cap=200, soft_token_cap=100)
        b.record(150, 0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.check()
            assert any("soft cap" in str(x.message) for x in w)
        # Second check should not re-warn (flag is set)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.check()
            assert not any("soft cap" in str(x.message) for x in w)

    def test_estimate_remaining(self):
        b = TokenBudget(hard_token_cap=1000, hard_usd_cap=10.0)
        b.record(300, 1.5)
        rem = b.estimate_remaining()
        assert rem["tokens_remaining"] == 700
        assert rem["usd_remaining"] == pytest.approx(8.5)

    def test_estimate_when_no_cap(self):
        b = TokenBudget()
        b.record(100, 1.0)
        rem = b.estimate_remaining()
        assert rem["tokens_remaining"] is None
        assert rem["usd_remaining"] is None

    def test_reset(self):
        b = TokenBudget(hard_token_cap=10)
        b.record(5, 1.0)
        b.reset()
        assert b.tokens_spent == 0
        assert b.usd_spent == 0.0
        assert b.calls_made == 0
        assert b.soft_warning_fired is False


class TestBudgetExceededMarker:
    def test_no_retry_marker(self):
        assert BudgetExceeded.__no_retry__ is True

    def test_carries_context(self):
        e = BudgetExceeded(spent_tokens=110, spent_usd=0.05, token_cap=100, usd_cap=None)
        msg = str(e)
        assert "tokens=110/100" in msg


# ───────────────────────────────────────────────────────────────────────────
# CostGateway
# ───────────────────────────────────────────────────────────────────────────


class TestCostGateway:
    def test_meters_each_call(self):
        b = TokenBudget()
        report = CostReport()
        gw = CostGateway(
            StubLLM(model="m", max_retries=1),
            budget=b,
            stage_name="gen",
            report=report,
        )
        gw.generate([{"role": "user", "content": "q"}])
        gw.generate([{"role": "user", "content": "q2"}])
        assert b.tokens_spent == 140
        assert b.calls_made == 2
        assert len(report.records) == 2

    def test_per_stage_attribution(self):
        b = TokenBudget()
        report = CostReport()
        a = CostGateway(StubLLM(model="m", max_retries=1), b, "gen", report)
        c = CostGateway(StubLLM(model="m", max_retries=1), b, "judge", report)
        a.generate([{"role": "user", "content": "q"}])
        c.generate([{"role": "user", "content": "q"}])
        per = report.per_stage()
        assert "gen" in per and per["gen"]["calls"] == 1
        assert "judge" in per and per["judge"]["calls"] == 1

    def test_hard_cap_propagates_past_retry(self):
        b = TokenBudget(hard_token_cap=100)
        gw = CostGateway(
            StubLLM(model="m", max_retries=3),
            budget=b,
            stage_name="gen",
            report=CostReport(),
        )
        # First call records 70 tokens, second pushes to 140 > 100 → halt
        gw.generate([{"role": "user", "content": "q1"}])
        with pytest.raises(BudgetExceeded):
            gw.generate([{"role": "user", "content": "q2"}])

    def test_records_carry_stage_and_model(self):
        b = TokenBudget()
        report = CostReport()
        gw = CostGateway(StubLLM(model="custom-m", max_retries=1), b, "qa", report)
        gw.generate([{"role": "user", "content": "q"}])
        r = report.records[0]
        assert r.stage == "qa"
        assert r.model == "custom-m"
        assert r.total_tokens == 70


# ───────────────────────────────────────────────────────────────────────────
# CostReport
# ───────────────────────────────────────────────────────────────────────────


class TestCostReport:
    @pytest.fixture
    def report(self):
        r = CostReport()
        r.add(
            CallRecord(
                stage="gen",
                model="m1",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cost_usd=0.01,
                latency_seconds=0.5,
            )
        )
        r.add(
            CallRecord(
                stage="gen",
                model="m1",
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
                cost_usd=0.02,
                latency_seconds=0.6,
            )
        )
        r.add(
            CallRecord(
                stage="judge",
                model="m2",
                prompt_tokens=5,
                completion_tokens=2,
                total_tokens=7,
                cost_usd=0.001,
                latency_seconds=0.1,
            )
        )
        return r

    def test_totals(self, report):
        t = report.totals()
        assert t["calls"] == 3
        assert t["total_tokens"] == 52
        assert t["cost_usd"] == pytest.approx(0.031)

    def test_per_stage(self, report):
        per = report.per_stage()
        assert per["gen"]["calls"] == 2
        assert per["gen"]["total_tokens"] == 45
        assert per["judge"]["calls"] == 1

    def test_per_model(self, report):
        per = report.per_model()
        assert per["m1"]["calls"] == 2
        assert per["m2"]["calls"] == 1

    def test_halted_state(self, report):
        report.mark_halted("budget cap hit")
        assert report.halted is True
        d = report.to_dict()
        assert d["halted"] is True
        assert d["halt_reason"] == "budget cap hit"

    def test_render_markdown_has_tables(self, report):
        md = report.render_markdown()
        assert "Per-stage Attribution" in md
        assert "Per-model Attribution" in md
        assert "$0.0310" in md

    def test_write(self, report, tmp_path):
        p = report.write(tmp_path)
        assert p.exists()
        assert p.read_text().startswith("# Cost Attribution Report")


# ───────────────────────────────────────────────────────────────────────────
# BillOfMaterials
# ───────────────────────────────────────────────────────────────────────────


class TestBillOfMaterials:
    @pytest.fixture
    def populated_dir(self, tmp_path):
        src = tmp_path / "src.jsonl"
        src.write_text('{"a":1}\n')
        (tmp_path / "sft_alpaca.jsonl").write_text('{"x":1}\n')
        (tmp_path / "rejected.jsonl").write_text("")
        (tmp_path / "dataset_card.md").write_text("# card")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"pipeline_config_hash": "abc123"}))
        return tmp_path, src, manifest

    def test_build_hashes_everything(self, populated_dir):
        out, src, manifest = populated_dir
        bom = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc123",
        )
        assert len(bom.sources) == 1
        assert len(bom.outputs) >= 2  # sft_alpaca + rejected + dataset_card
        assert len(bom.manifest_sha256) == 64
        assert len(bom.bom_signature) == 64

    def test_write_then_load_roundtrip(self, populated_dir):
        out, src, manifest = populated_dir
        bom = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc123",
        )
        path = bom.write(out)
        loaded = BillOfMaterials.load(path)
        assert loaded.bom_signature == bom.bom_signature
        assert len(loaded.sources) == len(bom.sources)
        assert len(loaded.outputs) == len(bom.outputs)

    def test_verify_clean(self, populated_dir):
        out, src, manifest = populated_dir
        bom = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc123",
        )
        v = bom.verify(manifest, out)
        assert v["all_ok"] is True
        assert v["signature_ok"] is True
        assert v["manifest_ok"] is True

    def test_verify_detects_output_tamper(self, populated_dir):
        out, src, manifest = populated_dir
        bom = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc123",
        )
        # Tamper with an output
        (out / "sft_alpaca.jsonl").write_text('{"x":999}\n')
        v = bom.verify(manifest, out)
        assert v["all_ok"] is False
        assert "sft_alpaca.jsonl" in v["outputs_ok"]
        assert v["outputs_ok"]["sft_alpaca.jsonl"] is False

    def test_verify_detects_source_tamper(self, populated_dir):
        out, src, manifest = populated_dir
        bom = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc123",
        )
        src.write_text('{"a":999}\n')
        v = bom.verify(manifest, out)
        assert v["all_ok"] is False
        assert v["sources_ok"][str(src)] is False

    def test_verify_detects_manifest_tamper(self, populated_dir):
        out, src, manifest = populated_dir
        bom = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc123",
        )
        manifest.write_text('{"tampered":true}')
        v = bom.verify(manifest, out)
        assert v["all_ok"] is False
        assert v["manifest_ok"] is False

    def test_verify_detects_signature_tamper(self, populated_dir):
        out, src, manifest = populated_dir
        bom = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc123",
        )
        # Pretend an attacker edited the BoM body without re-signing
        bom.outputs[0].sha256 = "0" * 64
        v = bom.verify(manifest, out)
        assert v["signature_ok"] is False
        assert v["all_ok"] is False

    def test_signature_recomputes_on_change(self, populated_dir):
        out, src, manifest = populated_dir
        bom1 = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="abc",
        )
        bom2 = BillOfMaterials.build(
            manifest_path=manifest,
            output_dir=out,
            source_files=[src],
            pipeline_config_hash="xyz",
        )
        assert bom1.bom_signature != bom2.bom_signature
