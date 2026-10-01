"""
Tests for MagpieTask, PersonaDrivenTask, JointBundleTask and the Prometheus judge.

Each test uses a scripted stub LLM (no network) so the suite runs offline.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from curatorkit.generators.joint_bundle import JointBundleTask, _parse_score
from curatorkit.generators.magpie import (
    CANONICAL_PRE_QUERY_TEMPLATES,
    MagpieTask,
)
from curatorkit.generators.persona import PersonaDrivenTask
from curatorkit.judges import (
    ABSOLUTE_GRADING_RUBRICS,
    PrometheusJudge,
    PrometheusRewardGate,
)
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class ScriptedLLM(BaseLLM):
    """Returns a queued list of responses in order."""

    def __init__(self, responses: list[str], model: str = "stub"):
        super().__init__(model=model, max_retries=1)
        self.responses = list(responses)
        self.call_count = 0
        # Capture each call's args for assertions about per-call config
        self.calls: list[dict] = []

    def _call(self, messages, **kwargs):
        self.call_count += 1
        self.calls.append({"messages": messages, **kwargs})
        text = self.responses.pop(0) if self.responses else "ok"
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )


# ───────────────────────────────────────────────────────────────────────────
# MagpieTask
# ───────────────────────────────────────────────────────────────────────────


class TestMagpie:
    def test_canonical_templates_include_llama3(self):
        # The paper validates Llama-3 most; the template must be in our registry.
        assert "llama3" in CANONICAL_PRE_QUERY_TEMPLATES
        assert "<|start_header_id|>user<|end_header_id|>" in CANONICAL_PRE_QUERY_TEMPLATES["llama3"]

    def test_cold_start_no_input_required(self):
        llm = ScriptedLLM(
            [
                "Write a Python fibonacci function.",
                "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
            ]
        )
        task = MagpieTask(llm=llm, n_samples=1)
        out = task.run([])  # empty input — cold-start mode
        assert len(out) == 1
        assert out[0].instruction.startswith("Write a Python fibonacci")
        assert out[0].output.startswith("def fib")
        assert out[0].source_uri == "magpie_cold_start"

    def test_two_calls_per_sample(self):
        llm = ScriptedLLM(["instruction text", "response text"])
        task = MagpieTask(llm=llm, n_samples=1)
        task.run([])
        # Magpie is always two calls per produced sample
        assert llm.call_count == 2

    def test_step1_short_filter(self):
        llm = ScriptedLLM(["ok"])  # below skip_short_instructions=10
        task = MagpieTask(llm=llm, n_samples=1, skip_short_instructions=10)
        out = task.run([])
        assert out == []
        rejected = task.flush_rejected()
        assert len(rejected) == 1
        assert "step1_empty_or_short" in rejected[0].rejection_reason

    def test_step2_empty_filter(self):
        llm = ScriptedLLM(["Write a Python program please.", ""])
        task = MagpieTask(llm=llm, n_samples=1, skip_short_instructions=5)
        out = task.run([])
        assert out == []
        assert task.flush_rejected()[0].rejection_reason.endswith("step2_empty")

    def test_step1_strips_wrapping_quotes(self):
        # Some chat models wrap their output in quotes despite instructions.
        llm = ScriptedLLM(['"Write a Python fibonacci function."', "def f(): pass"])
        task = MagpieTask(llm=llm, n_samples=1)
        out = task.run([])
        assert not out[0].instruction.startswith('"')

    def test_raw_prefix_template_routed(self):
        llm = ScriptedLLM(["Q?", "A."])
        task = MagpieTask(llm=llm, n_samples=1, raw_prefix=True, template_id="llama3")
        task.run([])
        # First call should carry the pre-query template as a system message
        first_call_messages = llm.calls[0]["messages"]
        assert first_call_messages[0]["role"] == "system"
        assert "<|start_header_id|>user<|end_header_id|>" in first_call_messages[0]["content"]

    def test_async_path(self):
        llm = ScriptedLLM(
            [
                "Q1",
                "A1",
                "Q2",
                "A2",
            ]
        )
        task = MagpieTask(llm=llm, n_samples=2, skip_short_instructions=1, concurrency=2)
        out = asyncio.run(task.run_async([]))
        assert len(out) == 2

    def test_provenance_records_two_calls(self):
        llm = ScriptedLLM(["Write a Python fibonacci function.", "def f(): pass"])
        task = MagpieTask(llm=llm, n_samples=1)
        out = task.run([])
        rec = next(r for r in out[0].provenance_chain if r.step_name == "MagpieTask")
        assert rec.notes["total_llm_calls"] == 2


# ───────────────────────────────────────────────────────────────────────────
# PersonaDrivenTask
# ───────────────────────────────────────────────────────────────────────────


class TestPersona:
    def test_instruction_two_pass(self):
        llm = ScriptedLLM(
            [
                "User prompt: How do I balance H2 + O2?",
                "Step 1: Count atoms...",
            ]
        )
        task = PersonaDrivenTask(llm=llm, style="instruction")
        sample = DataSample(source_uri="persona_hub", instruction="a chemistry teacher")
        out = task.run([sample])
        assert len(out) == 1
        # "User prompt:" prefix is stripped
        assert not out[0].instruction.startswith("User prompt:")
        assert out[0].instruction.startswith("How do I balance")
        assert out[0].output.startswith("Step 1")
        assert out[0].metadata["two_pass"] is True

    def test_single_pass_no_answer(self):
        llm = ScriptedLLM(["User prompt: solve fizzbuzz"])
        task = PersonaDrivenTask(llm=llm, style="instruction", generate_answer=False)
        out = task.run([DataSample(source_uri="x", instruction="a developer")])
        assert out[0].output == ""
        assert out[0].metadata["two_pass"] is False
        # Only ONE call made
        assert llm.call_count == 1

    def test_math_style_prefix_strip(self):
        llm = ScriptedLLM(
            [
                "Math problem: If 3x + 5 = 20, find x.",
                "x = 5",
            ]
        )
        task = PersonaDrivenTask(llm=llm, style="math")
        out = task.run([DataSample(source_uri="x", instruction="a math tutor")])
        assert out[0].instruction.startswith("If 3x")
        assert out[0].metadata["style"] == "math"

    def test_knowledge_style_passage_in_output(self):
        llm = ScriptedLLM(["Passage: Stoichiometry governs ratios..."])
        task = PersonaDrivenTask(llm=llm, style="knowledge", generate_answer=False)
        out = task.run([DataSample(source_uri="x", instruction="a chemist")])
        assert out[0].instruction == ""
        assert out[0].output.startswith("Stoichiometry")
        assert out[0].task_type == "language_modeling"

    def test_missing_persona_rejected(self):
        task = PersonaDrivenTask(llm=ScriptedLLM([]), style="instruction")
        out = task.run([DataSample(source_uri="x", instruction="")])
        assert out == []
        assert task.flush_rejected()[0].rejection_reason == "persona_missing"

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="style must be"):
            PersonaDrivenTask(llm=ScriptedLLM([]), style="not_a_style")

    def test_metadata_persona_field(self):
        # Read persona from metadata.persona instead of instruction
        llm = ScriptedLLM(["User prompt: my question", "answer"])
        task = PersonaDrivenTask(
            llm=llm,
            style="instruction",
            persona_field="metadata.persona",
        )
        s = DataSample(
            source_uri="x",
            instruction="ignored",
            metadata={"persona": "a marine biologist"},
        )
        out = task.run([s])
        assert len(out) == 1
        assert "marine biologist" in llm.calls[0]["messages"][0]["content"]


# ───────────────────────────────────────────────────────────────────────────
# JointBundleTask
# ───────────────────────────────────────────────────────────────────────────


class TestJointBundle:
    BUNDLE_RESPONSE = json.dumps(
        {
            "instruction": "What is photosynthesis?",
            "chosen": "Photosynthesis is the process by which plants convert light into glucose.",
            "rejected": "Photosynthesis is how plants eat sunlight.",
            "rollouts": ["A1", "A2", "A3", "A4"],
        }
    )

    def test_emits_three_linked_samples(self):
        llm = ScriptedLLM(
            [self.BUNDLE_RESPONSE] + ["0.9", "0.8", "0.6", "0.7"],
        )
        task = JointBundleTask(llm=llm, num_rollouts=4, score_rollouts=True)
        src = DataSample(
            source_uri="bio.pdf",
            output="Plants use light for energy.",
            task_type="language_modeling",
        )
        out = task.run([src])
        assert len(out) == 3
        roles = {s.metadata["role_in_bundle"] for s in out}
        assert roles == {"sft", "dpo", "grpo"}

    def test_shared_bundle_id(self):
        llm = ScriptedLLM(
            [self.BUNDLE_RESPONSE, "0.5", "0.5", "0.5", "0.5"],
        )
        task = JointBundleTask(llm=llm, num_rollouts=4)
        out = task.run([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        bundle_ids = {s.metadata["bundle_id"] for s in out}
        assert len(bundle_ids) == 1

    def test_dpo_carries_chosen_rejected(self):
        llm = ScriptedLLM([self.BUNDLE_RESPONSE, "0.9", "0.5", "0.5", "0.5"])
        task = JointBundleTask(llm=llm, num_rollouts=4)
        out = task.run([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        dpo = next(s for s in out if s.metadata["role_in_bundle"] == "dpo")
        assert dpo.chosen and dpo.rejected
        assert dpo.task_type == "preference"

    def test_grpo_has_responses_and_scores(self):
        llm = ScriptedLLM([self.BUNDLE_RESPONSE, "1.0", "0.5", "0.3", "0.1"])
        task = JointBundleTask(llm=llm, num_rollouts=4)
        out = task.run([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        grpo = next(s for s in out if s.metadata["role_in_bundle"] == "grpo")
        assert len(grpo.responses) == 4
        assert grpo.reward_scores == [1.0, 0.5, 0.3, 0.1]

    def test_score_rollouts_false_skips_scoring_calls(self):
        llm = ScriptedLLM([self.BUNDLE_RESPONSE])
        task = JointBundleTask(llm=llm, num_rollouts=4, score_rollouts=False)
        out = task.run([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        # 3 samples emitted, but only 1 LLM call total (no scoring pass)
        assert len(out) == 3
        assert llm.call_count == 1

    def test_parse_error_rejected(self):
        llm = ScriptedLLM(["this isn't valid JSON"])
        task = JointBundleTask(llm=llm, num_rollouts=4, score_rollouts=False)
        out = task.run([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        assert out == []
        rejected = task.flush_rejected()
        assert len(rejected) == 1
        assert "parse_error" in rejected[0].rejection_reason

    def test_missing_required_fields_rejected(self):
        bad_bundle = json.dumps({"instruction": "Q", "chosen": "A"})  # missing rejected + rollouts
        llm = ScriptedLLM([bad_bundle])
        task = JointBundleTask(llm=llm, num_rollouts=4, score_rollouts=False)
        out = task.run([DataSample(source_uri="x", output="src", task_type="language_modeling")])
        assert out == []
        assert task.flush_rejected()

    def test_score_parser_handles_scales(self):
        assert _parse_score("0.85") == 0.85
        assert _parse_score("8.5") == 0.85  # 0-10 → 0-1
        assert _parse_score("Score: 0.7 thanks") == 0.7
        assert _parse_score("") == 0.5  # default

    def test_async_path(self):
        async def _run():
            llm = ScriptedLLM([self.BUNDLE_RESPONSE, "0.9", "0.5", "0.5", "0.5"])
            task = JointBundleTask(llm=llm, num_rollouts=4, score_rollouts=True, concurrency=2)
            return await task.run_async(
                [
                    DataSample(source_uri="x", output="src", task_type="language_modeling"),
                ]
            )

        out = asyncio.run(_run())
        assert len(out) == 3


# ───────────────────────────────────────────────────────────────────────────
# PrometheusJudge
# ───────────────────────────────────────────────────────────────────────────


class TestPrometheus:
    def test_all_four_canonical_rubrics_present(self):
        # The paper validates these dimensions; we ship all four.
        assert set(ABSOLUTE_GRADING_RUBRICS) == {
            "helpfulness",
            "factuality",
            "honesty",
            "harmlessness",
        }

    def test_rubric_renders_all_five_descriptions(self):
        r = ABSOLUTE_GRADING_RUBRICS["helpfulness"]
        rendered = r.render()
        for level in ("Score 1:", "Score 2:", "Score 3:", "Score 4:", "Score 5:"):
            assert level in rendered

    def test_canonical_format_parsed(self):
        llm = ScriptedLLM(["Feedback: Good answer. [RESULT] 4"])
        judge = PrometheusJudge(llm=llm, rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"])
        r = judge.judge("Q?", "A.")
        assert r.score_1_to_5 == 4
        assert r.score_normalized == 0.75
        assert "Good answer" in r.feedback

    def test_score_out_of_range_clamped(self):
        # Even if the model returns [RESULT] 9, we clamp
        llm = ScriptedLLM(["Feedback: x [RESULT] 9"])
        judge = PrometheusJudge(llm=llm, rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"])
        r = judge.judge("Q?", "A.")
        # 9 isn't a valid digit in [1-5] regex → fallback to trailing-int → 9
        # → fall through to default integer in last regex match → score=9→clamp
        # Actually 9 doesn't match [1-5] so it falls back to trailing match
        # which is the same regex. The output will be the default of 3 because
        # no [1-5] regex matched anywhere. Either way, must be in 1..5.
        assert 1 <= r.score_1_to_5 <= 5

    def test_fallback_to_trailing_integer(self):
        llm = ScriptedLLM(["Pretty good overall. 3"])
        judge = PrometheusJudge(llm=llm, rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"])
        r = judge.judge("Q?", "A.")
        assert r.score_1_to_5 == 3

    def test_default_for_malformed(self):
        llm = ScriptedLLM(["Nothing relevant here."])
        judge = PrometheusJudge(llm=llm, rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"])
        r = judge.judge("Q?", "A.")
        assert r.score_1_to_5 == 3  # neutral default

    def test_reference_answer_included_in_prompt(self):
        llm = ScriptedLLM(["Feedback: x [RESULT] 5"])
        judge = PrometheusJudge(
            llm=llm,
            rubric=ABSOLUTE_GRADING_RUBRICS["factuality"],
            reference_answer="The correct answer is 42.",
        )
        judge.judge("Q?", "A.")
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "Reference Answer" in prompt
        assert "The correct answer is 42." in prompt


# ───────────────────────────────────────────────────────────────────────────
# PrometheusRewardGate
# ───────────────────────────────────────────────────────────────────────────


class TestPrometheusRewardGate:
    def test_pass_above_threshold(self):
        llm = ScriptedLLM(["Feedback: Great. [RESULT] 5"])
        judge = PrometheusJudge(llm=llm, rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"])
        gate = PrometheusRewardGate(judge=judge, threshold=0.6)
        s = DataSample(source_uri="x", instruction="q", output="a")
        p, r = gate.run([s])
        assert len(p) == 1 and not r
        assert p[0].label == 1.0  # 5/5 normalized

    def test_reject_below_threshold(self):
        llm = ScriptedLLM(["Feedback: Poor. [RESULT] 2"])
        judge = PrometheusJudge(llm=llm, rubric=ABSOLUTE_GRADING_RUBRICS["factuality"])
        gate = PrometheusRewardGate(judge=judge, threshold=0.6)
        s = DataSample(source_uri="x", instruction="q", output="a")
        p, r = gate.run([s])
        assert not p and len(r) == 1
        assert "prometheus_below_threshold" in r[0].rejection_reason

    def test_no_response_passes_through(self):
        # No output, chosen, or responses → skip with audit, don't reject
        llm = ScriptedLLM([])
        judge = PrometheusJudge(llm=llm, rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"])
        gate = PrometheusRewardGate(judge=judge)
        s = DataSample(source_uri="x", instruction="q", output="")
        p, r = gate.run([s])
        assert len(p) == 1 and not r
        rec = next(rec for rec in p[0].provenance_chain if rec.step_name == "PrometheusRewardGate")
        assert rec.notes["skipped"] is True

    def test_llm_failure_passes_with_warning(self):
        # On judge failure, sample passes through with a passed_on_error audit
        class FailLLM(BaseLLM):
            def _call(self, messages, **kw):
                raise RuntimeError("oops")

        judge = PrometheusJudge(
            llm=FailLLM(model="fail", max_retries=1),
            rubric=ABSOLUTE_GRADING_RUBRICS["helpfulness"],
        )
        gate = PrometheusRewardGate(judge=judge)
        s = DataSample(source_uri="x", instruction="q", output="a long enough output")
        p, r = gate.run([s])
        assert len(p) == 1
        rec = next(rec for rec in p[0].provenance_chain if rec.step_name == "PrometheusRewardGate")
        assert rec.notes.get("passed_on_error") is True
