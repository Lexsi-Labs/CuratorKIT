"""
Tests for the five SOTA modules added in v0.4.0:

  SelfRewardingTask              (Yuan et al., ICML 2024)
  ConstitutionalRevisionTask     (Bai et al., 2022)
  ConstitutionalPreferenceTask   (Bai et al., 2022 — DPO pair variant)
  DecontaminationGate            (GPT-3 §4.0 13-gram protocol)
  FActScoreJudge / FActScoreGate (Min et al., EMNLP 2023)
  BestOfNTask                    (Tulu 3 / Stiennon et al.)

Every test uses a scripted stub LLM so the suite runs fully offline.
"""

from __future__ import annotations

import json

import pytest

from curatorkit.gates.decontamination import (
    DecontaminationGate,
    _ngrams,
    _tokenize,
)
from curatorkit.generators.best_of_n import (
    BestOfNTask,
    make_length_scorer,
)
from curatorkit.generators.constitutional import (
    CANONICAL_CONSTITUTION,
    ConstitutionalPreferenceTask,
    ConstitutionalRevisionTask,
)
from curatorkit.generators.self_rewarding import (
    SelfRewardingTask,
    _parse_score,
)
from curatorkit.judges.factscore import (
    FActScoreGate,
    FActScoreJudge,
)
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class ScriptedLLM(BaseLLM):
    """Returns a queued list of responses in order. Records every call."""

    def __init__(self, responses: list[str]):
        super().__init__(model="stub", max_retries=1)
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _call(self, messages, **kw):
        text = self.responses.pop(0) if self.responses else "ok"
        self.calls.append({"messages": messages, **kw})
        return LLMResponse(text=text, model=self.model)


# ───────────────────────────────────────────────────────────────────────────
# Self-Rewarding (Yuan et al. ICML 2024)
# ───────────────────────────────────────────────────────────────────────────


class TestSelfRewarding:
    def test_score_parser_canonical(self):
        # Paper's canonical format: "Score: <total points>"
        assert _parse_score("Score: 4") == 4
        assert _parse_score("Some justification.\nScore: 5") == 5

    def test_score_parser_clamps_high_values(self):
        # Additive 5-point rubric in paper; anything above 5 is clamped down.
        assert _parse_score("Score: 9") == 5
        assert _parse_score("Score: 100") == 5

    def test_score_parser_trailing_int_fallback(self):
        assert _parse_score("That's a 3") == 3

    def test_score_parser_default_for_garbage(self):
        assert _parse_score("no score here") == 0

    def test_emits_dpo_pair_with_chosen_lowest(self):
        # 4 candidates × 3 judge calls per = 12 + 4 = 16 LLM calls
        responses = [f"candidate{i}" for i in range(4)]
        judge_scores = [
            "Score: 5",
            "Score: 5",
            "Score: 5",  # cand 0 → avg 5
            "Score: 1",
            "Score: 1",
            "Score: 2",  # cand 1 → avg 1
            "Score: 3",
            "Score: 3",
            "Score: 3",  # cand 2 → avg 3
            "Score: 4",
            "Score: 4",
            "Score: 4",  # cand 3 → avg 4
        ]
        llm = ScriptedLLM(responses + judge_scores)
        task = SelfRewardingTask(llm=llm, n_candidates=4, k_judge_samples=3)
        out = task.run([DataSample(source_uri="x", instruction="Q?")])
        assert len(out) == 1
        assert out[0].task_type == "preference"
        # candidate0 is highest (5), candidate1 is lowest (1)
        assert out[0].chosen == "candidate0"
        assert out[0].rejected == "candidate1"
        assert out[0].metadata["chosen_score"] == 5
        assert out[0].metadata["rejected_score"] == 1

    def test_tie_rejected_when_no_score_delta(self):
        # All 4 candidates score identically
        llm = ScriptedLLM(
            ["c0", "c1", "c2", "c3"] + ["Score: 3"] * 12  # 4 candidates × 3 judge calls
        )
        task = SelfRewardingTask(llm=llm, n_candidates=4, k_judge_samples=3)
        out = task.run([DataSample(source_uri="x", instruction="Q?")])
        assert out == []
        rejected = task.flush_rejected()
        assert len(rejected) == 1
        assert "score_tie" in rejected[0].rejection_reason

    def test_missing_instruction_rejected(self):
        task = SelfRewardingTask(llm=ScriptedLLM([]))
        out = task.run([DataSample(source_uri="x")])
        assert out == []
        assert task.flush_rejected()[0].rejection_reason == "self_rewarding:no_instruction"

    def test_label_normalised_to_0_1(self):
        responses = ["a", "b", "c", "d"]
        judge_scores = ["Score: 5"] * 3 + ["Score: 1"] * 3 + ["Score: 3"] * 3 + ["Score: 3"] * 3
        llm = ScriptedLLM(responses + judge_scores)
        task = SelfRewardingTask(llm=llm, n_candidates=4, k_judge_samples=3)
        out = task.run([DataSample(source_uri="x", instruction="Q?")])
        assert out[0].label == 1.0  # 5/5


# ───────────────────────────────────────────────────────────────────────────
# Constitutional AI (Bai et al. 2022)
# ───────────────────────────────────────────────────────────────────────────


class TestConstitutional:
    def test_canonical_constitution_has_five_principles(self):
        assert len(CANONICAL_CONSTITUTION) == 5
        names = {p.name for p in CANONICAL_CONSTITUTION}
        assert {"harmlessness", "honesty", "helpfulness", "non_evasion", "privacy"} == names

    def test_principle_dataclass_is_frozen(self):
        p = CANONICAL_CONSTITUTION[0]
        with pytest.raises(Exception):
            p.name = "changed"  # frozen dataclass

    def test_critique_revise_loop_applies_revision(self):
        llm = ScriptedLLM(
            [
                "Critique: response is too verbose.",  # critique
                "Concise revised response.",  # revision
            ]
        )
        task = ConstitutionalRevisionTask(
            llm=llm,
            principles=CANONICAL_CONSTITUTION[:1],
        )
        s = DataSample(source_uri="x", instruction="Q", output="Long initial response.")
        out = task.run([s])
        assert len(out) == 1
        assert out[0].output == "Concise revised response."
        assert out[0].metadata["revisions_applied"] == 1
        assert out[0].metadata["original_response"] == "Long initial response."

    def test_no_revision_token_skips_revision(self):
        llm = ScriptedLLM(["NO_REVISION_NEEDED"])
        task = ConstitutionalRevisionTask(
            llm=llm,
            principles=CANONICAL_CONSTITUTION[:1],
        )
        out = task.run([DataSample(source_uri="x", instruction="Q", output="Already good.")])
        assert out[0].output == "Already good."
        assert out[0].metadata["revisions_applied"] == 0

    def test_multiple_principles_applied_in_order(self):
        # 2 principles → 2 critiques + 2 revisions (each principle revises)
        llm = ScriptedLLM(
            [
                "Has harm.",
                "Step 1 revision.",
                "Has dishonesty.",
                "Step 2 revision.",
            ]
        )
        task = ConstitutionalRevisionTask(
            llm=llm,
            principles=CANONICAL_CONSTITUTION[:2],
        )
        out = task.run([DataSample(source_uri="x", instruction="Q", output="orig")])
        assert out[0].metadata["revisions_applied"] == 2
        assert out[0].output == "Step 2 revision."
        trace = out[0].metadata["revision_trace"]
        assert len(trace) == 2
        assert trace[0]["principle"] == "harmlessness"
        assert trace[1]["principle"] == "honesty"

    def test_early_stop_on_no_revision(self):
        llm = ScriptedLLM(["NO_REVISION_NEEDED"])  # only one principle's critique
        task = ConstitutionalRevisionTask(
            llm=llm,
            principles=CANONICAL_CONSTITUTION[:3],
            early_stop_on_no_revision=True,
        )
        out = task.run([DataSample(source_uri="x", instruction="Q", output="good")])
        # Only the first principle is evaluated; loop stops
        assert len(out[0].metadata["revision_trace"]) == 1

    def test_preference_task_emits_dpo_pair(self):
        llm = ScriptedLLM(["Has issues.", "Better response."])
        task = ConstitutionalPreferenceTask(
            llm=llm,
            principles=CANONICAL_CONSTITUTION[:1],
        )
        out = task.run([DataSample(source_uri="x", instruction="Q", output="Original.")])
        assert len(out) == 1
        assert out[0].task_type == "preference"
        assert out[0].chosen == "Better response."
        assert out[0].rejected == "Original."

    def test_preference_task_rejects_when_no_revision(self):
        llm = ScriptedLLM(["NO_REVISION_NEEDED"])
        task = ConstitutionalPreferenceTask(
            llm=llm,
            principles=CANONICAL_CONSTITUTION[:1],
        )
        out = task.run([DataSample(source_uri="x", instruction="Q", output="Already perfect.")])
        assert out == []
        rejected = task.flush_rejected()
        assert len(rejected) == 1
        assert "no_revision_made" in rejected[0].rejection_reason

    def test_initial_generation_when_output_empty(self):
        llm = ScriptedLLM(
            [
                "Initial generated response.",  # initial gen
                "NO_REVISION_NEEDED",  # critique (no revision)
            ]
        )
        task = ConstitutionalRevisionTask(
            llm=llm,
            principles=CANONICAL_CONSTITUTION[:1],
        )
        out = task.run([DataSample(source_uri="x", instruction="Q", output="")])
        assert out[0].output == "Initial generated response."


# ───────────────────────────────────────────────────────────────────────────
# Decontamination (GPT-3 §4.0)
# ───────────────────────────────────────────────────────────────────────────


class TestDecontamination:
    def test_tokenize_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_ngrams_correct_count(self):
        toks = ["a", "b", "c", "d"]
        grams = list(_ngrams(toks, 2))
        assert grams == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_ngrams_empty_when_text_shorter_than_n(self):
        toks = ["a", "b"]
        assert list(_ngrams(toks, 5)) == []

    def test_rejects_exact_match(self, tmp_path):
        shard = tmp_path / "test_shard.jsonl"
        shard.write_text(json.dumps({"question": "Paris is the capital of France"}) + "\n")
        gate = DecontaminationGate(shards={"mmlu": [shard]}, n=5, max_ngram_overlap=0)
        bad = DataSample(
            source_uri="x", instruction="Paris is the capital of France", output="confirmed"
        )
        good = DataSample(
            source_uri="x",
            instruction="A completely unrelated topic about cooking pasta",
            output="ok",
        )
        passed, rejected = gate.run([bad, good])
        assert len(passed) == 1 and len(rejected) == 1
        assert "decontamination:mmlu" in rejected[0].rejection_reason

    def test_text_file_shard_supported(self, tmp_path):
        shard = tmp_path / "test_shard.txt"
        shard.write_text("The quick brown fox jumps over the lazy dog.")
        gate = DecontaminationGate(shards={"corpus": [shard]}, n=5, max_ngram_overlap=0)
        stats = gate.shard_stats()
        assert stats["corpus"]["n_documents"] == 1
        assert stats["corpus"]["n_ngrams"] > 0

    def test_threshold_above_zero_allows_short_overlap(self, tmp_path):
        shard = tmp_path / "shard.jsonl"
        shard.write_text(
            json.dumps(
                {
                    "question": "What is the capital of France",
                }
            )
            + "\n"
        )
        # With max_ngram_overlap=2, a tiny coincidental overlap doesn't trigger
        gate = DecontaminationGate(shards={"s": [shard]}, n=4, max_ngram_overlap=2)
        # "the capital of France" has 1 4-gram from the shard
        sample = DataSample(
            source_uri="x", instruction="In a discussion of the capital of France today", output=""
        )
        passed, rejected = gate.run([sample])
        assert len(passed) == 1  # below threshold

    def test_provenance_attached(self, tmp_path):
        shard = tmp_path / "s.txt"
        shard.write_text("alpha beta gamma delta epsilon")
        gate = DecontaminationGate(shards={"s": [shard]}, n=3, max_ngram_overlap=0)
        sample = DataSample(source_uri="x", instruction="zeta eta theta", output="iota kappa")
        passed, _ = gate.run([sample])
        steps = [r.step_name for r in passed[0].provenance_chain]
        assert "DecontaminationGate" in steps

    def test_missing_shard_file_silently_skipped(self):
        # No exception when a shard path doesn't exist
        gate = DecontaminationGate(
            shards={"bogus": ["nonexistent_file_path.jsonl"]},
            n=5,
        )
        stats = gate.shard_stats()
        assert stats["bogus"]["n_ngrams"] == 0


# ───────────────────────────────────────────────────────────────────────────
# FActScore (Min et al. EMNLP 2023)
# ───────────────────────────────────────────────────────────────────────────


class TestFActScore:
    def test_factscore_one_for_all_supported(self):
        llm = ScriptedLLM(
            [
                "Fact one.\nFact two.",  # decompose
                "SUPPORTED\nDirect support.",  # verify f1
                "SUPPORTED\nDirect support.",  # verify f2
            ]
        )
        judge = FActScoreJudge(llm=llm)
        r = judge.judge("Q", "Some claim.", "Some source containing the claim.")
        assert r.factscore == 1.0
        assert r.n_supported == 2
        assert r.n_unsupported == 0
        assert r.n_contradicted == 0

    def test_factscore_partial(self):
        llm = ScriptedLLM(
            [
                "Fact one.\nFact two.\nFact three.",
                "SUPPORTED\njust.",
                "UNSUPPORTED\nnot in source.",
                "SUPPORTED\njust.",
            ]
        )
        judge = FActScoreJudge(llm=llm)
        r = judge.judge("Q", "claim", "source")
        assert r.factscore == 2 / 3
        assert r.n_supported == 2 and r.n_unsupported == 1

    def test_no_facts_token_returns_empty(self):
        llm = ScriptedLLM(["NO_FACTS"])
        judge = FActScoreJudge(llm=llm)
        r = judge.judge("Q", "I cannot answer.", "src")
        assert len(r.facts) == 0
        assert r.factscore == 1.0  # vacuous truth

    def test_max_facts_caps_decomposition(self):
        # Decompose returns 10 facts, max_facts=3 caps to 3
        llm = ScriptedLLM(
            [
                "\n".join(f"Fact {i}." for i in range(10)),
                "SUPPORTED\n.",
                "SUPPORTED\n.",
                "SUPPORTED\n.",
            ]
        )
        judge = FActScoreJudge(llm=llm, max_facts=3)
        r = judge.judge("Q", "claim", "src")
        assert len(r.facts) == 3

    def test_gate_passes_above_threshold(self):
        llm = ScriptedLLM(
            [
                "Fact a.\nFact b.",
                "SUPPORTED\n.",
                "SUPPORTED\n.",
            ]
        )
        judge = FActScoreJudge(llm=llm)
        gate = FActScoreGate(judge=judge, threshold=0.7)
        sample = DataSample(
            source_uri="x", instruction="Q", input="some source text here", output="claim text"
        )
        passed, rejected = gate.run([sample])
        assert len(passed) == 1 and not rejected
        assert passed[0].label == 1.0

    def test_gate_rejects_below_threshold(self):
        llm = ScriptedLLM(
            [
                "Fact a.\nFact b.",
                "UNSUPPORTED\nnope.",
                "UNSUPPORTED\nnope.",
            ]
        )
        judge = FActScoreJudge(llm=llm)
        gate = FActScoreGate(judge=judge, threshold=0.7)
        sample = DataSample(source_uri="x", instruction="Q", input="source", output="claim")
        passed, rejected = gate.run([sample])
        assert not passed and len(rejected) == 1
        assert "factscore_below_threshold" in rejected[0].rejection_reason

    def test_contradicted_hard_fail(self):
        llm = ScriptedLLM(
            [
                "Fact a.\nFact b.",
                "SUPPORTED\n.",
                "CONTRADICTED\nopposite.",  # one contradiction is hard fail
            ]
        )
        judge = FActScoreJudge(llm=llm)
        gate = FActScoreGate(
            judge=judge,
            threshold=0.4,  # score=0.5 passes
            contradicted_is_hard_fail=True,
        )
        sample = DataSample(source_uri="x", instruction="Q", input="source", output="claim")
        passed, rejected = gate.run([sample])
        # Even though factscore=0.5 > threshold=0.4, contradiction triggers reject
        assert not passed and len(rejected) == 1
        assert "contradicted" in rejected[0].rejection_reason

    def test_missing_source_skipped(self):
        gate = FActScoreGate(judge=FActScoreJudge(llm=ScriptedLLM([])))
        sample = DataSample(source_uri="x", instruction="Q", input="", output="claim")
        passed, _ = gate.run([sample])
        assert len(passed) == 1
        rec = next(r for r in passed[0].provenance_chain if r.step_name == "FActScoreGate")
        assert rec.notes.get("skipped") is True


# ───────────────────────────────────────────────────────────────────────────
# Best-of-N (Tulu 3 / Stiennon et al.)
# ───────────────────────────────────────────────────────────────────────────


class TestBestOfN:
    def test_top_k_1_emits_sft(self):
        llm = ScriptedLLM([f"candidate {i} " * (i + 1) for i in range(4)])
        task = BestOfNTask(
            llm=llm,
            scorer=make_length_scorer(),
            n_candidates=4,
            top_k=1,
        )
        out = task.run([DataSample(source_uri="x", instruction="Q")])
        assert len(out) == 1
        assert out[0].task_type == "instruction_following"
        # Highest-scoring candidate kept
        assert "candidate 3" in out[0].output

    def test_top_k_2_emits_dpo_vs_lowest(self):
        # Space-separated so .split() in the length scorer differentiates.
        llm = ScriptedLLM(
            [
                "c0",
                "c1 c1",
                "c2 c2 c2",
                "c3 c3 c3 c3",
            ]
        )
        task = BestOfNTask(
            llm=llm,
            scorer=make_length_scorer(),
            n_candidates=4,
            top_k=2,
            dpo_mode="vs_lowest",
        )
        out = task.run([DataSample(source_uri="x", instruction="Q")])
        assert out[0].task_type == "preference"
        # Chosen = longest, rejected = shortest
        assert out[0].chosen.startswith("c3")
        assert out[0].rejected == "c0"

    def test_top_k_2_emits_dpo_consecutive(self):
        llm = ScriptedLLM(
            [
                "c0",
                "c1 c1",
                "c2 c2 c2",
                "c3 c3 c3 c3",
            ]
        )
        task = BestOfNTask(
            llm=llm,
            scorer=make_length_scorer(),
            n_candidates=4,
            top_k=2,
            dpo_mode="consecutive",
        )
        out = task.run([DataSample(source_uri="x", instruction="Q")])
        # Rejected = second-highest scoring
        assert out[0].rejected.startswith("c2")

    def test_top_k_greater_than_2_emits_grpo(self):
        llm = ScriptedLLM([f"r{i}" for i in range(8)])
        task = BestOfNTask(
            llm=llm,
            scorer=make_length_scorer(),
            n_candidates=8,
            top_k=4,
        )
        out = task.run([DataSample(source_uri="x", instruction="Q")])
        assert out[0].task_type == "grpo"
        assert len(out[0].responses) == 4
        assert len(out[0].reward_scores) == 4

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            BestOfNTask(
                llm=ScriptedLLM([]),
                scorer=make_length_scorer(),
                n_candidates=4,
                top_k=10,
            )

    def test_invalid_dpo_mode_raises(self):
        with pytest.raises(ValueError, match="dpo_mode"):
            BestOfNTask(
                llm=ScriptedLLM([]),
                scorer=make_length_scorer(),
                n_candidates=4,
                top_k=2,
                dpo_mode="bogus",
            )

    def test_no_candidates_rejected(self):
        class FailLLM(BaseLLM):
            def _call(self, messages, **kw):
                raise RuntimeError("network")

        task = BestOfNTask(
            llm=FailLLM(model="fail", max_retries=1),
            scorer=make_length_scorer(),
            n_candidates=4,
            top_k=1,
        )
        out = task.run([DataSample(source_uri="x", instruction="Q")])
        assert out == []
        rejected = task.flush_rejected()
        assert len(rejected) == 1
        assert "no_candidates_generated" in rejected[0].rejection_reason

    def test_provenance_records_total_calls(self):
        llm = ScriptedLLM([f"c{i}" for i in range(4)])
        task = BestOfNTask(
            llm=llm,
            scorer=make_length_scorer(),
            n_candidates=4,
            top_k=1,
        )
        out = task.run([DataSample(source_uri="x", instruction="Q")])
        rec = next(r for r in out[0].provenance_chain if r.step_name == "BestOfNTask")
        assert rec.notes["n_candidates"] == 4
        assert rec.notes["top_k"] == 1
