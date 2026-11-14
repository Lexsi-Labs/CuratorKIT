"""
Two-pass preference and adversarial preference temperatures are settings, not hard-coded numbers,
and every sample built from several LLM calls says how it was made.

Before: `PreferenceGenerationTask(mode="two_pass")` always sampled the rejected answer at
`temperature + 0.3` and `AdversarialPreferenceTask` at 0.8 / 0.9, with no way to change them;
no sample recorded the temperatures; and two_pass, turn-by-turn multi-turn, GRPO and
evolved-answer samples carried no generator provenance record at all (a single_call sample has
one with the model, token counts and prompt hash).
"""

from __future__ import annotations

import asyncio

import pytest

from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask
from curatorkit.generators.evol_instruct import EvolInstructTask
from curatorkit.generators.grpo_rollout import GRPORolloutTask
from curatorkit.generators.multiturn_gen import MultiTurnTask
from curatorkit.generators.preference_gen import PreferenceGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample

_FAITHFUL = '[{"question": "Q1?", "answer": "A1."}]'
_EVOLVED = '{"evolved_instruction": "Explain X in depth", "strategy_applied": "s"}'


class _RecordingLLM(BaseLLM):
    """Records the temperature of every call; answers with fixed token counts."""

    def __init__(self, temperature=0.7, text="an answer"):
        super().__init__(model="fake/rec", temperature=temperature, max_retries=0)
        self.text = text
        self.temps: list[float] = []

    def _call(self, messages, **kwargs):
        self.temps.append(kwargs["temperature"])
        text = self.text
        content = messages[0]["content"]
        if "adversarial_preference" in self.text:
            text = _FAITHFUL
        if "evolved_instruction" in content:
            text = _EVOLVED
        return LLMResponse(
            text=text, model=self.model, prompt_tokens=10, completion_tokens=5, total_tokens=15
        )


def _instr() -> DataSample:
    return DataSample(source_uri="t://", instruction="q", task_type="instruction_following")


def _passage() -> DataSample:
    return DataSample(source_uri="t://", output="Passage: fact.", task_type="language_modeling")


def _record(ds: DataSample, step: str):
    return [p for p in ds.provenance_chain if p.step_name == step][-1]


# ── PreferenceGenerationTask two_pass ────────────────────────────────────────


@pytest.mark.parametrize(
    "base, offset, expected_rejected",
    [(0.7, None, 1.0), (0.7, 0.0, 0.7), (0.7, 0.5, 1.2), (0.7, 2.0, 1.5), (0.2, -0.5, 0.0)],
)
def test_two_pass_rejected_temperature_follows_the_offset(base, offset, expected_rejected):
    llm = _RecordingLLM(temperature=base)
    kw = {} if offset is None else {"rejected_temperature_offset": offset}
    task = PreferenceGenerationTask(llm=llm, mode="two_pass", **kw)
    [ds] = task.run([_instr()])
    assert llm.temps == [pytest.approx(base), pytest.approx(expected_rejected)]
    assert ds.metadata["chosen_temperature"] == pytest.approx(base)
    assert ds.metadata["rejected_temperature"] == pytest.approx(expected_rejected)


def test_two_pass_async_uses_the_offset_and_records_temperatures():
    llm = _RecordingLLM(temperature=0.7)
    task = PreferenceGenerationTask(llm=llm, mode="two_pass", rejected_temperature_offset=0.0)
    [ds] = asyncio.run(task.run_async([_instr()]))
    assert llm.temps == [0.7, 0.7]
    assert (ds.metadata["chosen_temperature"], ds.metadata["rejected_temperature"]) == (0.7, 0.7)


def test_two_pass_sample_has_a_generation_provenance_record():
    llm = _RecordingLLM(temperature=0.7)
    [ds] = PreferenceGenerationTask(llm=llm, mode="two_pass").run([_instr()])
    notes = _record(ds, "PreferenceGenerationTask").notes
    assert notes["llm_model"] == "fake/rec"
    assert notes["llm_calls"] == 2
    assert notes["total_tokens"] == 30  # both calls
    assert notes["chosen_temperature"] == pytest.approx(0.7)
    assert notes["rejected_temperature"] == pytest.approx(1.0)
    assert notes["rejected_temperature_offset"] == 0.3
    assert notes["source_sample_id"] == ds.metadata["source_sample_id"]
    assert len(notes["prompt_hash"]) == 12


def test_two_pass_corpus_mode_is_one_call_without_an_offset():
    corpus = '{"question": "Q?", "chosen": "good", "rejected": "meh"}'
    llm = _RecordingLLM(temperature=0.7, text=corpus)
    [ds] = PreferenceGenerationTask(llm=llm, mode="two_pass").run(
        [DataSample(source_uri="t://", output="A passage.", task_type="language_modeling")]
    )
    assert llm.temps == [0.7]
    assert ds.metadata["chosen_temperature"] == ds.metadata["rejected_temperature"] == 0.7
    assert _record(ds, "PreferenceGenerationTask").notes["llm_calls"] == 1


# ── AdversarialPreferenceTask ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "injection_rate, kw, expected",
    [
        (1.0, {}, 0.8),
        (0.0, {}, 0.9),
        (1.0, {"adversarial_temperature": 0.2}, 0.2),
        (0.0, {"naive_rejected_temperature": 0.0}, 0.0),
    ],
)
def test_adversarial_preference_rejected_temperatures_are_settings(injection_rate, kw, expected):
    llm = _RecordingLLM(temperature=0.7, text="adversarial_preference")
    task = AdversarialPreferenceTask(
        llm=llm, injection_rate=injection_rate, injection_types=["contradicts_source"], **kw
    )
    [ds] = task.run([_passage()])
    assert llm.temps == [0.7, pytest.approx(expected)]
    assert ds.metadata["chosen_temperature"] == 0.7
    assert ds.metadata["rejected_temperature"] == pytest.approx(expected)
    assert any(
        p.notes.get("rejected_temperature") == expected
        for p in ds.provenance_chain
        if p.step_name == "AdversarialPreferenceTask"
    )


def test_adversarial_preference_async_uses_the_settings():
    llm = _RecordingLLM(temperature=0.7, text="adversarial_preference")
    task = AdversarialPreferenceTask(
        llm=llm, injection_rate=1.0, injection_types=["contradicts_source"],
        adversarial_temperature=0.1,
    )
    asyncio.run(task.run_async([_passage()]))
    assert llm.temps == [0.7, 0.1]


# ── provenance on the other multi-call paths ─────────────────────────────────


def test_turn_by_turn_multiturn_sample_has_a_provenance_record():
    llm = _RecordingLLM(temperature=0.7)
    task = MultiTurnTask(llm=llm, num_turns=3)
    [ds] = task.run([_instr()])
    notes = _record(ds, "MultiTurnTask").notes
    assert notes["llm_calls"] == len(llm.temps) >= 2
    assert notes["temperatures_used"] == llm.temps
    assert notes["total_tokens"] == 15 * len(llm.temps)


def test_turn_by_turn_multiturn_async_sample_has_a_provenance_record():
    llm = _RecordingLLM(temperature=0.7)
    [ds] = asyncio.run(MultiTurnTask(llm=llm, num_turns=3).run_async([_instr()]))
    assert _record(ds, "MultiTurnTask").notes["llm_calls"] == len(llm.temps)


def test_grpo_rollout_sample_has_a_provenance_record():
    llm = _RecordingLLM(temperature=0.7)
    task = GRPORolloutTask(llm=llm, num_responses=3, score_responses=False, temperature_spread=0.4)
    [ds] = task.run([_instr()])
    notes = _record(ds, "GRPORolloutTask").notes
    assert notes["llm_calls"] == 3
    assert notes["temperatures_used"] == ds.metadata["temperatures_used"] == llm.temps
    assert notes["total_tokens"] == 45


def test_grpo_rollout_async_sample_has_a_provenance_record():
    llm = _RecordingLLM(temperature=0.7)
    task = GRPORolloutTask(llm=llm, num_responses=2, score_responses=False)
    [ds] = asyncio.run(task.run_async([_instr()]))
    assert _record(ds, "GRPORolloutTask").notes["llm_calls"] == 2


def test_evolved_answer_pass_adds_its_own_record():
    llm = _RecordingLLM(temperature=0.7)
    [ds] = EvolInstructTask(llm=llm, num_evolutions=1, generate_answers=True).run([_instr()])
    records = [p for p in ds.provenance_chain if p.step_name == "EvolInstructTask"]
    assert [r.notes.get("stage") for r in records] == [None, "answer"]
    assert records[1].notes["llm_calls"] == 1
    assert records[1].notes["temperatures_used"] == [0.7]


# ── configuration reaches the tasks ──────────────────────────────────────────


def test_curator_config_reaches_the_tasks(monkeypatch):
    from curatorkit.curator import Curator, CuratorConfig

    monkeypatch.setattr(Curator, "_build_gen_llm", lambda self, override: _RecordingLLM())
    cfg = CuratorConfig(dataset="d.jsonl", generation_task="preference", preference_mode="two_pass")
    assert Curator(cfg)._build_generator().rejected_temperature_offset == 0.3
    cfg = CuratorConfig(
        dataset="d.jsonl", generation_task="preference", preference_mode="two_pass",
        rejected_temperature_offset=0.0,
    )
    assert Curator(cfg)._build_generator().rejected_temperature_offset == 0.0

    cfg = CuratorConfig(dataset="d.jsonl", generation_task="adversarial_preference")
    task = Curator(cfg)._build_generator()
    assert (task.adversarial_temperature, task.naive_rejected_temperature) == (0.8, 0.9)
    cfg = CuratorConfig(
        dataset="d.jsonl", generation_task="adversarial_preference",
        adversarial_temperature=0.3, naive_rejected_temperature=0.4,
    )
    task = Curator(cfg)._build_generator()
    assert (task.adversarial_temperature, task.naive_rejected_temperature) == (0.3, 0.4)


def test_yaml_generator_config_defaults_and_overrides():
    from curatorkit.config import GenerationConfig

    g = GenerationConfig(type="preference")
    assert (g.rejected_temperature_offset, g.adversarial_temperature, g.naive_rejected_temperature) == (0.3, 0.8, 0.9)
    g = GenerationConfig(type="preference", rejected_temperature_offset=0.0)
    assert g.rejected_temperature_offset == 0.0
