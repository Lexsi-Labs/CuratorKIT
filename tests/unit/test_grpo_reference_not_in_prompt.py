"""An instruction row's gold answer (`output`) must never reach the rollout prompt or the
rollout sample's `input`: the policy would copy it and the rewards would be meaningless."""

from __future__ import annotations

from curatorkit.generators.grpo_rollout import GRPORolloutTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class _FakeLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="fake", temperature=0.7, max_tokens=100)
        self.prompts: list[str] = []

    def _call(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return LLMResponse(text="a response", model="fake")


def _task():
    llm = _FakeLLM()
    return GRPORolloutTask(llm=llm, num_responses=2, score_responses=False), llm


def test_gold_answer_is_not_in_prompt_or_input():
    task, llm = _task()
    seed = DataSample(source_uri="t://", instruction="Compute 2+2.", output="SECRET-GOLD-4")
    (rollout,) = task.run([seed])
    assert all("SECRET-GOLD-4" not in p for p in llm.prompts)
    assert rollout.input == ""


def test_explicit_input_is_still_the_source_passage():
    task, llm = _task()
    seed = DataSample(
        source_uri="t://", instruction="Who wrote it?", input="Passage by Ada.", output="Ada"
    )
    (rollout,) = task.run([seed])
    assert "Passage by Ada." in llm.prompts[0]
    assert "Source passage" in llm.prompts[0]
    assert rollout.input == "Passage by Ada."


def test_corpus_chunk_text_is_still_the_source_passage():
    task, llm = _task()
    chunk = DataSample(source_uri="t://", output="Corpus text.", task_type="language_modeling")
    (rollout,) = task.run([chunk])
    assert "Corpus text." in llm.prompts[0]
    assert rollout.input == "Corpus text."
