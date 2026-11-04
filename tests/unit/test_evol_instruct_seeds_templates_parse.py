"""Regression tests for three silent EvolInstruct footguns.

1. Instruction-only seeds were rejected by the pre-generation SchemaGate
   (``missing_field:output``) even though the generation step fills ``output``,
   so an evol run over prompt seeds produced 0 rows with no error.
2. A custom ``answer_prompt_template`` was ignored whenever the evolved sample
   carried source context (``input`` set, or ``output`` used as context), so the
   obvious workaround for (1) - filling ``output`` - silently disabled it.
3. An unparseable generator reply (e.g. a refusal) became the evolved
   instruction verbatim instead of being retried / rejected.
"""

from __future__ import annotations

import json

from curatorkit.gates.schema import SchemaGate
from curatorkit.generators.evol_instruct import EvolInstructTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class _ScriptedLLM(BaseLLM):
    """Returns ``evol_text`` for evolution prompts and records answer prompts."""

    def __init__(self, evol_text: str, **kw):
        super().__init__(**kw)
        self.evol_text = evol_text
        self.answer_prompts: list[str] = []

    def _call(self, messages, **kw):
        content = messages[0]["content"]
        if "instruction evolution expert" in content:
            return LLMResponse(text=self.evol_text, model="fake")
        self.answer_prompts.append(content)
        return LLMResponse(text="an answer", model="fake")

    async def _acall(self, messages, **kw):
        return self._call(messages, **kw)


def _llm(evol_text: str) -> _ScriptedLLM:
    return _ScriptedLLM(evol_text, model="fake", temperature=0.0, max_tokens=50)


_GOOD = json.dumps({"evolved_instruction": "Explain X in depth", "strategy_applied": "deepen"})


# ── 1. SchemaGate ahead of generation ────────────────────────────────────────
def _seed(**kw) -> DataSample:
    return DataSample(source_uri="t://", task_type="instruction_following", **kw)


def test_schema_gate_default_still_requires_output():
    passed, rejected = SchemaGate(min_tokens=1).run(
        [_seed(instruction="How do I bake bread today?")]
    )
    assert not passed and rejected[0].rejection_reason == "missing_field:output"


def test_schema_gate_allows_instruction_only_seeds_before_generation():
    gate = SchemaGate(min_tokens=1, output_generated_later=True)
    passed, rejected = gate.run([_seed(instruction="How do I bake bread today?")])
    assert len(passed) == 1 and not rejected
    # instruction is still required
    passed, rejected = gate.run([_seed(instruction="", output="something")])
    assert not passed and rejected[0].rejection_reason == "missing_field:instruction"


def test_curator_passes_output_generated_later_when_generation_task_set(tmp_path):
    from curatorkit.curator import Curator, CuratorConfig

    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(json.dumps({"instruction": "How do I bake bread today?"}) + "\n")

    cfg = CuratorConfig(dataset=str(seeds), generation_task="evol", output_dir=str(tmp_path / "o"))
    cur = Curator(cfg)
    gates = [s for s in _all_steps(cur) if isinstance(s, SchemaGate)]
    assert gates and all(g.output_generated_later for g in gates)

    cfg2 = CuratorConfig(dataset=str(seeds), output_dir=str(tmp_path / "o2"))
    gates2 = [s for s in _all_steps(Curator(cfg2)) if isinstance(s, SchemaGate)]
    assert gates2 and not any(g.output_generated_later for g in gates2)


def _all_steps(cur):
    for name in ("_build_pipeline", "build_pipeline", "_build_steps"):
        fn = getattr(cur, name, None)
        if fn:
            built = fn()
            return getattr(built, "steps", built)
    raise AssertionError("could not find Curator pipeline builder")


# ── 2. custom answer template ────────────────────────────────────────────────
def test_custom_answer_template_used_even_with_source_context():
    llm = _llm(_GOOD)
    task = EvolInstructTask(
        llm=llm, num_evolutions=1, answer_prompt_template="CUSTOM: {instruction} | ctx={context}"
    )
    task.run([_seed(instruction="How to X?", output="seed answer used as context")])
    assert llm.answer_prompts and all(p.startswith("CUSTOM: ") for p in llm.answer_prompts)
    assert "ctx=seed answer used as context" in llm.answer_prompts[0]


def test_custom_answer_template_without_context_placeholder():
    llm = _llm(_GOOD)
    task = EvolInstructTask(llm=llm, num_evolutions=1, answer_prompt_template="ONLY {instruction}")
    task.run([_seed(instruction="How to X?", input="a passage")])
    assert llm.answer_prompts == ["ONLY Explain X in depth"]


def test_default_answer_prompt_still_grounded_with_context():
    llm = _llm(_GOOD)
    EvolInstructTask(llm=llm, num_evolutions=1).run(
        [_seed(instruction="How to X?", input="a passage")]
    )
    assert "a passage" in llm.answer_prompts[0]


# ── 3. unparseable replies ───────────────────────────────────────────────────
def test_refusal_is_not_used_as_instruction():
    llm = _llm("I'm sorry, but I can't help with that.")
    task = EvolInstructTask(llm=llm, num_evolutions=1, generate_answers=False)
    out = task.run([_seed(instruction="How to X?")])
    assert out == []
    assert task._rejected and task._rejected[0].rejection_reason.startswith(
        "generation_parse_failed"
    )
    assert "I'm sorry" in task._rejected[0].metadata["raw_llm_response"]


def test_json_wrapped_in_prose_is_recovered():
    llm = _llm("Sure! Here you go:\n" + _GOOD + "\nHope that helps.")
    out = EvolInstructTask(llm=llm, num_evolutions=1, generate_answers=False).run(
        [_seed(instruction="How to X?")]
    )
    assert [s.instruction for s in out] == ["Explain X in depth"]


# ── 4. answer template may use only {context} ────────────────────────────────
def test_answer_template_with_only_context_is_accepted():
    llm = _llm(_GOOD)
    task = EvolInstructTask(
        llm=llm, num_evolutions=1, answer_prompt_template="Translate: {context}"
    )
    task.run([_seed(instruction="How to X?", output="the seed answer")])
    assert llm.answer_prompts == ["Translate: the seed answer"]


def test_answer_template_without_any_placeholder_is_rejected():
    import pytest

    with pytest.raises(ValueError, match=r"\{instruction\} or \{context\}"):
        EvolInstructTask(llm=_llm(_GOOD), answer_prompt_template="no placeholders here")


# ── 5. instruction + input rows (no output) keep their input ─────────────────
def test_prompt_only_rows_keep_input_column(tmp_path):
    from curatorkit.connectors.json_reader import JSONReader

    path = tmp_path / "seeds.jsonl"
    path.write_text(
        json.dumps({"instruction": "Translate the passage.", "input": "a passage"}) + "\n"
    )
    samples, rejected = JSONReader(str(path)).read()
    assert not rejected
    assert samples[0].task_type == "prompt_only"
    assert samples[0].instruction == "Translate the passage."
    assert samples[0].input == "a passage"
