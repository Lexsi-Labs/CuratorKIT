"""Judge gates fail closed: judge errors and unparseable scores never pass a sample."""

import asyncio

import pytest

from curatorkit.gates.hallucination import HallucinationGate
from curatorkit.gates.reward import RewardGate
from curatorkit.llm.base import LLMResponse
from curatorkit.schema import DataSample


class FakeJudge:
    """Duck-typed judge LLM: raises `exc` or returns `text`."""

    model = "fake-judge"
    temperature = 0.0
    max_tokens = 256

    def __init__(self, text: str = "", exc: Exception | None = None) -> None:
        self.text, self.exc = text, exc

    def generate(self, messages, **kw):
        if self.exc:
            raise self.exc
        return LLMResponse(text=self.text)

    async def agenerate(self, messages, **kw):
        return self.generate(messages, **kw)


def _sft():
    return DataSample(source_uri="t", instruction="What is 2+2?", input="x" * 60, output="4")


def _pair():
    return DataSample(source_uri="t", instruction="Q?", chosen="good", rejected="bad",
                      task_type="preference")


GATES = [
    (RewardGate, "RewardGate", _sft),
    (RewardGate, "RewardGate", _pair),
    (HallucinationGate, "HallucinationGate", _sft),
]
JUDGES = {
    "raises": FakeJudge(exc=TimeoutError("judge down")),
    "unparseable": FakeJudge(text="I cannot rate this answer."),
}


def _run(gate, sample, use_async):
    if use_async:
        return asyncio.run(gate.run_async([sample]))
    return gate.run([sample])


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("judge", sorted(JUDGES))
@pytest.mark.parametrize("cls,step,make", GATES)
def test_judge_failure_rejects_by_default(cls, step, make, judge, use_async):
    gate = cls(llm=JUDGES[judge], threshold=0.1)  # low threshold: old 0.5 fallback passed
    with pytest.warns(UserWarning, match="judge failed on 1/1"):
        passed, rejected = _run(gate, make(), use_async)
    assert passed == []
    assert len(rejected) == 1
    rej = rejected[0]
    expected = "TimeoutError" if judge == "raises" else "JudgeParseError"
    assert rej.rejection_reason == f"judge_error:{expected}"
    assert rej.rejecting_step == step
    notes = rej.provenance_chain[-1].notes
    assert notes["error_type"] == expected and notes["passed_on_error"] is False
    assert notes["error"]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("cls,step,make", GATES)
def test_on_judge_error_pass_keeps_legacy_behaviour(cls, step, make, use_async):
    gate = cls(llm=JUDGES["raises"], threshold=0.9, on_judge_error="pass")
    with pytest.warns(UserWarning, match="judge failed"):
        passed, rejected = _run(gate, make(), use_async)
    assert rejected == [] and len(passed) == 1
    notes = passed[0].provenance_chain[-1].notes
    assert notes["passed_on_error"] is True and notes["error_type"] == "TimeoutError"


def test_parseable_judge_still_scores():
    gate = RewardGate(llm=FakeJudge(text='{"overall_score": 0.8}'), threshold=0.7)
    passed, rejected = gate.run([_sft()])
    assert len(passed) == 1 and passed[0].label == 0.8


def test_invalid_on_judge_error_raises():
    with pytest.raises(ValueError, match="on_judge_error"):
        RewardGate(llm=FakeJudge(), on_judge_error="ignore")


def test_curator_run_records_judge_errors(tmp_path, monkeypatch):
    """End to end: judge errors reach manifest.json and rejected.jsonl."""
    import json

    from curatorkit.curator import Curator, CuratorConfig

    seed = tmp_path / "seed.jsonl"
    seed.write_text("".join(json.dumps({
        "instruction": f"Explain topic number {i} in a few clear and simple sentences please.",
        "output": f"Topic {i} is explained here in several plain words for the reader to follow.",
    }) + "\n" for i in range(4)), encoding="utf-8")
    monkeypatch.setattr(Curator, "_build_reward_backend",
                        lambda self, task_llm: FakeJudge(exc=TimeoutError("judge down")))
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="judge failed on 4/4"):
        result = Curator(CuratorConfig(dataset=str(seed), llm_model="openai/x",
                                       reward_threshold=0.5, output_dir=str(out))).run()
    assert result.passed == [] and len(result.rejected) == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rejected_breakdown"] == {"judge_error:TimeoutError": 4}
    rows = [json.loads(line) for line in (out / "rejected.jsonl").read_text().splitlines()]
    assert {r["rejection_reason"] for r in rows} == {"judge_error:TimeoutError"}


def test_probe_and_refiner_skip_judge_errors(tmp_path):
    """A failing judge can't re-judge regenerations: no probe or refiner calls."""
    from curatorkit.diagnostic.probe import DiagnosticProbe
    from curatorkit.diagnostic.reward_refine import RewardRefiner
    from curatorkit.interfaces import BaseReader
    from curatorkit.pipeline import Pipeline

    class OneSample(BaseReader):
        def read(self):
            return [_sft()], []

    generator = FakeJudge(exc=AssertionError("generator must not be called"))
    gate = RewardGate(llm=FakeJudge(exc=TimeoutError("down")))
    gate.probe = DiagnosticProbe(generator_llm=generator, gate=gate)
    for use_async in (False, True):
        pipe = Pipeline([OneSample(), gate], output_dir=tmp_path / f"p{use_async}")
        with pytest.warns(UserWarning, match="judge failed"):
            res = asyncio.run(pipe.run_async()) if use_async else pipe.run()
        assert [r.rejection_reason for r in res.rejected] == ["judge_error:TimeoutError"]
        diag = res.rejected[0].diagnosis
        assert diag.probe_calls == 0 and diag.notes == {"skip_reason": "judge_error"}

    refiner = RewardRefiner(generator_llm=generator, reward_gate=gate)
    recovered, still = refiner.refine(res.rejected)
    assert recovered == [] and still == res.rejected
