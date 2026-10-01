"""
Curator wiring for the generation tasks and gates ported from the v0.4.0
branch: every new generation_task value builds its task, the new gates land
in the step list in the right order, and multi-type tasks resolve their
export formats from the generated samples.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install curatorkit[generation]")

from curatorkit import Curator, CuratorConfig  # noqa: E402
from curatorkit.kg.multihop_qa import MultiHopQATask  # noqa: E402
from curatorkit.kg.store import KnowledgeGraph  # noqa: E402


@pytest.mark.parametrize(
    "task, cls_name",
    [
        ("magpie", "MagpieTask"),
        ("persona", "PersonaDrivenTask"),
        ("joint_bundle", "JointBundleTask"),
        ("self_rewarding", "SelfRewardingTask"),
        ("constitutional", "ConstitutionalRevisionTask"),
        ("constitutional_preference", "ConstitutionalPreferenceTask"),
        ("best_of_n", "BestOfNTask"),
    ],
)
def test_curator_builds_new_generation_tasks(task, cls_name):

    cfg = CuratorConfig(dataset="", llm_model="openai/gpt-4o-mini", generation_task=task)
    generator = Curator(cfg)._build_generator()
    assert type(generator).__name__ == cls_name


def test_curator_kg_multihop_requires_path():

    cfg = CuratorConfig(dataset="", llm_model="openai/gpt-4o-mini", generation_task="kg_multihop")
    with pytest.raises(ValueError, match="kg_path"):
        Curator(cfg)._build_generator()


def test_curator_kg_multihop_loads_graph(tmp_path):

    kg = KnowledgeGraph()
    kg.add_triple("Marie Curie", "discovered", "polonium", source_uri="curie.pdf")
    path = kg.save(tmp_path / "kg.json")
    cfg = CuratorConfig(
        dataset="",
        llm_model="openai/gpt-4o-mini",
        generation_task="kg_multihop",
        kg_path=str(path),
        kg_mode="atomic",
    )
    generator = Curator(cfg)._build_generator()
    assert isinstance(generator, MultiHopQATask)
    assert len(generator.kg) == 1


def test_multi_type_tasks_resolve_exports_from_data():
    """joint_bundle emits SFT + DPO + GRPO, so exports come from the samples."""

    c = Curator(CuratorConfig(dataset="", llm_model="m", generation_task="joint_bundle"))
    assert c._export_formats_need_data()
    formats = c._resolve_export_formats({"instruction_following", "preference", "grpo"})
    assert set(formats) == {"alpaca", "sharegpt", "messages", "dpo", "grpo"}
    c = Curator(CuratorConfig(dataset="", llm_model="m", generation_task="self_rewarding"))
    assert not c._export_formats_need_data()
    assert c._resolve_export_formats() == ["dpo"]


def test_curator_wires_new_gates(tmp_path):

    shard = tmp_path / "eval.jsonl"
    shard.write_text(json.dumps({"text": "a b c d e f g h i j k l m n"}) + "\n")
    cfg = CuratorConfig(
        dataset=str(tmp_path / "seed.jsonl"),
        llm_model="openai/gpt-4o-mini",
        pii_gate="redact",
        jailbreak_gate=True,
        jailbreak_use_model=False,
        prometheus_threshold=0.6,
        factscore_threshold=0.7,
        output_safety_gate=True,
        decontamination_shards={"eval": [str(shard)]},
        export_formats=["alpaca", "argilla"],
    )
    (tmp_path / "seed.jsonl").write_text(json.dumps({"instruction": "q", "output": "a"}) + "\n")
    names = [type(s).__name__ for s in Curator(cfg)._build_steps()]
    for expected in (
        "PIIGate",
        "JailbreakGate",
        "PrometheusRewardGate",
        "FActScoreGate",
        "OutputSafetyGate",
        "DecontaminationGate",
        "ArgillaExporter",
    ):
        assert expected in names, expected
    # hygiene gates run before any LLM-backed gate; output checks after
    assert names.index("PIIGate") < names.index("PrometheusRewardGate")
    assert names.index("OutputSafetyGate") > names.index("FActScoreGate")


def test_best_of_n_scorer_choice():
    cfg = CuratorConfig(
        dataset="",
        llm_model="openai/gpt-4o-mini",
        generation_task="best_of_n",
        bon_scorer="prometheus",
    )
    assert type(Curator(cfg)._build_generator()).__name__ == "BestOfNTask"
    cfg.bon_scorer = "bogus"
    with pytest.raises(ValueError, match="bon_scorer"):
        Curator(cfg)._build_generator()


def test_argilla_is_never_misaligned():
    from curatorkit.exporters.compatibility import misaligned_formats

    assert misaligned_formats("qa", ["alpaca", "argilla"]) == []
    assert misaligned_formats("qa", ["dpo", "argilla"]) == ["dpo"]
