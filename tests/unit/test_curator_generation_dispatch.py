"""
Regression tests for Curator._build_generator()'s adversarial task wiring.

adversarial_qa used to coerce each injection_types string via
`InjectionType(t)`, where InjectionType is a `typing.Literal[...]` alias —
Literal aliases are not callable, so this raised TypeError at pipeline-build
time (before any LLM call), for every run that combined generation_task=
"adversarial_qa" with an explicit injection_types list. adversarial_preference
was never affected — it passes injection_types through unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install curatorkit[generation]")

from curatorkit.curator import Curator, CuratorConfig


def _config(**overrides) -> CuratorConfig:
    defaults = {
        "dataset": "dummy.jsonl",
        "llm_model": "openai/gpt-4o-mini",
    }
    return CuratorConfig(**{**defaults, **overrides})


class TestAdversarialQADispatch:
    def test_builds_with_explicit_injection_types(self):
        cfg = _config(
            generation_task="adversarial_qa",
            injection_types=["contradicts_source", "parametric_drift"],
        )
        gen = Curator(cfg)._build_generator()

        from curatorkit.generators.adversarial_qa_generator import AdversarialQAGenerationTask

        assert isinstance(gen, AdversarialQAGenerationTask)
        assert gen.injection_types == ["contradicts_source", "parametric_drift"]

    def test_builds_with_default_injection_types(self):
        cfg = _config(generation_task="adversarial_qa")
        gen = Curator(cfg)._build_generator()

        from curatorkit.generators.adversarial_qa_generator import ALL_INJECTION_TYPES

        assert gen.injection_types == ALL_INJECTION_TYPES


class TestAdversarialPreferenceDispatch:
    def test_builds_with_explicit_injection_types(self):
        cfg = _config(
            generation_task="adversarial_preference",
            injection_types=["contradicts_source", "domain_mismatch"],
        )
        gen = Curator(cfg)._build_generator()

        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        assert isinstance(gen, AdversarialPreferenceTask)
        assert gen.injection_types == ["contradicts_source", "domain_mismatch"]


@pytest.mark.parametrize("task", ["adversarial_qa", "adversarial_preference"])
def test_injection_types_never_raises_at_build_time(task):
    cfg = _config(generation_task=task, injection_types=["contradicts_source"])
    Curator(cfg)._build_generator()
