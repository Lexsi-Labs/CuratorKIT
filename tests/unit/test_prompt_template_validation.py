"""
Prompt-template placeholder validation and wiring fixes.

Covers:
  - the shared curatorkit.utils.prompt_validation.validate_prompt_template
  - the previously-missing _validate_template()/validate_prompt_template()
    calls added to QAGenerationTask, AdversarialPreferenceTask,
    HallucinationGate, RewardGate, RewardRefiner, and DiagnosticProbe
  - the EvolInstructTask KeyError('context') crash on context-less samples
  - CuratorConfig.multiturn_mode actually reaching MultiTurnTask
  - GRPORolloutTask/ChainOfThoughtTask now pairing a custom prompt_template
    with source context via {context_section}, instead of silently
    swapping it out for a different, non-customizable built-in prompt
    whenever the sample happens to carry context
"""

from __future__ import annotations

import pytest

from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample
from curatorkit.utils.prompt_validation import validate_prompt_template


class _FakeLLM(BaseLLM):
    def _call(self, messages, **kw):
        return LLMResponse(text="{}", model="fake", prompt_tokens=1, completion_tokens=1)

    async def _acall(self, messages, **kw):
        return LLMResponse(text="{}", model="fake", prompt_tokens=1, completion_tokens=1)


def _llm() -> _FakeLLM:
    return _FakeLLM(model="fake", temperature=0.7, max_tokens=100)


class TestValidatePromptTemplate:
    def test_passes_when_all_placeholders_present(self):
        validate_prompt_template("Q: {question} A: {answer}", ["question", "answer"])

    def test_raises_on_missing_placeholder(self):
        with pytest.raises(ValueError, match=r"missing required placeholder.*\{answer\}"):
            validate_prompt_template("Q: {question}", ["question", "answer"])

    def test_error_names_the_field(self):
        with pytest.raises(ValueError, match="custom_field_name"):
            validate_prompt_template("no placeholders here", ["x"], "custom_field_name")


class TestQAGenerationTaskValidation:
    def test_missing_context_raises(self):
        from curatorkit.generators.qa_generator import QAGenerationTask

        with pytest.raises(ValueError, match=r"\{context\}"):
            QAGenerationTask(llm=_llm(), prompt_template="Generate {num_questions} questions.")

    def test_missing_num_questions_raises(self):
        from curatorkit.generators.qa_generator import QAGenerationTask

        with pytest.raises(ValueError, match=r"\{num_questions\}"):
            QAGenerationTask(llm=_llm(), prompt_template="Source: {context}")

    def test_valid_template_constructs(self):
        from curatorkit.generators.qa_generator import QAGenerationTask

        QAGenerationTask(
            llm=_llm(), prompt_template="Source: {context}\nWrite {num_questions} questions."
        )

    def test_table_prompt_template_validated(self):
        from curatorkit.generators.qa_generator import QAGenerationTask

        with pytest.raises(ValueError, match="table_prompt_template"):
            QAGenerationTask(llm=_llm(), table_prompt_template="no placeholders")


class TestAdversarialPreferenceTaskValidation:
    def test_missing_faithful_placeholder_raises(self):
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        with pytest.raises(ValueError, match="faithful_prompt_template"):
            AdversarialPreferenceTask(llm=_llm(), faithful_prompt_template="no placeholders")

    def test_missing_adversarial_placeholder_raises(self):
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        with pytest.raises(ValueError, match="adversarial_prompt_template"):
            AdversarialPreferenceTask(llm=_llm(), adversarial_prompt_template="no placeholders")

    def test_valid_templates_construct(self):
        from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask

        AdversarialPreferenceTask(
            llm=_llm(),
            faithful_prompt_template="Source: {context}\n{num_questions} questions.",
            adversarial_prompt_template="Source: {context}\nQuestion: {question}",
        )


class TestHallucinationGateValidation:
    def test_missing_placeholder_raises(self):
        from curatorkit.gates.hallucination import HallucinationGate

        with pytest.raises(ValueError, match="hallucination_prompt_template"):
            HallucinationGate(llm=_llm(), prompt_template='{{"grounding_score": 0.5}}')

    def test_valid_template_constructs(self):
        from curatorkit.gates.hallucination import HallucinationGate

        HallucinationGate(
            llm=_llm(),
            prompt_template=(
                "Source: {source_text}\nQ: {question}\nA: {answer}\n"
                '{{"grounding_score": 0.XX}}'
            ),
        )


class TestRewardGateValidation:
    def test_missing_placeholder_raises(self):
        from curatorkit.gates.reward import RewardGate

        with pytest.raises(ValueError, match="reward_prompt_template"):
            RewardGate(llm=_llm(), prompt_template='{{"overall_score": 0.5}}')

    def test_valid_template_constructs(self):
        from curatorkit.gates.reward import RewardGate

        RewardGate(
            llm=_llm(),
            prompt_template='Instruction: {instruction}\nResponse: {response}\n{{"overall_score": 0.XX}}',
        )


class TestRewardRefinerValidation:
    def test_missing_refine_placeholder_raises(self):
        from curatorkit.diagnostic.reward_refine import RewardRefiner
        from curatorkit.gates.reward import RewardGate

        with pytest.raises(ValueError, match="reward_refine_prompt_template"):
            RewardRefiner(
                generator_llm=_llm(),
                reward_gate=RewardGate(llm=_llm()),
                refine_prompt_template="no placeholders",
            )

    def test_missing_instruction_refine_placeholder_raises(self):
        from curatorkit.diagnostic.reward_refine import RewardRefiner
        from curatorkit.gates.reward import RewardGate

        with pytest.raises(ValueError, match="reward_instruction_refine_template"):
            RewardRefiner(
                generator_llm=_llm(),
                reward_gate=RewardGate(llm=_llm()),
                instruction_refine_template="no placeholders",
            )

    def test_valid_templates_construct(self):
        from curatorkit.diagnostic.reward_refine import RewardRefiner
        from curatorkit.gates.reward import RewardGate

        RewardRefiner(
            generator_llm=_llm(),
            reward_gate=RewardGate(llm=_llm()),
            refine_prompt_template=(
                "{axis} {weakness} {instruction} {source} {answer}"
            ),
            instruction_refine_template="{weakness} {source} {question}",
        )


class TestDiagnosticProbeValidation:
    def test_missing_placeholder_in_extra_template_raises(self):
        from curatorkit.diagnostic.probe import DiagnosticProbe
        from curatorkit.gates.hallucination import HallucinationGate

        with pytest.raises(ValueError, match=r"probe_extra_templates\['my_template'\]"):
            DiagnosticProbe(
                generator_llm=_llm(),
                gate=HallucinationGate(llm=_llm()),
                extra_templates={"my_template": "no placeholders here"},
            )

    def test_valid_extra_template_constructs(self):
        from curatorkit.diagnostic.probe import DiagnosticProbe
        from curatorkit.gates.hallucination import HallucinationGate

        DiagnosticProbe(
            generator_llm=_llm(),
            gate=HallucinationGate(llm=_llm()),
            extra_templates={"my_template": "Source: {source}\nQuestion: {question}"},
        )


class TestEvolInstructContextCrashFix:
    """EvolInstructTask required {context} in any custom prompt_template (it's
    used in the with-context branch), but the no-context branch called
    template.format(instruction=, strategy=) without passing context= at
    all — a template satisfying validation would still raise
    KeyError('context') the moment a sample had no source text.
    """

    def test_no_longer_crashes_on_context_less_sample(self):
        from curatorkit.generators.evol_instruct import EvolInstructTask

        task = EvolInstructTask(
            llm=_llm(), prompt_template="Evolve: {instruction} via {strategy}. Ctx: {context}"
        )
        sample = DataSample(source_uri="t://", instruction="What is 2+2?", task_type="prompt_only")
        messages = task._build_messages(sample)
        assert messages[0]["content"] == "Evolve: What is 2+2? via add_constraints. Ctx: "

    def test_still_works_with_context(self):
        from curatorkit.generators.evol_instruct import EvolInstructTask

        task = EvolInstructTask(
            llm=_llm(), prompt_template="Evolve: {instruction} via {strategy}. Ctx: {context}"
        )
        sample = DataSample(
            source_uri="t://",
            instruction="What is 2+2?",
            input="Source passage text.",
            task_type="instruction_following",
        )
        messages = task._build_messages(sample)
        assert "Source passage text." in messages[0]["content"]


class TestGRPOContextSectionPairing:
    """grpo_prompt_template used to be dropped entirely (in favor of a
    different, hardcoded built-in prompt) the moment a sample had source
    context, even with an instruction present. It must now always be used
    together with an optional {context_section}.
    """

    def test_requires_context_section_placeholder(self):
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        with pytest.raises(ValueError, match=r"\{context_section\}"):
            GRPORolloutTask(llm=_llm(), response_prompt="Answer: {instruction}")

    def test_custom_template_used_with_context(self):
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        task = GRPORolloutTask(
            llm=_llm(), response_prompt="In bullet points.\n{context_section}Q: {instruction}"
        )
        sample = DataSample(
            source_uri="t://",
            instruction="What is X?",
            input="X is a concept.",
            task_type="instruction_following",
        )
        content = task._build_messages(sample)[0]["content"]
        assert content.startswith("In bullet points.")
        assert "X is a concept." in content
        assert "Q: What is X?" in content

    def test_custom_template_used_without_context(self):
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        task = GRPORolloutTask(
            llm=_llm(), response_prompt="In bullet points.\n{context_section}Q: {instruction}"
        )
        sample = DataSample(source_uri="t://", instruction="What is X?", task_type="prompt_only")
        content = task._build_messages(sample)[0]["content"]
        assert content == "In bullet points.\nQ: What is X?"

    def test_no_instruction_pure_corpus_mode_unaffected(self):
        """No instruction at all -> nothing for response_prompt to fill in;
        this still uses the built-in from-scratch corpus prompt."""
        from curatorkit.generators.grpo_rollout import GRPORolloutTask

        task = GRPORolloutTask(
            llm=_llm(), response_prompt="In bullet points.\n{context_section}Q: {instruction}"
        )
        sample = DataSample(source_uri="t://", input="Some passage.", task_type="language_modeling")
        content = task._build_messages(sample)[0]["content"]
        assert "In bullet points." not in content
        assert "Some passage." in content


class TestChainOfThoughtContextSectionPairing:
    """cot_prompt_template (mode="generate") had the same bypass problem as
    GRPO — fixed the same way, via {context_section}."""

    def test_requires_context_section_placeholder_in_generate_mode(self):
        from curatorkit.generators.cot_generator import ChainOfThoughtTask

        with pytest.raises(ValueError, match=r"\{context_section\}"):
            ChainOfThoughtTask(llm=_llm(), prompt_template="Solve: {instruction}")

    def test_wrap_mode_still_requires_answer_not_context_section(self):
        from curatorkit.generators.cot_generator import ChainOfThoughtTask

        # wrap mode never branches on context, so it must not demand
        # {context_section} — only {instruction} and {answer}.
        ChainOfThoughtTask(
            llm=_llm(), mode="wrap", prompt_template="Q: {instruction}\nA: {answer}"
        )
        with pytest.raises(ValueError, match=r"\{answer\}"):
            ChainOfThoughtTask(llm=_llm(), mode="wrap", prompt_template="Q: {instruction}")

    def test_custom_template_used_with_context(self):
        from curatorkit.generators.cot_generator import ChainOfThoughtTask

        task = ChainOfThoughtTask(
            llm=_llm(), prompt_template="Think.\n{context_section}Q: {instruction}"
        )
        sample = DataSample(
            source_uri="t://",
            instruction="Why?",
            input="Background info.",
            task_type="instruction_following",
        )
        content = task._build_messages(sample)[0]["content"]
        assert content.startswith("Think.")
        assert "Background info." in content
        assert "Q: Why?" in content

    def test_custom_template_used_without_context(self):
        from curatorkit.generators.cot_generator import ChainOfThoughtTask

        task = ChainOfThoughtTask(
            llm=_llm(), prompt_template="Think.\n{context_section}Q: {instruction}"
        )
        sample = DataSample(source_uri="t://", instruction="Why?", task_type="prompt_only")
        content = task._build_messages(sample)[0]["content"]
        assert content == "Think.\nQ: Why?"


pytest.importorskip("litellm", reason="litellm not installed — install curatorkit[generation]")


class TestMultiturnModeWiring:
    """multiturn_prompt_template had zero effect: MultiTurnTask only reads
    prompt_template in mode="single_call", but Curator never passed a mode
    at all, so it always fell back to the "turn_by_turn" constructor
    default. CuratorConfig.multiturn_mode now reaches the generator.
    """

    def test_default_mode_is_turn_by_turn(self):
        from curatorkit.curator import Curator, CuratorConfig

        cfg = CuratorConfig(dataset="unused", llm_model="openai/gpt-4o-mini", generation_task="multiturn")
        gen = Curator(cfg)._build_generator()
        assert gen.mode == "turn_by_turn"

    def test_single_call_mode_reaches_generator(self):
        from curatorkit.curator import Curator, CuratorConfig

        cfg = CuratorConfig(
            dataset="unused",
            llm_model="openai/gpt-4o-mini",
            generation_task="multiturn",
            multiturn_mode="single_call",
            multiturn_prompt_template=(
                "{num_turns} turns. {context_section} Start: {initial_question}"
            ),
        )
        gen = Curator(cfg)._build_generator()
        assert gen.mode == "single_call"
        assert gen.prompt_template == cfg.multiturn_prompt_template

    def test_single_call_mode_validates_prompt_template(self):
        from curatorkit.curator import Curator, CuratorConfig

        cfg = CuratorConfig(
            dataset="unused",
            llm_model="openai/gpt-4o-mini",
            generation_task="multiturn",
            multiturn_mode="single_call",
            multiturn_prompt_template="missing all placeholders",
        )
        with pytest.raises(ValueError):
            Curator(cfg)._build_generator()
