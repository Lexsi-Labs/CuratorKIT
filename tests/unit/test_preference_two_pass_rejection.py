"""
Regression test: PreferenceGenerationTask's two_pass mode (both run() and
run_async()) crashed instead of cleanly rejecting a sample whenever chosen/
rejected generation failed (empty response, or corpus parse/incomplete).

The rejection-path constructed RejectedSample as:

    RejectedSample(
        **sample.model_dump(),
        ...
        metadata={**sample.metadata, "partial_result": str(err)},
    )

sample.model_dump() already includes "metadata", so passing metadata= too
raised "RejectedSample() got multiple values for keyword argument
'metadata'" — this crashed the whole curation run instead of just adding
one entry to result.rejected.
"""

from __future__ import annotations

import asyncio

from curatorkit.generators.preference_gen import PreferenceGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample


class _EmptyResponseLLM(BaseLLM):
    """Always returns an empty completion, forcing the two_pass path's
    'chosen_text/rejected_text empty' rejection branch."""

    def _call(self, messages, **kw):
        return LLMResponse(text="", model="fake")

    async def _acall(self, messages, **kw):
        return LLMResponse(text="", model="fake")


def _llm() -> _EmptyResponseLLM:
    return _EmptyResponseLLM(model="fake", temperature=0.7, max_tokens=100)


class TestPreferenceTwoPassRejectionDoesNotCrash:
    def test_run_rejects_cleanly_instead_of_raising(self):
        task = PreferenceGenerationTask(llm=_llm(), mode="two_pass")
        samples = [
            DataSample(source_uri="t://", instruction="q", task_type="instruction_following")
        ]

        results = task.run(samples)  # must not raise TypeError

        assert results == []
        assert len(task._rejected) == 1
        assert task._rejected[0].rejection_reason.startswith("generation_parse_failed")

    def test_run_async_rejects_cleanly_instead_of_raising(self):
        task = PreferenceGenerationTask(llm=_llm(), mode="two_pass")
        samples = [
            DataSample(source_uri="t://", instruction="q", task_type="instruction_following")
        ]

        results = asyncio.run(task.run_async(samples))  # must not raise TypeError

        assert results == []
        assert len(task._rejected) == 1
        assert task._rejected[0].rejection_reason.startswith("generation_parse_failed")
