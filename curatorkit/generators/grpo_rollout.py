"""
GRPORolloutTask — generate N diverse responses per prompt for GRPO training.

Corpus-aware: when the input sample carries raw source text (task_type=
'language_modeling'), the generator includes the source in the response
prompt so all N rollouts are grounded in the passage.  The output DataSample
always carries input=source_context so HallucinationGate can verify grounding.

Group Relative Policy Optimization requires multiple candidate responses
per prompt, each scored by a reward signal. This task:
  1. Generates N responses per prompt using temperature-varied sampling
  2. Optionally scores each response using a separate LLM-as-judge call
  3. Outputs DataSample with populated responses[] and reward_scores[]

The output format matches what TRL's GRPOTrainer expects.
"""

from __future__ import annotations

import json
import re
import uuid

from tqdm import tqdm

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample

_DEFAULT_RESPONSE_PROMPT = """Answer the following instruction to the best of your ability.

{instruction}"""

_RESPONSE_PROMPT_WITH_CONTEXT = """Answer the following instruction based on the source passage provided.

Source passage:
---
{context}
---

Instruction:
{instruction}"""

_CORPUS_RESPONSE_PROMPT = """Based on the following source passage, provide a thorough and accurate response covering the key points.

Source passage:
---
{context}
---

Respond with a detailed, well-organized answer about the main topics in the passage."""

_DEFAULT_SCORING_PROMPT = """Rate the following response on a scale of 0.0 to 1.0 based on:
- Accuracy and correctness
- Completeness (does it address all aspects?)
- Clarity and coherence
- Helpfulness

Instruction: {instruction}

Response: {response}

Respond with ONLY a JSON object:
{{"score": 0.XX, "reasoning": "brief justification"}}"""


class GRPORolloutTask(BaseGenerationTask):
    """
    Generate N diverse responses per prompt for GRPO training.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend for response generation.
    num_responses : int
        Number of responses to generate per prompt.
    scoring_llm : BaseLLM | None
        Separate LLM for scoring. If None, uses the same LLM.
        Set to None and score_responses=False for unscored rollouts.
    score_responses : bool
        Whether to score each response.
    temperature_spread : float
        Temperature variation across responses. Responses are generated
        at temperatures from (base - spread/2) to (base + spread/2).
    response_prompt : str | None
        Custom template for response generation.
    scoring_prompt : str | None
        Custom template for response scoring.
    """

    def __init__(
        self,
        llm: BaseLLM,
        num_responses: int = 4,
        scoring_llm: BaseLLM | None = None,
        score_responses: bool = True,
        temperature_spread: float = 0.0,
        temperatures: list[float] | None = None,
        response_prompt: str | None = None,
        scoring_prompt: str | None = None,
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, concurrency=concurrency)
        self.num_responses = max(2, num_responses)
        self.scoring_llm = scoring_llm or llm
        self.score_responses = score_responses
        self.temperature_spread = temperature_spread
        self._explicit_temperatures = temperatures
        self.response_prompt = response_prompt or _DEFAULT_RESPONSE_PROMPT
        self.scoring_prompt = scoring_prompt or _DEFAULT_SCORING_PROMPT
        if response_prompt:
            self._validate_template(response_prompt, ["instruction"])
        if scoring_prompt:
            self._validate_template(scoring_prompt, ["instruction", "response"], "scoring_prompt")

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        """Build messages for a single response generation."""
        source_context = self._get_source_context(sample)
        instruction = sample.instruction

        if not instruction and source_context:
            prompt = _CORPUS_RESPONSE_PROMPT.format(context=source_context)
        elif source_context:
            prompt = _RESPONSE_PROMPT_WITH_CONTEXT.format(
                instruction=instruction,
                context=source_context,
            )
        else:
            prompt = self.response_prompt.format(instruction=instruction)

        return [{"role": "user", "content": prompt}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        """Not used directly — run() handles multi-response generation."""
        return []

    def _get_temperatures(self) -> list[float]:
        """Return per-rollout temperatures.

        Priority: explicit list > spread calculation.
        An explicit list shorter than num_responses is cycled; longer is truncated.
        """
        n = self.num_responses
        if self._explicit_temperatures is not None:
            t = self._explicit_temperatures
            if len(t) >= n:
                return t[:n]
            return [t[i % len(t)] for i in range(n)]

        base = self.llm.temperature
        spread = self.temperature_spread
        low = max(0.0, base - spread / 2)
        high = min(2.0, base + spread / 2)

        if n == 1:
            return [base]

        step = (high - low) / (n - 1)
        return [round(low + i * step, 2) for i in range(n)]

    def _score_single(self, instruction: str, response_text: str) -> float:
        """Score a single response using the scoring LLM."""
        prompt = self.scoring_prompt.format(
            instruction=instruction,
            response=response_text,
        )

        try:
            resp = self.scoring_llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            text = resp.text.strip()
            text = re.sub(r"```(?:json)?\s*", "", text)
            text = re.sub(r"```\s*$", "", text)

            parsed = json.loads(text)
            score = float(parsed.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError):
            return 0.5

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        """
        Generate N responses per prompt with optional scoring.

        Each input sample produces exactly one output sample with
        populated responses[] and reward_scores[] lists.
        """
        self._rejected = []
        results: list[DataSample] = []
        temperatures = self._get_temperatures()

        for sample in tqdm(samples, desc="[GRPORollout] generating", unit="sample"):
            try:
                source_context = self._get_source_context(sample)
                # Use generic instruction for corpus-mode samples
                instruction = (
                    sample.instruction
                    or "Analyze and discuss the key points from the source passage."
                )
                messages = self._build_messages(sample)
                responses: list[str] = []

                response_status: list[str] = []
                for temp in temperatures:
                    resp = self.llm.generate(messages, temperature=temp)
                    text = resp.text.strip()
                    if text:
                        responses.append(text)
                        response_status.append("ok")
                    else:
                        response_status.append("empty")

                if not responses:
                    continue

                scores: list[float] = []
                if self.score_responses:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=len(responses)) as pool:
                        score_futures = [
                            pool.submit(self._score_single, instruction, rt) for rt in responses
                        ]
                        scores = [f.result() for f in score_futures]
                else:
                    scores = [0.0] * len(responses)

                results.append(
                    DataSample(
                        id=str(uuid.uuid4()),
                        source_uri=sample.source_uri,
                        instruction=instruction,
                        input=source_context,
                        responses=responses,
                        reward_scores=scores,
                        task_type="grpo",
                        metadata={
                            "generation_source": "grpo_rollout",
                            "num_responses": len(responses),
                            "temperatures_used": temperatures[: len(responses)],
                            "response_status": response_status,
                            "scored": self.score_responses,
                            "corpus_mode": not sample.instruction and bool(source_context),
                            "source_sample_id": sample.id,
                        },
                        provenance_chain=list(sample.provenance_chain),
                    )
                )

            except Exception as e:
                from curatorkit.schema import RejectedSample

                self._rejected.append(
                    RejectedSample(
                        **sample.model_dump(),
                        rejection_reason=f"generation_failed:{type(e).__name__}:{e}",
                        rejecting_step=self.task_name,
                    )
                )

        return results
