"""
MultiTurnTask — extend a prompt or text chunk into a multi-turn conversation.

Two modes
---------
turn_by_turn (default)
    Each turn is generated independently, conditioned on all prior turns.
    User follow-ups see the actual assistant responses; assistant answers see
    the actual user questions.  N turns = 2N LLM calls.  This mirrors how
    real RLHF conversation data is collected and avoids the distribution
    collapse of single-call generation.

single_call
    Entire conversation generated in one LLM call.  Cheaper but quality
    degrades for longer conversations (>4 turns) and turn N is not truly
    conditioned on independently sampled turn N-1.  Kept for backward compat.
"""

from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, RejectedSample

# ── Single-call prompt ────────────────────────────────────────────────────────

_DEFAULT_MULTITURN_PROMPT = """Generate a natural multi-turn conversation with exactly {num_turns} total exchanges (user-assistant pairs).

{context_section}

The conversation should:
- Start with: "{initial_question}"
- Each follow-up should naturally build on the previous response
- Follow-ups should explore different aspects, ask for clarification, or go deeper
- Responses should be thorough but not overly long (2-5 sentences each)
- The conversation should feel natural, not scripted

Respond in JSON format ONLY:
{{
  "turns": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}"""

# ── Turn-by-turn prompts ──────────────────────────────────────────────────────

_INITIAL_QUESTION_PROMPT = """Based on the source passage below, generate one specific, focused question that opens a learning conversation.

Source passage:
---
{context}
---

Return ONLY the question text, nothing else."""

_USER_FOLLOWUP_PROMPT = """You are a curious learner. Based on the source passage and the conversation so far, write ONE natural follow-up question.

Source passage:
---
{context}
---

Conversation so far:
{conversation}

Write a follow-up that digs deeper, asks for clarification, or explores a related aspect.
Return ONLY the question text, nothing else."""

_ASSISTANT_RESPONSE_PROMPT = """You are a helpful assistant. Answer the user's question based on the source passage.

Source passage:
---
{context}
---

Conversation so far:
{conversation}

Answer the latest user question accurately and concisely (2-5 sentences), grounded in the passage.
Return ONLY the answer text, nothing else."""


class MultiTurnTask(BaseGenerationTask):
    """
    Generate multi-turn conversations from prompts or text chunks.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend for generation.
    num_turns : int
        Number of user-assistant exchange pairs.
    mode : str
        "turn_by_turn" (default) — each turn is a separate LLM call,
        conditioned on all prior real turns.
        "single_call" — one LLM call generates the full conversation.
    prompt_template : str | None
        Custom template for single_call mode.
        Required vars: {num_turns}, {context_section}, {initial_question}.
    include_context : bool
        If True, include source text as grounding context.
    """

    def __init__(
        self,
        llm: BaseLLM,
        num_turns: int = 3,
        mode: str = "turn_by_turn",
        prompt_template: str | None = None,
        include_context: bool = True,
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=prompt_template, concurrency=concurrency)
        self.num_turns = max(2, num_turns)
        self.mode = mode
        self.include_context = include_context
        if prompt_template and mode == "single_call":
            self._validate_template(
                prompt_template,
                ["num_turns", "context_section", "initial_question"],
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_context(self, sample: DataSample) -> str:
        return self._get_source_context(sample)

    def _get_initial_question(self, sample: DataSample) -> str:
        if sample.instruction:
            return sample.instruction
        return "Can you explain this topic in detail?"

    @staticmethod
    def _format_conversation(turns: list[dict]) -> str:
        lines = []
        for t in turns:
            role = t.get("role", "").capitalize()
            content = t.get("content", "").strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ── Single-call path (BaseGenerationTask.run delegates here) ──────────────

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        template = self.prompt_template or _DEFAULT_MULTITURN_PROMPT
        context = self._get_context(sample)
        context_section = (
            f"Base the conversation on this source text:\n---\n{context}\n---"
            if self.include_context and context
            else ""
        )
        prompt = template.format(
            num_turns=self.num_turns,
            context_section=context_section,
            initial_question=self._get_initial_question(sample),
        )
        return [{"role": "user", "content": prompt}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        text = response.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        turns = self._extract_turns(text)
        if not turns or len(turns) < 2:
            return []
        return [self._turns_to_sample(sample, turns)]

    # ── Turn-by-turn path ─────────────────────────────────────────────────────

    def _generate_turn_by_turn(self, sample: DataSample) -> DataSample | None:
        context = self._get_context(sample)
        turns: list[dict] = []

        # Turn 0 user: use existing instruction or generate from source
        if sample.instruction:
            first_q = sample.instruction
        elif context:
            resp = self.llm.generate(
                [{"role": "user", "content": _INITIAL_QUESTION_PROMPT.format(context=context)}],
                temperature=0.7,
                max_tokens=128,
            )
            first_q = resp.text.strip() or "Can you explain this topic in detail?"
        else:
            first_q = "Can you explain this topic in detail?"

        turns.append({"role": "user", "content": first_q})

        for turn_idx in range(self.num_turns):
            conv_str = self._format_conversation(turns)

            if turn_idx == self.num_turns - 1 and len(turns) % 2 == 1:
                # Last turn is an assistant response
                prompt = _ASSISTANT_RESPONSE_PROMPT.format(
                    context=context,
                    conversation=conv_str,
                )
                resp = self.llm.generate(
                    [{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512,
                )
                turns.append({"role": "assistant", "content": resp.text.strip()})
                break

            if len(turns) % 2 == 1:
                # Need assistant response
                prompt = _ASSISTANT_RESPONSE_PROMPT.format(
                    context=context,
                    conversation=conv_str,
                )
                resp = self.llm.generate(
                    [{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512,
                )
                turns.append({"role": "assistant", "content": resp.text.strip()})
            else:
                # Need user follow-up
                prompt = _USER_FOLLOWUP_PROMPT.format(
                    context=context,
                    conversation=conv_str,
                )
                resp = self.llm.generate(
                    [{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=128,
                )
                turns.append({"role": "user", "content": resp.text.strip()})

        if len(turns) < 2:
            return None
        return self._turns_to_sample(sample, turns)

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        if self.mode == "single_call":
            return super().run(samples)

        # turn_by_turn — concurrent across samples, sequential within each
        self._rejected = []
        results_map: dict[int, DataSample] = {}

        def _process(idx_sample):
            idx, sample = idx_sample
            try:
                ds = self._generate_turn_by_turn(sample)
                return idx, ds, None
            except Exception as e:
                return idx, None, str(e)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(_process, (i, s)): s for i, s in enumerate(samples)}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[MultiTurn] turn_by_turn",
                unit="sample",
            ):
                sample = futures[future]
                idx, ds, err = future.result()
                if ds is not None:
                    results_map[idx] = ds
                else:
                    self._rejected.append(
                        RejectedSample(
                            **sample.model_dump(),
                            rejection_reason=f"generation_failed:{self.task_name}:{err or 'empty'}",
                            rejecting_step=self.task_name,
                        )
                    )

        return [results_map[i] for i in sorted(results_map)]

    # ── Shared output builder ─────────────────────────────────────────────────

    def _turns_to_sample(self, source: DataSample, turns: list[dict]) -> DataSample:
        first_user = ""
        first_assistant = ""
        extra_turns = []
        for i, t in enumerate(turns):
            role = t.get("role", "").lower()
            content = t.get("content", "").strip()
            if i == 0 and role == "user":
                first_user = content
            elif i == 1 and role == "assistant":
                first_assistant = content
            else:
                sharegpt_role = "human" if role == "user" else "gpt"
                extra_turns.append({"from": sharegpt_role, "value": content})

        return DataSample(
            id=str(uuid.uuid4()),
            source_uri=source.source_uri,
            instruction=first_user,
            input=self._get_source_context(source),
            output=first_assistant,
            task_type="conversational",
            metadata={
                "turns": extra_turns,
                "total_turns": len(turns),
                "generation_source": "multiturn_generator",
                "generation_mode": self.mode,
                "source_sample_id": source.id,
                **{
                    k: v
                    for k, v in source.metadata.items()
                    if k in ("page", "parent_heading", "source_file")
                },
            },
            provenance_chain=list(source.provenance_chain),
        )

    def _extract_turns(self, text: str) -> list[dict[str, str]]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                turns = parsed.get("turns", parsed.get("conversation", []))
            elif isinstance(parsed, list):
                turns = parsed
            else:
                return []
            if isinstance(turns, list):
                return [t for t in turns if isinstance(t, dict) and "role" in t and "content" in t]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                turns = json.loads(match.group())
                if isinstance(turns, list):
                    return [
                        t for t in turns if isinstance(t, dict) and "role" in t and "content" in t
                    ]
            except json.JSONDecodeError:
                pass
        return []
