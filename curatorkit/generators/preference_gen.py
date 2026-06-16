"""
PreferenceGenerationTask — generate chosen/rejected pairs for DPO training.

Corpus-aware: when the input sample carries raw source text (task_type=
'language_modeling'), the generator derives a question from the source and
produces a grounded chosen + quality-degraded rejected pair.  When the input
already has an instruction (task_type='instruction_following' etc.) the
generator uses that instruction directly.

Either way the output DataSample always carries:
  instruction = the question / prompt
  input       = source context (for downstream grounding gates)
  chosen      = high-quality, source-grounded response
  rejected    = lower-quality response (not hallucinated — just less helpful)
"""

from __future__ import annotations

import json
import re
import uuid

from tqdm import tqdm

from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, RejectedSample

# ── Corpus-mode (no pre-existing instruction) ────────────────────────────────

_CORPUS_PAIR_PROMPT = """You are a data generation expert creating preference training pairs from a source passage.

Source passage:
---
{context}
---

Generate:
1. A specific, focused question answerable from this passage
2. A HIGH-QUALITY chosen response: thorough, accurate, cites specific details from the passage
3. A LOWER-QUALITY rejected response — use EXACTLY ONE of these degradation patterns:
   - Answer only the surface-level question, omitting the most important specific detail the passage provides
   - Use vague language ("some factors", "various aspects") where the passage is concrete
   - Miss a key distinction or nuance that the passage explicitly makes
   Both responses must be factually correct and on-topic. The difference is depth and specificity, not accuracy.

Return JSON only:
{{
  "question": "...",
  "chosen": "thorough response citing specific passage details...",
  "rejected": "shallower response that omits key specifics...",
  "degradation_pattern": "which pattern was used for rejected"
}}"""

# ── Instruction-mode (pre-existing instruction) ──────────────────────────────

_DEFAULT_PAIR_PROMPT = """You are a data generation expert creating preference training pairs.

Given the following instruction, generate TWO responses with a CLEAR quality contrast:
1. A HIGH-QUALITY response (chosen): thorough, accurate, addresses all aspects with specifics
2. A LOWER-QUALITY response (rejected): use EXACTLY ONE degradation pattern:
   - Omit the most important specific detail that a good answer would include
   - Use vague language ("some", "various", "certain") where the chosen response is concrete
   - Answer only part of a multi-part question
   Both responses must be factually correct. The difference is depth and completeness, not accuracy.

Instruction:
---
{instruction}
---

{context_section}
Respond in JSON format ONLY:
{{
  "chosen": "thorough response with specific details...",
  "rejected": "shallower response omitting key specifics...",
  "degradation_pattern": "which pattern was used"
}}"""

_DEFAULT_CHOSEN_PROMPT = """You are an expert assistant. Provide a thorough, accurate, well-structured response.

Requirements:
- Be comprehensive but concise
- Use clear structure (paragraphs, examples where helpful)
- Ensure factual accuracy
- Address all aspects of the instruction

{context_section}
Instruction:
{instruction}"""

_DEFAULT_REJECTED_PROMPT = """Respond to the following instruction, but deliberately include ONE or TWO of these quality issues:
- Be vague or overly general where specifics are needed
- Miss one aspect of a multi-part question
- Use an awkward or confusing structure
- Provide a surface-level answer that lacks depth

The response should still be relevant and on-topic (not nonsensical), but noticeably worse.

{context_section}
Instruction:
{instruction}"""


class PreferenceGenerationTask(BaseGenerationTask):
    """
    Generate preference pairs (chosen/rejected) for DPO training.

    Parameters
    ----------
    llm : BaseLLM
    prompt_template : str | None
        Custom single-call template. Must contain {instruction} and
        optionally {context_section}.
    mode : str
        "single_call" — one LLM call generates both chosen and rejected.
        "two_pass"    — separate LLM calls for chosen and rejected.
    chosen_prompt : str | None
        Two-pass template for chosen (must contain {instruction},
        optionally {context_section}).
    rejected_prompt : str | None
        Two-pass template for rejected.
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str | None = None,
        mode: str = "single_call",
        chosen_prompt: str | None = None,
        rejected_prompt: str | None = None,
        concurrency: int = 10,
    ) -> None:
        super().__init__(llm=llm, prompt_template=prompt_template, concurrency=concurrency)
        self.mode = mode
        self.chosen_prompt = chosen_prompt or _DEFAULT_CHOSEN_PROMPT
        self.rejected_prompt = rejected_prompt or _DEFAULT_REJECTED_PROMPT
        if prompt_template:
            self._validate_template(prompt_template, ["instruction", "context_section"])
        if chosen_prompt:
            self._validate_template(
                chosen_prompt, ["instruction", "context_section"], "chosen_prompt"
            )
        if rejected_prompt:
            self._validate_template(
                rejected_prompt, ["instruction", "context_section"], "rejected_prompt"
            )

    # ── Corpus detection ─────────────────────────────────────────────────────

    def _is_corpus_mode(self, sample: DataSample) -> bool:
        return not sample.instruction and bool(self._get_source_context(sample))

    @staticmethod
    def _context_section(source_context: str) -> str:
        if not source_context:
            return ""
        return f"Source passage:\n---\n{source_context}\n---\n\n"

    # ── Message building ─────────────────────────────────────────────────────

    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        source_context = self._get_source_context(sample)

        if self._is_corpus_mode(sample):
            prompt = _CORPUS_PAIR_PROMPT.format(context=source_context)
        else:
            template = self.prompt_template or _DEFAULT_PAIR_PROMPT
            prompt = template.format(
                instruction=sample.instruction,
                context_section=self._context_section(source_context),
            )
        return [{"role": "user", "content": prompt}]

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        source_context = self._get_source_context(sample)
        corpus_mode = self._is_corpus_mode(sample)

        text = response.text.strip()
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        chosen = rejected = question = ""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                chosen = parsed.get("chosen", "")
                rejected = parsed.get("rejected", "")
                question = parsed.get("question", "")
        except json.JSONDecodeError:
            return self._fallback_parse(sample, text, source_context, corpus_mode)

        if not chosen or not rejected:
            return []

        instruction = question if corpus_mode else sample.instruction
        if not instruction:
            return []

        return [
            DataSample(
                id=str(uuid.uuid4()),
                source_uri=sample.source_uri,
                instruction=instruction,
                input=source_context,
                chosen=chosen,
                rejected=rejected,
                task_type="preference",
                metadata={
                    "generation_source": "preference_generator",
                    "generation_mode": self.mode,
                    "corpus_mode": corpus_mode,
                    "source_sample_id": sample.id,
                },
                provenance_chain=list(sample.provenance_chain),
            )
        ]

    def _fallback_parse(
        self, sample: DataSample, text: str, source_context: str, corpus_mode: bool
    ) -> list[DataSample]:
        chosen_match = re.search(
            r"(?:chosen|high.quality|better)[:\s]*(.*?)(?:(?:rejected|low.quality|worse)[:\s]|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        rejected_match = re.search(
            r"(?:rejected|low.quality|worse)[:\s]*(.*?)$",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        chosen = chosen_match.group(1).strip() if chosen_match else ""
        rejected = rejected_match.group(1).strip() if rejected_match else ""

        if not chosen or not rejected:
            return []

        instruction = sample.instruction if not corpus_mode else ""
        if corpus_mode and not instruction:
            return []

        return [
            DataSample(
                id=str(uuid.uuid4()),
                source_uri=sample.source_uri,
                instruction=instruction,
                input=source_context,
                chosen=chosen,
                rejected=rejected,
                task_type="preference",
                metadata={
                    "generation_source": "preference_generator",
                    "generation_mode": "fallback_parse",
                    "corpus_mode": corpus_mode,
                    "source_sample_id": sample.id,
                },
                provenance_chain=list(sample.provenance_chain),
            )
        ]

    # ── run() ────────────────────────────────────────────────────────────────

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        if self.mode == "single_call":
            return super().run(samples)

        # Two-pass mode — separate chosen / rejected calls
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._rejected = []
        results_map: dict[int, DataSample] = {}

        def _generate_pair(idx_sample):
            idx, sample = idx_sample
            try:
                source_context = self._get_source_context(sample)
                ctx_section = self._context_section(source_context)
                corpus_mode = self._is_corpus_mode(sample)

                # For corpus mode derive question inline from corpus prompt
                if corpus_mode:
                    corpus_resp = self.llm.generate(
                        [
                            {
                                "role": "user",
                                "content": _CORPUS_PAIR_PROMPT.format(context=source_context),
                            }
                        ]
                    )
                    text = corpus_resp.text.strip()
                    text = re.sub(r"```(?:json)?\s*", "", text)
                    text = re.sub(r"```\s*$", "", text)
                    try:
                        parsed = json.loads(text)
                        instruction = parsed.get("question", "")
                        chosen_text = parsed.get("chosen", "")
                        rejected_text = parsed.get("rejected", "")
                    except json.JSONDecodeError:
                        return idx, None, "corpus_parse_failed"
                    if not instruction or not chosen_text or not rejected_text:
                        return idx, None, "corpus_incomplete"
                else:
                    instruction = sample.instruction
                    chosen_prompt = self.chosen_prompt.format(
                        instruction=instruction,
                        context_section=ctx_section,
                    )
                    rejected_prompt = self.rejected_prompt.format(
                        instruction=instruction,
                        context_section=ctx_section,
                    )
                    chosen_resp = self.llm.generate([{"role": "user", "content": chosen_prompt}])
                    chosen_text = chosen_resp.text.strip()
                    rejected_resp = self.llm.generate(
                        [{"role": "user", "content": rejected_prompt}],
                        temperature=min(self.llm.temperature + 0.3, 1.5),
                    )
                    rejected_text = rejected_resp.text.strip()

                if not chosen_text or not rejected_text:
                    return idx, None, {"chosen": chosen_text, "rejected": rejected_text}

                return (
                    idx,
                    DataSample(
                        id=str(uuid.uuid4()),
                        source_uri=sample.source_uri,
                        instruction=instruction,
                        input=source_context,
                        chosen=chosen_text,
                        rejected=rejected_text,
                        task_type="preference",
                        metadata={
                            "generation_source": "preference_generator",
                            "generation_mode": "two_pass",
                            "corpus_mode": corpus_mode,
                            "source_sample_id": sample.id,
                        },
                        provenance_chain=list(sample.provenance_chain),
                    ),
                    None,
                )
            except Exception as e:
                return idx, None, str(e)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(_generate_pair, (i, s)): s for i, s in enumerate(samples)}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[PreferenceGen] two_pass",
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
                            rejection_reason=f"generation_parse_failed:{self.task_name}",
                            rejecting_step=self.task_name,
                            metadata={**sample.metadata, "partial_result": str(err)},
                        )
                    )

        return [results_map[i] for i in sorted(results_map)]
