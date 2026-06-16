"""
BaseGenerationTask — abstract base for LLM-powered generation steps.

Extends BaseNormalizer interface so it slots into the pipeline without
any wiring changes. A generation task takes DataSamples (typically
text chunks or prompts) and returns enriched DataSamples (QA pairs,
preferences, rollouts, etc.).

Design:
  - Each task owns only the prompt template and output parser.
  - The LLM is a swappable backend passed at construction time.
  - Tasks append ProvenanceRecord with model info and prompt hash.
  - Failed generations produce RejectedSample objects collected via
    the task's .rejected property (flushed by the pipeline runner).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import abstractmethod
from datetime import UTC, datetime

from tqdm import tqdm

from curatorkit.interfaces import BaseNormalizer
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

STEP_VERSION = "1.0.0"


class BaseGenerationTask(BaseNormalizer):
    """
    Abstract base for LLM generation tasks.

    Subclasses must implement:
      _build_messages(sample) -> list[dict[str, str]]
          Build the prompt messages for the LLM call.

      _parse_response(sample, response) -> list[DataSample]
          Parse the LLM output into one or more DataSamples.

    Parameters
    ----------
    llm : BaseLLM
        LLM backend to use for generation.
    prompt_template : str | None
        Custom prompt template. If None, uses the task's default.
    concurrency : int
        Number of concurrent async LLM calls (used by run_async).
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str | None = None,
        concurrency: int = 10,
        max_parse_retries: int = 1,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.concurrency = concurrency
        self.max_parse_retries = max_parse_retries
        self._rejected: list[RejectedSample] = []

    @property
    def task_name(self) -> str:
        """Human-readable task name for provenance."""
        return type(self).__name__

    @property
    def rejected(self) -> list[RejectedSample]:
        """Samples that failed generation. Flushed by pipeline after run()."""
        return self._rejected

    def flush_rejected(self) -> list[RejectedSample]:
        """Return and clear accumulated rejected samples."""
        out = list(self._rejected)
        self._rejected = []
        return out

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_template(
        template: str, required_vars: list[str], name: str = "prompt_template"
    ) -> None:
        """Raise ValueError if any required {var} placeholder is absent from template.

        Call this in __init__ when the user supplies a custom prompt_template so
        they get a clear error at construction time rather than a KeyError at
        generation time.
        """
        missing = [v for v in required_vars if "{" + v + "}" not in template]
        if missing:
            raise ValueError(
                f"Custom {name} is missing required placeholder(s): "
                + ", ".join(f"{{{v}}}" for v in missing)
                + f". Available: {', '.join('{' + v + '}' for v in required_vars)}"
            )

    def _get_source_context(self, sample: DataSample) -> str:
        """Extract source text from any input format.

        Corpus chunks arrive as task_type='language_modeling' with source text
        in sample.output. Instruction-following samples carry source in sample.input.
        All generators should use this so downstream grounding gates receive
        a populated sample.input regardless of the upstream data format.
        """
        if sample.task_type == "language_modeling" and sample.output:
            return sample.output
        return sample.input or sample.output or ""

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_messages(self, sample: DataSample) -> list[dict[str, str]]:
        """Build the LLM prompt messages for this sample."""
        ...

    @abstractmethod
    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        """Parse the LLM response into one or more DataSamples."""
        ...

    # ------------------------------------------------------------------
    # Sync pipeline interface
    # ------------------------------------------------------------------

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        """
        Synchronous generation. Calls the LLM once per input sample.

        Returns enriched DataSamples. Failed generations are collected
        into self.rejected (not returned in the output list).
        """
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC)

        for sample in tqdm(samples, desc=f"[{self.task_name}] generating", unit="sample"):
            try:
                messages = self._build_messages(sample)
                response = self.llm.generate(messages)
                generated = self._parse_response(sample, response)

                # Retry on parse failure up to max_parse_retries times
                attempt = 0
                while not generated and attempt < self.max_parse_retries:
                    attempt += 1
                    response = self.llm.generate(messages)
                    generated = self._parse_response(sample, response)

                if not generated:
                    # Store raw LLM response so failures can be diagnosed
                    self._rejected.append(
                        RejectedSample(
                            **sample.model_dump(exclude={"metadata"}),
                            rejection_reason=f"generation_parse_failed:{self.task_name}",
                            rejecting_step=self.task_name,
                            metadata={
                                **sample.metadata,
                                "raw_llm_response": response.text[:2000],
                                "parse_attempts": attempt + 1,
                            },
                        )
                    )
                    continue

                prompt_hash = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:12]

                for gen_sample in generated:
                    gen_sample.append_provenance(
                        ProvenanceRecord(
                            step_name=self.task_name,
                            step_version=STEP_VERSION,
                            timestamp=ts,
                            config_hash=self.llm.config_hash(),
                            notes={
                                **response.to_provenance_dict(),
                                "prompt_hash": prompt_hash,
                                "source_sample_id": sample.id,
                                "generation_task": self.task_name,
                            },
                        )
                    )
                    results.append(gen_sample)

            except Exception as e:
                self._rejected.append(
                    RejectedSample(
                        **sample.model_dump(),
                        rejection_reason=f"generation_failed:{type(e).__name__}:{e}",
                        rejecting_step=self.task_name,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Async pipeline interface
    # ------------------------------------------------------------------

    async def run_async(self, samples: list[DataSample]) -> list[DataSample]:
        """
        Async generation with concurrency control.

        Fires up to self.concurrency LLM calls in parallel.
        """
        self._rejected = []
        results: list[DataSample] = []
        ts = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _process_one(sample: DataSample) -> list[DataSample]:
            async with semaphore:
                try:
                    messages = self._build_messages(sample)
                    response = await self.llm.agenerate(messages)
                    generated = self._parse_response(sample, response)

                    attempt = 0
                    while not generated and attempt < self.max_parse_retries:
                        attempt += 1
                        response = await self.llm.agenerate(messages)
                        generated = self._parse_response(sample, response)

                    if not generated:
                        self._rejected.append(
                            RejectedSample(
                                **sample.model_dump(exclude={"metadata"}),
                                rejection_reason=f"generation_parse_failed:{self.task_name}",
                                rejecting_step=self.task_name,
                                metadata={
                                    **sample.metadata,
                                    "raw_llm_response": response.text[:2000],
                                    "parse_attempts": attempt + 1,
                                },
                            )
                        )
                        return []

                    prompt_hash = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:12]

                    for gen_sample in generated:
                        gen_sample.append_provenance(
                            ProvenanceRecord(
                                step_name=self.task_name,
                                step_version=STEP_VERSION,
                                timestamp=ts,
                                config_hash=self.llm.config_hash(),
                                notes={
                                    **response.to_provenance_dict(),
                                    "prompt_hash": prompt_hash,
                                    "source_sample_id": sample.id,
                                    "generation_task": self.task_name,
                                },
                            )
                        )
                    return generated

                except Exception as e:
                    self._rejected.append(
                        RejectedSample(
                            **sample.model_dump(),
                            rejection_reason=f"generation_failed:{type(e).__name__}:{e}",
                            rejecting_step=self.task_name,
                        )
                    )
                    return []

        # Process in chunks to bound the number of live coroutine objects.
        # The semaphore already caps concurrent API calls; chunking caps memory
        # from coroutine objects + pending result lists for very large corpora.
        chunk_size = max(self.concurrency * 8, 256)
        pbar_desc = f"[{self.task_name}] generating"
        with tqdm(total=len(samples), desc=pbar_desc, unit="sample") as pbar:
            for chunk_start in range(0, len(samples), chunk_size):
                chunk = samples[chunk_start : chunk_start + chunk_size]
                chunk_tasks = [_process_one(s) for s in chunk]
                chunk_results = await asyncio.gather(*chunk_tasks)
                for generated in chunk_results:
                    results.extend(generated)
                pbar.update(len(chunk))
                del chunk_tasks, chunk_results

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _config_hash(self) -> str:
        """Config hash combining task name and LLM config."""
        payload = json.dumps(
            {
                "task": self.task_name,
                "llm": self.llm.config_hash(),
                "prompt_template": self.prompt_template or "default",
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
