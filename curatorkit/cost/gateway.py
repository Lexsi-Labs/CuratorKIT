"""
CostGateway — wraps any BaseLLM with token / cost / budget enforcement.

  - Pre-call: checks the shared TokenBudget for room.
  - Post-call: records the call's tokens + USD into the budget and a
    CostReport call log.
  - Mid-run: raises BudgetExceeded if a hard cap is hit. The Curator
    catches this and finalises the pipeline gracefully.

The gateway delegates the actual LLM call to the wrapped BaseLLM, so any
backend (LiteLLMBackend, OllamaBackend, custom) works without subclassing.

Cost estimation: LiteLLM's `completion_cost()` for known models. When the
model is unknown (Ollama, vLLM-local, custom OpenAI-compatible endpoint),
the gateway records 0.0 USD but still tracks tokens.
"""

from __future__ import annotations

import time
from typing import Any

from curatorkit.cost.budget import TokenBudget
from curatorkit.cost.report import CallRecord, CostReport
from curatorkit.llm.base import BaseLLM, LLMResponse


def _estimate_call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort cost estimate. Returns 0.0 if cost data is unavailable."""
    try:
        import litellm  # type: ignore

        # LiteLLM keeps a per-model {input, output}-token price table.
        # Older versions exposed `completion_cost` with prompt + completion
        # kwargs; newer versions accept a `response` object. Cover both.
        try:
            return float(
                litellm.completion_cost(  # type: ignore[attr-defined]
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        except (TypeError, AttributeError):
            # Last resort: per-token table direct lookup
            cost = litellm.model_cost.get(model)  # type: ignore[attr-defined]
            if not cost:
                return 0.0
            return float(
                cost.get("input_cost_per_token", 0.0) * prompt_tokens
                + cost.get("output_cost_per_token", 0.0) * completion_tokens
            )
    except Exception:
        return 0.0


class CostGateway(BaseLLM):
    """
    BaseLLM-compatible wrapper that meters every call.

    Parameters
    ----------
    inner : BaseLLM
        The underlying backend the gateway delegates to.
    budget : TokenBudget
        Shared budget for the run. A single TokenBudget instance can be
        shared across multiple gateways (one per generator / gate) — the
        budget tracks global spend.
    stage_name : str
        Label used for per-stage attribution in CostReport. Generators set
        this to their class name; gates do the same.
    report : Optional[CostReport]
        Where call records are appended. Pass a shared report so every
        gateway logs into the same run-wide report.
    """

    def __init__(
        self,
        inner: BaseLLM,
        budget: TokenBudget,
        stage_name: str = "generic",
        report: CostReport | None = None,
    ) -> None:
        # Mirror enough of BaseLLM's surface that retry / temperature
        # introspection still works on the gateway directly.
        super().__init__(
            model=inner.model,
            temperature=inner.temperature,
            max_tokens=inner.max_tokens,
            api_key=getattr(inner, "api_key", None),
            timeout=getattr(inner, "timeout", 120.0),
            max_retries=getattr(inner, "max_retries", 3),
        )
        self.inner = inner
        self.budget = budget
        self.stage_name = stage_name
        self.report = report or CostReport()

    # ─────────────────────────────────────────────────────────────────────
    # BaseLLM hooks
    # ─────────────────────────────────────────────────────────────────────

    def _call(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        # Pre-flight check — raises BudgetExceeded if cap already hit.
        self.budget.check()

        t0 = time.monotonic()
        response = self.inner._call(messages, **kwargs)
        latency = time.monotonic() - t0

        self._post_call(response, latency)
        return response

    async def _acall(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.budget.check()
        t0 = time.monotonic()
        response = await self.inner._acall(messages, **kwargs)
        latency = time.monotonic() - t0
        self._post_call(response, latency)
        return response

    # ─────────────────────────────────────────────────────────────────────
    # Accounting
    # ─────────────────────────────────────────────────────────────────────

    def _post_call(self, response: LLMResponse, latency: float) -> None:
        cost = _estimate_call_cost(
            self.inner.model,
            response.prompt_tokens,
            response.completion_tokens,
        )
        self.budget.record(response.total_tokens, cost)
        self.report.add(
            CallRecord(
                stage=self.stage_name,
                model=self.inner.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                cost_usd=cost,
                latency_seconds=latency,
            )
        )
        # Post-call check — also raises if the call itself crossed the cap.
        # Done after recording so the report stays consistent.
        self.budget.check()

    # ─────────────────────────────────────────────────────────────────────
    # Convenience constructors
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def wrap(
        cls,
        inner: BaseLLM,
        budget: TokenBudget,
        stage_name: str,
        report: CostReport,
    ) -> CostGateway:
        """Idempotent: if `inner` is already a CostGateway, rebind its stage."""
        if isinstance(inner, cls):
            inner.stage_name = stage_name
            inner.budget = budget
            inner.report = report
            return inner
        return cls(inner=inner, budget=budget, stage_name=stage_name, report=report)
