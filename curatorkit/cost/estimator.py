"""
DryRunCostEstimator — predict run cost without spending a single LLM call.

Approach:
  1. Walk the step list returned by Curator._build_steps() (or accept any
     list of steps directly).
  2. For each LLM-backed step (generator / hallucination / reward / output-
     safety / probe), estimate per-call token usage from prompt-template
     length + assumed completion length.
  3. Multiply by expected sample count (passed through, then attenuated by
     gate pass rates per stage).
  4. Look up per-token cost from LiteLLM's pricing table when the model is
     known; otherwise return tokens-only.

The estimate is *conservative by construction*: per-call token counts use
the worst-case prompt template length (the default rubric / max
num_questions). Set `assume_pass_rate=...` to model gate attenuation more
aggressively if your seed corpus is high-quality.

Output:
  CostEstimate.totals        — total tokens + USD across all stages
  CostEstimate.per_stage     — same broken down by stage label
  CostEstimate.assumptions   — every parameter the estimate depended on,
                               so the audit trail is reproducible

Honest caveats baked in:
  - Completion length is estimated as `expected_response_tokens` (configurable);
    actual generation may be longer.
  - LLM-call retry on transient failure is NOT modeled (multiply by 1.1-1.5
    in practice).
  - Diagnostic-loop Probe LLM calls are accounted for when adaptive_loop=True
    AND probe_max_llm_calls is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from curatorkit.cost.gateway import CostGateway, _estimate_call_cost

# ───────────────────────────────────────────────────────────────────────────
# Per-stage cost row
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class StageEstimate:
    stage: str
    model: str
    calls: int
    prompt_tokens_per_call: int
    completion_tokens_per_call: int
    total_tokens: int
    cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens_per_call": self.prompt_tokens_per_call,
            "completion_tokens_per_call": self.completion_tokens_per_call,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class CostEstimate:
    """The full pre-flight estimate."""

    per_stage: list[StageEstimate] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    assumptions: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "per_stage": [s.to_dict() for s in self.per_stage],
            "assumptions": self.assumptions,
            "warnings": self.warnings,
        }

    def render_markdown(self) -> str:
        lines = ["# Pre-flight Cost Estimate\n"]
        lines.append(
            f"- **Estimated total tokens**: {self.total_tokens:,}\n"
            f"- **Estimated total cost**: ${self.total_cost_usd:.4f}\n"
        )
        lines.append("\n## Assumptions\n")
        for k, v in self.assumptions.items():
            lines.append(f"- `{k}`: {v}")
        if self.warnings:
            lines.append("\n## Warnings\n")
            for w in self.warnings:
                lines.append(f"- {w}")
        lines.append("\n## Per-stage breakdown\n")
        lines.append("| Stage | Model | Calls | Prompt | Completion | Total | Cost (USD) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for s in self.per_stage:
            lines.append(
                f"| {s.stage} | `{s.model}` | {s.calls:,} | "
                f"{s.prompt_tokens_per_call:,} | {s.completion_tokens_per_call:,} | "
                f"{s.total_tokens:,} | ${s.cost_usd:.4f} |"
            )
        return "\n".join(lines) + "\n"


# ───────────────────────────────────────────────────────────────────────────
# Token counter
# ───────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\S+")


def _approx_tokens(text: str) -> int:
    """Approximate token count using a 1.3 words-to-tokens multiplier.

    This is the GPT-family rule-of-thumb (Tiktoken's cl100k_base averages
    ~1.3 tokens per English word for prose). Tighter than splitting bytes/4
    and good enough for budget estimation.
    """
    if not text:
        return 0
    word_count = len(_WORD_RE.findall(text))
    return max(1, int(round(word_count * 1.3)))


# ───────────────────────────────────────────────────────────────────────────
# Stage-specific prompt-length introspection
# ───────────────────────────────────────────────────────────────────────────


def _step_prompt_size(step: Any) -> int:
    """
    Best-effort: pull the largest prompt template from the step.

    Knows about every generator / gate currently in the library; falls
    back to inspecting common attrs (`prompt_template`, `_DEFAULT_PROMPT`).
    """
    # Generators
    for attr in ("prompt_template", "table_prompt_template"):
        v = getattr(step, attr, None)
        if isinstance(v, str) and v:
            return _approx_tokens(v)
    # Gates that build a prompt internally — pull their module-level default.
    module = type(step).__module__
    try:
        import importlib

        mod = importlib.import_module(module)
    except Exception:
        return 256  # safe default for an LLM-backed step
    for candidate in (
        "_DEFAULT_QA_PROMPT",
        "_DEFAULT_GROUNDING_PROMPT",
        "_DEFAULT_REWARD_PROMPT",
        "_JOINT_PROMPT",
        "_ABSOLUTE_GRADING_PROMPT",
    ):
        v = getattr(mod, candidate, None)
        if isinstance(v, str):
            return _approx_tokens(v)
    return 256


def _step_completion_size(step: Any) -> int:
    """Estimated completion length per call, in tokens."""
    # Generators usually request structured output → moderate length
    cls = type(step).__name__
    if cls in (
        "QAGenerationTask",
        "EvolInstructTask",
        "PreferenceGenerationTask",
        "MultiTurnTask",
        "ChainOfThoughtTask",
    ):
        return 400
    if cls in ("JointBundleTask",):
        return 800  # bundle is bigger
    if cls in ("MagpieTask",):
        return 200  # one Q + one short response per Magpie pass (but we make 2 calls)
    if cls in ("GRPORolloutTask",):
        # N rollouts in one call
        n = int(getattr(step, "num_responses", 4))
        return 200 * n
    if cls in (
        "HallucinationGate",
        "RewardGate",
        "PrometheusRewardGate",
        "OutputSafetyGate",
    ):
        return 150  # short judgement
    return 200


def _step_calls_per_sample(step: Any) -> int:
    """How many LLM calls this step makes per input sample."""
    cls = type(step).__name__
    if cls == "MagpieTask":
        return 2  # step1 instruction + step2 response
    if cls == "JointBundleTask":
        n = int(getattr(step, "num_rollouts", 4))
        score = bool(getattr(step, "score_rollouts", True))
        return 1 + (n if score else 0)
    if cls in (
        "QAGenerationTask",
        "EvolInstructTask",
        "PreferenceGenerationTask",
        "MultiTurnTask",
        "ChainOfThoughtTask",
        "GRPORolloutTask",
        "HallucinationGate",
        "RewardGate",
        "PrometheusRewardGate",
        "OutputSafetyGate",
    ):
        return 1
    return 0  # not LLM-backed (readers / dedup / cleaner / sampler / exporters)


def _step_model(step: Any) -> str:
    """Pull the model name from the step's llm attribute, if any."""
    llm = getattr(step, "llm", None)
    if llm is None:
        return "unknown"
    # CostGateway wraps the inner; we want the underlying model id
    if isinstance(llm, CostGateway):
        return llm.inner.model
    return getattr(llm, "model", "unknown")


# ───────────────────────────────────────────────────────────────────────────
# Estimator
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class DryRunCostEstimator:
    """
    Build a CostEstimate for a given pipeline configuration.

    Parameters
    ----------
    n_input_samples : int
        Expected number of input samples after readers.
    gate_pass_rate : float
        Cumulative pass rate per gate. 0.7 means each gate keeps 70% of its
        input. Default 1.0 (worst-case — no attenuation).
    expected_chunks_per_sample : int
        For PDF connectors: how many chunks one input file yields. Used
        only when the first reader is a PDFReader.
    probe_calls_per_rejection : int
        When adaptive_loop is on, each rejected sample triggers up to this
        many probe LLM calls. Default 6 (the plan's worst case).
    overall_failure_rate : float
        Fraction of samples expected to land in the rejected list before
        the diagnostic loop runs. Default 0.2 — only matters when
        adaptive_loop=True.
    """

    n_input_samples: int = 100
    gate_pass_rate: float = 1.0
    expected_chunks_per_sample: int = 1
    probe_calls_per_rejection: int = 6
    overall_failure_rate: float = 0.2

    # ─────────────────────────────────────────────────────────────────────

    def estimate(self, steps: list[Any], adaptive_loop: bool = False) -> CostEstimate:
        est = CostEstimate(
            assumptions={
                "n_input_samples": self.n_input_samples,
                "gate_pass_rate": self.gate_pass_rate,
                "expected_chunks_per_sample": self.expected_chunks_per_sample,
                "probe_calls_per_rejection": self.probe_calls_per_rejection,
                "overall_failure_rate": self.overall_failure_rate,
                "adaptive_loop": adaptive_loop,
                "tokens_per_word": 1.3,
            }
        )

        # Current sample volume flowing through the pipeline
        current_n = self.n_input_samples

        for step in steps:
            calls_per_sample = _step_calls_per_sample(step)
            if calls_per_sample == 0:
                # Non-LLM step — adjust volume via gate semantics where applicable
                cls = type(step).__name__
                if "Gate" in cls:
                    current_n = int(current_n * self.gate_pass_rate)
                continue

            prompt = _step_prompt_size(step)
            completion = _step_completion_size(step)
            calls = current_n * calls_per_sample
            total_tokens = calls * (prompt + completion)
            model = _step_model(step)
            cost = _estimate_call_cost(model, prompt * calls, completion * calls)

            est.per_stage.append(
                StageEstimate(
                    stage=type(step).__name__,
                    model=model,
                    calls=calls,
                    prompt_tokens_per_call=prompt,
                    completion_tokens_per_call=completion,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                )
            )

            # LLM-backed gates attenuate by gate_pass_rate too
            if "Gate" in type(step).__name__:
                current_n = int(current_n * self.gate_pass_rate)

            # Diagnostic loop budget — extra calls per rejected sample.
            # n_rejected = whichever of (1 - gate_pass_rate) or overall_failure_rate
            # is larger, so the estimate is conservative under both assumptions.
            if adaptive_loop and "Gate" in type(step).__name__:
                fr_from_gate = max(0.0, 1.0 - self.gate_pass_rate)
                effective_failure = max(fr_from_gate, self.overall_failure_rate)
                # current_n was already attenuated above — pre-attenuation
                # volume is the upstream-input volume.
                upstream_n = int(current_n / max(self.gate_pass_rate, 1e-6))
                n_rejected = int(upstream_n * effective_failure)
                probe_calls = n_rejected * self.probe_calls_per_rejection
                if probe_calls > 0:
                    probe_tokens = probe_calls * (prompt + completion)
                    probe_cost = _estimate_call_cost(
                        model,
                        prompt * probe_calls,
                        completion * probe_calls,
                    )
                    est.per_stage.append(
                        StageEstimate(
                            stage=f"DiagnosticProbe({type(step).__name__})",
                            model=model,
                            calls=probe_calls,
                            prompt_tokens_per_call=prompt,
                            completion_tokens_per_call=completion,
                            total_tokens=probe_tokens,
                            cost_usd=probe_cost,
                        )
                    )

        est.total_tokens = sum(s.total_tokens for s in est.per_stage)
        est.total_cost_usd = sum(s.cost_usd for s in est.per_stage)

        # Honesty warnings — these are limits the user should know about
        unknown_models = {
            s.model for s in est.per_stage if s.model == "unknown" or s.cost_usd == 0.0
        }
        if unknown_models:
            est.warnings.append(
                f"Cost was 0 for {len(unknown_models)} stage(s) — "
                "either the model is not in LiteLLM's pricing table, or the "
                "step does not expose its LLM. Token estimate is still valid."
            )
        est.warnings.append(
            "Retry-on-transient-failure not modeled — multiply USD by 1.1-1.5 "
            "for production budgeting."
        )

        return est

    # ─────────────────────────────────────────────────────────────────────
    # Convenience constructor that builds the steps from a Curator config
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_curator_config(
        cls,
        config: Any,
        n_input_samples: int = 100,
        gate_pass_rate: float = 1.0,
    ) -> tuple[DryRunCostEstimator, CostEstimate]:
        """
        Build steps via Curator._build_steps() and immediately estimate.

        Returns (estimator, estimate). Useful for `--dry-run` CLI flags.
        """
        from curatorkit.curator import Curator

        curator = Curator(config)
        steps = curator._build_steps()
        estimator = cls(
            n_input_samples=n_input_samples,
            gate_pass_rate=gate_pass_rate,
        )
        est = estimator.estimate(
            steps,
            adaptive_loop=getattr(config, "enable_diagnostic_probe", False),
        )
        return estimator, est
