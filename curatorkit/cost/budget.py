"""
TokenBudget — pre-flight estimate + mid-run enforcement.

The plan calls for two budget controls:
  1. Pre-run token-budget estimation (warn before spending begins)
  2. Mid-run enforcement that halts gracefully — exports what was produced
     instead of crashing with partial in-memory state.

`BudgetExceeded` is the controlled exception the gateway raises when a
hard cap fires. Curator catches it, stops further LLM calls, and lets the
pipeline finish exporting whatever data has accumulated so far.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised by CostGateway when the TokenBudget hard cap is reached.

    Carries the spent / cap counts so callers can render an honest
    cost-attribution report at halt time.

    Marked __no_retry__ so BaseLLM.generate() lets it propagate instead
    of swallowing it in the retry loop.
    """

    __no_retry__ = True

    def __init__(
        self,
        spent_tokens: int,
        spent_usd: float,
        token_cap: int | None,
        usd_cap: float | None,
    ) -> None:
        self.spent_tokens = spent_tokens
        self.spent_usd = spent_usd
        self.token_cap = token_cap
        self.usd_cap = usd_cap
        bits: list[str] = []
        if token_cap is not None:
            bits.append(f"tokens={spent_tokens}/{token_cap}")
        if usd_cap is not None:
            bits.append(f"usd=${spent_usd:.4f}/${usd_cap:.4f}")
        super().__init__(f"BudgetExceeded ({', '.join(bits) or 'no caps'})")


@dataclass
class TokenBudget:
    """
    Per-run token + USD spend cap.

    Either or both caps may be set. Soft / hard caps let callers distinguish
    a warning from a halt — typical usage is `soft = 0.8 * hard`.

    Thread-safe: increments are guarded by a Lock so the async pipeline can
    update the same budget from multiple workers without races.

    Counters:
      tokens_spent  — sum of total_tokens from every gateway call.
      usd_spent     — sum of cost estimates (LiteLLM cost_per_token table).
      calls_made    — number of LLM calls made through any wrapped gateway.
      calls_halted  — calls that were rejected because the hard cap was hit.
    """

    hard_token_cap: int | None = None
    soft_token_cap: int | None = None
    hard_usd_cap: float | None = None
    soft_usd_cap: float | None = None

    tokens_spent: int = 0
    usd_spent: float = 0.0
    calls_made: int = 0
    calls_halted: int = 0
    soft_warning_fired: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ─────────────────────────────────────────────────────────────────────
    # Spend bookkeeping
    # ─────────────────────────────────────────────────────────────────────

    def record(self, tokens: int, usd: float) -> None:
        """Atomically add the cost of one completed call."""
        with self._lock:
            self.tokens_spent += int(tokens)
            self.usd_spent += float(usd)
            self.calls_made += 1

    def check(self) -> None:
        """Raise BudgetExceeded if any hard cap is breached. Soft cap → warn."""
        with self._lock:
            if self.hard_token_cap is not None and self.tokens_spent >= self.hard_token_cap:
                self.calls_halted += 1
                raise BudgetExceeded(
                    self.tokens_spent,
                    self.usd_spent,
                    self.hard_token_cap,
                    self.hard_usd_cap,
                )
            if self.hard_usd_cap is not None and self.usd_spent >= self.hard_usd_cap:
                self.calls_halted += 1
                raise BudgetExceeded(
                    self.tokens_spent,
                    self.usd_spent,
                    self.hard_token_cap,
                    self.hard_usd_cap,
                )

            if not self.soft_warning_fired:
                soft_token_hit = (
                    self.soft_token_cap is not None and self.tokens_spent >= self.soft_token_cap
                )
                soft_usd_hit = self.soft_usd_cap is not None and self.usd_spent >= self.soft_usd_cap
                if soft_token_hit or soft_usd_hit:
                    import warnings

                    warnings.warn(
                        f"TokenBudget soft cap reached: "
                        f"tokens={self.tokens_spent} usd=${self.usd_spent:.4f}",
                        stacklevel=3,
                    )
                    self.soft_warning_fired = True

    # ─────────────────────────────────────────────────────────────────────
    # Pre-flight estimate
    # ─────────────────────────────────────────────────────────────────────

    def estimate_remaining(self) -> dict[str, float | int | None]:
        """Return how much room is left (None where no cap is set)."""
        with self._lock:
            return {
                "tokens_spent": self.tokens_spent,
                "usd_spent": round(self.usd_spent, 6),
                "tokens_remaining": (
                    None
                    if self.hard_token_cap is None
                    else max(0, self.hard_token_cap - self.tokens_spent)
                ),
                "usd_remaining": (
                    None
                    if self.hard_usd_cap is None
                    else round(max(0.0, self.hard_usd_cap - self.usd_spent), 6)
                ),
            }

    def reset(self) -> None:
        with self._lock:
            self.tokens_spent = 0
            self.usd_spent = 0.0
            self.calls_made = 0
            self.calls_halted = 0
            self.soft_warning_fired = False
