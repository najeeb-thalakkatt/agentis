"""CostTracker — token and cost tracking with budget limits.

Tracks per-turn token usage and estimated costs. Can enforce a
spending budget. Opt-in via extensions=[CostTracker(budget_usd=5.0)].
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentis.types import CostRecord, TokenUsage

logger = logging.getLogger("agentis")

# Conservative default pricing (USD per 1M tokens)
# Users can override by subclassing or setting _pricing
_DEFAULT_PRICING = {
    "input": 3.0 / 1_000_000,   # $3 per 1M input tokens
    "output": 15.0 / 1_000_000,  # $15 per 1M output tokens
}


class CostTracker:
    """Extension that tracks token usage and estimated costs.

    Implements the Extension protocol. Records usage after each turn
    and can enforce a spending budget.
    """

    name: str = "cost_tracker"

    def __init__(
        self,
        budget_usd: float | None = None,
        input_price_per_token: float = _DEFAULT_PRICING["input"],
        output_price_per_token: float = _DEFAULT_PRICING["output"],
    ) -> None:
        self._budget = budget_usd
        self._input_price = input_price_per_token
        self._output_price = output_price_per_token
        self._records: list[CostRecord] = []
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_cost: float = 0.0

    async def on_runtime_start(self, runtime: Any) -> None:
        """Called when the runtime starts."""

    async def on_turn_end(self, runtime: Any, result: Any) -> None:
        """Record token usage and cost after each turn."""
        usage = getattr(result, "usage", None)
        if not isinstance(usage, TokenUsage):
            return

        cost = (
            usage.input_tokens * self._input_price
            + usage.output_tokens * self._output_price
        )

        self._records.append(CostRecord(
            provider="unknown",
            model="unknown",
            usage=usage,
            estimated_cost_usd=cost,
            timestamp=time.time(),
        ))

        self._total_input += usage.input_tokens
        self._total_output += usage.output_tokens
        self._total_cost += cost

        if self._budget is not None and self._total_cost > self._budget:
            logger.warning(
                "Cost budget exceeded: $%.4f / $%.4f",
                self._total_cost, self._budget,
            )

    async def on_idle(self, runtime: Any, idle_seconds: float) -> None:
        """No-op for cost tracker."""

    async def on_runtime_stop(self, runtime: Any) -> None:
        """Called when the runtime stops."""

    def total_cost(self) -> float:
        """Return total estimated cost in USD."""
        return self._total_cost

    def total_tokens(self) -> TokenUsage:
        """Return aggregated token usage."""
        return TokenUsage(
            input_tokens=self._total_input,
            output_tokens=self._total_output,
        )

    def is_over_budget(self) -> bool:
        """Check if spending has exceeded the budget."""
        if self._budget is None:
            return False
        return self._total_cost > self._budget

    def get_report(self) -> dict[str, Any]:
        """Return a summary report of costs and usage."""
        return {
            "total_cost_usd": self._total_cost,
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "num_turns": len(self._records),
            "budget_usd": self._budget,
            "over_budget": self.is_over_budget(),
        }
