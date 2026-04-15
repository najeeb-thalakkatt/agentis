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
    and can enforce a spending budget. Includes stall detection to
    signal when the agent is spinning without progress.
    """

    name: str = "cost_tracker"

    def __init__(
        self,
        budget_usd: float | None = None,
        input_price_per_token: float = _DEFAULT_PRICING["input"],
        output_price_per_token: float = _DEFAULT_PRICING["output"],
        stall_turn_threshold: int = 3,
        stall_budget_fraction: float = 0.7,
    ) -> None:
        self._budget = budget_usd
        self._input_price = input_price_per_token
        self._output_price = output_price_per_token
        self._records: list[CostRecord] = []
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_cost: float = 0.0

        # Stall detection
        self._stall_turn_threshold = stall_turn_threshold
        self._stall_budget_fraction = stall_budget_fraction
        self._seen_tools: set[str] = set()
        self._stall_turns: int = 0
        self._should_force_conclude: bool = False
        self._stall_fires: int = 0

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

        # Stall detection: check tool calls in this turn's response
        tool_calls = getattr(result, "tool_calls", None) or []
        tool_signatures: list[str] = []
        for tc in tool_calls:
            if hasattr(tc, "name"):
                # Use (name, sorted args) as a signature — same tool with
                # different arguments counts as progress, not a stall.
                args_key = ""
                if hasattr(tc, "arguments") and tc.arguments:
                    args_key = str(sorted(tc.arguments.items()))
                tool_signatures.append(f"{tc.name}:{args_key}")
        self._record_turn_for_stall(tool_signatures, self._total_cost)

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
            "stall_fires": self._stall_fires,
        }

    # ── Stall Detection ────────────────────────────────────

    def _record_turn_for_stall(
        self,
        tool_names: list[str],
        cost_so_far: float,
    ) -> None:
        """Update stall tracking after a turn.

        A turn is "stalling" if it introduces no previously unseen tool.
        After ``stall_turn_threshold`` consecutive stall turns AND the
        cost exceeds ``stall_budget_fraction`` of budget, the tracker
        signals the runtime to force a conclusion.
        """
        if self._budget is None:
            return

        # Check if any new tool appeared this turn
        new_tools = set(tool_names) - self._seen_tools
        self._seen_tools.update(tool_names)

        if new_tools:
            self._stall_turns = 0
        else:
            self._stall_turns += 1

        # Signal if stalling AND over budget fraction
        if (
            self._stall_turns >= self._stall_turn_threshold
            and cost_so_far >= self._budget * self._stall_budget_fraction
        ):
            if not self._should_force_conclude:
                self._stall_fires += 1
            self._should_force_conclude = True
            logger.warning(
                "Stall detected: %d turns without new tools, "
                "cost $%.4f / $%.4f (%.0f%% of budget)",
                self._stall_turns,
                cost_so_far,
                self._budget,
                (cost_so_far / self._budget) * 100,
            )

    @property
    def should_force_conclude(self) -> bool:
        """Whether the runtime should force the agent to conclude.

        This is a consume-once flag: reading it resets the signal so
        subsequent calls return False until a new stall is detected.
        """
        if self._should_force_conclude:
            self._should_force_conclude = False
            return True
        return False

    @property
    def force_conclude_message(self) -> str:
        """Message to inject when forcing conclusion."""
        pct = 0.0
        if self._budget and self._budget > 0:
            pct = (self._total_cost / self._budget) * 100
        return (
            f"You have used {pct:.0f}% of your budget without new findings "
            f"in the last {self._stall_turns} turns. Write your best answer "
            f"with what you have. Do not make additional tool calls."
        )
