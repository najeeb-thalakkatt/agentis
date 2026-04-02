"""Tests for agentis.extensions.cost_tracker — CostTracker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentis.extensions.cost_tracker import CostTracker
from agentis.types import ProviderResponse, TokenUsage


def _mock_result(input_tokens: int = 100, output_tokens: int = 50) -> ProviderResponse:
    return ProviderResponse(
        content="response",
        tool_calls=[],
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class TestCostTrackerBasics:
    def test_name(self) -> None:
        ct = CostTracker()
        assert ct.name == "cost_tracker"

    def test_initial_state(self) -> None:
        ct = CostTracker()
        assert ct.total_cost() == 0.0
        assert ct.total_tokens().input_tokens == 0
        assert ct.total_tokens().output_tokens == 0

    def test_not_over_budget_initially(self) -> None:
        ct = CostTracker(budget_usd=10.0)
        assert ct.is_over_budget() is False

    def test_no_budget_never_over(self) -> None:
        ct = CostTracker()  # No budget set
        assert ct.is_over_budget() is False


class TestCostTrackerTracking:
    @pytest.mark.asyncio
    async def test_on_turn_end_records_usage(self) -> None:
        ct = CostTracker()
        await ct.on_turn_end(MagicMock(), _mock_result(100, 50))

        tokens = ct.total_tokens()
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 50

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate(self) -> None:
        ct = CostTracker()
        await ct.on_turn_end(MagicMock(), _mock_result(100, 50))
        await ct.on_turn_end(MagicMock(), _mock_result(200, 100))

        tokens = ct.total_tokens()
        assert tokens.input_tokens == 300
        assert tokens.output_tokens == 150

    @pytest.mark.asyncio
    async def test_cost_estimated(self) -> None:
        ct = CostTracker()
        await ct.on_turn_end(MagicMock(), _mock_result(1000, 500))

        # Should have some estimated cost > 0
        assert ct.total_cost() > 0.0

    @pytest.mark.asyncio
    async def test_budget_exceeded(self) -> None:
        ct = CostTracker(budget_usd=0.001)  # Very low budget
        # Large usage to exceed budget
        await ct.on_turn_end(MagicMock(), _mock_result(100_000, 50_000))

        assert ct.is_over_budget() is True


class TestCostTrackerReport:
    @pytest.mark.asyncio
    async def test_report_structure(self) -> None:
        ct = CostTracker()
        await ct.on_turn_end(MagicMock(), _mock_result(100, 50))
        await ct.on_turn_end(MagicMock(), _mock_result(200, 100))

        report = ct.get_report()
        assert "total_cost_usd" in report
        assert "total_input_tokens" in report
        assert "total_output_tokens" in report
        assert "num_turns" in report
        assert report["num_turns"] == 2

    @pytest.mark.asyncio
    async def test_report_empty(self) -> None:
        ct = CostTracker()
        report = ct.get_report()
        assert report["num_turns"] == 0
        assert report["total_cost_usd"] == 0.0


class TestCostTrackerLifecycle:
    @pytest.mark.asyncio
    async def test_on_runtime_start(self) -> None:
        ct = CostTracker()
        await ct.on_runtime_start(MagicMock())  # Should not raise

    @pytest.mark.asyncio
    async def test_on_runtime_stop(self) -> None:
        ct = CostTracker()
        await ct.on_runtime_stop(MagicMock())  # Should not raise

    @pytest.mark.asyncio
    async def test_on_idle(self) -> None:
        ct = CostTracker()
        await ct.on_idle(MagicMock(), idle_seconds=30.0)  # Should not raise

    @pytest.mark.asyncio
    async def test_handles_non_provider_response(self) -> None:
        """on_turn_end should handle unexpected result types gracefully."""
        ct = CostTracker()
        # Pass a non-ProviderResponse — should not crash
        await ct.on_turn_end(MagicMock(), MagicMock(spec=[]))
        assert ct.total_tokens().input_tokens == 0
