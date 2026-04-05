"""Tests for Improvement 2: Per-tool use_utility flag.

Tools marked with use_utility=True should be routed through the
utility (cheaper) provider when one is available. Tools without
the flag always use the primary provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool
from agentis.types import Permission


# ── Tests: Decorator ───────────────────────────────────────


class TestUseUtilityDecorator:
    """Test that @tool accepts and stores use_utility flag."""

    def test_use_utility_defaults_to_false(self) -> None:
        """Without use_utility, the flag defaults to False."""

        @tool()
        async def basic_tool(query: str) -> str:
            """A basic tool."""
            return query

        assert basic_tool.use_utility is False

    def test_use_utility_true(self) -> None:
        """use_utility=True is stored on the FunctionTool."""

        @tool(use_utility=True)
        async def cheap_tool(query: str) -> str:
            """A cheap tool."""
            return query

        assert cheap_tool.use_utility is True

    def test_use_utility_false_explicit(self) -> None:
        """use_utility=False is stored on the FunctionTool."""

        @tool(use_utility=False)
        async def smart_tool(claim: str) -> str:
            """Needs the smart model."""
            return claim

        assert smart_tool.use_utility is False

    def test_use_utility_with_permission(self) -> None:
        """use_utility works alongside permission setting."""

        @tool(permission=Permission.MUTATING, use_utility=True)
        async def cheap_mutator(data: str) -> bool:
            """Cheap mutating tool."""
            return True

        assert cheap_mutator.permission == Permission.MUTATING
        assert cheap_mutator.use_utility is True

    def test_use_utility_with_name_and_description(self) -> None:
        """use_utility works alongside all other decorator params."""

        @tool(name="my_tool", description="Custom desc", use_utility=True)
        async def fn(x: str) -> str:
            return x

        assert fn.name == "my_tool"
        assert fn.description == "Custom desc"
        assert fn.use_utility is True


# ── Tests: FunctionTool Direct Construction ────────────────


class TestFunctionToolUseUtility:
    """Test FunctionTool class stores use_utility correctly."""

    def test_function_tool_use_utility_default(self) -> None:
        """FunctionTool defaults use_utility to False."""

        async def dummy(x: str) -> str:
            return x

        ft = FunctionTool(
            fn=dummy,
            name="test",
            description="test tool",
            permission=Permission.READ_ONLY,
            parameter_schema={"type": "object", "properties": {}},
        )
        assert ft.use_utility is False

    def test_function_tool_use_utility_explicit(self) -> None:
        """FunctionTool stores use_utility when passed."""

        async def dummy(x: str) -> str:
            return x

        ft = FunctionTool(
            fn=dummy,
            name="test",
            description="test tool",
            permission=Permission.READ_ONLY,
            parameter_schema={"type": "object", "properties": {}},
            use_utility=True,
        )
        assert ft.use_utility is True

    def test_repr_unchanged(self) -> None:
        """use_utility does not break repr."""

        async def dummy(x: str) -> str:
            return x

        ft = FunctionTool(
            fn=dummy,
            name="test",
            description="test",
            permission=Permission.READ_ONLY,
            parameter_schema={},
            use_utility=True,
        )
        assert "test" in repr(ft)
