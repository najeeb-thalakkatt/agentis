"""Shared test fixtures for the Agentis test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentis.hooks.registry import HookRegistry
from agentis.memory.index import MemoryIndex
from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.tools.decorator import tool
from agentis.types import (
    Message,
    Permission,
    ProviderResponse,
    TokenUsage,
)

# ── Mock Provider ───────────────────────────────────────────


class MockProvider:
    """Reusable mock LLM provider for tests.

    Accepts a list of scripted responses. Falls back to a default
    text response when the script is exhausted.
    """

    def __init__(
        self,
        responses: list[ProviderResponse] | None = None,
        default_text: str = "Mock response.",
    ) -> None:
        self._responses = responses or []
        self._default_text = default_text
        self._index = 0
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        self.calls.append(list(messages))
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return ProviderResponse(
            content=self._default_text,
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield self._default_text

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="mock", max_context_tokens=128_000)


# ── Sample Tools ────────────────────────────────────────────


@tool()
async def sample_read(path: str) -> str:
    """A sample read-only tool for tests."""
    return f"contents of {path}"


@tool(permission=Permission.MUTATING)
async def sample_write(path: str, content: str) -> bool:
    """A sample mutating tool for tests."""
    return True


@tool(permission=Permission.DANGEROUS)
async def sample_dangerous(command: str) -> str:
    """A sample dangerous tool for tests."""
    return f"executed: {command}"


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def mock_provider() -> MockProvider:
    """A fresh MockProvider with default text response."""
    return MockProvider()


@pytest.fixture
def sample_tools() -> list:
    """A list of sample tools (read, write, dangerous)."""
    return [sample_read, sample_write, sample_dangerous]


@pytest.fixture
def hook_registry() -> HookRegistry:
    """A fresh HookRegistry with no hooks registered."""
    return HookRegistry()


@pytest.fixture
async def tmp_memory(tmp_path) -> MemoryIndex:
    """A MemoryIndex using a temporary directory."""
    return MemoryIndex(workspace=str(tmp_path))
