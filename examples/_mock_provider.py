"""Shared MockProvider for examples — no API keys needed.

All examples import from here so they run out of the box.
Replace with a real provider (AnthropicProvider, OpenAIProvider)
to connect to an actual LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agentis import (
    Message,
    ProviderCapabilities,
    ProviderResponse,
    TokenUsage,
    ToolSchema,
)


class MockProvider:
    """A scripted LLM provider for running examples without API keys.

    Pass a list of ProviderResponse objects. Each call to complete()
    returns the next response in the script. When the script is
    exhausted, returns a default text response.
    """

    def __init__(
        self,
        responses: list[ProviderResponse] | None = None,
        default_text: str = "Done.",
    ) -> None:
        self._responses = responses or []
        self._index = 0
        self._default_text = default_text

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return ProviderResponse(
            content=self._default_text,
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield self._default_text

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="mock", max_context_tokens=128_000)
