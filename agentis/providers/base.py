"""BaseProvider — abstract base class for LLM providers."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.types import Message, ProviderResponse, ToolCall

logger = logging.getLogger("agentis")


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Handles common functionality: message/tool conversion, silent retry
    for rate limits. Subclasses implement provider-specific API calls.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key or ""
        self._base_url = base_url

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send a completion request to the LLM."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from the LLM."""
        ...

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the provider's capability declaration."""
        ...

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert agentis Messages to a generic dict format.

        Subclasses may override for provider-specific formats.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            entry: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        return result

    def _convert_tools(self, tools: list[ToolSchema] | None) -> list[dict[str, Any]]:
        """Convert agentis ToolSchemas to a generic dict format."""
        if not tools:
            return []
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in tools
        ]

    async def _retry_with_backoff(
        self,
        fn: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """Retry a coroutine with exponential backoff for transient errors.

        Used for silent rate-limit retry. Non-transient errors are raised
        immediately.
        """
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await fn()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.debug(
                        "Retry %d/%d after %.1fs: %s",
                        attempt + 1, max_retries, delay, e,
                    )
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]
