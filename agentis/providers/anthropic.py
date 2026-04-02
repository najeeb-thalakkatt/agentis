"""AnthropicProvider — Anthropic Claude API integration."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from agentis.errors import ConfigError, ProviderError
from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.providers.base import BaseProvider
from agentis.types import Message, ProviderResponse, TokenUsage, ToolCall

logger = logging.getLogger("agentis")

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude models.

    Supports prompt caching, tool use, and streaming.
    Requires the ``anthropic`` package: ``pip install agentis[anthropic]``
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not _HAS_ANTHROPIC:
            raise ConfigError(
                "The anthropic SDK is required for AnthropicProvider. "
                "Install it with: pip install agentis[anthropic]"
            )
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    def capabilities(self) -> ProviderCapabilities:
        """Return Anthropic provider capabilities."""
        return ProviderCapabilities(
            name="anthropic",
            supports_tool_use=True,
            supports_streaming=True,
            supports_prompt_caching=True,
            supports_image_input=True,
            max_context_tokens=200_000,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send a completion request to the Anthropic API."""
        system_prompt, converted_messages = self._prepare_messages(messages)
        converted_tools = self._convert_tools_anthropic(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}") from e

        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream tokens from the Anthropic API."""
        system_prompt, converted_messages = self._prepare_messages(messages)
        converted_tools = self._convert_tools_anthropic(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise ProviderError(f"Anthropic streaming error: {e}") from e

    def _prepare_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Separate system messages and convert to Anthropic format.

        Anthropic requires system content as a separate parameter,
        not in the messages array.
        """
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "tool":
                # Convert tool results to Anthropic's tool_result format
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.metadata.get("tool_use_id", ""),
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool use blocks
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        system = "\n\n".join(system_parts)
        return system, converted

    def _convert_tools_anthropic(
        self, tools: list[ToolSchema] | None
    ) -> list[dict[str, Any]]:
        """Convert tool schemas to Anthropic's tool format."""
        if not tools:
            return []
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _parse_response(self, response: Any) -> ProviderResponse:
        """Parse an Anthropic API response into a ProviderResponse."""
        content_text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(
                response.usage, "cache_read_input_tokens", 0
            ) or 0,
            cache_write_tokens=getattr(
                response.usage, "cache_creation_input_tokens", 0
            ) or 0,
        )

        return ProviderResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=response.stop_reason,
        )
