"""Protocol definitions for the Agentis framework.

All pluggable interfaces are defined here as typing.Protocol classes.
Concrete implementations live in their respective modules.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentis.types import (
        Message,
        Permission,
        ProviderResponse,
        ToolResult,
    )


# ── Tool Protocols ──────────────────────────────────────────


@dataclass(slots=True)
class ToolSchema:
    """JSON Schema description of a tool for LLM consumption."""

    name: str
    description: str
    parameters: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Protocol for all tools in the system."""

    name: str
    description: str
    permission: Permission

    def get_schema(self) -> ToolSchema:
        """Return the JSON schema for this tool's parameters."""
        ...

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given arguments."""
        ...


# ── Provider Protocols ──────────────────────────────────────


@dataclass(slots=True)
class ProviderCapabilities:
    """Declares what features a provider supports.

    Used for feature negotiation — callers check capabilities before
    using provider-specific features like prompt caching.
    """

    name: str
    supports_tool_use: bool = True
    supports_streaming: bool = True
    supports_prompt_caching: bool = False
    supports_extended_thinking: bool = False
    supports_image_input: bool = False
    supports_system_message: bool = True
    max_context_tokens: int = 128_000
    max_tools: int = 128
    parallel_tool_calls: bool = True


@runtime_checkable
class Provider(Protocol):
    """Protocol for LLM providers."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send a completion request to the LLM."""
        ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from the LLM."""
        ...

    def capabilities(self) -> ProviderCapabilities:
        """Return the provider's capability declaration."""
        ...


# ── Isolation Protocols ─────────────────────────────────────


@runtime_checkable
class IsolationStrategy(Protocol):
    """Protocol for agent workspace isolation."""

    async def setup(self) -> Path:
        """Set up the isolated workspace and return its path."""
        ...

    async def cleanup(self) -> None:
        """Clean up the isolated workspace."""
        ...

    def working_dir(self) -> Path:
        """Return the current working directory of the isolated workspace."""
        ...


# ── Mailbox Protocols ───────────────────────────────────────


@runtime_checkable
class MailboxBackend(Protocol):
    """Protocol for inter-agent communication backends."""

    async def send(self, to: str, message: dict[str, Any]) -> None:
        """Send a message to another agent's mailbox."""
        ...

    async def receive(self, agent_name: str) -> list[dict[str, Any]]:
        """Receive all pending messages for an agent."""
        ...

    async def peek(self, agent_name: str) -> list[dict[str, Any]]:
        """Peek at pending messages without consuming them."""
        ...

    async def clear(self, agent_name: str) -> None:
        """Clear all messages for an agent."""
        ...


# ── Extension Protocols ─────────────────────────────────────


@runtime_checkable
class Extension(Protocol):
    """Protocol for runtime extensions (4-method lifecycle).

    Extensions hook into the runtime lifecycle for opt-in behavior
    like cost tracking or background memory consolidation.
    """

    name: str

    async def on_runtime_start(self, runtime: Any) -> None:
        """Called when the runtime is initialized."""
        ...

    async def on_turn_end(self, runtime: Any, result: Any) -> None:
        """Called after each turn (LLM call + tool execution)."""
        ...

    async def on_idle(self, runtime: Any, idle_seconds: float) -> None:
        """Called when the user is idle (for background maintenance)."""
        ...

    async def on_runtime_stop(self, runtime: Any) -> None:
        """Called when the runtime is shutting down."""
        ...
