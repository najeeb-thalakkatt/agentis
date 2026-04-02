"""Core types for the Agentis framework.

Dataclasses and enums used throughout the system. These are internal hot-path
types — dataclasses (not Pydantic) for speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


# ── Enums ────────────────────────────────────────────────────


class Priority(IntEnum):
    """Context entry priority for compaction decisions."""

    CRITICAL = 0   # Never evict (system prompt, memory index)
    HIGH = 1       # Recent edits, current errors
    MEDIUM = 2     # Earlier tool results, file reads
    LOW = 3        # Old conversation turns
    EPHEMERAL = 4  # Stale tool outputs, superseded info


class Permission(Enum):
    """Tool permission level controlling concurrency and approval."""

    READ_ONLY = "read_only"     # Can run in parallel
    MUTATING = "mutating"       # Must run serially with a lock
    DANGEROUS = "dangerous"     # Requires human approval


class HookAction(Enum):
    """Result action from a lifecycle hook."""

    ALLOW = "allow"
    DENY = "deny"
    ASK_HUMAN = "ask_human"


class LifecycleEvent(Enum):
    """Lifecycle events that hooks can intercept."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_COMPACT = "pre_compact"
    ON_ERROR = "on_error"


# ── Dataclasses ──────────────────────────────────────────────


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    """Structured result from a tool execution.

    Tools return this instead of raising — the LLM always gets structured
    feedback.
    """

    success: bool
    data: Any
    summary: str
    full_output: str
    tokens_used: int
    error: str | None = None


@dataclass(slots=True)
class Message:
    """A single message in the conversation history."""

    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextEntry:
    """An entry in the context window, tagged with priority for compaction."""

    content: str
    priority: Priority
    token_count: int
    timestamp: float
    entry_type: str  # "tool_result", "conversation", "file_read", "summary"


@dataclass(slots=True)
class ApprovalRequest:
    """Request for human approval of a dangerous operation."""

    tool_name: str
    arguments: dict[str, Any]
    reason: str
    agent_id: str = ""
    session_id: str = ""
    conversation_summary: str = ""
    turn_number: int = 0


@dataclass(slots=True)
class HookContext:
    """Context passed to lifecycle hooks."""

    event: LifecycleEvent
    tool_name: str | None
    arguments: dict[str, Any] | None
    agent_id: str
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HookResponse:
    """Response returned by a lifecycle hook."""

    action: HookAction
    reason: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class TokenUsage:
    """Token usage reported by a provider response."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(slots=True)
class ProviderResponse:
    """Structured response from an LLM provider."""

    content: str
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str | None = None


@dataclass(slots=True)
class CostRecord:
    """A record of cost incurred by a single LLM call."""

    provider: str
    model: str
    usage: TokenUsage
    estimated_cost_usd: float
    timestamp: float
