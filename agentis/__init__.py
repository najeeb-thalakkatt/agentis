"""Agentis — Production-grade, provider-agnostic framework for building agentic AI systems."""

from __future__ import annotations

from agentis.memory.index import MemoryIndex, MemoryPointer
from agentis.memory.recall_tool import RecallTool
from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool
from agentis.tools.orchestrator import ToolOrchestrator
from agentis.providers.base import BaseProvider
from agentis.providers.anthropic import AnthropicProvider
from agentis.providers.openai import OpenAIProvider
from agentis.providers.openai_compatible import OpenAICompatibleProvider
from agentis.compaction.compactor import CompactionResult, ContextCompactor
from agentis.compaction.dedup import DedupResult, FileDeduplicator

from agentis.errors import (
    AgentisError,
    CompactionError,
    ConfigError,
    HookDeniedError,
    ProviderError,
    SessionError,
    ToolExecutionError,
)
from agentis.protocols import (
    Extension,
    IsolationStrategy,
    MailboxBackend,
    Provider,
    ProviderCapabilities,
    Tool,
    ToolSchema,
)
from agentis.token_utils import estimate_messages_tokens, estimate_tokens
from agentis.types import (
    ApprovalRequest,
    ContextEntry,
    CostRecord,
    HookAction,
    HookContext,
    HookResponse,
    LifecycleEvent,
    Message,
    Permission,
    Priority,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolResult,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Memory
    "MemoryIndex",
    "MemoryPointer",
    "RecallTool",
    # Types
    "ApprovalRequest",
    "ContextEntry",
    "CostRecord",
    "HookAction",
    "HookContext",
    "HookResponse",
    "LifecycleEvent",
    "Message",
    "Permission",
    "Priority",
    "ProviderResponse",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    # Protocols
    "Extension",
    "IsolationStrategy",
    "MailboxBackend",
    "Provider",
    "ProviderCapabilities",
    "Tool",
    "ToolSchema",
    # Tools
    "FunctionTool",
    "ToolOrchestrator",
    "tool",
    # Providers
    "BaseProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    # Compaction
    "CompactionResult",
    "ContextCompactor",
    "DedupResult",
    "FileDeduplicator",
    # Errors
    "AgentisError",
    "CompactionError",
    "ConfigError",
    "HookDeniedError",
    "ProviderError",
    "SessionError",
    "ToolExecutionError",
    # Utils
    "estimate_messages_tokens",
    "estimate_tokens",
]
