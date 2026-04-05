"""RecallTool — the built-in tool for fetching full memory content."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agentis.protocols import ToolSchema
from agentis.token_utils import estimate_tokens
from agentis.types import Permission, ToolResult

if TYPE_CHECKING:
    from agentis.memory.index import MemoryIndex


class RecallTool:
    """Built-in tool for recalling full content from the memory index.

    This is the only tool that ships with agentis core. It enables
    the LLM to fetch full topic content from the memory index on demand.
    """

    name: str = "recall"
    description: str = (
        "Fetch full content for a memory topic by its ID. "
        "Use this when you need details beyond what the memory index shows."
    )
    permission: Permission = Permission.READ_ONLY

    def __init__(self, memory: MemoryIndex) -> None:
        self._memory = memory

    def get_schema(self) -> ToolSchema:
        """Return the JSON schema for this tool's parameters."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "topic_id": {
                        "type": "string",
                        "description": "The topic ID from the memory index.",
                    },
                },
                "required": ["topic_id"],
            },
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the recall tool.

        Args:
            topic_id: The topic ID to recall.

        Returns:
            ToolResult with the full topic content or an error.
        """
        topic_id = kwargs.get("topic_id", "")

        try:
            content = await self._memory.recall(topic_id)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                summary=f"Failed to recall topic {topic_id}",
                full_output="",
                tokens_used=10,
                error=str(e),
            )

        if content is None:
            return ToolResult(
                success=False,
                data=None,
                summary=f"Topic {topic_id} not found in memory",
                full_output="",
                tokens_used=10,
                error=f"No topic with ID '{topic_id}' exists in memory.",
            )

        # Estimate tokens
        tokens = estimate_tokens(content)

        # Add freshness metadata
        freshness_suffix = ""
        age_label = "unknown time"
        pointer = self._memory._pointers.get(topic_id)
        if pointer and pointer.created_at > 0:
            age_label = self._memory._format_age(time.time() - pointer.created_at)
            freshness_suffix = f"\n\n[Memory saved {age_label}]"

        return ToolResult(
            success=True,
            data=content,
            summary=f"Recalled topic {topic_id} ({tokens} tokens, saved {age_label})",
            full_output=content + freshness_suffix,
            tokens_used=tokens,
        )
