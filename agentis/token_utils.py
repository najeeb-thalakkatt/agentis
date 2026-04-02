"""Token estimation utilities.

Uses a chars/3 heuristic for fast approximation without external dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentis.types import Message


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses chars/3 heuristic — fast, no dependencies, roughly accurate
    for English text across most tokenizers.
    """
    return len(text) // 3


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate total tokens across a list of messages."""
    return sum(estimate_tokens(msg.content) for msg in messages)
