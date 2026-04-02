"""Tests for agentis.token_utils — token estimation."""

from __future__ import annotations

from agentis.token_utils import estimate_messages_tokens, estimate_tokens
from agentis.types import Message


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 1  # chars/3 + 1

    def test_short_string(self) -> None:
        result = estimate_tokens("hello")
        assert result == len("hello") // 3 + 1

    def test_longer_string(self) -> None:
        text = "a" * 300
        result = estimate_tokens(text)
        assert result == 101  # 300 / 3 + 1

    def test_returns_int(self) -> None:
        result = estimate_tokens("ab")  # 2 / 3 + 1 = 1
        assert isinstance(result, int)


class TestEstimateMessagesTokens:
    def test_empty_list(self) -> None:
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self) -> None:
        msg = Message(role="user", content="hello world")
        result = estimate_messages_tokens([msg])
        assert result == estimate_tokens("hello world")

    def test_multiple_messages_sums(self) -> None:
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Do something."),
        ]
        result = estimate_messages_tokens(msgs)
        expected = estimate_tokens("You are helpful.") + estimate_tokens("Do something.")
        assert result == expected
