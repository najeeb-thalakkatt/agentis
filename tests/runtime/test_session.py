"""Tests for agentis.runtime.session — Session management."""

from __future__ import annotations

import pytest

from agentis.runtime.session import Session
from agentis.types import Message


class TestSessionBasics:
    def test_construction_default_id(self) -> None:
        s = Session()
        assert s.session_id
        assert len(s.session_id) > 0
        assert s.turn_count == 0

    def test_construction_explicit_id(self) -> None:
        s = Session(session_id="my-session")
        assert s.session_id == "my-session"

    def test_messages_empty_initially(self) -> None:
        s = Session()
        assert s.get_messages() == []


class TestSessionMessages:
    def test_add_message(self) -> None:
        s = Session()
        s.add_message(Message(role="user", content="hello"))
        assert len(s.get_messages()) == 1
        assert s.get_messages()[0].content == "hello"

    def test_add_multiple_messages(self) -> None:
        s = Session()
        s.add_message(Message(role="user", content="hi"))
        s.add_message(Message(role="assistant", content="hello"))
        assert len(s.get_messages()) == 2

    def test_turn_count_increments(self) -> None:
        s = Session()
        s.increment_turn()
        s.increment_turn()
        assert s.turn_count == 2


class TestSessionReset:
    def test_reset_clears_messages(self) -> None:
        s = Session()
        s.add_message(Message(role="user", content="hi"))
        s.add_message(Message(role="assistant", content="hello"))
        s.increment_turn()
        s.reset()
        assert s.get_messages() == []
        assert s.turn_count == 0

    def test_reset_keeps_session_id(self) -> None:
        s = Session(session_id="keep-me")
        s.add_message(Message(role="user", content="hi"))
        s.reset()
        assert s.session_id == "keep-me"


class TestSessionFork:
    def test_fork_creates_copy(self) -> None:
        s = Session(session_id="parent")
        s.add_message(Message(role="user", content="hello"))
        s.increment_turn()

        forked = s.fork()
        assert forked.session_id != s.session_id  # new ID
        assert len(forked.get_messages()) == 1
        assert forked.get_messages()[0].content == "hello"
        assert forked.turn_count == 1

    def test_fork_is_isolated(self) -> None:
        s = Session()
        s.add_message(Message(role="user", content="original"))

        forked = s.fork()
        forked.add_message(Message(role="assistant", content="forked reply"))

        # Parent should be unchanged
        assert len(s.get_messages()) == 1
        assert len(forked.get_messages()) == 2

    def test_fork_prefix(self) -> None:
        s = Session(session_id="parent-123")
        forked = s.fork()
        assert forked.session_id.startswith("fork-")
