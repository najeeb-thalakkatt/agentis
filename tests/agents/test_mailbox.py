"""Tests for agentis.agents.mailbox — InMemoryMailbox and FileMailbox."""

from __future__ import annotations

import os

import pytest

from agentis.agents.mailbox import FileMailbox, InMemoryMailbox


class TestInMemoryMailbox:
    @pytest.mark.asyncio
    async def test_send_and_receive(self) -> None:
        mb = InMemoryMailbox()
        await mb.send("agent-b", {"text": "hello from A"})
        msgs = await mb.receive("agent-b")
        assert len(msgs) == 1
        assert msgs[0]["text"] == "hello from A"

    @pytest.mark.asyncio
    async def test_receive_consumes_messages(self) -> None:
        mb = InMemoryMailbox()
        await mb.send("agent-b", {"n": 1})
        await mb.send("agent-b", {"n": 2})

        msgs = await mb.receive("agent-b")
        assert len(msgs) == 2

        # Second receive should be empty
        msgs2 = await mb.receive("agent-b")
        assert msgs2 == []

    @pytest.mark.asyncio
    async def test_peek_does_not_consume(self) -> None:
        mb = InMemoryMailbox()
        await mb.send("agent-b", {"text": "peek me"})

        peeked = await mb.peek("agent-b")
        assert len(peeked) == 1

        # Should still be there
        msgs = await mb.receive("agent-b")
        assert len(msgs) == 1

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        mb = InMemoryMailbox()
        await mb.send("agent-b", {"n": 1})
        await mb.send("agent-b", {"n": 2})
        await mb.clear("agent-b")

        msgs = await mb.receive("agent-b")
        assert msgs == []

    @pytest.mark.asyncio
    async def test_separate_mailboxes(self) -> None:
        mb = InMemoryMailbox()
        await mb.send("alice", {"for": "alice"})
        await mb.send("bob", {"for": "bob"})

        alice_msgs = await mb.receive("alice")
        bob_msgs = await mb.receive("bob")
        assert len(alice_msgs) == 1
        assert alice_msgs[0]["for"] == "alice"
        assert len(bob_msgs) == 1
        assert bob_msgs[0]["for"] == "bob"

    @pytest.mark.asyncio
    async def test_receive_empty_mailbox(self) -> None:
        mb = InMemoryMailbox()
        msgs = await mb.receive("nobody")
        assert msgs == []


class TestFileMailbox:
    @pytest.mark.asyncio
    async def test_send_and_receive(self, tmp_path: object) -> None:
        mb = FileMailbox(base_dir=str(tmp_path))
        await mb.send("agent-b", {"text": "hello"})

        msgs = await mb.receive("agent-b")
        assert len(msgs) == 1
        assert msgs[0]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_receive_consumes_messages(self, tmp_path: object) -> None:
        mb = FileMailbox(base_dir=str(tmp_path))
        await mb.send("agent-b", {"n": 1})
        await mb.send("agent-b", {"n": 2})

        msgs = await mb.receive("agent-b")
        assert len(msgs) == 2

        msgs2 = await mb.receive("agent-b")
        assert msgs2 == []

    @pytest.mark.asyncio
    async def test_peek_does_not_consume(self, tmp_path: object) -> None:
        mb = FileMailbox(base_dir=str(tmp_path))
        await mb.send("agent-b", {"text": "peeked"})

        peeked = await mb.peek("agent-b")
        assert len(peeked) == 1

        msgs = await mb.receive("agent-b")
        assert len(msgs) == 1

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path: object) -> None:
        mb = FileMailbox(base_dir=str(tmp_path))
        await mb.send("agent-b", {"n": 1})
        await mb.clear("agent-b")

        msgs = await mb.receive("agent-b")
        assert msgs == []

    @pytest.mark.asyncio
    async def test_messages_survive_new_instance(self, tmp_path: object) -> None:
        """File mailbox persists — new instance reads old messages."""
        mb1 = FileMailbox(base_dir=str(tmp_path))
        await mb1.send("agent-b", {"persistent": True})

        mb2 = FileMailbox(base_dir=str(tmp_path))
        msgs = await mb2.receive("agent-b")
        assert len(msgs) == 1
        assert msgs[0]["persistent"] is True

    @pytest.mark.asyncio
    async def test_separate_agents(self, tmp_path: object) -> None:
        mb = FileMailbox(base_dir=str(tmp_path))
        await mb.send("alice", {"for": "alice"})
        await mb.send("bob", {"for": "bob"})

        assert len(await mb.receive("alice")) == 1
        assert len(await mb.receive("bob")) == 1
