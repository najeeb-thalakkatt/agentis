"""Tests for agentis.agents.teammate — TeammateAgent."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentis.agents.mailbox import InMemoryMailbox
from agentis.agents.teammate import TeammateAgent
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import AgentRuntime
from agentis.types import ProviderResponse, TokenUsage


class MockProvider:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["Done."]
        self._idx = 0

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return ProviderResponse(
            content=text, tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, messages, **kw) -> AsyncIterator[str]:
        yield "done"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="mock")


class TestTeammateAgent:
    @pytest.mark.asyncio
    async def test_construction(self) -> None:
        mb = InMemoryMailbox()
        rt = AgentRuntime(provider=MockProvider())
        t = TeammateAgent(name="worker-1", runtime=rt, mailbox=mb)
        assert t.name == "worker-1"

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        mb = InMemoryMailbox()
        rt = AgentRuntime(provider=MockProvider())
        t = TeammateAgent(name="alice", runtime=rt, mailbox=mb)

        await t.send("bob", {"task": "review PR"})
        msgs = await mb.receive("bob")
        assert len(msgs) == 1
        assert msgs[0]["from"] == "alice"
        assert msgs[0]["content"]["task"] == "review PR"

    @pytest.mark.asyncio
    async def test_check_mail(self) -> None:
        mb = InMemoryMailbox()
        rt = AgentRuntime(provider=MockProvider())
        t = TeammateAgent(name="bob", runtime=rt, mailbox=mb)

        # Send message to bob from outside
        await mb.send("bob", {"from": "alice", "content": {"info": "ready"}})
        mail = await t.check_mail()
        assert len(mail) == 1

    @pytest.mark.asyncio
    async def test_run_completes(self) -> None:
        mb = InMemoryMailbox()
        rt = AgentRuntime(provider=MockProvider(["Task complete."]))
        t = TeammateAgent(name="worker", runtime=rt, mailbox=mb)

        result = await t.run("Do the thing")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_two_teammates_communicate(self) -> None:
        """Two teammates sharing a mailbox can exchange messages."""
        mb = InMemoryMailbox()
        rt1 = AgentRuntime(provider=MockProvider())
        rt2 = AgentRuntime(provider=MockProvider())

        alice = TeammateAgent(name="alice", runtime=rt1, mailbox=mb)
        bob = TeammateAgent(name="bob", runtime=rt2, mailbox=mb)

        await alice.send("bob", {"status": "frontend done"})
        bob_mail = await bob.check_mail()
        assert len(bob_mail) == 1
        assert bob_mail[0]["content"]["status"] == "frontend done"

        await bob.send("alice", {"status": "backend done"})
        alice_mail = await alice.check_mail()
        assert len(alice_mail) == 1
        assert alice_mail[0]["content"]["status"] == "backend done"
