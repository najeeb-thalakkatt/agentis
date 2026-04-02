"""Tests for agentis.memory.skeptical — trust-but-verify discipline."""

from __future__ import annotations

import pytest

from agentis.memory.index import MemoryIndex
from agentis.memory.skeptical import SkepticalMemory
from agentis.types import ToolResult


class TestSkepticalMemoryPrompt:
    def test_get_verification_prompt(self) -> None:
        sm = SkepticalMemory()
        prompt = sm.get_verification_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "verify" in prompt.lower() or "read" in prompt.lower()
        assert "memory" in prompt.lower() or "hint" in prompt.lower()

    def test_prompt_mentions_key_rules(self) -> None:
        sm = SkepticalMemory()
        prompt = sm.get_verification_prompt()
        # Should instruct to re-read before editing
        assert "edit" in prompt.lower() or "write" in prompt.lower()
        # Should warn about stale memory
        assert "stale" in prompt.lower() or "changed" in prompt.lower()


class TestRememberIfSuccessful:
    @pytest.mark.asyncio
    async def test_remembers_on_success(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        sm = SkepticalMemory()
        result = ToolResult(
            success=True, data="wrote 50 lines",
            summary="Edited auth.py", full_output="", tokens_used=10,
        )

        topic_id = await sm.remember_if_successful(
            memory=memory,
            topic="auth.py edit",
            content="Added JWT validation",
            result=result,
        )
        assert topic_id is not None
        assert len(memory.entries) == 1

    @pytest.mark.asyncio
    async def test_skips_on_failure(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        sm = SkepticalMemory()
        result = ToolResult(
            success=False, data=None,
            summary="Edit failed", full_output="",
            tokens_used=5, error="old_text not found",
        )

        topic_id = await sm.remember_if_successful(
            memory=memory,
            topic="failed edit",
            content="Tried to edit but failed",
            result=result,
        )
        assert topic_id is None
        assert len(memory.entries) == 0

    @pytest.mark.asyncio
    async def test_passes_tags(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        sm = SkepticalMemory()
        result = ToolResult(
            success=True, data="ok", summary="Done",
            full_output="", tokens_used=5,
        )

        topic_id = await sm.remember_if_successful(
            memory=memory,
            topic="tagged entry",
            content="Content",
            result=result,
            tags=["verified", "auth"],
        )
        assert topic_id is not None
        assert memory.entries[0].tags == ["verified", "auth"]


class TestShouldVerify:
    def test_recent_context_no_verify(self) -> None:
        sm = SkepticalMemory()
        assert sm.should_verify(tool_name="file_edit", context_age_turns=1) is False

    def test_old_context_should_verify(self) -> None:
        sm = SkepticalMemory()
        assert sm.should_verify(tool_name="file_edit", context_age_turns=10) is True

    def test_read_tools_dont_need_verify(self) -> None:
        sm = SkepticalMemory()
        # Read tools are fetching fresh data, no need to verify
        assert sm.should_verify(tool_name="file_read", context_age_turns=20) is False
        assert sm.should_verify(tool_name="grep", context_age_turns=20) is False

    def test_edit_tools_verify_after_threshold(self) -> None:
        sm = SkepticalMemory()
        assert sm.should_verify(tool_name="file_edit", context_age_turns=3) is False
        assert sm.should_verify(tool_name="file_edit", context_age_turns=5) is True

    def test_custom_threshold(self) -> None:
        sm = SkepticalMemory(verify_after_turns=2)
        assert sm.should_verify(tool_name="file_edit", context_age_turns=1) is False
        assert sm.should_verify(tool_name="file_edit", context_age_turns=2) is True


class TestSkepticalRuntimeIntegration:
    """Test that AgentRuntime uses skeptical memory when enabled."""

    @pytest.mark.asyncio
    async def test_runtime_includes_skeptical_prompt(self) -> None:
        from collections.abc import AsyncIterator

        from agentis.protocols import ProviderCapabilities, ToolSchema
        from agentis.runtime.agent import AgentRuntime
        from agentis.types import Message, ProviderResponse, TokenUsage

        captured_messages: list[list[Message]] = []

        class CapturingProvider:
            async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
                captured_messages.append(list(messages))
                return ProviderResponse(
                    content="Done.", tool_calls=[],
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                )

            async def stream(self, messages, **kw) -> AsyncIterator[str]:
                yield "done"

            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(name="mock")

        rt = AgentRuntime(provider=CapturingProvider(), skeptical=True)
        await rt.run("Hello")

        # The system messages sent to the provider should contain the skeptical prompt
        all_content = " ".join(m.content for m in captured_messages[0] if m.role == "system")
        assert "hint" in all_content.lower() or "verify" in all_content.lower()

    @pytest.mark.asyncio
    async def test_runtime_skeptical_false_no_prompt(self) -> None:
        from collections.abc import AsyncIterator

        from agentis.protocols import ProviderCapabilities
        from agentis.runtime.agent import AgentRuntime
        from agentis.types import Message, ProviderResponse, TokenUsage

        captured_messages: list[list[Message]] = []

        class CapturingProvider:
            async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
                captured_messages.append(list(messages))
                return ProviderResponse(
                    content="Done.", tool_calls=[],
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                )

            async def stream(self, messages, **kw) -> AsyncIterator[str]:
                yield "done"

            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(name="mock")

        rt = AgentRuntime(provider=CapturingProvider(), skeptical=False)
        await rt.run("Hello")

        all_content = " ".join(m.content for m in captured_messages[0] if m.role == "system")
        assert "Memory Protocol" not in all_content
