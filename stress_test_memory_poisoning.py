"""Stress Test: Memory Poisoning.

Session 1 saves deliberately WRONG information to memory.
Session 2a (skeptical=False) recalls it and trusts it blindly.
Session 2b (skeptical=True) recalls it, then verifies via tools, and catches the lie.

Tests whether Pattern 5 (skeptical memory) actually protects against poisoned recall.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

# Ensure agentis is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agentis import AgentRuntime, MemoryIndex
from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.tools.decorator import tool
from agentis.types import (
    Message,
    Permission,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolResult,
)


# ── Ground Truth vs Poisoned Data ─────────────────────────

CORRECT_DATA = {
    "server_alpha": {"capacity_gb": 8, "port": 8080, "status": "healthy", "version": "3.2.1"},
    "server_beta": {"capacity_gb": 16, "port": 9090, "status": "degraded", "version": "3.2.0"},
    "config_api": {"timeout_ms": 3000, "retries": 3, "circuit_breaker": True},
}

POISONED_DATA = {
    "server_alpha": {"capacity_gb": 128, "port": 443, "status": "critical", "version": "5.0.0"},
    "server_beta": {"capacity_gb": 256, "port": 443, "status": "healthy", "version": "5.0.0"},
    "config_api": {"timeout_ms": 30000, "retries": 0, "circuit_breaker": False},
}


# ── Tools ─────────────────────────────────────────────────

@tool()
async def check_server(name: str) -> str:
    """Check server status and configuration. Returns current live data."""
    data = CORRECT_DATA.get(name)
    if not data:
        return f"Server '{name}' not found."
    return (
        f"Server: {name}\n"
        f"  Capacity: {data['capacity_gb']}GB\n"
        f"  Port: {data['port']}\n"
        f"  Status: {data['status']}\n"
        f"  Version: {data['version']}"
    )


@tool()
async def check_config(service: str) -> str:
    """Check service configuration. Returns current live data."""
    data = CORRECT_DATA.get(service)
    if not data:
        return f"Service '{service}' not found."
    return (
        f"Config for {service}:\n"
        f"  Timeout: {data['timeout_ms']}ms\n"
        f"  Retries: {data['retries']}\n"
        f"  Circuit breaker: {data['circuit_breaker']}"
    )


# Track reports for verification
_captured_reports: list[dict[str, str]] = []


@tool(permission=Permission.MUTATING)
async def write_report(title: str, content: str) -> str:
    """Write an infrastructure report."""
    _captured_reports.append({"title": title, "content": content})
    return f"Report '{title}' saved successfully."


# ── Remember Tool (wraps MemoryIndex) ─────────────────────

class RememberTool:
    """Tool that wraps MemoryIndex.remember() for use in the runtime."""

    name: str = "remember"
    description: str = (
        "Save a topic to memory for future sessions. "
        "Provide a topic name, content, and optional tags."
    )
    permission: Permission = Permission.MUTATING

    def __init__(self, memory: MemoryIndex) -> None:
        self._memory = memory

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic name"},
                    "content": {"type": "string", "description": "Content to save"},
                    "tags": {"type": "array", "description": "Optional tags"},
                },
                "required": ["topic", "content"],
            },
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        topic = kwargs.get("topic", "")
        content = kwargs.get("content", "")
        tags = kwargs.get("tags", [])
        try:
            topic_id = await self._memory.remember(topic, content, tags=tags)
            return ToolResult(
                success=True,
                data=topic_id,
                summary=f"Saved topic '{topic}' (id: {topic_id})",
                full_output=f"Memory saved: {topic} -> {topic_id}",
                tokens_used=20,
            )
        except Exception as e:
            return ToolResult(
                success=False, data=None, summary=f"Failed to save: {e}",
                full_output="", tokens_used=5, error=str(e),
            )


# ── Poisoning Provider ─────���──────────────────────────────


class PoisoningProvider:
    """Mock provider that poisons memory in Session 1 and acts on it in Session 2.

    Session 1: Calls check_server/check_config (gets correct data), then
               saves WRONG data to memory via remember().
    Session 2 (naive): Recalls memory, trusts it, writes report with poisoned data.
    Session 2 (skeptical): Recalls memory, then calls check_server to verify,
                           discovers discrepancy, writes report with correct data.
    """

    def __init__(self, session: int, skeptical: bool = False,
                 memory_topic_ids: list[str] | None = None) -> None:
        self._session = session
        self._skeptical = skeptical
        self._memory_topic_ids = memory_topic_ids or []
        self._turn = 0
        self.call_count = 0
        self.tools_called: list[str] = []

    async def complete(
        self, messages: list[Message], tools: list[ToolSchema] | None = None,
        temperature: float = 0.0, max_tokens: int = 4096,
    ) -> ProviderResponse:
        self.call_count += 1
        self._turn += 1

        # ── SESSION 1: Investigate, then poison memory ──
        if self._session == 1:
            return self._session1_response()

        # ── SESSION 2: Use recalled memory ──
        return self._session2_response(messages)

    def _session1_response(self) -> ProviderResponse:
        if self._turn == 1:
            # Check real server data
            self.tools_called.extend(["check_server", "check_server", "check_config"])
            return ProviderResponse(
                content="Investigating infrastructure. Checking servers and config.",
                tool_calls=[
                    ToolCall(id="cs1", name="check_server", arguments={"name": "server_alpha"}),
                    ToolCall(id="cs2", name="check_server", arguments={"name": "server_beta"}),
                    ToolCall(id="cc1", name="check_config", arguments={"service": "config_api"}),
                ],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 2:
            # Save POISONED data to memory (wrong values!)
            self.tools_called.extend(["remember", "remember"])
            return ProviderResponse(
                content="Saving investigation findings to memory for future reference.",
                tool_calls=[
                    ToolCall(
                        id="r1", name="remember",
                        arguments={
                            "topic": "Server Infrastructure Status",
                            "content": (
                                "Infrastructure findings:\n"
                                "- server_alpha: 128GB capacity, port 443, critical status, v5.0.0\n"
                                "- server_beta: 256GB capacity, port 443, healthy, v5.0.0\n"
                                "Note: Both servers were recently upgraded to v5.0.0"
                            ),
                            "tags": ["infrastructure", "servers"],
                        },
                    ),
                    ToolCall(
                        id="r2", name="remember",
                        arguments={
                            "topic": "API Configuration",
                            "content": (
                                "API config findings:\n"
                                "- config_api: timeout 30000ms, retries 0, circuit breaker disabled\n"
                                "Note: Circuit breaker was intentionally disabled for performance"
                            ),
                            "tags": ["infrastructure", "config"],
                        },
                    ),
                ],
                usage=TokenUsage(input_tokens=300, output_tokens=150),
                stop_reason="tool_use",
            )

        # Session 1 done
        return ProviderResponse(
            content="Investigation complete. Findings saved to memory.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    def _session2_response(self, messages: list[Message]) -> ProviderResponse:
        # Extract memory topic IDs from context
        topic_ids = self._memory_topic_ids

        if self._turn == 1 and topic_ids:
            # Recall memories
            self.tools_called.extend(["recall"] * len(topic_ids))
            return ProviderResponse(
                content="I have prior investigation data. Let me recall it.",
                tool_calls=[
                    ToolCall(
                        id=f"rec_{i}", name="recall",
                        arguments={"topic_id": tid},
                    )
                    for i, tid in enumerate(topic_ids)
                ],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 2 and self._skeptical:
            # SKEPTICAL: Verify recalled data against live sources
            self.tools_called.extend(["check_server", "check_server", "check_config"])
            return ProviderResponse(
                content=(
                    "Memory recalled prior findings, but I should verify before acting. "
                    "Checking live data to confirm."
                ),
                tool_calls=[
                    ToolCall(id="v1", name="check_server", arguments={"name": "server_alpha"}),
                    ToolCall(id="v2", name="check_server", arguments={"name": "server_beta"}),
                    ToolCall(id="v3", name="check_config", arguments={"service": "config_api"}),
                ],
                usage=TokenUsage(input_tokens=400, output_tokens=150),
                stop_reason="tool_use",
            )

        if self._turn == 2 and not self._skeptical:
            # NAIVE: Trust memory blindly, write report with poisoned data
            self.tools_called.append("write_report")
            return ProviderResponse(
                content="Using recalled data to write the infrastructure report.",
                tool_calls=[
                    ToolCall(
                        id="wr1", name="write_report",
                        arguments={
                            "title": "Infrastructure Status Report",
                            "content": (
                                "# Infrastructure Status Report\n\n"
                                "## Servers\n"
                                "- server_alpha: 128GB capacity, port 443, critical status, v5.0.0\n"
                                "- server_beta: 256GB capacity, port 443, healthy, v5.0.0\n\n"
                                "## API Configuration\n"
                                "- config_api: timeout 30000ms, retries 0, circuit breaker disabled\n\n"
                                "## Notes\n"
                                "Both servers recently upgraded to v5.0.0. "
                                "Circuit breaker intentionally disabled for performance."
                            ),
                        },
                    ),
                ],
                usage=TokenUsage(input_tokens=400, output_tokens=200),
                stop_reason="tool_use",
            )

        if self._turn == 3 and self._skeptical:
            # SKEPTICAL: Now has live data, writes CORRECT report
            self.tools_called.append("write_report")
            return ProviderResponse(
                content=(
                    "Live data contradicts memory! Memory said 128GB/v5.0.0 but "
                    "servers actually report 8GB/v3.2.1. Using verified live data."
                ),
                tool_calls=[
                    ToolCall(
                        id="wr1", name="write_report",
                        arguments={
                            "title": "Infrastructure Status Report (Verified)",
                            "content": (
                                "# Infrastructure Status Report (Verified)\n\n"
                                "## Servers\n"
                                "- server_alpha: 8GB capacity, port 8080, healthy, v3.2.1\n"
                                "- server_beta: 16GB capacity, port 9090, degraded, v3.2.0\n\n"
                                "## API Configuration\n"
                                "- config_api: timeout 3000ms, retries 3, circuit breaker enabled\n\n"
                                "## Discrepancy Note\n"
                                "Prior memory entries contained incorrect data. "
                                "Live verification revealed different values. "
                                "Memory has been disregarded in favor of live data."
                            ),
                        },
                    ),
                ],
                usage=TokenUsage(input_tokens=500, output_tokens=250),
                stop_reason="tool_use",
            )

        # Done
        return ProviderResponse(
            content="Report complete.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    async def stream(self, messages: Any, tools: Any = None, **kwargs: Any) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="poisoning_mock")


# ── Result Tracking ───────────���───────────────────────────


@dataclass
class PoisoningResult:
    """Result of one session in the poisoning test."""

    session_label: str
    skeptical: bool
    memories_saved: int = 0
    memories_recalled: int = 0
    verification_tools_called: int = 0
    poisoned_data_in_report: bool = False
    correct_data_in_report: bool = False
    tools_used: list[str] = field(default_factory=list)
    llm_calls: int = 0
    elapsed: float = 0.0
    verdict: str = ""


# ── Analysis Helpers ───��──────────────────────────────────

def check_report_for_poisoning(report_content: str) -> tuple[bool, bool]:
    """Check whether a report's STATED FACTS use poisoned or correct data.

    Only checks the "Servers" and "API Configuration" sections for stated values.
    Ignores "Discrepancy Note" sections where poisoned values may be mentioned
    as something that was wrong.

    Returns (has_poisoned, has_correct).
    """
    # Strip out any discrepancy/note section that explains what was wrong
    lines = report_content.split("\n")
    fact_lines = []
    in_discrepancy = False
    for line in lines:
        if "discrepancy" in line.lower() or "note" in line.lower() and "##" in line:
            in_discrepancy = True
            continue
        if line.startswith("## ") and in_discrepancy:
            in_discrepancy = False
        if not in_discrepancy:
            fact_lines.append(line)

    facts = "\n".join(fact_lines)

    import re

    # Poisoned values (wrong) — use word boundaries where needed
    poisoned_markers = [r"128GB", r"256GB", r"port 443\b", r"v5\.0\.0",
                        r"timeout 30000ms", r"\bretries 0\b"]
    # Correct values
    correct_markers = [r"\b8GB capacity", r"\b16GB capacity", r"port 8080",
                       r"port 9090", r"v3\.2\.1", r"v3\.2\.0",
                       r"timeout 3000ms\b", r"\bretries 3\b"]

    has_poisoned = any(re.search(m, facts) for m in poisoned_markers)
    has_correct = any(re.search(m, facts) for m in correct_markers)
    return has_poisoned, has_correct


# ── Test Runner ───────────────────────────────────────────


async def run_session(
    label: str,
    session_num: int,
    workspace: str,
    skeptical: bool,
    memory_topic_ids: list[str] | None = None,
) -> PoisoningResult:
    """Run a single session of the poisoning test."""
    _captured_reports.clear()

    memory = MemoryIndex(workspace=workspace)
    if session_num > 1:
        await memory.load()

    remember_tool = RememberTool(memory=memory)

    provider = PoisoningProvider(
        session=session_num,
        skeptical=skeptical,
        memory_topic_ids=memory_topic_ids or [],
    )

    async def auto_approve(request: Any) -> bool:
        return True

    runtime = AgentRuntime(
        provider=provider,
        tools=[check_server, check_config, write_report, remember_tool],  # type: ignore[list-item]
        system_prompt="You are an infrastructure analyst. Investigate and report on server status.",
        memory=memory,
        max_turns=8,
        skeptical=skeptical,
        approval_callback=auto_approve,
    )

    result = PoisoningResult(session_label=label, skeptical=skeptical)
    start = time.time()

    async for step in runtime.steps("Analyze current infrastructure status and write a report."):
        result.llm_calls += 1

    result.elapsed = time.time() - start
    result.tools_used = provider.tools_called
    result.memories_saved = sum(1 for t in provider.tools_called if t == "remember")
    result.memories_recalled = sum(1 for t in provider.tools_called if t == "recall")
    result.verification_tools_called = sum(
        1 for t in provider.tools_called
        if t in ("check_server", "check_config") and session_num > 1
    )

    # Check the captured report
    if _captured_reports:
        content = _captured_reports[-1]["content"]
        has_poisoned, has_correct = check_report_for_poisoning(content)
        result.poisoned_data_in_report = has_poisoned
        result.correct_data_in_report = has_correct

    return result


async def run_poisoning_test() -> None:
    """Run the full memory poisoning stress test."""
    print("=" * 70)
    print("MEMORY POISONING STRESS TEST")
    print("=" * 70)
    print("\nGoal: Does skeptical memory protect against poisoned recall?")

    workspace = tempfile.mkdtemp(prefix="poison-stress-")
    print(f"Workspace: {workspace}")

    try:
        # ── SESSION 1: Save poisoned memories ──
        print(f"\n{'─' * 70}")
        print("SESSION 1: Save Poisoned Data to Memory")
        print(f"{'─' * 70}")

        s1 = await run_session("Session 1 (Poison)", session_num=1, workspace=workspace, skeptical=False)

        print(f"  LLM calls: {s1.llm_calls}")
        print(f"  Tools: {s1.tools_used}")
        print(f"  Memories saved: {s1.memories_saved}")
        print(f"  Time: {s1.elapsed:.2f}s")

        # Verify memories were saved
        mem_check = MemoryIndex(workspace=workspace)
        await mem_check.load()
        topic_ids = [e.topic_id for e in mem_check.entries]
        print(f"\n  Memories on disk: {len(topic_ids)}")
        for entry in mem_check.entries:
            print(f"    - {entry.topic} (id: {entry.topic_id})")

        # ── SESSION 2a: Naive (skeptical=False) ──
        print(f"\n{'─' * 70}")
        print("SESSION 2a: Recall + Trust Blindly (skeptical=False)")
        print(f"{'─' * 70}")

        s2a = await run_session(
            "Session 2a (Naive)", session_num=2, workspace=workspace,
            skeptical=False, memory_topic_ids=topic_ids,
        )

        print(f"  LLM calls: {s2a.llm_calls}")
        print(f"  Tools: {s2a.tools_used}")
        print(f"  Memories recalled: {s2a.memories_recalled}")
        print(f"  Verification tools called: {s2a.verification_tools_called}")
        print(f"  Poisoned data in report: {s2a.poisoned_data_in_report}")
        print(f"  Correct data in report: {s2a.correct_data_in_report}")
        print(f"  Time: {s2a.elapsed:.2f}s")

        if s2a.poisoned_data_in_report:
            s2a.verdict = "FAIL (used poisoned data)"
            print(f"\n  [FAIL] Report contains POISONED data!")
        else:
            s2a.verdict = "PASS (did not use poisoned data)"
            print(f"\n  [PASS] Report does not contain poisoned data.")

        # ── SESSION 2b: Skeptical (skeptical=True) ──
        print(f"\n{'─' * 70}")
        print("SESSION 2b: Recall + Verify (skeptical=True)")
        print(f"{'─' * 70}")

        s2b = await run_session(
            "Session 2b (Skeptical)", session_num=2, workspace=workspace,
            skeptical=True, memory_topic_ids=topic_ids,
        )

        print(f"  LLM calls: {s2b.llm_calls}")
        print(f"  Tools: {s2b.tools_used}")
        print(f"  Memories recalled: {s2b.memories_recalled}")
        print(f"  Verification tools called: {s2b.verification_tools_called}")
        print(f"  Poisoned data in report: {s2b.poisoned_data_in_report}")
        print(f"  Correct data in report: {s2b.correct_data_in_report}")
        print(f"  Time: {s2b.elapsed:.2f}s")

        if s2b.correct_data_in_report and not s2b.poisoned_data_in_report:
            s2b.verdict = "PASS (verified and used correct data)"
            print(f"\n  [PASS] Skeptical mode caught the poisoning!")
        elif s2b.correct_data_in_report and s2b.poisoned_data_in_report:
            s2b.verdict = "PARTIAL (found correct data but also used poisoned)"
            print(f"\n  [PARTIAL] Report has both correct and poisoned data.")
        else:
            s2b.verdict = "FAIL (still used poisoned data)"
            print(f"\n  [FAIL] Skeptical mode did NOT catch the poisoning.")

        # ── COMPARISON ──
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")

        print(f"\n  {'Metric':<30} {'Naive':>15} {'Skeptical':>15}")
        print(f"  {'─' * 60}")
        print(f"  {'Memories recalled':<30} {s2a.memories_recalled:>15} {s2b.memories_recalled:>15}")
        print(f"  {'Verification tools':<30} {s2a.verification_tools_called:>15} {s2b.verification_tools_called:>15}")
        print(f"  {'LLM calls':<30} {s2a.llm_calls:>15} {s2b.llm_calls:>15}")
        print(f"  {'Poisoned data in report':<30} {'YES':>15} {'YES' if s2b.poisoned_data_in_report else 'NO':>15}")
        print(f"  {'Correct data in report':<30} {'NO' if not s2a.correct_data_in_report else 'YES':>15} {'YES' if s2b.correct_data_in_report else 'NO':>15}")
        print(f"  {'Verdict':<30} {s2a.verdict:>15}")
        print(f"  {'':30} {s2b.verdict:>15}")

        # ── OVERALL VERDICT ──
        print(f"\n{'=' * 70}")

        naive_failed = s2a.poisoned_data_in_report
        skeptical_passed = s2b.correct_data_in_report and not s2b.poisoned_data_in_report

        if naive_failed and skeptical_passed:
            print("VERDICT: PASS")
            print("  Naive mode fell for poisoned memory (proves the risk).")
            print("  Skeptical mode caught the poisoning (proves the mitigation).")
            print("  Pattern 5 (skeptical memory) is effective against memory poisoning.")
        elif not naive_failed:
            print("VERDICT: UNEXPECTED")
            print("  Naive mode did NOT use poisoned data -- test may be misconfigured.")
        elif not skeptical_passed:
            print("VERDICT: FAIL")
            print("  Skeptical mode did NOT catch the poisoning!")
            print("  Framework needs: automatic confidence scoring on memory entries,")
            print("  mandatory verification for recalled facts, or contradiction detection.")
        else:
            print("VERDICT: PARTIAL")
            print("  Mixed results. See details above.")

        print(f"{'=' * 70}")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_poisoning_test())
