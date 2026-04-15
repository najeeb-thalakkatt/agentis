"""Stress Test: Stale Memory Degradation.

Run 10 sessions over "simulated weeks" where system info changes between sessions.
Memory from early sessions becomes outdated. Does the agent trust stale memory
over fresh tool data?

Expected: FAIL — MemoryIndex has no freshness tracking (created_at lost on load,
no TTL, no staleness indicator). Old memory entries look identical to new ones.
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


# ── Evolving System State ─────────────────────────────────

SYSTEM_STATE_BY_WEEK: dict[int, dict[str, Any]] = {
    1: {"version": "3.2.1", "replicas": 3, "config": "A", "status": "stable",
        "db_version": "14.2", "cache_size_mb": 512},
    2: {"version": "3.2.1", "replicas": 3, "config": "A", "status": "stable",
        "db_version": "14.2", "cache_size_mb": 512},
    3: {"version": "4.0.0", "replicas": 5, "config": "B", "status": "migrating",
        "db_version": "15.0", "cache_size_mb": 1024},
    4: {"version": "4.0.0", "replicas": 5, "config": "B", "status": "stable",
        "db_version": "15.0", "cache_size_mb": 1024},
    5: {"version": "4.1.0", "replicas": 5, "config": "B", "status": "stable",
        "db_version": "15.1", "cache_size_mb": 1024},
    6: {"version": "4.1.0", "replicas": 5, "config": "B", "status": "stable",
        "db_version": "15.1", "cache_size_mb": 1024},
    7: {"version": "5.0.0", "replicas": 8, "config": "C", "status": "migrating",
        "db_version": "16.0", "cache_size_mb": 2048},
    8: {"version": "5.0.0", "replicas": 8, "config": "C", "status": "stable",
        "db_version": "16.0", "cache_size_mb": 2048},
    9: {"version": "5.1.0", "replicas": 10, "config": "C", "status": "stable",
        "db_version": "16.0", "cache_size_mb": 2048},
    10: {"version": "5.2.0", "replicas": 10, "config": "D", "status": "stable",
         "db_version": "16.1", "cache_size_mb": 4096},
}

# Current simulated week (set by test runner)
_current_week = 1


@tool()
async def get_system_status() -> str:
    """Get current system status including version, replicas, and config."""
    state = SYSTEM_STATE_BY_WEEK[_current_week]
    return (
        f"System Status (week {_current_week}):\n"
        f"  Version: {state['version']}\n"
        f"  Replicas: {state['replicas']}\n"
        f"  Config profile: {state['config']}\n"
        f"  Status: {state['status']}\n"
        f"  Database: v{state['db_version']}\n"
        f"  Cache size: {state['cache_size_mb']}MB"
    )


@tool()
async def get_current_config() -> str:
    """Get current configuration details."""
    state = SYSTEM_STATE_BY_WEEK[_current_week]
    return (
        f"Config Profile {state['config']}:\n"
        f"  App version: {state['version']}\n"
        f"  Replica count: {state['replicas']}\n"
        f"  DB version: {state['db_version']}\n"
        f"  Cache: {state['cache_size_mb']}MB\n"
        f"  Status: {state['status']}"
    )


_captured_reports: list[dict[str, str]] = []


@tool(permission=Permission.MUTATING)
async def write_report(title: str, content: str) -> str:
    """Write a system status report."""
    _captured_reports.append({"title": title, "content": content})
    return f"Report '{title}' saved."


# ── Remember Tool ─────────────────────────────────────────

class RememberTool:
    name: str = "remember"
    description: str = "Save findings to memory for future sessions."
    permission: Permission = Permission.MUTATING

    def __init__(self, memory: MemoryIndex) -> None:
        self._memory = memory

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name, description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array"},
                },
                "required": ["topic", "content"],
            },
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        topic = kwargs.get("topic", "")
        content = kwargs.get("content", "")
        tags = kwargs.get("tags", [])
        try:
            tid = await self._memory.remember(topic, content, tags=tags)
            return ToolResult(
                success=True, data=tid,
                summary=f"Saved '{topic}' (id: {tid})",
                full_output=f"Memory saved: {topic} -> {tid}",
                tokens_used=20,
            )
        except Exception as e:
            return ToolResult(
                success=False, data=None, summary=str(e),
                full_output="", tokens_used=5, error=str(e),
            )


# ── Staleness Provider ────────────────────────────────────


class StalenessProvider:
    """Provider that simulates sessions across different weeks.

    Each session: recall memories -> check live tools -> write report -> save to memory.

    Behavior:
    - If memories exist, recalls them first
    - Always calls get_system_status to get fresh data
    - Writes a report using whatever data it has
    - Saves current state to memory
    - In "trusting" mode: report uses memory data even if it contradicts tools
    - In "verifying" mode: report uses tool data, notes discrepancies
    """

    def __init__(self, week: int, memory_topic_ids: list[str] | None = None,
                 trusting: bool = True) -> None:
        self._week = week
        self._memory_topic_ids = memory_topic_ids or []
        self._trusting = trusting
        self._turn = 0
        self.call_count = 0
        self.tools_called: list[str] = []
        self._recalled_content: str = ""

    async def complete(
        self, messages: list[Message], tools: list[ToolSchema] | None = None,
        temperature: float = 0.0, max_tokens: int = 4096,
    ) -> ProviderResponse:
        self.call_count += 1
        self._turn += 1

        # Extract memory topic IDs from context
        topic_ids = self._memory_topic_ids
        if not topic_ids:
            for msg in messages:
                if msg.role == "system" and "memory index" in msg.content.lower():
                    for line in msg.content.split("\n"):
                        if "topics/" in line:
                            parts = line.split("topics/")
                            if len(parts) > 1:
                                tid = parts[1].split(".md")[0]
                                if tid not in topic_ids:
                                    topic_ids.append(tid)

        # Turn 1: Recall memories if available
        if self._turn == 1 and topic_ids:
            self.tools_called.extend(["recall"] * len(topic_ids))
            return ProviderResponse(
                content="Recalling prior system knowledge.",
                tool_calls=[
                    ToolCall(id=f"rec_{i}", name="recall", arguments={"topic_id": tid})
                    for i, tid in enumerate(topic_ids)
                ],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            )

        # Turn 2 (or 1 if no memories): Check live system
        check_turn = 2 if topic_ids else 1
        if self._turn == check_turn:
            self.tools_called.extend(["get_system_status", "get_current_config"])
            return ProviderResponse(
                content="Checking current system status.",
                tool_calls=[
                    ToolCall(id="gs1", name="get_system_status", arguments={}),
                    ToolCall(id="gc1", name="get_current_config", arguments={}),
                ],
                usage=TokenUsage(input_tokens=300, output_tokens=100),
                stop_reason="tool_use",
            )

        # Turn 3 (or 2): Write report
        report_turn = check_turn + 1
        if self._turn == report_turn:
            state = SYSTEM_STATE_BY_WEEK[self._week]

            if self._trusting and topic_ids:
                # TRUSTING MODE: Use a mix of memory and live data
                # Simulates a model that doesn't notice contradictions
                # Uses the FIRST memory entry's version info (which is stale)
                # but uses live replica count
                report_content = (
                    f"# System Status Report (Week {self._week})\n\n"
                    f"## Current State\n"
                    f"Based on prior knowledge and current checks:\n"
                    f"- Version: 3.2.1 (from memory)\n"
                    f"- Replicas: {state['replicas']} (confirmed live)\n"
                    f"- Config: A (from memory)\n"
                    f"- Database: v14.2 (from memory)\n"
                    f"- Cache: 512MB (from memory)\n\n"
                    f"## Assessment\n"
                    f"System appears to be running the same version as last checked."
                )
            else:
                # VERIFYING MODE: Use live data
                report_content = (
                    f"# System Status Report (Week {self._week})\n\n"
                    f"## Current State (Live)\n"
                    f"- Version: {state['version']}\n"
                    f"- Replicas: {state['replicas']}\n"
                    f"- Config: {state['config']}\n"
                    f"- Database: v{state['db_version']}\n"
                    f"- Cache: {state['cache_size_mb']}MB\n\n"
                    f"## Assessment\n"
                    f"Live data used. System status verified."
                )

            self.tools_called.append("write_report")
            return ProviderResponse(
                content="Writing system status report.",
                tool_calls=[ToolCall(
                    id="wr1", name="write_report",
                    arguments={"title": f"Week {self._week} Status", "content": report_content},
                )],
                usage=TokenUsage(input_tokens=400, output_tokens=200),
                stop_reason="tool_use",
            )

        # Turn 4 (or 3): Save current state to memory
        save_turn = report_turn + 1
        if self._turn == save_turn:
            state = SYSTEM_STATE_BY_WEEK[self._week]
            self.tools_called.append("remember")
            return ProviderResponse(
                content="Saving current system state to memory.",
                tool_calls=[ToolCall(
                    id="mem1", name="remember",
                    arguments={
                        "topic": "System Status",
                        "content": (
                            f"System state as of week {self._week}:\n"
                            f"- Version: {state['version']}\n"
                            f"- Replicas: {state['replicas']}\n"
                            f"- Config: {state['config']}\n"
                            f"- Database: v{state['db_version']}\n"
                            f"- Cache: {state['cache_size_mb']}MB\n"
                            f"- Status: {state['status']}"
                        ),
                        "tags": ["system", "status", f"week_{self._week}"],
                    },
                )],
                usage=TokenUsage(input_tokens=300, output_tokens=150),
                stop_reason="tool_use",
            )

        return ProviderResponse(
            content="Session complete.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    async def stream(self, messages: Any, tools: Any = None, **kwargs: Any) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="staleness_mock")


# ── Result Tracking ───────────────────────────────────────

@dataclass
class SessionResult:
    week: int
    memories_available: int = 0
    memories_stale: int = 0  # Memories whose version differs from current
    stale_data_in_report: bool = False
    fresh_data_in_report: bool = False
    recall_calls: int = 0
    tool_calls: int = 0
    elapsed: float = 0.0


# ���─ Test Runner ───────────────────────────────────────────


def check_report_staleness(report: str, current_state: dict[str, Any]) -> tuple[bool, bool]:
    """Check if report uses stale or fresh data.

    Returns (has_stale, has_fresh).
    """
    import re
    version = current_state["version"]
    config = current_state["config"]
    db = current_state["db_version"]
    cache = str(current_state["cache_size_mb"])

    has_fresh = (
        version in report and
        f"Config: {config}" in report
    )

    # Check for any old version numbers
    old_versions = {"3.2.1", "3.2.0", "4.0.0", "4.1.0", "5.0.0", "5.1.0", "5.2.0"}
    old_versions.discard(version)

    has_stale = any(v in report for v in old_versions)
    # Also check old configs
    old_configs = {"A", "B", "C", "D"}
    old_configs.discard(config)
    if any(f"Config: {c}" in report for c in old_configs):
        has_stale = True

    return has_stale, has_fresh


async def run_stale_memory_test() -> None:
    """Run the stale memory degradation stress test."""
    global _current_week

    print("=" * 70)
    print("STALE MEMORY DEGRADATION STRESS TEST")
    print("=" * 70)
    print("\nGoal: Does old memory become a liability as info changes?")
    print("      Does the framework detect or handle staleness?")

    workspace = tempfile.mkdtemp(prefix="stale-mem-")
    # Run at weeks 1, 3, 5, 7, 10 (5 sessions with significant changes between)
    test_weeks = [1, 3, 5, 7, 10]
    results: list[SessionResult] = []

    try:
        for i, week in enumerate(test_weeks):
            _current_week = week
            _captured_reports.clear()

            print(f"\n{'─' * 70}")
            print(f"SESSION {i+1}: Week {week}")
            print(f"{'─' * 70}")

            current_state = SYSTEM_STATE_BY_WEEK[week]
            print(f"  Live state: v{current_state['version']}, "
                  f"config {current_state['config']}, "
                  f"{current_state['replicas']} replicas, "
                  f"DB {current_state['db_version']}")

            # Load memory from prior sessions
            memory = MemoryIndex(workspace=workspace)
            await memory.load()
            remember_tool = RememberTool(memory=memory)

            topic_ids = [e.topic_id for e in memory.entries]

            # Check how many memories are stale
            stale_count = 0
            for entry in memory.entries:
                content = await memory.recall(entry.topic_id)
                if content and current_state["version"] not in content:
                    stale_count += 1

            sr = SessionResult(week=week, memories_available=len(topic_ids), memories_stale=stale_count)

            print(f"  Memories available: {sr.memories_available} ({sr.memories_stale} stale)")

            # Use trusting mode for sessions after the first
            trusting = i > 0 and len(topic_ids) > 0

            provider = StalenessProvider(
                week=week,
                memory_topic_ids=topic_ids,
                trusting=trusting,
            )

            async def auto_approve(request: Any) -> bool:
                return True

            runtime = AgentRuntime(
                provider=provider,
                tools=[get_system_status, get_current_config, write_report,
                       remember_tool],  # type: ignore[list-item]
                system_prompt="You are a system administrator. Check system status and write a report.",
                memory=memory,
                max_turns=8,
                skeptical=False,
                approval_callback=auto_approve,
            )

            start = time.time()
            async for step in runtime.steps(f"Check system status for week {week} and write a report."):
                pass
            sr.elapsed = time.time() - start

            sr.recall_calls = sum(1 for t in provider.tools_called if t == "recall")
            sr.tool_calls = len(provider.tools_called)

            # Check the report
            if _captured_reports:
                report = _captured_reports[-1]["content"]
                has_stale, has_fresh = check_report_staleness(report, current_state)
                sr.stale_data_in_report = has_stale
                sr.fresh_data_in_report = has_fresh
                print(f"  Report uses stale data: {has_stale}")
                print(f"  Report uses fresh data: {has_fresh}")
                if has_stale:
                    print(f"    [STALE] Report references outdated version/config!")
            else:
                print(f"  No report generated.")

            results.append(sr)

        # ── Memory Index Analysis ──
        print(f"\n{'─' * 70}")
        print("MEMORY INDEX ANALYSIS")
        print(f"{'─' * 70}")

        mem_final = MemoryIndex(workspace=workspace)
        await mem_final.load()
        for entry in mem_final.entries:
            content = await mem_final.recall(entry.topic_id)
            created = entry.created_at
            print(f"  [{entry.topic_id}] {entry.topic}")
            print(f"    created_at: {created} ({'lost (0.0)' if created == 0.0 else time.ctime(created)})")
            print(f"    tags: {entry.tags}")
            if content:
                # Show first data line
                for line in content.split("\n"):
                    if "Version:" in line:
                        print(f"    {line.strip()}")
                        break

        # ── SUMMARY ──
        print(f"\n{'=' * 70}")
        print("DEGRADATION CURVE")
        print(f"{'=' * 70}")

        print(f"\n  {'Week':>6} {'Memories':>10} {'Stale':>8} {'Stale in Report':>16} {'Fresh in Report':>16}")
        print(f"  {'─' * 56}")
        for sr in results:
            stale_icon = "YES" if sr.stale_data_in_report else "no"
            fresh_icon = "YES" if sr.fresh_data_in_report else "no"
            print(f"  {sr.week:>6} {sr.memories_available:>10} {sr.memories_stale:>8} "
                  f"{stale_icon:>16} {fresh_icon:>16}")

        total_stale_reports = sum(1 for sr in results if sr.stale_data_in_report)
        sessions_with_memory = sum(1 for sr in results if sr.memories_available > 0)

        # ─�� VERDICT ──
        print(f"\n{'=' * 70}")

        # Check framework-level staleness detection
        has_freshness_tracking = False
        mem_check = MemoryIndex(workspace=workspace)
        await mem_check.load()
        for entry in mem_check.entries:
            if entry.created_at > 0.0:
                has_freshness_tracking = True
                break

        if total_stale_reports == 0:
            print("VERDICT: PASS")
            print("  No stale data leaked into reports.")
        elif has_freshness_tracking:
            print("VERDICT: PARTIAL")
            print(f"  {total_stale_reports} report(s) used stale data, but framework tracks freshness.")
        else:
            print("VERDICT: FAIL (design gap)")
            print(f"  {total_stale_reports}/{sessions_with_memory} sessions with memory used stale data in reports.")
            print(f"  MemoryIndex has NO freshness tracking:")
            print(f"    - created_at is lost on load() (set to 0.0)")
            print(f"    - No TTL mechanism")
            print(f"    - get_context_payload() shows all entries equally")
            print(f"    - No staleness indicator in memory index")
            print(f"")
            print(f"  Motivated improvements:")
            print(f"    1. Preserve created_at timestamps through save/load")
            print(f"    2. Add freshness_score or TTL to MemoryPointer")
            print(f"    3. Add [stale?] indicator for entries older than threshold")
            print(f"    4. Add expire() or prune(max_age) to MemoryIndex")
            print(f"    5. When memory contradicts live data, flag it automatically")

        print(f"{'=' * 70}")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_stale_memory_test())
