"""Stress Test #3: Cross-session memory.

Run two InsightForge sessions on related topics with a shared memory workspace.
Session 1 discovers findings and saves them to memory.
Session 2 should recall Session 1's findings and benefit from prior knowledge.

Measures: tool calls, LLM calls, score, and whether recall was used.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "insight", "insightforge-simulator"))

from agentis import AgentRuntime, CostTracker, MemoryIndex
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import StepResult
from agentis.types import (
    Message,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from phase2.hooks import build_safety_hooks
from phase2.prompts import SYSTEM_PROMPT, format_query
from phase2.tools import build_agentis_tools
from simulator.engine import ResearchSimulatorEngine as SimulatorEngine
from simulator.scenarios import get_all_scenarios


# ── Memory-Aware Provider ──────────────────────────────────


class MemoryAwareProvider:
    """Dummy LLM that knows about the memory system.

    On Session 1: follows normal research flow (search, read, write).
    On Session 2: checks memory index first, uses recall if memories exist,
    then does less searching (benefits from prior knowledge).
    """

    def __init__(
        self,
        scenario_query: str,
        session_num: int = 1,
        memory_topics: list[str] | None = None,
    ) -> None:
        self._query = scenario_query
        self._session = session_num
        self._memory_topics = memory_topics or []
        self._turn = 0
        self.call_count = 0
        self.tools_called: list[str] = []
        self.recall_used = False

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.call_count += 1
        self._turn += 1

        # Check if memory index has entries (look in system messages)
        has_memories = False
        memory_topic_ids: list[str] = []
        for msg in messages:
            if msg.role == "system" and "memory index" in msg.content.lower():
                if "(empty)" not in msg.content:
                    has_memories = True
                    # Extract topic IDs from memory index
                    for line in msg.content.split("\n"):
                        if line.strip().startswith("- ") and "topics/" in line:
                            # "- Topic Name: ./topics/abcd1234.md [tags]"
                            parts = line.split("topics/")
                            if len(parts) > 1:
                                tid = parts[1].split(".md")[0]
                                memory_topic_ids.append(tid)

        # ── Session 2 with memories: use recall first ──
        if self._session == 2 and has_memories and self._turn == 1:
            self.recall_used = True
            # Recall all available memories before searching
            recall_calls = [
                ToolCall(
                    id=f"recall_{i}",
                    name="recall",
                    arguments={"topic_id": tid},
                )
                for i, tid in enumerate(memory_topic_ids[:3])
            ]
            self.tools_called.extend(["recall"] * len(recall_calls))
            return ProviderResponse(
                content="I have prior research memories. Let me recall them before starting new research.",
                tool_calls=recall_calls,
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            )

        # ── Session 2 after recall: targeted search (fewer calls) ──
        if self._session == 2 and self.recall_used and self._turn == 2:
            self.tools_called.append("search_sources")
            return ProviderResponse(
                content="Based on prior research, I need to fill in gaps. Doing targeted search.",
                tool_calls=[ToolCall(
                    id=f"s2_{self._turn}",
                    name="search_sources",
                    arguments={"query": self._query},
                )],
                usage=TokenUsage(input_tokens=300, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._session == 2 and self.recall_used and self._turn == 3:
            # Read fewer sources (already have context from memory)
            self.tools_called.extend(["read_source", "read_source"])
            return ProviderResponse(
                content="Reading sources to supplement prior knowledge.",
                tool_calls=[
                    ToolCall(id="rs1", name="read_source", arguments={"source_id": "src_001"}),
                    ToolCall(id="rs2", name="read_source", arguments={"source_id": "src_002"}),
                ],
                usage=TokenUsage(input_tokens=400, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._session == 2 and self.recall_used and self._turn == 4:
            # Write report (builds on prior knowledge)
            self.tools_called.append("write_report")
            return ProviderResponse(
                content="Writing report combining prior research with new findings.",
                tool_calls=[ToolCall(
                    id="wr1",
                    name="write_report",
                    arguments={"title": "Market Analysis Report", "content": (
                        "# Market Analysis\n\n"
                        "## Market Size\n"
                        "Based on prior research and new data, the market is valued at "
                        "$45B in 2026 [Source: src_001]. Growth rate is 28% CAGR.\n\n"
                        "## Growth Rate\n"
                        "The CAGR of 28% [Source: src_002] is consistent with prior findings.\n\n"
                        "## Key Players\n"
                        "GitHub Copilot (40%), Cursor (15%), Amazon CodeWhisperer (12%) "
                        "[Source: src_001].\n\n"
                        "## Adoption\n"
                        "Enterprise adoption at 65% [Source: src_002].\n"
                    )},
                )],
                usage=TokenUsage(input_tokens=500, output_tokens=200),
                stop_reason="tool_use",
            )

        # ── Session 1: Normal research flow ────────────
        if self._turn == 1:
            self.tools_called.extend(["search_sources", "search_sources"])
            return ProviderResponse(
                content="Starting research. Searching for relevant sources.",
                tool_calls=[
                    ToolCall(id="s1", name="search_sources", arguments={"query": self._query}),
                    ToolCall(id="s2", name="search_sources", arguments={"query": self._query + " market size"}),
                ],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 2:
            self.tools_called.extend(["read_source"] * 4)
            return ProviderResponse(
                content="Found relevant sources. Reading them now.",
                tool_calls=[
                    ToolCall(id="r1", name="read_source", arguments={"source_id": "src_001"}),
                    ToolCall(id="r2", name="read_source", arguments={"source_id": "src_002"}),
                    ToolCall(id="r3", name="read_source", arguments={"source_id": "src_003"}),
                    ToolCall(id="r4", name="read_source", arguments={"source_id": "src_004"}),
                ],
                usage=TokenUsage(input_tokens=400, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 3:
            self.tools_called.extend(["verify_fact", "check_source_freshness"])
            return ProviderResponse(
                content="Verifying key claims and checking source freshness.",
                tool_calls=[
                    ToolCall(id="v1", name="verify_fact", arguments={"claim": "Market size is $45B", "source_ids": "src_001,src_002"}),
                    ToolCall(id="cf1", name="check_source_freshness", arguments={"source_id": "src_003"}),
                ],
                usage=TokenUsage(input_tokens=500, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 4:
            self.tools_called.append("write_report")
            return ProviderResponse(
                content="Writing the final report.",
                tool_calls=[ToolCall(
                    id="wr1",
                    name="write_report",
                    arguments={"title": "AI Coding Tools TAM Report", "content": (
                        "# Market Analysis\n\n"
                        "## Market Size\n"
                        "The total addressable market is $45B in 2026 [Source: src_001].\n\n"
                        "## Growth Rate\n"
                        "CAGR of 28% through 2030 [Source: src_002].\n\n"
                        "## Key Players\n"
                        "Top players: GitHub Copilot (40% share), Cursor (15%), "
                        "Amazon CodeWhisperer (12%) [Source: src_001].\n\n"
                        "## Adoption\n"
                        "Enterprise adoption at 65%, developer satisfaction at 78% "
                        "[Source: src_002].\n"
                    )},
                )],
                usage=TokenUsage(input_tokens=600, output_tokens=200),
                stop_reason="tool_use",
            )

        # Final response (both sessions)
        return ProviderResponse(
            content="Research complete. Report has been written.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="memory_test_mock")


# ── Session Result ─────────────────────────────────────────


@dataclass
class SessionResult:
    """Metrics from one research session."""

    session_num: int
    topic: str
    llm_calls: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    recall_used: bool = False
    memories_before: int = 0
    memories_after: int = 0
    elapsed: float = 0.0
    score: float = 0.0


# ── Stress Test Runner ─────────────────────────────────────


async def run_session(
    scenario_idx: int,
    session_num: int,
    memory: MemoryIndex,
    memory_topics: list[str],
) -> SessionResult:
    """Run one research session."""
    scenarios = get_all_scenarios()
    scenario = scenarios[scenario_idx % len(scenarios)]
    sim = SimulatorEngine(scenario, simulate_latency=False)

    provider = MemoryAwareProvider(
        scenario_query=scenario.research_query,
        session_num=session_num,
        memory_topics=memory_topics,
    )
    tools = build_agentis_tools(sim)
    hooks = build_safety_hooks()

    result = SessionResult(
        session_num=session_num,
        topic=scenario.name,
        memories_before=len(memory.entries),
    )

    runtime = AgentRuntime(
        provider=provider,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        hooks=hooks,
        memory=memory,
        max_turns=8,
        skeptical=False,
    )

    start = time.time()
    async for step in runtime.steps(format_query(scenario.research_query)):
        result.llm_calls += 1
        for tr in step.tool_results:
            result.tool_calls += 1

    result.elapsed = time.time() - start
    result.tools_used = provider.tools_called
    result.recall_used = provider.recall_used

    # After session: save key findings to memory for next session
    if session_num == 1:
        report = sim.get_report_content() if hasattr(sim, 'get_report_content') else ""
        # Save research findings as memory
        await memory.remember(
            topic=f"Research: {scenario.name}",
            content=(
                f"Research on: {scenario.research_query}\n\n"
                f"Key findings:\n"
                f"- Market size: $45B (2026)\n"
                f"- Growth rate: 28% CAGR\n"
                f"- Key players: GitHub Copilot (40%), Cursor (15%), Amazon CodeWhisperer (12%)\n"
                f"- Enterprise adoption: 65%\n"
                f"- Source quality: 4 sources verified, 1 deprecated\n"
            ),
            tags=["market_research", scenario.category],
        )
        await memory.remember(
            topic=f"Sources: {scenario.name}",
            content=(
                f"Verified sources for {scenario.name}:\n"
                f"- src_001: Market sizing report (fresh, reliable)\n"
                f"- src_002: Industry survey (fresh, reliable)\n"
                f"- src_003: Older analysis (check freshness)\n"
                f"- src_004: Competitive landscape (fresh)\n"
            ),
            tags=["sources", scenario.category],
        )

    result.memories_after = len(memory.entries)

    # Score the session
    sim_metrics = sim.get_metrics_summary()
    result.score = sim_metrics.get("report_score", 0)

    return result


async def run_cross_session_test() -> None:
    """Run the cross-session memory stress test."""
    print("=" * 70)
    print("CROSS-SESSION MEMORY STRESS TEST")
    print("=" * 70)

    # Create shared memory workspace
    workspace = tempfile.mkdtemp(prefix="memory-stress-")
    print(f"\nShared workspace: {workspace}")

    try:
        # ── SESSION 1: Fresh research (no prior memory) ──
        print(f"\n{'─' * 70}")
        print("SESSION 1: Fresh Research (no prior memory)")
        print(f"{'─' * 70}")

        memory1 = MemoryIndex(workspace=workspace)
        s1 = await run_session(
            scenario_idx=0,  # AI Coding Tools TAM
            session_num=1,
            memory=memory1,
            memory_topics=[],
        )

        print(f"  Topic: {s1.topic}")
        print(f"  LLM calls: {s1.llm_calls}")
        print(f"  Tool calls: {s1.tool_calls}")
        print(f"  Tools: {s1.tools_used}")
        print(f"  Recall used: {s1.recall_used}")
        print(f"  Memories before/after: {s1.memories_before} -> {s1.memories_after}")
        print(f"  Time: {s1.elapsed:.2f}s")

        # Verify memories were saved
        print(f"\n  Memory index after Session 1:")
        for entry in memory1.entries:
            print(f"    - {entry.topic} [{', '.join(entry.tags)}]")

        # ── SESSION 2: Related research (should benefit from memory) ──
        print(f"\n{'─' * 70}")
        print("SESSION 2: Related Research (with prior memory)")
        print(f"{'─' * 70}")

        # Create new MemoryIndex pointing to same workspace
        memory2 = MemoryIndex(workspace=workspace)
        await memory2.load()  # Load Session 1's memories

        print(f"  Memories loaded from Session 1: {len(memory2.entries)}")
        for entry in memory2.entries:
            print(f"    - {entry.topic} [{', '.join(entry.tags)}]")

        memory_topic_ids = [e.topic_id for e in memory2.entries]

        s2 = await run_session(
            scenario_idx=3,  # SaaS Pricing (related to AI tools)
            session_num=2,
            memory=memory2,
            memory_topics=memory_topic_ids,
        )

        print(f"\n  Topic: {s2.topic}")
        print(f"  LLM calls: {s2.llm_calls}")
        print(f"  Tool calls: {s2.tool_calls}")
        print(f"  Tools: {s2.tools_used}")
        print(f"  Recall used: {s2.recall_used}")
        print(f"  Memories before/after: {s2.memories_before} -> {s2.memories_after}")
        print(f"  Time: {s2.elapsed:.2f}s")

        # ── COMPARISON ──
        print(f"\n{'=' * 70}")
        print("COMPARISON: Session 1 (Cold) vs Session 2 (Warm)")
        print(f"{'=' * 70}")

        print(f"\n  {'Metric':<25} {'Session 1':>12} {'Session 2':>12} {'Delta':>12}")
        print(f"  {'─' * 61}")
        print(f"  {'LLM calls':<25} {s1.llm_calls:>12} {s2.llm_calls:>12} {s2.llm_calls - s1.llm_calls:>+12}")
        print(f"  {'Tool calls':<25} {s1.tool_calls:>12} {s2.tool_calls:>12} {s2.tool_calls - s1.tool_calls:>+12}")
        print(f"  {'search_sources':<25} {s1.tools_used.count('search_sources'):>12} {s2.tools_used.count('search_sources'):>12} {s2.tools_used.count('search_sources') - s1.tools_used.count('search_sources'):>+12}")
        print(f"  {'read_source':<25} {s1.tools_used.count('read_source'):>12} {s2.tools_used.count('read_source'):>12} {s2.tools_used.count('read_source') - s1.tools_used.count('read_source'):>+12}")
        print(f"  {'recall (memory)':<25} {s1.tools_used.count('recall'):>12} {s2.tools_used.count('recall'):>12} {s2.tools_used.count('recall') - s1.tools_used.count('recall'):>+12}")
        print(f"  {'Recall used':<25} {'No':>12} {'Yes' if s2.recall_used else 'No':>12}")
        print(f"  {'Memories available':<25} {s1.memories_before:>12} {s2.memories_before:>12}")

        # ── VERDICT ──
        print(f"\n{'=' * 70}")
        improvements = []
        if s2.recall_used:
            improvements.append("Memory recall was used")
        if s2.llm_calls < s1.llm_calls:
            improvements.append(f"Fewer LLM calls ({s2.llm_calls} vs {s1.llm_calls})")
        if s2.tool_calls < s1.tool_calls:
            improvements.append(f"Fewer tool calls ({s2.tool_calls} vs {s1.tool_calls})")
        if s2.tools_used.count("search_sources") < s1.tools_used.count("search_sources"):
            improvements.append("Fewer search calls (leveraged prior knowledge)")
        if s2.tools_used.count("read_source") < s1.tools_used.count("read_source"):
            improvements.append("Fewer source reads (already had context)")

        if improvements:
            print("PASS: Session 2 benefited from Session 1's memory:")
            for imp in improvements:
                print(f"  + {imp}")
        else:
            print("FAIL: Session 2 did not benefit from prior memory.")

        print(f"{'=' * 70}")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "insight", "insightforge-simulator"))
    asyncio.run(run_cross_session_test())
