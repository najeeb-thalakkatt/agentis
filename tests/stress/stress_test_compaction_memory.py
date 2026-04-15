"""Stress Test: Compaction + Memory Interaction.

Run a 50-turn session where:
  - Agent saves memories mid-session (turns 12, 20)
  - Compaction fires and evicts old turns (including remember-source turns)
  - Agent recalls memories later (turn 40+)

Question: Do memory pointers survive compaction? Is recall content still correct
even after the turns that created those memories have been evicted?

Expected: DEGRADED — recall works (file-based), but conversation context
explaining WHY those memories were saved is evicted.
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


# ── Verbose Tools (generate lots of context to fill window) ──

VERBOSE_POD_OUTPUT = (
    "NAME                           READY  STATUS    RESTARTS  AGE\n"
    "api-server-7d4f8b6c9-x2k8m    1/1    Running   0         45d\n"
    "api-server-7d4f8b6c9-p9n3j    1/1    Running   2         45d\n"
    "worker-5c8f7a2b1-m4k7n        1/1    Running   0         30d\n"
    "worker-5c8f7a2b1-r8t2v        0/1    OOMKilled 5         30d\n"
    "redis-master-0                 1/1    Running   0         90d\n"
    "redis-slave-0                  1/1    Running   1         90d\n"
    "redis-slave-1                  1/1    Running   0         90d\n"
    "ingress-nginx-8f7d6c5-q3w2e   1/1    Running   0         60d\n"
    "monitoring-prometheus-0        1/1    Running   0         45d\n"
    "monitoring-grafana-7b8c9d-z1x  1/1    Running   0         45d\n"
)

VERBOSE_METRICS_OUTPUT = (
    "=== CPU Usage (last 24h) ===\n"
    "api-server-x2k8m:  avg=45% peak=78% p99=72%\n"
    "api-server-p9n3j:  avg=42% peak=75% p99=70%\n"
    "worker-m4k7n:      avg=65% peak=92% p99=88%\n"
    "worker-r8t2v:      avg=95% peak=100% p99=100%  <- HIGH\n"
    "\n=== Memory Usage ===\n"
    "api-server-x2k8m:  2.1GB / 4GB (52%)\n"
    "api-server-p9n3j:  1.9GB / 4GB (47%)\n"
    "worker-m4k7n:      3.2GB / 4GB (80%) <- ELEVATED\n"
    "worker-r8t2v:      4.0GB / 4GB (100%) <- OOM\n"
    "\n=== Network I/O ===\n"
    "Total ingress: 145 req/s, avg latency 23ms, p99 89ms\n"
    "Error rate: 2.3% (normal < 1%)\n"
)

VERBOSE_LOGS_OUTPUT = (
    "[2026-04-05T10:15:32Z] ERROR worker-r8t2v OOMKilled: memory limit exceeded\n"
    "[2026-04-05T10:15:33Z] WARN  worker-r8t2v Container restarting (attempt 5/10)\n"
    "[2026-04-05T10:16:01Z] ERROR api-server-x2k8m Request timeout: /api/v2/reports\n"
    "[2026-04-05T10:16:02Z] WARN  api-server-x2k8m Upstream connection refused\n"
    "[2026-04-05T10:17:15Z] INFO  redis-master-0 Connected clients: 142\n"
    "[2026-04-05T10:17:16Z] WARN  redis-master-0 Memory usage approaching limit\n"
    "[2026-04-05T10:18:00Z] ERROR ingress-nginx 502 Bad Gateway: worker-r8t2v\n"
    "[2026-04-05T10:18:30Z] INFO  monitoring Alert fired: HighErrorRate\n"
    "[2026-04-05T10:19:00Z] ERROR worker-r8t2v OOMKilled: memory limit exceeded\n"
    "[2026-04-05T10:19:01Z] WARN  worker-r8t2v Container restarting (attempt 6/10)\n"
)

VERBOSE_EVENTS_OUTPUT = (
    "LAST SEEN   TYPE     REASON       OBJECT                MESSAGE\n"
    "5m          Warning  OOMKilled    pod/worker-r8t2v      Container killed due to OOM\n"
    "5m          Normal   Pulling      pod/worker-r8t2v      Pulling image 'worker:v3.2.1'\n"
    "4m          Normal   Pulled       pod/worker-r8t2v      Successfully pulled image\n"
    "4m          Normal   Created      pod/worker-r8t2v      Created container worker\n"
    "4m          Normal   Started      pod/worker-r8t2v      Started container worker\n"
    "3m          Warning  BackOff      pod/worker-r8t2v      Back-off restarting failed\n"
    "2m          Warning  Unhealthy    pod/worker-r8t2v      Readiness probe failed\n"
    "1m          Warning  OOMKilled    pod/worker-r8t2v      Container killed due to OOM\n"
)

VERBOSE_DEPLOYS_OUTPUT = (
    "NAME           READY  UP-TO-DATE  AVAILABLE  AGE  IMAGE\n"
    "api-server     2/2    2           2          45d  api-server:v3.2.1\n"
    "worker         1/2    2           1          30d  worker:v3.2.1\n"
    "redis-master   1/1    1           1          90d  redis:7.2\n"
    "redis-slave    2/2    2           2          90d  redis:7.2\n"
    "ingress-nginx  1/1    1           1          60d  nginx:1.25\n"
    "monitoring     2/2    2           2          45d  prometheus:2.47\n"
)


@tool()
async def get_pods(namespace: str) -> str:
    """Get pod status for a namespace."""
    return f"Pods in {namespace}:\n{VERBOSE_POD_OUTPUT}"


@tool()
async def get_metrics(pod: str) -> str:
    """Get resource metrics for a pod or all pods."""
    return f"Metrics for {pod}:\n{VERBOSE_METRICS_OUTPUT}"


@tool()
async def get_logs(pod: str, lines: int = 20) -> str:
    """Get recent log lines for a pod."""
    return f"Logs for {pod} (last {lines} lines):\n{VERBOSE_LOGS_OUTPUT}"


@tool()
async def get_events(namespace: str) -> str:
    """Get recent events in a namespace."""
    return f"Events in {namespace}:\n{VERBOSE_EVENTS_OUTPUT}"


@tool()
async def get_deploys(namespace: str) -> str:
    """Get deployment status."""
    return f"Deployments in {namespace}:\n{VERBOSE_DEPLOYS_OUTPUT}"


_final_reports: list[str] = []


@tool(permission=Permission.MUTATING)
async def write_diagnosis(title: str, content: str) -> str:
    """Write a final diagnosis report."""
    _final_reports.append(content)
    return f"Diagnosis '{title}' saved."


# ── Remember Tool ─────────────────────────────────────────

class RememberTool:
    """Wraps MemoryIndex.remember() for use in the runtime."""

    name: str = "remember"
    description: str = "Save a topic to memory for future reference."
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
            topic_id = await self._memory.remember(topic, content, tags=tags)
            return ToolResult(
                success=True, data=topic_id,
                summary=f"Saved '{topic}' (id: {topic_id})",
                full_output=f"Memory saved: {topic} -> {topic_id}",
                tokens_used=20,
            )
        except Exception as e:
            return ToolResult(
                success=False, data=None, summary=f"Failed: {e}",
                full_output="", tokens_used=5, error=str(e),
            )


# ── Compaction+Memory Provider ────────────────────────────


class CompactionMemoryProvider:
    """50-turn provider that saves memories mid-session.

    Turn 1-10: Investigation tool calls (verbose output fills context)
    Turn 12: remember("OOM Root Cause", ...)
    Turn 13-19: More investigation
    Turn 20: remember("Fix Plan", ...)
    Turn 21-38: Continued investigation (triggers compaction)
    Turn 40: recall() to retrieve saved memories
    Turn 42+: write_diagnosis using recalled info
    """

    def __init__(self) -> None:
        self._turn = 0
        self.call_count = 0
        self.tools_called: list[str] = []
        self._memory_topic_ids: list[str] = []  # Collected from tool results
        self.memory_index_present_every_call = True

    async def complete(
        self, messages: list[Message], tools: list[ToolSchema] | None = None,
        temperature: float = 0.0, max_tokens: int = 4096,
    ) -> ProviderResponse:
        self.call_count += 1
        self._turn += 1

        # Check if memory index is present in system messages
        found_memory = False
        for msg in messages:
            if msg.role == "system" and "memory index" in msg.content.lower():
                found_memory = True
                # Extract topic IDs
                for line in msg.content.split("\n"):
                    if "topics/" in line:
                        parts = line.split("topics/")
                        if len(parts) > 1:
                            tid = parts[1].split(".md")[0]
                            if tid not in self._memory_topic_ids:
                                self._memory_topic_ids.append(tid)
                break
        if not found_memory:
            self.memory_index_present_every_call = False

        # Check for remember results in tool messages
        for msg in messages:
            if msg.role == "tool" and "Memory saved:" in msg.content:
                parts = msg.content.split("-> ")
                if len(parts) > 1:
                    tid = parts[1].strip()
                    if tid not in self._memory_topic_ids:
                        self._memory_topic_ids.append(tid)

        # ── Turn-by-turn scripting ──
        if self._turn <= 3:
            # Phase 1: Initial investigation
            tool_name = ["get_pods", "get_metrics", "get_logs"][self._turn - 1]
            args = (
                {"namespace": "production"} if tool_name != "get_logs"
                else {"pod": "worker-r8t2v", "lines": 50}
            )
            if tool_name == "get_metrics":
                args = {"pod": "all"}
            self.tools_called.append(tool_name)
            return ProviderResponse(
                content=f"Investigating... ({tool_name})",
                tool_calls=[ToolCall(id=f"t{self._turn}", name=tool_name, arguments=args)],
                usage=TokenUsage(input_tokens=500, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 4:
            # Get events and deploys
            self.tools_called.extend(["get_events", "get_deploys"])
            return ProviderResponse(
                content="Checking events and deployments.",
                tool_calls=[
                    ToolCall(id="t4a", name="get_events", arguments={"namespace": "production"}),
                    ToolCall(id="t4b", name="get_deploys", arguments={"namespace": "production"}),
                ],
                usage=TokenUsage(input_tokens=600, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn >= 5 and self._turn <= 7:
            # More detailed investigation
            pods = ["api-server-x2k8m", "worker-m4k7n", "redis-master-0"]
            pod = pods[self._turn - 5]
            self.tools_called.extend(["get_logs", "get_metrics"])
            return ProviderResponse(
                content=f"Deep-diving into {pod}.",
                tool_calls=[
                    ToolCall(id=f"t{self._turn}a", name="get_logs", arguments={"pod": pod, "lines": 50}),
                    ToolCall(id=f"t{self._turn}b", name="get_metrics", arguments={"pod": pod}),
                ],
                usage=TokenUsage(input_tokens=700, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 8:
            # SAVE MEMORY #1: OOM Root Cause
            self.tools_called.append("remember")
            return ProviderResponse(
                content="I've identified the root cause. Saving to memory for persistence.",
                tool_calls=[ToolCall(
                    id="mem1", name="remember",
                    arguments={
                        "topic": "OOM Root Cause",
                        "content": (
                            "Root cause analysis:\n"
                            "- worker-r8t2v is OOM-killed repeatedly (5+ restarts)\n"
                            "- Memory usage at 100% (4GB/4GB)\n"
                            "- CPU at 95-100% sustained\n"
                            "- Root cause: memory leak in worker processing pipeline\n"
                            "- Triggered by: large batch job on 2026-04-05\n"
                            "- Impact: 2.3% error rate, upstream connection refused errors"
                        ),
                        "tags": ["incident", "oom", "worker"],
                    },
                )],
                usage=TokenUsage(input_tokens=800, output_tokens=200),
                stop_reason="tool_use",
            )

        if self._turn >= 9 and self._turn <= 14:
            # More investigation to fill context
            tool_name = ["get_pods", "get_metrics", "get_logs", "get_events", "get_metrics", "get_logs"][self._turn - 9]
            if tool_name == "get_pods":
                args = {"namespace": "production"}
            elif tool_name == "get_events":
                args = {"namespace": "production"}
            elif tool_name == "get_logs":
                args = {"pod": "worker-r8t2v", "lines": 100}
            else:
                args = {"pod": "worker-r8t2v"}
            self.tools_called.append(tool_name)
            return ProviderResponse(
                content=f"Continuing analysis (turn {self._turn})...",
                tool_calls=[ToolCall(id=f"t{self._turn}", name=tool_name, arguments=args)],
                usage=TokenUsage(input_tokens=600, output_tokens=80),
                stop_reason="tool_use",
            )

        if self._turn == 15:
            # SAVE MEMORY #2: Fix Plan
            self.tools_called.append("remember")
            return ProviderResponse(
                content="Devising fix plan. Saving to memory.",
                tool_calls=[ToolCall(
                    id="mem2", name="remember",
                    arguments={
                        "topic": "Worker OOM Fix Plan",
                        "content": (
                            "Remediation plan:\n"
                            "1. Immediate: Increase worker memory limit to 8GB\n"
                            "2. Short-term: Deploy memory leak fix (PR #4521)\n"
                            "3. Long-term: Add memory circuit breaker to worker pipeline\n"
                            "4. Monitoring: Set alert at 80% memory threshold\n"
                            "Impact if not fixed: cascading failures, data loss risk"
                        ),
                        "tags": ["incident", "remediation", "worker"],
                    },
                )],
                usage=TokenUsage(input_tokens=700, output_tokens=200),
                stop_reason="tool_use",
            )

        if self._turn >= 16 and self._turn <= 25:
            # Fill context to trigger compaction
            tools = ["get_pods", "get_metrics", "get_logs", "get_events", "get_deploys"]
            tool_name = tools[(self._turn - 16) % len(tools)]
            if tool_name in ("get_pods", "get_events", "get_deploys"):
                args = {"namespace": "production"}
            elif tool_name == "get_logs":
                args = {"pod": "worker-r8t2v", "lines": 50}
            else:
                args = {"pod": "all"}
            self.tools_called.append(tool_name)
            return ProviderResponse(
                content=f"Monitoring status (turn {self._turn})...",
                tool_calls=[ToolCall(id=f"t{self._turn}", name=tool_name, arguments=args)],
                usage=TokenUsage(input_tokens=500, output_tokens=60),
                stop_reason="tool_use",
            )

        if self._turn == 26:
            # RECALL memories to write final diagnosis
            recall_calls = []
            for i, tid in enumerate(self._memory_topic_ids[:3]):
                recall_calls.append(ToolCall(
                    id=f"rec_{i}", name="recall",
                    arguments={"topic_id": tid},
                ))
                self.tools_called.append("recall")

            if not recall_calls:
                # No memories found — this would indicate a bug
                self.tools_called.append("get_pods")
                return ProviderResponse(
                    content="No memories available. Checking current state.",
                    tool_calls=[ToolCall(id="fallback", name="get_pods", arguments={"namespace": "production"})],
                    usage=TokenUsage(input_tokens=300, output_tokens=50),
                    stop_reason="tool_use",
                )

            return ProviderResponse(
                content="Recalling prior findings to write final diagnosis.",
                tool_calls=recall_calls,
                usage=TokenUsage(input_tokens=400, output_tokens=100),
                stop_reason="tool_use",
            )

        if self._turn == 27:
            # Write final diagnosis using recalled memory
            self.tools_called.append("write_diagnosis")
            return ProviderResponse(
                content="Writing final diagnosis based on recalled findings.",
                tool_calls=[ToolCall(
                    id="diag", name="write_diagnosis",
                    arguments={
                        "title": "Worker OOM Incident Diagnosis",
                        "content": (
                            "# Worker OOM Incident Diagnosis\n\n"
                            "## Root Cause\n"
                            "worker-r8t2v has a memory leak triggered by large batch processing.\n"
                            "Memory usage hit 100% (4GB/4GB), causing OOMKilled events.\n\n"
                            "## Impact\n"
                            "- 2.3% error rate (normal < 1%)\n"
                            "- Upstream connection failures from api-server\n"
                            "- 5+ container restarts\n\n"
                            "## Fix Plan\n"
                            "1. Increase worker memory limit to 8GB (immediate)\n"
                            "2. Deploy memory leak fix PR #4521 (short-term)\n"
                            "3. Add memory circuit breaker (long-term)\n"
                        ),
                    },
                )],
                usage=TokenUsage(input_tokens=500, output_tokens=300),
                stop_reason="tool_use",
            )

        # Done
        return ProviderResponse(
            content="Investigation complete.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    async def stream(self, messages: Any, tools: Any = None, **kwargs: Any) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="compaction_memory_mock",
            max_context_tokens=16_000,
        )


# ── Result Tracking ───────────────────────────────────────

@dataclass
class CompactionMemoryResult:
    total_turns: int = 0
    compaction_events: int = 0
    tokens_before_compaction: list[int] = field(default_factory=list)
    tokens_after_compaction: list[int] = field(default_factory=list)
    memories_saved: int = 0
    memories_recalled: int = 0
    recall_returned_content: bool = False
    memory_index_always_present: bool = True
    source_turns_survived: bool = True  # Did remember() turns survive compaction?
    diagnosis_contains_findings: bool = False
    elapsed: float = 0.0


# ── Test Runner ───────────────────────────────────────────


async def run_compaction_memory_test() -> None:
    """Run the compaction + memory interaction stress test."""
    print("=" * 70)
    print("COMPACTION + MEMORY INTERACTION STRESS TEST")
    print("=" * 70)
    print("\nGoal: Do memory pointers survive compaction?")
    print("      Is recall content correct after source turns are evicted?")

    workspace = tempfile.mkdtemp(prefix="compact-mem-")
    _final_reports.clear()

    try:
        memory = MemoryIndex(workspace=workspace)
        remember_tool = RememberTool(memory=memory)
        provider = CompactionMemoryProvider()

        async def auto_approve(request: Any) -> bool:
            return True

        runtime = AgentRuntime(
            provider=provider,
            tools=[get_pods, get_metrics, get_logs, get_events, get_deploys,
                   write_diagnosis, remember_tool],  # type: ignore[list-item]
            system_prompt="You are an SRE investigating a production incident. Investigate, save findings to memory, and write a diagnosis.",
            memory=memory,
            max_turns=30,
            max_tokens=3_000,  # Very small window to force aggressive compaction
            skeptical=False,
            approval_callback=auto_approve,
        )

        result = CompactionMemoryResult()
        prev_est_tokens = 0
        prev_session_len = 0
        compaction_log: list[dict[str, Any]] = []

        start = time.time()
        async for step in runtime.steps("Investigate the worker OOM incident in production."):
            result.total_turns += 1

            # Track compaction via token estimation
            msgs = runtime._session.get_messages()
            current_session_len = len(msgs)
            est_tokens = sum(max(len(m.content) // 3, 1) for m in msgs)

            # Detect compaction: tokens dropped significantly (layer 1 summarizes in-place)
            if prev_est_tokens > 0 and est_tokens < prev_est_tokens * 0.7:
                result.compaction_events += 1
                compaction_log.append({
                    "turn": result.total_turns,
                    "tokens_before": prev_est_tokens,
                    "tokens_after": est_tokens,
                    "freed": prev_est_tokens - est_tokens,
                    "messages": current_session_len,
                })

            # Check if remember-source turns still present
            found_remember_in_session = any(
                "Memory saved:" in m.content for m in msgs
            )

            prev_est_tokens = est_tokens
            prev_session_len = current_session_len

        result.elapsed = time.time() - start
        result.memories_saved = sum(1 for t in provider.tools_called if t == "remember")
        result.memories_recalled = sum(1 for t in provider.tools_called if t == "recall")
        result.memory_index_always_present = provider.memory_index_present_every_call

        # Check if recall returned content
        for msg in runtime._session.get_messages():
            if msg.role == "tool" and "# OOM Root Cause" in msg.content:
                result.recall_returned_content = True
            if msg.role == "tool" and "Root cause" in msg.content and "recall" not in msg.content.lower():
                result.recall_returned_content = True

        # Check diagnosis
        if _final_reports:
            diag = _final_reports[-1]
            result.diagnosis_contains_findings = (
                "memory leak" in diag.lower() or
                "4GB" in diag or
                "OOM" in diag or
                "worker-r8t2v" in diag
            )

        # Check if remember-source turns survived compaction WITH their content.
        # Detection uses structural metadata (tool_name), not content string matching.
        session_msgs = runtime._session.get_messages()
        found_remember_full_content = False
        found_remember_summarized = False
        for msg in session_msgs:
            # Memory-anchor tool results carry tool_name="remember" in metadata
            if msg.metadata.get("tool_name") == "remember" and "Memory saved:" in msg.content:
                found_remember_full_content = True
            # Summarized (Layer 1 replaced it)
            if "lines" in msg.content and "summarized" in msg.content.lower():
                found_remember_summarized = True

        result.source_turns_survived = found_remember_full_content

        # ── REPORT ──
        print(f"\n  Workspace: {workspace}")
        print(f"  Max context: 3,000 tokens (very small to force aggressive compaction)")
        print(f"  Total turns: {result.total_turns}")
        print(f"  Tools called: {len(provider.tools_called)}")
        print(f"  Time: {result.elapsed:.2f}s")

        print(f"\n  --- Memory ---")
        print(f"  Memories saved: {result.memories_saved}")
        print(f"  Memories recalled: {result.memories_recalled}")
        print(f"  Memory index present every call: {result.memory_index_always_present}")
        print(f"  Recall returned content: {result.recall_returned_content}")

        print(f"\n  --- Compaction ---")
        print(f"  Compaction events: {result.compaction_events}")
        for event in compaction_log:
            print(f"    Turn {event['turn']}: ~{event['tokens_before']} -> ~{event['tokens_after']} tokens "
                  f"(freed ~{event['freed']}, {event['messages']} msgs)")

        print(f"\n  --- Source Turn Survival ---")
        print(f"  Remember tool results (full): {found_remember_full_content}")
        print(f"  Summarized entries present: {found_remember_summarized}")
        print(f"  Final session messages: {len(session_msgs)}")

        print(f"\n  --- Diagnosis ---")
        print(f"  Diagnosis contains findings: {result.diagnosis_contains_findings}")
        print(f"  Remember-source turns survived: {result.source_turns_survived}")

        # Memory entries on disk
        print(f"\n  --- Memory Index on Disk ---")
        mem_check = MemoryIndex(workspace=workspace)
        await mem_check.load()
        for entry in mem_check.entries:
            content = await mem_check.recall(entry.topic_id)
            content_preview = (content[:80] + "...") if content and len(content) > 80 else (content or "EMPTY")
            print(f"    [{entry.topic_id}] {entry.topic}")
            print(f"      Content: {content_preview}")

        # ── VERDICT ──
        print(f"\n{'=' * 70}")

        if (result.memory_index_always_present and
                result.recall_returned_content and
                result.diagnosis_contains_findings):
            if result.source_turns_survived:
                print("VERDICT: PASS")
                print("  Memory pointers survived compaction.")
                print("  Recall returned correct content (file-based, not session-based).")
                print("  Diagnosis correctly used recalled findings.")
                print("  Source context for remember() calls also survived.")
            else:
                print("VERDICT: DEGRADED (as expected)")
                print("  Memory pointers survived compaction: YES")
                print("  Recall returned correct content: YES (file-based)")
                print("  Diagnosis used recalled findings: YES")
                print("  Source turns for remember() survived: NO")
                print("")
                print("  The conversation context explaining WHY memories were saved")
                print("  has been evicted by compaction. The agent can recall the facts")
                print("  but has lost the reasoning chain that produced them.")
                print("")
                print("  Improvement: Add CRITICAL priority for memory-anchor turns,")
                print("  or store reasoning alongside content in topic files.")
        elif not result.recall_returned_content:
            print("VERDICT: FAIL")
            print("  Recall did NOT return content after compaction!")
            print("  Memory system broken when combined with compaction.")
        else:
            print("VERDICT: PARTIAL")
            if not result.memory_index_always_present:
                print("  Memory index was NOT present in every LLM call.")
            if not result.diagnosis_contains_findings:
                print("  Diagnosis did not use recalled findings.")

        print(f"{'=' * 70}")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_compaction_memory_test())
