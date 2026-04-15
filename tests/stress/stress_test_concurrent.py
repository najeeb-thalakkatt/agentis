"""Stress Test #6: Concurrent ForkAgent test.

Spin up multiple ForkAgents on a complex SentinelOps incident.
Tests:
  1. Do concurrent forks complete without errors?
  2. Do agents step on each other's state?
  3. Does the shared provider handle concurrent calls?
  4. What happens to cost and latency vs sequential?
  5. Are sessions truly isolated?
  6. Do results combine into a better diagnosis?
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agentis import AgentRuntime, CostTracker, ForkAgent, MemoryIndex
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import StepResult
from agentis.tools.decorator import tool
from agentis.types import (
    Message,
    Permission,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)


# ── Concurrent-Safe Mock Provider ──────────────────────────


class ConcurrentProvider:
    """Thread-safe mock provider that tracks concurrent access.

    Each call gets a unique sequence number. Tracks max concurrency,
    total calls, and per-session calls to detect state leakage.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._call_seq = 0
        self._active = 0
        self._max_concurrent = 0
        self.total_calls = 0
        self.calls_by_session: dict[str, int] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        async with self._lock:
            self._call_seq += 1
            self._active += 1
            self._max_concurrent = max(self._max_concurrent, self._active)
            self.total_calls += 1
            seq = self._call_seq

        # Simulate API latency (allows concurrency to show)
        await asyncio.sleep(0.05)

        # Track per-session calls
        session_id = "unknown"
        for msg in messages:
            if msg.role == "system" and "SUBTASK" not in msg.content:
                continue
            if msg.role == "user" and "SUBTASK" in msg.content:
                # Extract task from message to identify the fork
                session_id = msg.content[:50]
                break

        async with self._lock:
            self._active -= 1
            self.calls_by_session[session_id] = self.calls_by_session.get(session_id, 0) + 1

        # Determine response based on what tools are available and conversation state
        has_tools = tools and len(tools) > 0
        user_msgs = [m for m in messages if m.role == "user"]
        turn = len([m for m in messages if m.role == "assistant"]) + 1

        if has_tools and turn <= 2:
            # Investigation turn: call available tools
            tool_schemas = tools or []
            tool_name = tool_schemas[seq % len(tool_schemas)].name if tool_schemas else "investigate"
            input_tokens = 200 + seq * 20
            output_tokens = 100 + seq * 10

            async with self._lock:
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

            return ProviderResponse(
                content=f"Agent #{seq}: Investigating with {tool_name}.",
                tool_calls=[ToolCall(
                    id=f"call_{seq}",
                    name=tool_name,
                    arguments=_tool_args(tool_name),
                )],
                usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                stop_reason="tool_use",
            )
        else:
            # Final response
            input_tokens = 150
            output_tokens = 100
            async with self._lock:
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

            return ProviderResponse(
                content=(
                    f"Agent #{seq} findings: The auth-service pod is OOMKilled. "
                    f"Memory usage at 95%, heap exhaustion detected in logs. "
                    f"Root cause: memory leak in DataProcessor after deploy v3.2.1."
                ),
                tool_calls=[],
                usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="concurrent_mock", max_context_tokens=32_000)

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent


def _tool_args(tool_name: str) -> dict[str, Any]:
    """Generate plausible tool arguments."""
    return {
        "get_pods": {"namespace": "production"},
        "get_metrics": {"service": "auth-service", "window": "30m"},
        "get_logs": {"pod": "auth-service-main", "lines": 100},
        "get_events": {"namespace": "production"},
        "get_deploys": {"service": "auth-service"},
    }.get(tool_name, {"target": "auth-service"})


# ── SentinelOps-style Tools ───────────────────────────────


@tool()
async def get_pods(namespace: str) -> str:
    """Get pod status for a namespace."""
    await asyncio.sleep(0.01)  # Simulate I/O
    return (
        f"Pods in {namespace}:\n"
        f"  auth-service-main: CrashLoopBackOff, restarts=5, memory=512Mi/512Mi\n"
        f"  auth-service-backup: Running, restarts=0, memory=256Mi/512Mi\n"
        f"  api-gateway: Running, restarts=0, memory=128Mi/256Mi"
    )


@tool()
async def get_metrics(service: str, window: str = "30m") -> str:
    """Get service metrics."""
    await asyncio.sleep(0.01)
    return (
        f"Metrics for {service} ({window}):\n"
        f"  memory_usage: 512Mi/512Mi (100%) — CRITICAL\n"
        f"  cpu_usage: 78%\n  error_rate: 12.5%\n"
        f"  p99_latency: 2400ms\n  gc_pause: 340ms"
    )


@tool()
async def get_logs(pod: str, lines: int = 50) -> str:
    """Get pod logs."""
    await asyncio.sleep(0.01)
    return (
        f"Logs for {pod} (last {lines}):\n"
        f"  [ERROR] java.lang.OutOfMemoryError: Java heap space\n"
        f"  [ERROR] at DataProcessor.process(DataProcessor.java:142)\n"
        f"  [WARN] GC overhead limit exceeded — 98% time in GC\n"
        f"  [INFO] Heap: 490Mi/512Mi, OldGen: 95%"
    )


@tool()
async def get_events(namespace: str) -> str:
    """Get Kubernetes events."""
    await asyncio.sleep(0.01)
    return (
        f"Events in {namespace}:\n"
        f"  Warning OOMKilled pod/auth-service-main: container exceeded memory limit\n"
        f"  Warning BackOff pod/auth-service-main: restarting (5 times in 10m)\n"
        f"  Normal Scheduled pod/auth-service-main: assigned to node-3"
    )


@tool()
async def get_deploys(service: str) -> str:
    """Get deployment history."""
    await asyncio.sleep(0.01)
    return (
        f"Deploy history for {service}:\n"
        f"  v3.2.1 @ 14:30 — heap=512Mi, replicas=2 (CURRENT)\n"
        f"  v3.2.0 @ 10:00 — heap=256Mi, replicas=2 (PREVIOUS)\n"
        f"  Change: doubled heap limit, added new DataProcessor batch job"
    )


@tool(permission=Permission.MUTATING)
async def rollback_deployment(service: str, version: str) -> str:
    """Rollback a deployment (MUTATING — not available to forks)."""
    return f"Rolled back {service} to {version}"


# ── Result Tracking ────────────────────────────────────────


@dataclass
class ConcurrencyResult:
    """Results from the concurrency test."""

    mode: str  # "sequential" or "parallel"
    num_agents: int
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    max_concurrent_calls: int = 0
    wall_time: float = 0.0
    results: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sessions_seen: int = 0
    parent_messages_after: int = 0


# ── Test Runner ────────────────────────────────────────────


async def run_sequential(
    parent: AgentRuntime,
    tasks: list[str],
    provider: ConcurrentProvider,
) -> ConcurrencyResult:
    """Run tasks sequentially (baseline)."""
    result = ConcurrencyResult(mode="sequential", num_agents=len(tasks))
    start = time.time()

    for task in tasks:
        try:
            fork = ForkAgent(parent=parent, task=task)
            r = await fork.run()
            result.results.append(r)
        except Exception as e:
            result.errors.append(f"{task[:30]}: {e}")

    result.wall_time = time.time() - start
    result.total_llm_calls = provider.total_calls
    result.max_concurrent_calls = provider.max_concurrent
    result.sessions_seen = len(provider.calls_by_session)
    return result


async def run_parallel(
    parent: AgentRuntime,
    tasks: list[str],
    provider: ConcurrentProvider,
) -> ConcurrencyResult:
    """Run tasks in parallel using ForkAgent.parallel_investigate."""
    result = ConcurrencyResult(mode="parallel", num_agents=len(tasks))
    start = time.time()

    try:
        results = await ForkAgent.parallel_investigate(parent, tasks)
        result.results = results
    except Exception as e:
        result.errors.append(str(e))

    result.wall_time = time.time() - start
    result.total_llm_calls = provider.total_calls
    result.max_concurrent_calls = provider.max_concurrent
    result.sessions_seen = len(provider.calls_by_session)
    return result


async def run_concurrent_test() -> None:
    """Run the concurrent agent stress test."""
    print("=" * 80)
    print("CONCURRENT AGENT STRESS TEST")
    print("  ForkAgent.parallel_investigate on SentinelOps incident")
    print("=" * 80)

    # Investigation subtasks (what parallel forks would do)
    tasks = [
        "Check pod status and health in the production namespace. Report any crashes, restarts, or resource limits.",
        "Get metrics for auth-service: memory, CPU, error rate, latency. Identify anomalies.",
        "Pull logs from the auth-service pod. Find error messages, stack traces, OOM indicators.",
        "Check Kubernetes events for the production namespace. Look for OOMKilled, BackOff, scheduling failures.",
        "Review deployment history for auth-service. Identify recent changes that could cause the incident.",
    ]

    print(f"\n  Subtasks: {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"    [{i+1}] {t[:70]}...")

    all_tools = [get_pods, get_metrics, get_logs, get_events, get_deploys, rollback_deployment]

    # ── Run 1: Sequential (baseline) ──
    print(f"\n{'─' * 80}")
    print("RUN 1: SEQUENTIAL (baseline)")
    print(f"{'─' * 80}")

    provider_seq = ConcurrentProvider()
    workspace_seq = tempfile.mkdtemp(prefix="concurrent-seq-")
    memory_seq = MemoryIndex(workspace=workspace_seq)

    parent_seq = AgentRuntime(
        provider=provider_seq,
        tools=all_tools,
        system_prompt="You are SentinelOps, an SRE incident response agent.",
        memory=memory_seq,
        max_turns=5,
        skeptical=False,
    )

    # Seed the parent with an initial message
    await parent_seq.step("INCIDENT: Pod OOMKilled: auth-service in production. Investigate.")

    seq_result = await run_sequential(parent_seq, tasks, provider_seq)
    parent_seq_msgs = len(parent_seq._session.get_messages())

    print(f"  Completed: {len(seq_result.results)} / {len(tasks)}")
    print(f"  Errors: {len(seq_result.errors)}")
    print(f"  LLM calls: {seq_result.total_llm_calls}")
    print(f"  Max concurrent: {seq_result.max_concurrent_calls}")
    print(f"  Wall time: {seq_result.wall_time:.3f}s")
    print(f"  Sessions seen: {seq_result.sessions_seen}")
    print(f"  Parent session msgs after: {parent_seq_msgs}")

    # ── Run 2: Parallel (ForkAgent) ──
    print(f"\n{'─' * 80}")
    print("RUN 2: PARALLEL (ForkAgent.parallel_investigate)")
    print(f"{'─' * 80}")

    provider_par = ConcurrentProvider()
    workspace_par = tempfile.mkdtemp(prefix="concurrent-par-")
    memory_par = MemoryIndex(workspace=workspace_par)

    parent_par = AgentRuntime(
        provider=provider_par,
        tools=all_tools,
        system_prompt="You are SentinelOps, an SRE incident response agent.",
        memory=memory_par,
        max_turns=5,
        skeptical=False,
    )

    # Same seed
    await parent_par.step("INCIDENT: Pod OOMKilled: auth-service in production. Investigate.")

    par_result = await run_parallel(parent_par, tasks, provider_par)
    parent_par_msgs = len(parent_par._session.get_messages())

    print(f"  Completed: {len(par_result.results)} / {len(tasks)}")
    print(f"  Errors: {len(par_result.errors)}")
    print(f"  LLM calls: {par_result.total_llm_calls}")
    print(f"  Max concurrent: {par_result.max_concurrent_calls}")
    print(f"  Wall time: {par_result.wall_time:.3f}s")
    print(f"  Sessions seen: {par_result.sessions_seen}")
    print(f"  Parent session msgs after: {parent_par_msgs}")

    # ── Comparison ─────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print(f"{'=' * 80}")

    print(f"\n  {'Metric':<30} {'Sequential':>15} {'Parallel':>15} {'Speedup':>10}")
    print(f"  {'─' * 70}")

    speedup = seq_result.wall_time / max(par_result.wall_time, 0.001)
    print(f"  {'Wall time':<30} {seq_result.wall_time:>14.3f}s {par_result.wall_time:>14.3f}s {speedup:>9.1f}x")
    print(f"  {'LLM calls':<30} {seq_result.total_llm_calls:>15} {par_result.total_llm_calls:>15}")
    print(f"  {'Max concurrent provider calls':<30} {seq_result.max_concurrent_calls:>15} {par_result.max_concurrent_calls:>15}")
    print(f"  {'Unique sessions':<30} {seq_result.sessions_seen:>15} {par_result.sessions_seen:>15}")
    print(f"  {'Results collected':<30} {len(seq_result.results):>15} {len(par_result.results):>15}")
    print(f"  {'Errors':<30} {len(seq_result.errors):>15} {len(par_result.errors):>15}")
    print(f"  {'Parent msgs (unchanged?)':<30} {parent_seq_msgs:>15} {parent_par_msgs:>15}")

    # ── Isolation Check ────────────────────────────────
    print(f"\n{'─' * 80}")
    print("ISOLATION ANALYSIS")
    print(f"{'─' * 80}")

    issues: list[str] = []

    # Check: parent session should be unchanged by forks
    if parent_par_msgs != parent_seq_msgs:
        issues.append(f"Parent session modified: seq={parent_seq_msgs}, par={parent_par_msgs}")

    # Check: each fork should have independent results
    if len(set(par_result.results)) < len(par_result.results) - 1:
        issues.append("Fork results are too similar — possible state leakage")

    # Check: parallel had actual concurrency
    if par_result.max_concurrent_calls <= 1 and len(tasks) > 1:
        issues.append(f"No actual concurrency detected (max concurrent: {par_result.max_concurrent_calls})")

    # Check: all forks completed
    if len(par_result.results) != len(tasks):
        issues.append(f"Not all forks completed: {len(par_result.results)}/{len(tasks)}")

    if issues:
        print(f"\n  Issues found: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  No isolation issues detected.")
        print("  - Parent session unchanged after parallel forks")
        print("  - Each fork produced independent results")
        print(f"  - Actual concurrency achieved (max {par_result.max_concurrent_calls} simultaneous calls)")
        print(f"  - All {len(tasks)} forks completed")

    # ── Fork Results Preview ───────────────────────────
    print(f"\n{'─' * 80}")
    print("FORK RESULTS (parallel)")
    print(f"{'─' * 80}")
    for i, r in enumerate(par_result.results):
        preview = r[:100].replace("\n", " ")
        print(f"  Fork {i+1}: {preview}...")

    # ── Verdict ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    if not issues and speedup > 1.5 and len(par_result.errors) == 0:
        print(f"VERDICT: CONCURRENT AGENTS WORK CORRECTLY.")
        print(f"  {speedup:.1f}x speedup, {par_result.max_concurrent_calls} concurrent calls,")
        print(f"  zero state leakage, zero errors.")
    elif issues:
        print(f"VERDICT: ISSUES FOUND — {len(issues)} isolation problems.")
    else:
        print(f"VERDICT: WORKS but limited speedup ({speedup:.1f}x).")
    print(f"{'=' * 80}")

    # Cleanup
    import shutil
    shutil.rmtree(workspace_seq, ignore_errors=True)
    shutil.rmtree(workspace_par, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_concurrent_test())
