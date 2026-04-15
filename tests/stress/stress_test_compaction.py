"""Stress Test #2: Long-session compaction with dummy LLM.

Runs 30+ tool calls through a single AgentRuntime session with a small
context window. Tracks compaction events turn by turn, verifies context
stays bounded, and checks quality after compaction.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

from agentis.compaction.compactor import CompactionResult
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import AgentRuntime, StepResult
from agentis.tools.decorator import tool
from agentis.types import (
    Message,
    Permission,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)


# ── Dummy LLM Provider ────────────────────────────────────


class StressTestProvider:
    """Provider that simulates a long investigation session.

    Generates N tool-calling turns, each with verbose content,
    then a final response that references earlier findings.
    """

    def __init__(self, tool_turns: int = 30) -> None:
        self._tool_turns = tool_turns
        self._turn = 0
        self.call_count = 0
        self.context_sizes: list[int] = []

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.call_count += 1
        # Track context size sent to us
        total_chars = sum(len(m.content) for m in messages)
        self.context_sizes.append(total_chars)
        self._turn += 1

        if self._turn <= self._tool_turns:
            # Alternate between different tools to simulate real investigation
            tool_cycle = [
                ("get_pods", {"namespace": f"ns-{self._turn}"}),
                ("get_metrics", {"service": f"svc-{self._turn}"}),
                ("get_logs", {"pod": f"pod-{self._turn}"}),
                ("get_events", {"namespace": f"ns-{self._turn}"}),
                ("get_deploys", {"service": f"svc-{self._turn}"}),
            ]
            tool_name, tool_args = tool_cycle[self._turn % len(tool_cycle)]

            # Verbose reasoning text (like a real model would produce)
            reasoning = (
                f"Turn {self._turn}: Investigating {tool_name} for "
                f"{list(tool_args.values())[0]}. Based on previous findings, "
                f"I need to check additional metrics and correlate with "
                f"the patterns observed in earlier investigation steps. "
                f"The error pattern suggests a cascading failure that "
                f"started around turn {max(1, self._turn - 3)}. "
                + ("Additional analysis context. " * 20)
            )

            return ProviderResponse(
                content=reasoning,
                tool_calls=[ToolCall(
                    id=f"call_{self._turn}",
                    name=tool_name,
                    arguments=tool_args,
                )],
                usage=TokenUsage(
                    input_tokens=total_chars // 3,
                    output_tokens=len(reasoning) // 3,
                ),
                stop_reason="tool_use",
            )
        else:
            # Final diagnosis referencing earlier findings
            final = (
                "DIAGNOSIS COMPLETE.\n\n"
                "Root Cause: Memory leak in svc-15 caused cascading OOM kills.\n"
                "Evidence:\n"
                "- Pod restarts started at turn 5 (ns-5 showed CrashLoopBackOff)\n"
                "- Metrics from svc-10 showed memory climbing from 512MB to 2GB\n"
                "- Logs from pod-15 contained 'java.lang.OutOfMemoryError'\n"
                "- Deploy history shows svc-15 was deployed at 14:30 with new config\n\n"
                "Remediation: Rolled back svc-15 to previous version.\n"
                "Verification: Memory usage returned to baseline within 5 minutes."
            )
            return ProviderResponse(
                content=final,
                tool_calls=[],
                usage=TokenUsage(input_tokens=200, output_tokens=len(final) // 3),
            )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="stress_test",
            max_context_tokens=16_000,  # Small window to force compaction
        )


# ── Tools (simulate SentinelOps-style investigation) ───────


@tool()
async def get_pods(namespace: str) -> str:
    """Get pod status for a namespace."""
    return (
        f"Pods in {namespace}:\n"
        + "\n".join(
            f"  pod-{namespace}-{i}: Running, restarts={i % 3}, "
            f"memory={256 + i * 64}MB, cpu={10 + i * 5}%"
            for i in range(8)
        )
        + f"\nTotal: 8 pods, {sum(i % 3 for i in range(8))} restarts"
    )


@tool()
async def get_metrics(service: str) -> str:
    """Get service metrics."""
    return (
        f"Metrics for {service} (last 30min):\n"
        f"  p50_latency: 45ms\n  p99_latency: 890ms\n"
        f"  error_rate: 2.3%\n  request_rate: 1250 req/s\n"
        f"  memory_usage: 1.8GB / 2.0GB (90%)\n"
        f"  cpu_usage: 78%\n"
        f"  gc_pause_time: 120ms (WARNING: elevated)\n"
        f"  heap_used: 1.6GB\n  heap_max: 2.0GB\n"
        f"  thread_count: 245\n  connection_pool: 180/200\n"
        + ("  datapoint: " + "x" * 50 + "\n") * 10
    )


@tool()
async def get_logs(pod: str) -> str:
    """Get pod logs."""
    return (
        f"Logs for {pod} (last 100 lines):\n"
        + "\n".join(
            f"  2026-04-05T10:{i:02d}:00Z [{['INFO','WARN','ERROR'][i%3]}] "
            f"Request processed in {10 + i * 5}ms, "
            f"memory={512 + i * 32}MB, queue_depth={i * 2}"
            for i in range(30)
        )
        + f"\n  2026-04-05T10:30:00Z [ERROR] java.lang.OutOfMemoryError: "
        f"GC overhead limit exceeded\n"
        f"    at com.example.service.DataProcessor.process(DataProcessor.java:142)"
    )


@tool()
async def get_events(namespace: str) -> str:
    """Get Kubernetes events."""
    return (
        f"Events in {namespace}:\n"
        + "\n".join(
            f"  {['Normal','Warning'][i%2]} {['Scheduled','Pulling','Started','OOMKilled','BackOff'][i%5]} "
            f"pod/{namespace}-worker-{i}: message about event {i}"
            for i in range(15)
        )
    )


@tool()
async def get_deploys(service: str) -> str:
    """Get deployment history."""
    return (
        f"Deploy history for {service}:\n"
        f"  v3.2.1 @ 2026-04-05T14:30:00Z - config: heap=2G, replicas=3\n"
        f"  v3.2.0 @ 2026-04-04T10:00:00Z - config: heap=1G, replicas=3\n"
        f"  v3.1.9 @ 2026-04-01T08:00:00Z - config: heap=1G, replicas=2\n"
        f"  Changed: JAVA_OPTS heap size 1G -> 2G, added GC logging\n"
        f"  Rollback available: v3.2.0 (last known good)"
    )


# ── Stress Test Runner ─────────────────────────────────────


async def run_stress_test(
    tool_turns: int = 30,
    max_context: int = 16_000,
) -> None:
    """Run the compaction stress test."""
    print(f"=" * 70)
    print(f"COMPACTION STRESS TEST")
    print(f"  Tool-calling turns: {tool_turns}")
    print(f"  Max context tokens: {max_context:,}")
    print(f"  Compaction threshold: {int(max_context * 0.85):,} (85%)")
    print(f"=" * 70)

    provider = StressTestProvider(tool_turns=tool_turns)

    rt = AgentRuntime(
        provider=provider,
        tools=[get_pods, get_metrics, get_logs, get_events, get_deploys],
        system_prompt=(
            "You are SentinelOps, an SRE incident response agent. "
            "Investigate the incident thoroughly by checking pods, metrics, "
            "logs, events, and deployment history across all namespaces."
        ),
        max_turns=tool_turns + 2,
        skeptical=False,
    )

    # Track compaction events
    start = time.time()
    turn = 0
    compaction_events: list[dict[str, Any]] = []
    prev_tokens = 0
    prev_entries = 0

    print(f"\n{'Turn':>4} | {'Tokens':>7} | {'Entries':>7} | {'Compacted':>9} | Tools")
    print("-" * 70)

    async for step in rt.steps("Critical incident: multiple services down in production"):
        turn += 1
        current_tokens = rt._compactor.current_tokens()
        current_entries = len(rt._compactor.get_entries())

        # Detect compaction (tokens decreased or entries decreased)
        compacted = ""
        if current_tokens < prev_tokens and prev_tokens > 0:
            freed = prev_tokens - current_tokens
            removed = prev_entries - current_entries
            compacted = f"-{freed}tok"
            compaction_events.append({
                "turn": turn,
                "tokens_before": prev_tokens,
                "tokens_after": current_tokens,
                "freed": freed,
                "entries_before": prev_entries,
                "entries_after": current_entries,
                "removed": removed,
            })

        tools = [tc.name for tc in (step.response.tool_calls or [])]
        tool_str = ", ".join(tools) if tools else "(final response)"

        print(f"{turn:>4} | {current_tokens:>7,} | {current_entries:>7} | {compacted:>9} | {tool_str}")

        prev_tokens = current_tokens
        prev_entries = current_entries

    elapsed = time.time() - start

    # ── Results ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\nSession Stats:")
    print(f"  Total turns: {turn}")
    print(f"  LLM calls: {provider.call_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Final token count: {rt._compactor.current_tokens():,}")
    print(f"  Final entry count: {len(rt._compactor.get_entries())}")

    print(f"\nCompaction Events: {len(compaction_events)}")
    for i, evt in enumerate(compaction_events):
        print(f"  [{i+1}] Turn {evt['turn']}: "
              f"{evt['tokens_before']:,} -> {evt['tokens_after']:,} "
              f"(freed {evt['freed']:,} tokens, removed {evt['removed']} entries)")

    print(f"\nContext Size Sent to Provider (chars):")
    sizes = provider.context_sizes
    print(f"  Min: {min(sizes):,}")
    print(f"  Max: {max(sizes):,}")
    print(f"  Final: {sizes[-1]:,}")
    max_allowed = max_context * 4  # chars ~= tokens * 3, with margin
    exceeded = [s for s in sizes if s > max_allowed]
    if exceeded:
        print(f"  WARNING: {len(exceeded)} calls exceeded estimated limit!")
    else:
        print(f"  All calls within bounds (< {max_allowed:,} chars)")

    # Check final response quality
    messages = rt._session.get_messages()
    final_msg = messages[-1].content if messages else ""
    has_diagnosis = any(
        kw in final_msg.lower()
        for kw in ["root cause", "diagnosis", "memory leak", "oom", "remediation"]
    )
    print(f"\nQuality Check:")
    print(f"  Final response length: {len(final_msg)} chars")
    print(f"  Contains diagnosis keywords: {has_diagnosis}")
    print(f"  Session messages remaining: {len(messages)}")

    # Verdict
    print(f"\n{'=' * 70}")
    if len(compaction_events) > 0 and has_diagnosis:
        print("PASS: Compaction fired and agent produced a valid diagnosis.")
    elif len(compaction_events) == 0:
        print("WARN: Compaction never fired. Context may not have exceeded threshold.")
    else:
        print("WARN: Compaction fired but diagnosis quality unclear.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 16_000
    asyncio.run(run_stress_test(tool_turns=turns, max_context=ctx))
