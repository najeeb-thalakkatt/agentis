# Agentis

The harness your model is missing. Production-grade, provider-agnostic Python framework for building agentic AI systems.

## What Agentis Does

- **Agentic loop** with three APIs: `run()` (autonomous), `step()` (single turn), `steps()` (async iterator)
- **`@tool` decorator** with permission system: READ_ONLY (parallel), MUTATING (serial), DANGEROUS (approval required)
- **Safety hooks** that the LLM cannot bypass: block destructive commands, path sandboxing, human approval gates
- **Cost tracking** with per-turn token accounting and budget limits
- **Persistent memory** across sessions: lightweight index in context, full data on disk
- **Multi-agent**: ForkAgent for parallel read-only investigation, TeammateAgent for mailbox-based coordination
- **Context compaction**: five-layer tiered eviction so long sessions don't overflow
- **Provider-agnostic**: Anthropic, OpenAI, any OpenAI-compatible endpoint. Swap in one line.

## Install

```bash
pip install agentis                    # Core (aiofiles only)
pip install agentis[anthropic]         # + Anthropic SDK
pip install agentis[openai]            # + OpenAI SDK
pip install agentis[all]               # Everything
```

Requires Python 3.11+.

## Scaffold a new project

```bash
pip install agentis[anthropic]
agentis new myagent
cd myagent
cp .env.example .env                   # paste your ANTHROPIC_API_KEY
pip install -e '.[anthropic]'
python main.py "what is 17 + 25?"
```

`agentis doctor` prints a diagnostic summary (SDKs installed, env vars set). Exit code 0 means you're ready.

## Environment variables

Each provider declares an env var and exposes `from_env()` for explicit auto-construction:

| Provider | Env var |
|---|---|
| `AnthropicProvider` | `ANTHROPIC_API_KEY` |
| `OpenAIProvider`    | `OPENAI_API_KEY`    |

```python
from agentis import AnthropicProvider
provider = AnthropicProvider.from_env()   # raises ConfigError naming the var if missing
```

Auth, rate-limit, and network failures surface as `AuthenticationError`, `RateLimitError`, and `ProviderNetworkError` (all subclasses of `ProviderError`) — each with an actionable message, not a generic wrap.

## Quickstart

```python
import asyncio
from agentis import AgentRuntime, AnthropicProvider, tool, Permission

@tool()
async def lookup_weather(city: str) -> str:
    """Look up current weather for a city."""
    return f"72F and sunny in {city}"

@tool(permission=Permission.MUTATING)
async def save_report(content: str) -> bool:
    """Save a weather report."""
    print(f"Saved: {content}")
    return True

async def main():
    provider = AnthropicProvider.from_env()  # reads ANTHROPIC_API_KEY

    agent = AgentRuntime(
        provider=provider,
        tools=[lookup_weather, save_report],
        system_prompt="You are a weather assistant.",
    )
    result = await agent.run("What's the weather in Portland? Save a report.")
    print(result)

asyncio.run(main())
```

The agent will call `lookup_weather` (parallel-safe), then `save_report` (serialized behind a lock), then respond. All tool calls go through safety hooks. All results are structured — the LLM never parses raw output.

## Architecture

```
                    +------------------+
                    |    Your Code     |
                    +--------+---------+
                             |
                run() / step() / steps()
                             |
               +-------------v--------------+
               |       AgentRuntime          |
               |                             |
               |  Session    HookRegistry    |
               |  MemoryIndex  Compactor     |
               |  ToolOrchestrator           |
               +-------------+---------------+
                             |
               +-------------+-------------+
               |             |             |
         +-----v----+ +-----v----+ +------v-----+
         | Anthropic | |  OpenAI  | | Compatible |
         | Provider  | | Provider | | (Ollama,..)| 
         +----------+ +----------+ +------------+
```

## Features

### Custom Tools

```python
from agentis import tool, Permission

@tool()
async def search(query: str, max_results: int = 10) -> list[str]:
    """Search the knowledge base."""
    ...  # READ_ONLY by default — runs in parallel

@tool(permission=Permission.MUTATING)
async def write_file(path: str, content: str) -> bool:
    """Write to a file."""
    ...  # Serialized behind an asyncio.Lock

@tool(permission=Permission.DANGEROUS)
async def drop_table(name: str) -> str:
    """Drop a database table."""
    ...  # Requires human approval via callback
```

Type hints become JSON Schema automatically. Exceptions become `ToolResult(success=False)` — tools never crash the agent.

### Safety Hooks

```python
from agentis import HookRegistry, LifecycleEvent
from agentis.hooks.builtins import block_destructive_commands, create_path_sandbox

hooks = HookRegistry()

# Block rm -rf, DROP DATABASE, fork bombs, etc.
hooks.register(LifecycleEvent.PRE_TOOL_USE, block_destructive_commands)

# Restrict file access to the project directory
hooks.register(
    LifecycleEvent.PRE_TOOL_USE,
    create_path_sandbox(["/home/user/project"]),
)
```

Hooks are **code, not prompts**. The LLM cannot talk its way past them. `PRE_TOOL_USE` is fail-closed: if a hook crashes, the operation is denied.

### Cost Tracking

```python
from agentis import CostTracker

tracker = CostTracker(budget_usd=5.00)

agent = AgentRuntime(
    provider=provider,
    extensions=[tracker],
    ...
)
await agent.run("Do the thing")

print(tracker.total_cost())      # $0.0234
print(tracker.is_over_budget())  # False
print(tracker.get_report())      # Full breakdown
```

### Human Approval

```python
from agentis import ApprovalRequest

async def ask_user(request: ApprovalRequest) -> bool:
    answer = input(f"Allow {request.tool_name}({request.arguments})? [y/n] ")
    return answer.lower() == "y"

agent = AgentRuntime(
    provider=provider,
    tools=[dangerous_tool],
    approval_callback=ask_user,
)
```

DANGEROUS tools pause and ask. The callback receives rich context: tool name, arguments, reason, session ID, turn number.

### Multi-Agent

**ForkAgent** — cheap parallel clones for read-only investigation:

```python
from agentis import ForkAgent

results = await ForkAgent.parallel_investigate(
    parent=agent,
    tasks=["Check auth module", "Check database module", "Check API module"],
)
# 3 results, run concurrently. On providers with prompt caching: ~1x cost.
```

**TeammateAgent** — independent agents with mailbox coordination:

```python
from agentis import TeammateAgent, InMemoryMailbox

mailbox = InMemoryMailbox()
researcher = TeammateAgent(name="researcher", runtime=runtime1, mailbox=mailbox)
writer = TeammateAgent(name="writer", runtime=runtime2, mailbox=mailbox)

await researcher.send("writer", {"findings": "JWT bug in auth.py:42"})
mail = await writer.check_mail()  # [{"from": "researcher", "content": {...}}]
```

### Memory

```python
from agentis import MemoryIndex

memory = MemoryIndex(workspace=".agentis/memory")

# Store — only a pointer goes into LLM context
await memory.remember("Auth system", "Uses JWT with 15-min expiry", tags=["security"])

# Recall — full content fetched on demand
content = await memory.recall(topic_id)

# Search
results = await memory.search(tags=["security"])

# Persists across sessions
memory2 = MemoryIndex(workspace=".agentis/memory")
await memory2.load()  # Picks up where you left off
```

The memory index is always in context (~150 chars per entry). Full content lives on disk and is fetched via the built-in `recall` tool.

### Tool Packs

```python
from agentis.packs.filesystem import TOOLS as fs_tools   # file_read, file_write, file_edit, list_directory
from agentis.packs.coding import TOOLS as coding_tools    # grep, glob
from agentis.packs.shell import TOOLS as shell_tools      # bash (DANGEROUS)
from agentis.packs.web import TOOLS as web_tools          # http_get, http_post
from agentis.packs.data import TOOLS as data_tools        # query_json

agent = AgentRuntime(provider=provider, tools=fs_tools + coding_tools)
```

### Provider Swap

```python
# Anthropic Claude
from agentis import AnthropicProvider
provider = AnthropicProvider(model="claude-sonnet-4-20250514")

# OpenAI
from agentis import OpenAIProvider
provider = OpenAIProvider(model="gpt-4o")

# Any OpenAI-compatible endpoint (Ollama, vLLM, Together, Groq, LM Studio)
from agentis import OpenAICompatibleProvider
provider = OpenAICompatibleProvider(model="llama3", base_url="http://localhost:11434/v1")
```

Same agent code. Different provider. One line.

## Examples

Run any example without API keys (they use a built-in MockProvider):

| Example | What it shows |
|---------|---------------|
| [`01_quickstart.py`](examples/01_quickstart.py) | Basic loop, `@tool`, permissions, `run()` and `step()` |
| [`02_safety_and_cost.py`](examples/02_safety_and_cost.py) | Safety hooks, cost tracking, human approval, `steps()` |
| [`03_multi_agent.py`](examples/03_multi_agent.py) | ForkAgent parallel investigation, TeammateAgent coordination |
| [`04_memory_and_providers.py`](examples/04_memory_and_providers.py) | Persistent memory, provider swap |

```bash
python examples/01_quickstart.py
python examples/02_safety_and_cost.py
python examples/03_multi_agent.py
python examples/04_memory_and_providers.py
```

## API Quick Reference

| Class | Purpose |
|-------|---------|
| `AgentRuntime` | The agentic loop — `run()`, `step()`, `steps()` |
| `@tool` | Decorator to create tools from async functions |
| `Permission` | `READ_ONLY`, `MUTATING`, `DANGEROUS` |
| `HookRegistry` | Register lifecycle hooks (8 events) |
| `MemoryIndex` | Persistent memory index with `remember()`/`recall()` |
| `ForkAgent` | Parallel read-only investigation |
| `TeammateAgent` | Mailbox-based agent coordination |
| `WorktreeAgent` | Isolated execution with pluggable isolation strategy |
| `CostTracker` | Token/cost tracking with budget limits |
| `DreamExtension` | Background memory consolidation (opt-in) |
| `AnthropicProvider` | Anthropic Claude models |
| `OpenAIProvider` | OpenAI models |
| `OpenAICompatibleProvider` | Any OpenAI-compatible endpoint |

## License

MIT
