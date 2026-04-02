"""Tests for agentis.hooks.registry — HookRegistry."""

from __future__ import annotations

from agentis.hooks.registry import HookRegistry
from agentis.types import HookAction, HookContext, HookResponse, LifecycleEvent


def _ctx(
    event: LifecycleEvent = LifecycleEvent.PRE_TOOL_USE,
    tool_name: str | None = "grep",
    arguments: dict | None = None,
) -> HookContext:
    return HookContext(
        event=event,
        tool_name=tool_name,
        arguments=arguments or {},
        agent_id="agent-1",
        session_id="sess-1",
    )


def _allow_hook(ctx: HookContext) -> HookResponse:
    return HookResponse(action=HookAction.ALLOW)


def _deny_hook(ctx: HookContext) -> HookResponse:
    return HookResponse(action=HookAction.DENY, reason="blocked")


def _ask_human_hook(ctx: HookContext) -> HookResponse:
    return HookResponse(action=HookAction.ASK_HUMAN)


def _raising_hook(ctx: HookContext) -> HookResponse:
    raise RuntimeError("hook crashed")


class TestHookRegistryBasics:
    def test_construction(self) -> None:
        r = HookRegistry()
        assert r.list_hooks() == []

    def test_register_hook(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _allow_hook, name="my_hook")
        hooks = r.list_hooks(LifecycleEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0].name == "my_hook"

    def test_register_multiple_events(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _allow_hook, name="a")
        r.register(LifecycleEvent.POST_TOOL_USE, _allow_hook, name="b")
        assert len(r.list_hooks(LifecycleEvent.PRE_TOOL_USE)) == 1
        assert len(r.list_hooks(LifecycleEvent.POST_TOOL_USE)) == 1

    def test_list_all_hooks(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _allow_hook, name="a")
        r.register(LifecycleEvent.ON_ERROR, _allow_hook, name="b")
        all_hooks = r.list_hooks()
        assert len(all_hooks) == 2

    def test_unregister_hook(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _allow_hook, name="removable")
        assert r.unregister(LifecycleEvent.PRE_TOOL_USE, "removable") is True
        assert len(r.list_hooks(LifecycleEvent.PRE_TOOL_USE)) == 0

    def test_unregister_nonexistent(self) -> None:
        r = HookRegistry()
        assert r.unregister(LifecycleEvent.PRE_TOOL_USE, "ghost") is False


class TestHookRegistryRunHooks:
    def test_allow_when_no_hooks(self) -> None:
        r = HookRegistry()
        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx())
        assert result.action == HookAction.ALLOW

    def test_allow_when_all_allow(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _allow_hook, name="a")
        r.register(LifecycleEvent.PRE_TOOL_USE, _allow_hook, name="b")
        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx())
        assert result.action == HookAction.ALLOW

    def test_deny_short_circuits(self) -> None:
        """First DENY should short-circuit — later hooks don't run."""
        call_log: list[str] = []

        def logging_allow(ctx: HookContext) -> HookResponse:
            call_log.append("allow")
            return HookResponse(action=HookAction.ALLOW)

        def logging_deny(ctx: HookContext) -> HookResponse:
            call_log.append("deny")
            return HookResponse(action=HookAction.DENY, reason="nope")

        def logging_after(ctx: HookContext) -> HookResponse:
            call_log.append("after_deny")
            return HookResponse(action=HookAction.ALLOW)

        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, logging_allow, name="a")
        r.register(LifecycleEvent.PRE_TOOL_USE, logging_deny, name="b")
        r.register(LifecycleEvent.PRE_TOOL_USE, logging_after, name="c")

        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx())
        assert result.action == HookAction.DENY
        assert result.reason == "nope"
        assert "after_deny" not in call_log

    def test_ask_human_wins_if_no_deny(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _allow_hook, name="a")
        r.register(LifecycleEvent.PRE_TOOL_USE, _ask_human_hook, name="b")
        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx())
        assert result.action == HookAction.ASK_HUMAN

    def test_deny_beats_ask_human(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _ask_human_hook, name="a")
        r.register(LifecycleEvent.PRE_TOOL_USE, _deny_hook, name="b")
        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx())
        assert result.action == HookAction.DENY

    def test_deny_includes_reason(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _deny_hook, name="a")
        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx())
        assert result.reason == "blocked"


class TestHookRegistryToolFiltering:
    def test_hook_only_fires_for_matching_tools(self) -> None:
        call_count = 0

        def counting_hook(ctx: HookContext) -> HookResponse:
            nonlocal call_count
            call_count += 1
            return HookResponse(action=HookAction.DENY, reason="no bash")

        r = HookRegistry()
        r.register(
            LifecycleEvent.PRE_TOOL_USE,
            counting_hook,
            name="bash_blocker",
            tools={"bash"},
        )

        # Should NOT fire for grep
        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx(tool_name="grep"))
        assert result.action == HookAction.ALLOW
        assert call_count == 0

        # Should fire for bash
        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx(tool_name="bash"))
        assert result.action == HookAction.DENY
        assert call_count == 1

    def test_hook_with_no_tools_filter_fires_for_all(self) -> None:
        call_count = 0

        def counting_hook(ctx: HookContext) -> HookResponse:
            nonlocal call_count
            call_count += 1
            return HookResponse(action=HookAction.ALLOW)

        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, counting_hook, name="all_tools")

        r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx(tool_name="grep"))
        r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx(tool_name="bash"))
        assert call_count == 2

    def test_multiple_tool_filters(self) -> None:
        r = HookRegistry()
        r.register(
            LifecycleEvent.PRE_TOOL_USE,
            _deny_hook,
            name="fs_blocker",
            tools={"file_read", "file_write", "file_edit"},
        )

        result_read = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx(tool_name="file_read"))
        assert result_read.action == HookAction.DENY
        result_grep = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx(tool_name="grep"))
        assert result_grep.action == HookAction.ALLOW


class TestHookRegistryFailBehavior:
    def test_pre_tool_use_fail_closed(self) -> None:
        """PRE_TOOL_USE is safety-critical — hook exceptions result in DENY."""
        r = HookRegistry()
        r.register(LifecycleEvent.PRE_TOOL_USE, _raising_hook, name="broken")

        result = r.run_hooks(LifecycleEvent.PRE_TOOL_USE, _ctx())
        assert result.action == HookAction.DENY
        assert "hook crashed" in (result.reason or "")

    def test_other_events_fail_open(self) -> None:
        """Non-safety-critical events fail open — hook exceptions are logged, ALLOW continues."""
        r = HookRegistry()
        r.register(LifecycleEvent.POST_TOOL_USE, _raising_hook, name="broken")

        result = r.run_hooks(LifecycleEvent.POST_TOOL_USE, _ctx(event=LifecycleEvent.POST_TOOL_USE))
        assert result.action == HookAction.ALLOW

    def test_on_error_event_fail_open(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.ON_ERROR, _raising_hook, name="broken")

        result = r.run_hooks(LifecycleEvent.ON_ERROR, _ctx(event=LifecycleEvent.ON_ERROR))
        assert result.action == HookAction.ALLOW

    def test_session_start_fail_open(self) -> None:
        r = HookRegistry()
        r.register(LifecycleEvent.SESSION_START, _raising_hook, name="broken")

        result = r.run_hooks(LifecycleEvent.SESSION_START, _ctx(event=LifecycleEvent.SESSION_START))
        assert result.action == HookAction.ALLOW
