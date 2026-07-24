from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest
from msgspec import UNSET

from plyngent.agent import ChatAgent, ToolRegistry
from plyngent.agent.todo_nag import inject_todo_nag, parse_todo_nag_strategy
from plyngent.agent.todo_stack import TodoStack, parse_push_titles
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    DeveloperChatMessage,
    ToolChatMessage,
    UserChatMessage,
)
from plyngent.memory import MemoryStore
from plyngent.tools.context import SessionState
from plyngent.tools.todo import TODO_TOOLS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _session_for(stack: TodoStack) -> SessionState:
    """Bind a live stack via SessionState (no process-global set_todo_stack)."""
    return SessionState(session_id="test", todo=stack)


def _todo_registry(stack: TodoStack) -> ToolRegistry:
    return ToolRegistry(list(TODO_TOOLS), session_state=_session_for(stack))


def test_parse_push_titles() -> None:
    assert parse_push_titles("only") == ["only"]
    assert parse_push_titles("T1\nT2") == ["T1", "T2"]
    assert parse_push_titles("T1; T2; T3") == ["T1", "T2", "T3"]
    assert parse_push_titles('["A", "B"]') == ["A", "B"]


def test_ids_reuse_after_clear() -> None:
    stack = TodoStack()
    g = stack.push_group(["A", "B"])
    assert [i.id for i in g.items] == ["t1", "t2"]
    assert stack.clear() == 2
    g2 = stack.push_group(["C"])
    assert [i.id for i in g2.items] == ["t1"]


def test_ids_reuse_after_pop() -> None:
    stack = TodoStack()
    _ = stack.push_group(["A", "B"])  # t1, t2
    g2 = stack.push_group(["C"])  # t3
    assert g2.items[0].id == "t3"
    _ = stack.pop()  # drop t3
    g3 = stack.push_group(["D"])
    # Highest live id is t2 → next is t3 again
    assert g3.items[0].id == "t3"


def test_push_is_group_not_per_task_stack() -> None:
    """Multi-title push is one group; pop removes the whole group."""
    stack = TodoStack()
    g = stack.push_group(["T1", "T2"])
    assert stack.depth == 1
    assert [i.title for i in g.items] == ["T1", "T2"]
    assert stack.top_group is g
    # Not two stack levels of single tasks
    assert len(stack.groups) == 1

    g2 = stack.push_group(["T1.1", "T1.2"])
    assert stack.depth == 2
    assert [i.title for i in g2.items] == ["T1.1", "T1.2"]

    popped = stack.pop()
    assert popped is not None
    assert [i.title for i in popped.items] == ["T1.1", "T1.2"]
    assert stack.depth == 1
    assert stack.top_group is not None
    assert [i.title for i in stack.top_group.items] == ["T1", "T2"]


def test_dfs_breakdown_with_groups() -> None:
    """push [T1,T2] → push [T1.1,T1.2] → pop → push [T2.1]."""
    stack = TodoStack()
    root = stack.push_group(["T1", "T2"])
    children = stack.push_group(["T1.1", "T1.2"])
    assert stack.depth == 2
    stack.update(children.items[0].id, status="done")
    stack.update(children.items[1].id, status="done")
    _ = stack.pop()
    assert stack.depth == 1
    stack.update(root.items[0].id, status="done")
    _ = stack.push_group(["T2.1"])
    assert stack.depth == 2
    assert stack.top_group is not None
    assert stack.top_group.items[0].title == "T2.1"
    assert stack.groups[0].items[1].title == "T2"


def test_single_title_still_one_group() -> None:
    stack = TodoStack()
    item = stack.push("only")
    assert stack.depth == 1
    assert item.title == "only"
    g = stack.pop()
    assert g is not None and len(g.items) == 1
    assert stack.is_empty()


def test_todo_stack_needs_review() -> None:
    stack = TodoStack()
    assert not stack.needs_review()
    item = stack.push("work")
    stack.begin_turn()
    # Non-empty + untouched → nag (open or hygiene).
    assert stack.needs_review()
    # Any todo_* access this turn suppresses end-of-turn nag, even with open items.
    stack.mark_touched()
    assert not stack.needs_review()
    stack.update(item.id, status="done")
    stack.begin_turn()
    # Terminal-only stack: same gate (untouched only).
    assert stack.needs_review()
    stack.mark_touched()
    assert not stack.needs_review()


def test_parse_todo_nag_strategy() -> None:
    assert parse_todo_nag_strategy("developer") == "developer"
    assert parse_todo_nag_strategy("SYNTHETIC-TOOL") == "synthetic_tool"
    assert parse_todo_nag_strategy("nope") == "developer"
    assert parse_todo_nag_strategy(None) == "developer"
    # Legacy alias — mid-turn system was not a useful Responses channel.
    assert parse_todo_nag_strategy("system") == "developer"


def test_inject_todo_nag_strategies() -> None:
    from plyngent.agent.events import ToolCallEvent, ToolResultEvent
    from plyngent.agent.todo_nag import (
        inject_todo_nag_for_stack_with_events,
        inject_todo_nag_with_events,
        synthetic_todo_list_result,
    )

    body = "[TODO OPEN WORK] test"
    messages: list[AnyChatMessage] = []
    assert inject_todo_nag(messages, body, strategy="none") is False
    assert messages == []

    messages = []
    assert inject_todo_nag(messages, body, strategy="developer") is True
    assert isinstance(messages[0], DeveloperChatMessage)
    assert body in messages[0].content

    messages = []
    assert inject_todo_nag(messages, body, strategy="user") is True
    assert isinstance(messages[0], UserChatMessage)

    # synthetic_tool: real todo_list-shaped body (stack.render), not OPEN WORK prose
    stack = TodoStack()
    _ = stack.push("synthetic body item")
    real = synthetic_todo_list_result(stack)
    assert real == stack.render()
    assert "[TODO OPEN WORK]" not in real
    assert "synthetic body item" in real

    messages = []
    ok, events = inject_todo_nag_for_stack_with_events(
        messages,
        stack,
        kind="end_of_turn",
        strategy="synthetic_tool",
    )
    assert ok is True
    assert len(messages) == 2
    assert isinstance(messages[0], AssistantChatMessage)
    tool_calls = messages[0].tool_calls
    assert tool_calls is not UNSET and tool_calls
    assert isinstance(messages[1], ToolChatMessage)
    tool_content = messages[1].content
    assert isinstance(tool_content, str)
    assert tool_content == real
    assert messages[1].tool_call_id.startswith("todo-nag-")
    assert len(events) == 2
    assert isinstance(events[0], ToolCallEvent)
    assert isinstance(events[1], ToolResultEvent)
    result_event = events[1]
    assert isinstance(result_event, ToolResultEvent)
    result_content = result_event.message.content
    assert isinstance(result_content, str)
    assert result_content == real

    # Raw inject still accepts an explicit body (e.g. tests / custom)
    messages = []
    ok2, events2 = inject_todo_nag_with_events(messages, body, strategy="synthetic_tool")
    assert ok2
    assert len(events2) == 2
    assert isinstance(events2[1], ToolResultEvent)
    raw_content = events2[1].message.content
    assert isinstance(raw_content, str)
    assert body in raw_content


def test_todo_prompts_signal_undone_work() -> None:
    stack = TodoStack()
    item = stack.push("open work")
    reminder = stack.turn_reminder_prompt()
    assert "[TODO OPEN WORK]" in reminder
    assert "Stack not empty" in reminder
    assert "unfinished work" in reminder.lower()
    assert "open work" in reminder

    review = stack.review_prompt()
    assert "[TODO OPEN WORK]" in review
    assert "undone work" in review.lower() or "open item" in review.lower()
    assert "open work" in review
    assert "t1:open work" in review or "open work" in review

    stack.update(item.id, status="done")
    terminal_review = stack.review_prompt()
    assert "[TODO HYGIENE]" in terminal_review
    assert "done/cancelled" in terminal_review
    terminal_reminder = stack.turn_reminder_prompt()
    assert "[TODO HYGIENE]" in terminal_reminder


def test_legacy_flat_and_frames_migrate() -> None:
    flat = TodoStack.from_raw(
        {
            "items": [{"id": "t1", "title": "old", "status": "pending", "notes": ""}],
            "next_id": 2,
        }
    )
    assert flat.depth == 1
    assert flat.all_items()[0].title == "old"

    framed = TodoStack.from_raw(
        {
            "frames": [
                {"items": [{"id": "t1", "title": "T1", "status": "pending", "notes": ""}]},
                {"items": [{"id": "t2", "title": "T1.1", "status": "pending", "notes": ""}]},
            ],
            "next_id": 3,
        }
    )
    assert framed.depth == 2
    assert framed.top_group is not None
    assert framed.top_group.items[0].title == "T1.1"


def test_todo_stack_roundtrip_raw() -> None:
    stack = TodoStack()
    _ = stack.push_group(["x", "y"], notes="n")
    raw = stack.to_raw()
    assert "groups" in raw
    restored = TodoStack.from_raw(raw)
    assert restored.depth == 1
    assert [i.title for i in restored.groups[0].items] == ["x", "y"]


async def test_todo_tools_and_persist(tmp_path: object) -> None:
    del tmp_path
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="t")
        stack = TodoStack()
        registry = _todo_registry(stack)
        out = await registry.execute("todo_push", '{"titles": ["T1", "T2"]}')
        assert "group" in out.lower() or "pushed" in out
        assert stack.depth == 1
        assert [i.title for i in stack.groups[0].items] == ["T1", "T2"]
        out = await registry.execute("todo_push", '{"titles": ["T1", "T2"]}')
        assert stack.depth == 2
        out3 = await registry.execute("todo_pop", "{}")
        assert "popped" in out3
        assert stack.depth == 1
        _ = await memory.update_session_todo_stack(session.sid, stack.to_raw())
        loaded = await memory.get_session_todo_stack(session.sid)
        assert loaded is not None
        again = TodoStack.from_raw(loaded)
        assert again.depth == 1
        assert [i.title for i in again.groups[0].items] == ["T1", "T2"]
    finally:
        await memory.close()


class ScriptedClient:
    def __init__(self) -> None:
        self.calls = 0

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[False] = False
    ) -> ChatCompletionResponse: ...

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[True]
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: bool = False
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        del stream
        self.calls += 1
        text = "ok" if self.calls > 1 else "done without todos"
        for msg in param.messages:
            if isinstance(msg, DeveloperChatMessage) and "[TODO OPEN]" in msg.content:
                text = "reviewed stack"
                break
        return ChatCompletionResponse(
            id="1",
            object="chat.completion",
            created=0,
            model="m",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantChatMessage(content=text),
                    logprobs={},
                    finish_reason="stop",
                )
            ],
            system_fingerprint="",
            usage={},
        )


@pytest.mark.asyncio
async def test_loop_injects_todo_review_when_untouched() -> None:
    stack = TodoStack()
    _ = stack.push("open work")
    stack.begin_turn()
    client = ScriptedClient()
    agent = ChatAgent(
        client,  # type: ignore[arg-type]
        model="m",
        tools=_todo_registry(stack),
        stream=False,
        todo_stack=stack,
    )
    async for _event in agent.run("do stuff"):
        pass
    assert client.calls >= 2
    assert any(isinstance(m, DeveloperChatMessage) and "[TODO OPEN WORK]" in m.content for m in agent.messages)
    assert not any(isinstance(m, UserChatMessage) and "[TODO OPEN WORK]" in m.content for m in agent.messages)


class ScriptedClientWithTodoListThenStop:
    """First completion: todo_list tool call; second: stop (no further nag round)."""

    def __init__(self) -> None:
        self.calls = 0

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[False] = False
    ) -> ChatCompletionResponse: ...

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[True]
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: bool = False
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        del stream, param
        self.calls += 1
        if self.calls == 1:
            message = AssistantChatMessage(
                content="",
                tool_calls=[
                    AssistantFunctionToolCall(
                        id="tl1",
                        function=AssistantFunctionTool(name="todo_list", arguments="{}"),
                    )
                ],
            )
            finish: str = "tool_calls"
        else:
            message = AssistantChatMessage(content="done after list")
            finish = "stop"
        return ChatCompletionResponse(
            id="1",
            object="chat.completion",
            created=0,
            model="m",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    logprobs={},
                    finish_reason=finish,
                )
            ],
            system_fingerprint="",
            usage={},
        )


@pytest.mark.asyncio
async def test_loop_skips_todo_review_when_touched_even_if_open() -> None:
    """todo_* access this turn suppresses end-of-turn nag despite open items.

    begin_turn() resets touch at run start, so the model must actually call a
    todo tool mid-turn; then stop without a third (nag) completion.
    """
    stack = TodoStack()
    _ = stack.push("still open")

    client = ScriptedClientWithTodoListThenStop()
    agent = ChatAgent(
        client,  # type: ignore[arg-type]
        model="m",
        tools=_todo_registry(stack),
        stream=False,
        todo_stack=stack,
        # Avoid turn-start synthetic inject muddying counts; developer turn-start
        # is fine (prose only). End-of-turn is what we assert against.
        todo_nag_strategy="developer",
    )
    async for _event in agent.run("do stuff"):
        pass
    # call1: tool_calls todo_list; call2: stop — no call3 from end-of-turn nag.
    assert client.calls == 2
    assert stack.touched_this_turn
    assert not stack.needs_review()
    # End-of-turn OPEN WORK prose must not appear after a natural stop.
    end_nags = [m for m in agent.messages if isinstance(m, DeveloperChatMessage) and "You stopped with" in m.content]
    assert end_nags == []


@pytest.mark.asyncio
async def test_loop_synthetic_tool_nag_strategy() -> None:
    stack = TodoStack()
    _ = stack.push("open work")
    stack.begin_turn()
    client = ScriptedClient()
    agent = ChatAgent(
        client,  # type: ignore[arg-type]
        model="m",
        tools=_todo_registry(stack),
        stream=False,
        todo_stack=stack,
        todo_nag_strategy="synthetic_tool",
    )
    async for _event in agent.run("do stuff"):
        pass
    # Result body is real todo_list shape (render), not OPEN WORK prose.
    assert any(
        isinstance(m, ToolChatMessage) and "open work" in m.content and "[TODO OPEN WORK]" not in m.content
        for m in agent.messages
    )
    assert not any(isinstance(m, DeveloperChatMessage) and "[TODO OPEN WORK]" in m.content for m in agent.messages)


@pytest.mark.asyncio
async def test_loop_none_nag_strategy_skips_inject() -> None:
    stack = TodoStack()
    _ = stack.push("open work")
    stack.begin_turn()
    client = ScriptedClient()
    agent = ChatAgent(
        client,  # type: ignore[arg-type]
        model="m",
        tools=_todo_registry(stack),
        stream=False,
        todo_stack=stack,
        todo_nag_strategy="none",
    )
    async for _event in agent.run("do stuff"):
        pass
    # One completion only — no end-of-turn continue from nag.
    assert client.calls == 1
    assert not any(
        isinstance(m, (DeveloperChatMessage, ToolChatMessage)) and "[TODO OPEN WORK]" in getattr(m, "content", "")
        for m in agent.messages
    )


def test_refresh_synthetic_todo_nags_updates_stale_results() -> None:
    """Forged nags keep call ids; results track the live stack (not a frozen dirty snapshot)."""
    from plyngent.agent.todo_nag import (
        inject_todo_nag_for_stack,
        is_synthetic_todo_nag_call_id,
        refresh_synthetic_todo_nags,
    )

    stack = TodoStack()
    _ = stack.push("stale dirty item")
    messages: list[AnyChatMessage] = []
    assert inject_todo_nag_for_stack(messages, stack, kind="end_of_turn", strategy="synthetic_tool")
    assert any(
        isinstance(m, ToolChatMessage)
        and "stale dirty item" in m.content
        and is_synthetic_todo_nag_call_id(m.tool_call_id)
        for m in messages
    )

    _ = stack.clear()
    n = refresh_synthetic_todo_nags(messages, stack)
    assert n >= 1
    for m in messages:
        if isinstance(m, ToolChatMessage) and is_synthetic_todo_nag_call_id(m.tool_call_id):
            assert "stale dirty item" not in m.content
            assert "empty" in m.content.lower()


@pytest.mark.asyncio
async def test_loop_synthetic_tool_refreshes_after_stack_cleared() -> None:
    """After a dirty stack is cleaned, later turns must not re-show old nag text."""

    class CaptureClient:
        def __init__(self) -> None:
            self.calls = 0
            self.payloads: list[list[AnyChatMessage]] = []

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[False] = False
        ) -> ChatCompletionResponse: ...

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[True]
        ) -> AsyncIterator[ChatCompletionChunk]: ...

        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: bool = False
        ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
            del stream
            self.calls += 1
            self.payloads.append(list(param.messages))
            if self.calls == 1:
                # First stop → end-of-turn synthetic nag → second call clears.
                message = AssistantChatMessage(content="first stop")
                finish = "stop"
            elif self.calls == 2:
                message = AssistantChatMessage(
                    content="",
                    tool_calls=[
                        AssistantFunctionToolCall(
                            id="clr",
                            function=AssistantFunctionTool(name="todo_clear", arguments="{}"),
                        )
                    ],
                )
                finish = "tool_calls"
            else:
                message = AssistantChatMessage(content="after clear")
                finish = "stop"
            return ChatCompletionResponse(
                id="1",
                object="chat.completion",
                created=0,
                model="m",
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        logprobs={},
                        finish_reason=finish,
                    )
                ],
                system_fingerprint="",
                usage={},
            )

    stack = TodoStack()
    _ = stack.push("was dirty")
    client = CaptureClient()
    agent = ChatAgent(
        client,  # type: ignore[arg-type]
        model="m",
        tools=_todo_registry(stack),
        stream=False,
        todo_stack=stack,
        todo_nag_strategy="synthetic_tool",
    )
    async for _event in agent.run("turn1"):
        pass
    assert stack.is_empty()
    n_after_turn1 = client.calls
    async for _event in agent.run("turn2 clean"):
        pass
    # Later request payloads must not re-present the old dirty item via synth nags.
    for payload in client.payloads[n_after_turn1:]:
        for msg in payload:
            if isinstance(msg, ToolChatMessage) and msg.tool_call_id.startswith("todo-nag-"):
                assert "was dirty" not in msg.content
    # Durable history refreshed at turn start as well.
    for msg in agent.messages:
        if isinstance(msg, ToolChatMessage) and msg.tool_call_id.startswith("todo-nag-"):
            assert "was dirty" not in msg.content
