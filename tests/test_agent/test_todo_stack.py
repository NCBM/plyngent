from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest

from plyngent.agent import ChatAgent, ToolRegistry
from plyngent.agent.todo_stack import TodoStack, parse_push_titles
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    DeveloperChatMessage,
    UserChatMessage,
)
from plyngent.memory import MemoryStore
from plyngent.tools.todo import TODO_TOOLS, set_todo_stack

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
    # Open items always need review, even if todo_* was used this turn.
    assert stack.needs_review()
    stack.mark_touched()
    assert stack.needs_review()
    stack.update(item.id, status="done")
    stack.begin_turn()
    # Terminal-only stack: review only when untouched this turn.
    assert stack.needs_review()
    stack.mark_touched()
    assert not stack.needs_review()


def test_todo_prompts_signal_undone_work() -> None:
    stack = TodoStack()
    item = stack.push("open work")
    reminder = stack.turn_reminder_prompt()
    assert "[TODO REMINDER]" in reminder
    assert "Stack not empty" in reminder
    assert "unfinished" in reminder.lower()
    assert "open work" in reminder

    review = stack.review_prompt()
    assert "[TODO OPEN]" in review
    assert "Stack not empty" in review
    assert "undone" in review.lower() or "incomplete" in review.lower()
    assert "open work" in review

    stack.update(item.id, status="done")
    terminal_review = stack.review_prompt()
    assert "done/cancelled" in terminal_review or "Bookkeeping" in terminal_review


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
        set_todo_stack(stack, on_change=None)
        registry = ToolRegistry(list(TODO_TOOLS))
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
        set_todo_stack(None)
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
        tools=ToolRegistry(list(TODO_TOOLS)),
        stream=False,
        todo_stack=stack,
    )
    set_todo_stack(stack)
    try:
        async for _event in agent.run("do stuff"):
            pass
        assert client.calls >= 2
        assert any(isinstance(m, DeveloperChatMessage) and "[TODO REMINDER]" in m.content for m in agent.messages)
        assert any(isinstance(m, DeveloperChatMessage) and "[TODO OPEN]" in m.content for m in agent.messages)
        assert not any(isinstance(m, UserChatMessage) and "[TODO OPEN]" in m.content for m in agent.messages)
    finally:
        set_todo_stack(None)


@pytest.mark.asyncio
async def test_loop_injects_todo_review_when_open_after_touch() -> None:
    """Open items still trigger end-of-turn review even if todo_* ran this turn."""
    stack = TodoStack()
    _ = stack.push("still open")
    stack.begin_turn()
    stack.mark_touched()  # simulates todo_list earlier in the turn
    assert stack.needs_review()

    client = ScriptedClient()
    agent = ChatAgent(
        client,  # type: ignore[arg-type]
        model="m",
        tools=ToolRegistry(list(TODO_TOOLS)),
        stream=False,
        todo_stack=stack,
    )
    set_todo_stack(stack)
    try:
        async for _event in agent.run("do stuff"):
            pass
        assert client.calls >= 2
        assert any(isinstance(m, DeveloperChatMessage) and "[TODO OPEN]" in m.content for m in agent.messages)
    finally:
        set_todo_stack(None)
