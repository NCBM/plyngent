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


def test_lifo_not_queue() -> None:
    """Stack is LIFO: push A then B → top is B; pop is B then A — never FIFO."""
    stack = TodoStack()
    _ = stack.push("A")
    _ = stack.push("B")
    assert stack.top is not None
    assert stack.top.title == "B"
    first = stack.pop()
    second = stack.pop()
    assert first is not None and first.title == "B"
    assert second is not None and second.title == "A"
    assert stack.pop() is None


def test_dfs_breakdown_pattern() -> None:
    """push [T1,T2] → top T1; push [T1.1,T1.2] → top T1.1; pop children; then T2.1."""
    stack = TodoStack()
    root = stack.push_titles(["T1", "T2"])
    assert [i.title for i in root] == ["T1", "T2"]
    assert stack.top is not None
    assert stack.top.title == "T1"
    # bottom→top: T2, T1
    assert [i.title for i in stack.items] == ["T2", "T1"]

    children = stack.push_titles(["T1.1", "T1.2"])
    assert [i.title for i in children] == ["T1.1", "T1.2"]
    assert stack.top is not None and stack.top.title == "T1.1"
    # bottom→top: T2, T1, T1.2, T1.1
    assert [i.title for i in stack.items] == ["T2", "T1", "T1.2", "T1.1"]

    p1 = stack.pop()
    p2 = stack.pop()
    assert p1 is not None and p1.title == "T1.1"
    assert p2 is not None and p2.title == "T1.2"
    p3 = stack.pop()
    assert p3 is not None and p3.title == "T1"
    assert stack.top is not None and stack.top.title == "T2"
    _ = stack.push_titles(["T2.1"])
    assert stack.top is not None and stack.top.title == "T2.1"
    assert [i.title for i in stack.items] == ["T2", "T2.1"]


def test_todo_stack_needs_review() -> None:
    stack = TodoStack()
    assert not stack.needs_review()
    _ = stack.push("work")
    stack.begin_turn()
    assert stack.needs_review()
    stack.mark_touched()
    assert not stack.needs_review()


def test_legacy_frames_flatten() -> None:
    stack = TodoStack.from_raw(
        {
            "frames": [
                {"items": [{"id": "t1", "title": "T1", "status": "pending", "notes": ""}]},
                {"items": [{"id": "t2", "title": "T1.1", "status": "pending", "notes": ""}]},
            ],
            "next_id": 3,
        }
    )
    assert [i.title for i in stack.items] == ["T1", "T1.1"]
    assert stack.top is not None and stack.top.title == "T1.1"


def test_todo_stack_roundtrip_raw() -> None:
    stack = TodoStack()
    _ = stack.push_titles(["x", "y"], notes="n")
    raw = stack.to_raw()
    restored = TodoStack.from_raw(raw)
    assert [i.title for i in restored.items] == ["y", "x"]  # bottom y, top x
    assert restored.top is not None and restored.top.title == "x"


async def test_todo_tools_and_persist(tmp_path: object) -> None:
    del tmp_path
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="t")
        stack = TodoStack()
        set_todo_stack(stack, on_change=None)
        registry = ToolRegistry(list(TODO_TOOLS))
        out = await registry.execute("todo_push", '{"titles": "T1\\nT2"}')
        assert "top" in out.lower() or "pushed" in out
        assert stack.top is not None and stack.top.title == "T1"
        _ = await registry.execute("todo_push", '{"titles": "T1.1; T1.2"}')
        assert stack.top is not None and stack.top.title == "T1.1"
        out3 = await registry.execute("todo_pop", "{}")
        assert "popped" in out3
        assert stack.top is not None and stack.top.title == "T1.2"
        _ = await memory.update_session_todo_stack(session.sid, stack.to_raw())
        loaded = await memory.get_session_todo_stack(session.sid)
        assert loaded is not None
        again = TodoStack.from_raw(loaded)
        assert again.top is not None and again.top.title == "T1.2"
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
            if isinstance(msg, DeveloperChatMessage) and "Todo stack review" in msg.content:
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
        assert any(
            isinstance(m, DeveloperChatMessage) and "Todo stack review" in m.content
            for m in agent.messages
        )
        assert not any(
            isinstance(m, UserChatMessage) and "Todo stack review" in m.content for m in agent.messages
        )
    finally:
        set_todo_stack(None)
