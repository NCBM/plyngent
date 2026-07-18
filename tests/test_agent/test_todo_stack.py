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


def test_nested_push_pop_pattern() -> None:
    """(push)[T1,T2] (push)[T1.1,T1.2] (pop) (push)[T2.1] …"""
    stack = TodoStack()
    root = stack.push_titles(["T1", "T2"])
    assert [i.title for i in root] == ["T1", "T2"]
    assert stack.depth == 1

    children = stack.push_titles(["T1.1", "T1.2"])
    assert stack.depth == 2
    assert [i.title for i in children] == ["T1.1", "T1.2"]
    stack.update(children[0].id, status="done")
    stack.update(children[1].id, status="done")

    popped = stack.pop()
    assert popped is not None
    assert [i.title for i in popped.items] == ["T1.1", "T1.2"]
    assert stack.depth == 1
    assert [i.title for i in stack.frames[0].items] == ["T1", "T2"]

    stack.update(root[0].id, status="done")
    _ = stack.push_titles(["T2.1"])
    assert stack.depth == 2
    assert stack.frames[-1].items[0].title == "T2.1"


def test_todo_stack_needs_review() -> None:
    stack = TodoStack()
    assert not stack.needs_review()
    _ = stack.push("work")
    stack.begin_turn()
    assert stack.needs_review()
    stack.mark_touched()
    assert not stack.needs_review()


def test_todo_stack_legacy_flat_raw() -> None:
    stack = TodoStack.from_raw(
        {
            "items": [{"id": "t1", "title": "old", "status": "pending", "notes": ""}],
            "next_id": 2,
        }
    )
    assert stack.depth == 1
    assert stack.all_items()[0].title == "old"
    raw = stack.to_raw()
    assert "frames" in raw


def test_todo_stack_roundtrip_raw() -> None:
    stack = TodoStack()
    _ = stack.push_titles(["x", "y"], notes="n")
    raw = stack.to_raw()
    restored = TodoStack.from_raw(raw)
    assert restored.depth == 1
    assert [i.title for i in restored.frames[0].items] == ["x", "y"]


async def test_todo_tools_and_persist(tmp_path: object) -> None:
    del tmp_path
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="t")
        stack = TodoStack()
        set_todo_stack(stack, on_change=None)
        registry = ToolRegistry(list(TODO_TOOLS))
        out = await registry.execute("todo_push", '{"titles": "T1\\nT2"}')
        assert "pushed frame" in out
        assert stack.depth == 1
        out2 = await registry.execute("todo_push", '{"titles": "T1.1; T1.2"}')
        assert stack.depth == 2
        assert "T1.1" in out2
        out3 = await registry.execute("todo_pop", "{}")
        assert "popped frame" in out3
        assert stack.depth == 1
        _ = await memory.update_session_todo_stack(session.sid, stack.to_raw())
        loaded = await memory.get_session_todo_stack(session.sid)
        assert loaded is not None
        again = TodoStack.from_raw(loaded)
        assert again.depth == 1
        assert [i.title for i in again.frames[0].items] == ["T1", "T2"]
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
