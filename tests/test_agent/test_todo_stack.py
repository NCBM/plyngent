from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest

from plyngent.agent import ChatAgent, ToolRegistry
from plyngent.agent.todo_stack import TodoStack
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


def test_todo_stack_push_pop_update() -> None:
    stack = TodoStack()
    a = stack.push("one")
    b = stack.push("two")
    assert a.id == "t1"
    assert b.id == "t2"
    assert "one" in stack.render()
    stack.update("t1", status="done")
    popped = stack.pop()
    assert popped is not None
    assert popped.id == "t2"
    assert stack.open_items() == []


def test_todo_stack_needs_review() -> None:
    stack = TodoStack()
    assert not stack.needs_review()
    stack.push("work")
    stack.begin_turn()
    assert stack.needs_review()
    stack.mark_touched()
    assert not stack.needs_review()


def test_todo_stack_roundtrip_raw() -> None:
    stack = TodoStack()
    _ = stack.push("x", notes="n")
    raw = stack.to_raw()
    restored = TodoStack.from_raw(raw)
    assert len(restored.items) == 1
    assert restored.items[0].title == "x"
    assert restored.items[0].notes == "n"


async def test_todo_tools_and_persist(tmp_path: object) -> None:
    del tmp_path
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="t")
        stack = TodoStack()
        set_todo_stack(stack, on_change=None)
        registry = ToolRegistry(list(TODO_TOOLS))
        assert "pushed t1" in await registry.execute("todo_push", '{"title": "ship it"}')
        assert "t1" in await registry.execute("todo_list", "{}")
        assert "done" in await registry.execute(
            "todo_update", '{"item_id": "t1", "status": "done"}'
        )
        _ = await memory.update_session_todo_stack(session.sid, stack.to_raw())
        loaded = await memory.get_session_todo_stack(session.sid)
        assert loaded is not None
        again = TodoStack.from_raw(loaded)
        assert again.items[0].status == "done"
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
        # First call: finish without tools; second: after review inject.
        text = "ok" if self.calls > 1 else "done without todos"
        # Detect review message in history
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
