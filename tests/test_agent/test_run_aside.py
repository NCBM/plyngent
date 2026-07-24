"""ChatAgent.run_aside: side turns leave main transcript/memory alone."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pytest

from plyngent.agent import ChatAgent, ToolRegistry, ToolTag, tool
from plyngent.agent.todo_stack import TodoStack
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    UserChatMessage,
)
from plyngent.memory import MemoryStore
from plyngent.tools.context import InstanceState, SessionState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ScriptedClient:
    """Returns a fixed assistant string each call."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or ["aside-answer"])
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
        text = self.replies[(self.calls - 1) % len(self.replies)]
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
async def test_run_aside_does_not_mutate_main_or_memory() -> None:
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="main")
        client = ScriptedClient(["main-reply", "aside-reply", "main-again"])
        agent = ChatAgent(
            cast("Any", client),
            model="m",
            memory=memory,
            session_id=session.sid,
            stream=False,
            todo_stack=TodoStack(),
        )
        async for _ in agent.run("hello main"):
            pass
        main_len = len(agent.messages)
        main_snapshot = list(agent.messages)
        db_before = await memory.list_messages(session.sid)

        texts: list[str] = []
        async for event in agent.run_aside("side question?", include_history=True, tools=False):
            from plyngent.agent.events import AssistantMessageEvent

            if isinstance(event, AssistantMessageEvent) and event.message.content:
                texts.append(str(event.message.content))

        assert any("aside-reply" in t for t in texts)
        assert len(agent.messages) == main_len
        assert agent.messages == main_snapshot
        db_after = await memory.list_messages(session.sid)
        assert len(db_after) == len(db_before)
        # Aside request saw main history + side user.
        aside_payload = client.payloads[1]
        assert any(isinstance(m, UserChatMessage) and "side question" in m.content for m in aside_payload)
        assert any(isinstance(m, UserChatMessage) and "hello main" in m.content for m in aside_payload)
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_run_aside_fresh_skips_history() -> None:
    client = ScriptedClient(["only"])
    agent = ChatAgent(cast("Any", client), model="m", stream=False)
    agent.messages.append(UserChatMessage(content="prior"))
    async for _ in agent.run_aside("q", include_history=False, tools=False):
        pass
    payload = client.payloads[0]
    users = [m.content for m in payload if isinstance(m, UserChatMessage)]
    assert users == ["q"]


@pytest.mark.asyncio
async def test_run_aside_tools_clones_registry_with_fresh_session() -> None:
    hits: list[str] = []

    @tool(tags=ToolTag.LOCAL | ToolTag.SESSION_STATE, register=False)
    async def note_session() -> str:
        from plyngent.tools.context import get_session

        session = get_session()
        hits.append("ok" if session is not None else "none")
        if session is not None:
            session.extras["aside"] = True
        return "noted"

    main_session = SessionState(session_id="main")
    instance = InstanceState()
    registry = ToolRegistry(
        [note_session],
        auto_bind_state=True,
        instance_state=instance,
        session_state=main_session,
    )

    class ToolThenStop:
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
            from plyngent.lmproto.openai_compatible.model import (
                AssistantFunctionTool,
                AssistantFunctionToolCall,
            )

            if self.calls == 1:
                message = AssistantChatMessage(
                    content="",
                    tool_calls=[
                        AssistantFunctionToolCall(
                            id="c1",
                            function=AssistantFunctionTool(name="note_session", arguments="{}"),
                        )
                    ],
                )
                finish = "tool_calls"
            else:
                message = AssistantChatMessage(content="done")
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

    client = ToolThenStop()
    agent = ChatAgent(cast("Any", client), model="m", tools=registry, stream=False)
    async for _ in agent.run_aside(
        "use tool",
        tools=True,
        instance_state=instance,
        session_state=SessionState(session_id="aside"),
    ):
        pass
    assert hits == ["ok"]
    assert "aside" not in main_session.extras
