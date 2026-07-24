from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, overload

from msgspec import UNSET

from plyngent.agent import ChatAgent, ToolRegistry, tool
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ChunkChoice,
    DeltaMessage,
    StreamFunctionDelta,
    StreamToolCallDelta,
    ToolChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _response(message: AssistantChatMessage) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="1",
        object="chat.completion",
        created=0,
        model="t",
        choices=[ChatCompletionChoice(index=0, message=message, logprobs={}, finish_reason="stop")],
        system_fingerprint="",
        usage={},
    )


class ParallelScriptedClient:
    step: int

    def __init__(self) -> None:
        self.step = 0

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
        del param
        self.step += 1
        if self.step == 1:
            msg = AssistantChatMessage(
                content="",
                tool_calls=[
                    AssistantFunctionToolCall(
                        id="1",
                        function=AssistantFunctionTool(name="slow_a", arguments="{}"),
                    ),
                    AssistantFunctionToolCall(
                        id="2",
                        function=AssistantFunctionTool(name="slow_b", arguments="{}"),
                    ),
                ],
            )
        else:
            msg = AssistantChatMessage(content="done")
        response = _response(msg)
        if stream:
            return self._stream(response)
        return response

    async def _stream(self, response: ChatCompletionResponse) -> AsyncIterator[ChatCompletionChunk]:
        message = response.choices[0].message
        if isinstance(message.content, str) and message.content:
            yield ChatCompletionChunk(
                id="1",
                object="chat.completion.chunk",
                created=0,
                model="t",
                choices=[ChunkChoice(index=0, delta=DeltaMessage(content=message.content), finish_reason=None)],
            )
        tool_calls = message.tool_calls
        if tool_calls is not UNSET and tool_calls:
            deltas: list[StreamToolCallDelta] = []
            for i, call in enumerate(tool_calls):
                if isinstance(call, AssistantFunctionToolCall):
                    deltas.append(
                        StreamToolCallDelta(
                            index=i,
                            id=call.id,
                            type="function",
                            function=StreamFunctionDelta(
                                name=call.function.name,
                                arguments=call.function.arguments,
                            ),
                        )
                    )
            yield ChatCompletionChunk(
                id="1",
                object="chat.completion.chunk",
                created=0,
                model="t",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(tool_calls=deltas),
                        finish_reason="tool_calls",
                    )
                ],
            )


async def test_parallel_tool_calls_run_concurrently() -> None:
    started = 0
    expected_parallel = 2
    gate = asyncio.Event()
    wait_timeout = 2.0

    @tool(register=False)
    async def slow_a() -> str:
        nonlocal started
        started += 1
        if started >= expected_parallel:
            _ = gate.set()
        _ = await asyncio.wait_for(gate.wait(), timeout=wait_timeout)
        return "a"

    @tool(register=False)
    async def slow_b() -> str:
        nonlocal started
        started += 1
        if started >= expected_parallel:
            _ = gate.set()
        _ = await asyncio.wait_for(gate.wait(), timeout=wait_timeout)
        return "b"

    registry = ToolRegistry([slow_a, slow_b])
    agent = ChatAgent(
        ParallelScriptedClient(),
        model="m",
        tools=registry,
        stream=True,
        parallel_tools=True,
    )
    events = [e async for e in agent.run("go")]
    tool_msgs = [e for e in agent.messages if isinstance(e, ToolChatMessage)]
    assert {m.content for m in tool_msgs} == {"a", "b"}
    assert any(getattr(e, "content", None) == "done" for e in events)
