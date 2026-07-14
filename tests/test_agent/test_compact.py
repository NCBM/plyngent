from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from msgspec import UNSET

from plyngent.agent.budget import (
    compact_messages_for_request,
    estimate_messages_chars,
    truncate_tool_result,
)
from plyngent.agent.loop import run_chat_loop
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
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage


def test_truncate_tool_result_short() -> None:
    assert truncate_tool_result("hello", 100) == "hello"


def test_truncate_tool_result_long() -> None:
    text = "a" * 50
    out = truncate_tool_result(text, 20)
    assert out.startswith("a" * 20)
    assert "truncated" in out
    assert "30" in out


def test_compact_shrinks_old_tool_results() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="start"),
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="1",
                    function=AssistantFunctionTool(name="t", arguments="{}"),
                )
            ],
        ),
        ToolChatMessage(content="OLD" * 200, tool_call_id="1"),
        UserChatMessage(content="again"),
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="2",
                    function=AssistantFunctionTool(name="t", arguments="{}"),
                )
            ],
        ),
        ToolChatMessage(content="NEW" * 50, tool_call_id="2"),
    ]
    original_old = messages[2]
    assert isinstance(original_old, ToolChatMessage)
    original_len = len(original_old.content)

    compacted = compact_messages_for_request(
        messages,
        max_chars=estimate_messages_chars(messages) - 1,
        old_tool_result_chars=40,
        keep_recent_tool_results=1,
    )
    assert isinstance(compacted[2], ToolChatMessage)
    assert len(compacted[2].content) < original_len
    assert "truncated" in compacted[2].content
    # Full history unchanged
    assert isinstance(messages[2], ToolChatMessage)
    assert len(messages[2].content) == original_len
    # Recent tool kept
    assert isinstance(compacted[5], ToolChatMessage)
    assert compacted[5].content == "NEW" * 50


def test_compact_disabled_when_max_chars_zero() -> None:
    messages: list[AnyChatMessage] = [
        ToolChatMessage(content="x" * 500, tool_call_id="1"),
    ]
    out = compact_messages_for_request(messages, max_chars=0)
    assert out[0] is messages[0] or (
        isinstance(out[0], ToolChatMessage) and out[0].content == "x" * 500
    )


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


class CaptureClient:
    _responses: list[ChatCompletionResponse]
    calls: list[ChatCompletionsParam]

    def __init__(self, responses: list[ChatCompletionResponse]) -> None:
        self._responses = list(responses)
        self.calls = []

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
        self.calls.append(param)
        response = self._responses.pop(0)
        if stream:

            async def as_stream() -> AsyncIterator[ChatCompletionChunk]:
                message = response.choices[0].message
                if isinstance(message.content, str) and message.content:
                    yield ChatCompletionChunk(
                        id="1",
                        object="chat.completion.chunk",
                        created=0,
                        model="t",
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=DeltaMessage(content=message.content),
                                finish_reason=None,
                            )
                        ],
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

            return as_stream()
        return response


async def test_loop_sends_compacted_request_not_history() -> None:
    big = "Z" * 500
    history: list[AnyChatMessage] = [
        UserChatMessage(content="u"),
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="1",
                    function=AssistantFunctionTool(name="t", arguments="{}"),
                )
            ],
        ),
        ToolChatMessage(content=big, tool_call_id="1"),
        UserChatMessage(content="next"),
    ]
    client = CaptureClient([_response(AssistantChatMessage(content="ok"))])
    _ = [
        e
        async for e in run_chat_loop(
            client,
            history,
            model="m",
            stream=False,
            max_context_chars=200,
            max_tool_result_chars=50,
        )
    ]
    assert client.calls
    sent = client.calls[0].messages
    tool_sent = next(m for m in sent if isinstance(m, ToolChatMessage))
    assert len(tool_sent.content) < len(big)
    # In-memory history still full for the old tool result
    assert isinstance(history[2], ToolChatMessage)
    assert history[2].content == big
