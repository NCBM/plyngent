from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from msgspec import UNSET

from plyngent.agent.budget import (
    compact_messages_for_request,
    estimate_messages_tokens,
    measure_messages_tokens,
    reasoning_trim_policy,
)
from plyngent.agent.events import UsageEvent
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
        max_tokens=max(1, estimate_messages_tokens(messages) - 1),
        old_tool_result_chars=40,
        keep_recent_tool_results=1,
    )
    assert isinstance(compacted[2], ToolChatMessage)
    assert len(compacted[2].content) < original_len
    assert "Truncated" in compacted[2].content
    # A shrunk result names its document (read_document / search_document).
    assert "document_id=1" in compacted[2].content
    # Full history unchanged
    assert isinstance(messages[2], ToolChatMessage)
    assert len(messages[2].content) == original_len
    # Recent tool kept
    assert isinstance(compacted[5], ToolChatMessage)
    assert compacted[5].content == "NEW" * 50


def test_compact_disabled_when_max_tokens_zero() -> None:
    messages: list[AnyChatMessage] = [
        ToolChatMessage(content="x" * 500, tool_call_id="1"),
    ]
    out = compact_messages_for_request(messages, max_tokens=0)
    assert out[0] is messages[0] or (isinstance(out[0], ToolChatMessage) and out[0].content == "x" * 500)


def test_measure_messages_tokens_calibrates_to_api_hint() -> None:
    messages: list[AnyChatMessage] = [UserChatMessage(content="a" * 40)]
    raw = estimate_messages_tokens(messages)
    # If char-est was 10 and API said 100, scale 2x content → ~200
    calibrated = measure_messages_tokens(
        messages,
        prompt_tokens_hint=raw * 10,
        sent_estimate_tokens=raw,
    )
    assert calibrated == raw * 10


def test_compact_uses_api_calibration() -> None:
    """With a high API hint scale, compact triggers earlier than raw char-est."""
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="start"),
        ToolChatMessage(content="OLD" * 400, tool_call_id="1"),
        ToolChatMessage(content="NEW" * 20, tool_call_id="2"),
    ]
    est = estimate_messages_tokens(messages)
    # Raw est under budget → no compact
    no_api = compact_messages_for_request(messages, max_tokens=est + 100)
    assert isinstance(no_api[1], ToolChatMessage)
    assert "Truncated" not in no_api[1].content
    # Calibrate so measured size is 5x → over a mid budget → shrink old tool
    compacted = compact_messages_for_request(
        messages,
        max_tokens=max(1, est * 2),
        prompt_tokens_hint=est * 5,
        sent_estimate_tokens=est,
        old_tool_result_chars=40,
        keep_recent_tool_results=1,
    )
    assert isinstance(compacted[1], ToolChatMessage)
    old = messages[1]
    assert isinstance(old, ToolChatMessage)
    assert "Truncated" in compacted[1].content or len(compacted[1].content) < len(old.content)


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
    events = [
        e
        async for e in run_chat_loop(
            client,
            history,
            model="m",
            stream=False,
            max_context_tokens=50,
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
    # The usage sample carries the estimate of the payload it describes, so a
    # calibration can divide the two instead of pairing them with the history.
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert usage_events
    assert usage_events[0].request_estimate_tokens == estimate_messages_tokens(sent)


def test_estimation_counts_chain_of_thought() -> None:
    plain = [AssistantChatMessage(content="answer")]
    reasoning = [AssistantChatMessage(content="answer", reasoning_content="R" * 400)]
    blocks = [
        AssistantChatMessage(
            content="answer",
            provider_blocks=[{"type": "thinking", "thinking": "T" * 400, "signature": "sig"}],
        )
    ]
    assert estimate_messages_tokens(reasoning) > estimate_messages_tokens(plain)
    assert estimate_messages_tokens(blocks) > estimate_messages_tokens(plain)


def test_compact_rewrites_older_reasoning_for_chat_completions() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="start"),
        AssistantChatMessage(content="first", reasoning_content="A" * 4000),
        UserChatMessage(content="again"),
        AssistantChatMessage(content="second", reasoning_content="B" * 400),
    ]
    compacted = compact_messages_for_request(
        messages,
        max_tokens=1,
        reasoning_policy=reasoning_trim_policy("chat_completions", keep_recent=1, max_chars=50),
    )
    old = compacted[1]
    assert isinstance(old, AssistantChatMessage)
    assert isinstance(old.reasoning_content, str)
    assert old.reasoning_content.startswith("A" * 50)
    assert "reasoning trimmed: 3950 chars omitted" in old.reasoning_content
    # The most recent round is protected, and the history is untouched.
    recent = compacted[3]
    assert isinstance(recent, AssistantChatMessage)
    assert recent.reasoning_content == "B" * 400
    stored = messages[1]
    assert isinstance(stored, AssistantChatMessage)
    assert stored.reasoning_content == "A" * 4000


def test_compact_blanks_older_reasoning_for_responses() -> None:
    """A Responses reasoning item needs non-empty text, so the text is dropped, not rewritten."""
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="start"),
        AssistantChatMessage(content="first", reasoning_content="A" * 4000),
        UserChatMessage(content="again"),
        AssistantChatMessage(content="second", reasoning_content="B" * 400),
    ]
    compacted = compact_messages_for_request(
        messages,
        max_tokens=1,
        reasoning_policy=reasoning_trim_policy("responses", keep_recent=1),
    )
    old = compacted[1]
    assert isinstance(old, AssistantChatMessage)
    assert old.reasoning_content is UNSET
    recent = compacted[3]
    assert isinstance(recent, AssistantChatMessage)
    assert recent.reasoning_content == "B" * 400


def test_compact_drops_older_thinking_blocks_for_messages() -> None:
    """A thinking block's signature cannot be edited, so the block goes and the rest stays."""
    search_block = {
        "type": "web_search_tool_result",
        "content": [{"type": "web_search_result", "encrypted_content": "opaque"}],
    }
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="start"),
        AssistantChatMessage(
            content="",
            provider_blocks=[{"type": "thinking", "thinking": "T" * 4000, "signature": "sig"}, search_block],
        ),
        ToolChatMessage(content="result", tool_call_id="1"),
        UserChatMessage(content="again"),
        AssistantChatMessage(
            content="second",
            provider_blocks=[{"type": "thinking", "thinking": "U" * 400, "signature": "sig2"}],
        ),
    ]
    compacted = compact_messages_for_request(
        messages,
        max_tokens=1,
        reasoning_policy=reasoning_trim_policy("messages", keep_recent=1),
    )
    old = compacted[1]
    assert isinstance(old, AssistantChatMessage)
    assert old.provider_blocks == [search_block]
    # The round keeps its text and calls; only the verified block went.
    assert old.content == ""
    recent = compacted[4]
    assert isinstance(recent, AssistantChatMessage)
    assert recent.provider_blocks is not UNSET
    assert recent.provider_blocks == [{"type": "thinking", "thinking": "U" * 400, "signature": "sig2"}]


def test_compact_leaves_reasoning_alone_without_a_policy() -> None:
    """Without a surface policy (the default) chain-of-thought is never rewritten."""
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="start"),
        AssistantChatMessage(content="first", reasoning_content="A" * 4000),
    ]
    compacted = compact_messages_for_request(messages, max_tokens=1)
    kept = compacted[1]
    assert isinstance(kept, AssistantChatMessage)
    assert kept.reasoning_content == "A" * 4000


def test_reasoning_policy_follows_the_client_kind() -> None:
    assert reasoning_trim_policy("chat_completions").mode == "trim_text"
    assert reasoning_trim_policy("responses").mode == "blank"
    assert reasoning_trim_policy("messages").mode == "drop_blocks"
    # An unknown surface would be a guess about what the provider accepts back.
    assert reasoning_trim_policy("something-else").mode == "none"


async def test_loop_trims_old_reasoning_in_request_only() -> None:
    history: list[AnyChatMessage] = [UserChatMessage(content="start")]
    for i in range(6):
        history.append(AssistantChatMessage(content=f"a{i}", reasoning_content="R" * 4000))
        history.append(UserChatMessage(content=f"u{i}"))
    client = CaptureClient([_response(AssistantChatMessage(content="ok"))])
    rounds = [m for m in history if isinstance(m, AssistantChatMessage)]
    _ = [e async for e in run_chat_loop(client, history, model="m", stream=False, max_context_tokens=1000)]
    sent = client.calls[0].messages
    trimmed = [
        m
        for m in sent
        if isinstance(m, AssistantChatMessage)
        and isinstance(m.reasoning_content, str)
        and "reasoning trimmed" in m.reasoning_content
    ]
    # Six rounds, the four most recent protected.
    assert len(trimmed) == 2
    assert isinstance(trimmed[0].reasoning_content, str)
    assert len(trimmed[0].reasoning_content) < 4000
    # In-memory history keeps every round's chain-of-thought.
    assert len(rounds) == 6
    assert all(message.reasoning_content == "R" * 4000 for message in rounds)
