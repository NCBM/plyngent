from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from plyngent.agent.compact import (
    build_compacted_seed_messages,
    format_transcript,
    soft_compact_transcript,
    summarize_messages,
)
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def test_format_transcript() -> None:
    text = format_transcript(
        [
            UserChatMessage(content="hi"),
            AssistantChatMessage(content="yo"),
            ToolChatMessage(content="out", tool_call_id="1"),
        ]
    )
    assert "[user] hi" in text
    assert "[assistant] yo" in text
    assert "[tool 1] out" in text


def test_soft_compact_transcript_shrinks_tools() -> None:
    big = "Z" * 2000
    messages = [
        UserChatMessage(content="u"),
        ToolChatMessage(content=big, tool_call_id="1"),
        ToolChatMessage(content="recent", tool_call_id="2"),
    ]
    out = soft_compact_transcript(messages, max_tokens=100)
    assert "truncated" in out or len(out) < len(big) + 50
    assert "recent" in out


def test_build_compacted_seed_messages() -> None:
    seed = build_compacted_seed_messages("summary text", system_prompt="sys", source_session_id=3)
    assert isinstance(seed[0], SystemChatMessage)
    assert seed[0].content == "sys"
    assert isinstance(seed[1], UserChatMessage)
    assert "summary text" in seed[1].content
    assert "session 3" in seed[1].content


class SummaryClient:
    last: ChatCompletionsParam | None

    def __init__(self) -> None:
        self.last = None

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
        self.last = param
        return ChatCompletionResponse(
            id="1",
            object="chat.completion",
            created=0,
            model="t",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantChatMessage(content="  done summary  "),
                    logprobs={},
                    finish_reason="stop",
                )
            ],
            system_fingerprint="",
            usage={},
        )


async def test_summarize_messages() -> None:
    client = SummaryClient()
    summary = await summarize_messages(
        client,
        [UserChatMessage(content="hello"), AssistantChatMessage(content="world")],
        model="m",
    )
    assert summary == "done summary"
    assert client.last is not None
    assert client.last.model == "m"
    from msgspec import UNSET

    assert client.last.tools is UNSET
