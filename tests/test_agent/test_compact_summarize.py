from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from plyngent.agent.compact import (
    DEFAULT_TRANSCRIPT_ROW_CHARS,
    build_compacted_seed_messages,
    format_transcript,
    soft_compact_transcript,
    summarize_messages,
)
from plyngent.lmproto.openai_compatible.model import (
    IMAGE_MARKER,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ChatImagePart,
    ChatImageUrl,
    ChatTextPart,
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


def test_transcript_clips_a_long_row() -> None:
    """One row only has to say what happened: a long answer collapses to its head."""
    messages = [
        UserChatMessage(content="brief"),
        AssistantChatMessage(content="A" * 20_000),
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="1",
                    function=AssistantFunctionTool(name="write_file", arguments="B" * 20_000),
                )
            ],
        ),
    ]
    rows = format_transcript(messages, row_chars=100).splitlines()
    assert "[user] brief" in rows
    assert all(len(row) < 200 for row in rows)
    assert "chars)" in rows[1]
    assert "chars)" in rows[2]


def test_transcript_drops_the_oldest_rows_to_fit() -> None:
    """A long session is bounded as a whole, keeping the task statement and the tail."""
    messages = [UserChatMessage(content="task statement")]
    messages.extend(AssistantChatMessage(content=f"row-{i}-" + "x" * 500) for i in range(100))
    messages.append(UserChatMessage(content="latest question"))
    out = soft_compact_transcript(messages, max_tokens=300, row_chars=600)
    assert len(out) <= 300 * 4
    assert out.startswith("[… ")
    assert "earlier rows dropped" in out
    assert "task statement" in out
    assert "latest question" in out
    assert "row-0-" not in out


def test_transcript_renders_a_user_part_array_as_text() -> None:
    """A part array becomes its prose: its repr would carry a base64 data URL."""
    messages = [
        UserChatMessage(
            content=[
                ChatTextPart(text="look at this"),
                ChatImagePart(image_url=ChatImageUrl(url="data:image/png;base64," + "Q" * 20_000)),
            ]
        )
    ]
    out = format_transcript(messages)
    assert "look at this" in out
    assert IMAGE_MARKER in out
    assert "base64" not in out
    assert "ChatImagePart" not in out


async def test_summarize_messages_bounds_the_request() -> None:
    """The summarization request itself never exceeds the configured budget."""
    client = SummaryClient()
    messages = [UserChatMessage(content="task statement")]
    messages.extend(UserChatMessage(content=f"paste {i}: " + "P" * 5000) for i in range(100))
    _ = await summarize_messages(client, messages, model="m", max_context_tokens=1_000)
    assert client.last is not None
    user_msg = client.last.messages[1]
    assert isinstance(user_msg, UserChatMessage)
    content = user_msg.content
    assert isinstance(content, str)
    assert len(content) <= 1_000 * 4
    assert "task statement" in content
    assert "100 earlier rows dropped" in content


def test_transcript_row_cap_is_a_default_not_a_floor() -> None:
    assert DEFAULT_TRANSCRIPT_ROW_CHARS > 0


def test_build_compacted_seed_messages() -> None:
    seed = build_compacted_seed_messages("summary text", system_prompt="sys", source_session_id=3)
    assert isinstance(seed[0], SystemChatMessage)
    assert seed[0].content == "sys"
    assert isinstance(seed[1], AssistantChatMessage)
    content = seed[1].content
    assert isinstance(content, str)
    assert "summary text" in content
    assert "session 3" in content


def test_build_compacted_seed_custom_template_and_braces_in_summary() -> None:
    seed = build_compacted_seed_messages(
        "use {braces} carefully",
        source_session_id=1,
        seed_text="From {src}:\n{summary}",
    )
    assert isinstance(seed[0], AssistantChatMessage)
    content = seed[0].content
    assert isinstance(content, str)
    assert "session 1" in content
    assert "use {braces} carefully" in content


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
    user_msg = client.last.messages[1]
    assert isinstance(user_msg, UserChatMessage)
    assert "hello" in user_msg.content
    assert "world" in user_msg.content


async def test_summarize_default_prompt_says_tools_are_unavailable() -> None:
    client = SummaryClient()
    _ = await summarize_messages(client, [UserChatMessage(content="hi")], model="m")
    assert client.last is not None
    sys_msg = client.last.messages[0]
    assert isinstance(sys_msg, SystemChatMessage)
    assert "tool calls are unavailable" in sys_msg.content.lower()


async def test_summarize_custom_prompts() -> None:
    client = SummaryClient()
    _ = await summarize_messages(
        client,
        [UserChatMessage(content="hello")],
        model="m",
        system_prompt="SYS-CUSTOM",
        user_prefix="PREFIX:\n{transcript}\nEND",
    )
    assert client.last is not None
    sys_msg = client.last.messages[0]
    user_msg = client.last.messages[1]
    assert isinstance(sys_msg, SystemChatMessage)
    assert sys_msg.content == "SYS-CUSTOM"
    assert isinstance(user_msg, UserChatMessage)
    assert user_msg.content.startswith("PREFIX:")
    assert "hello" in user_msg.content
    assert user_msg.content.endswith("END")
