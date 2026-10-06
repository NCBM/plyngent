"""Tests for the DeepSeek chat-completions client's wire quirks.

``thinking`` (an explicit flag the API wants) and the ``reasoning_content``
field a tool-call message must carry back in thinking mode.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import msgspec
import pytest
from msgspec import UNSET

from plyngent.lmproto.deepseek import DeepseekOpenAIClient
from plyngent.lmproto.openai_compatible.config import OpenAIConfig
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, ReasoningEffort
    from plyngent.typedef import Unset


async def _posted_body(
    monkeypatch: pytest.MonkeyPatch,
    effort: ReasoningEffort | Unset = UNSET,
    *,
    messages: list[AnyChatMessage] | None = None,
) -> dict[str, Any]:
    """Run one streamed request and return the JSON body the client posted."""
    client = DeepseekOpenAIClient(OpenAIConfig(access_key_or_token="sk-test", base_url="https://api.deepseek.com"))
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200

        async def iter_lines(self) -> AsyncIterator[bytes]:
            yield b"data: [DONE]"

        def close(self) -> None:
            return None

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        assert path == "/chat/completions"
        captured["data"] = kwargs.get("data")
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    param = ChatCompletionsParam(
        model="deepseek-v4-pro",
        messages=messages if messages is not None else [UserChatMessage(content="hi")],
        reasoning_effort=effort,
    )
    # Production dispatches the base param through the OpenAI-compatible surface
    # (not the DeepSeek-specific annotation), so mirror that here.
    cc: Any = client
    stream = cast("AsyncIterator[Any]", await cc.chat_completions(param, stream=True))
    async for _chunk in stream:
        pass
    data = captured["data"]
    assert isinstance(data, bytes)
    return json.loads(data)


@pytest.mark.asyncio
async def test_thinking_enabled_without_configured_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    body = await _posted_body(monkeypatch)
    assert body["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in body


@pytest.mark.asyncio
async def test_thinking_enabled_with_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    body = await _posted_body(monkeypatch, "high")
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_thinking_disabled_for_none_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    body = await _posted_body(monkeypatch, "none")
    assert body["thinking"] == {"type": "disabled"}
    assert body["reasoning_effort"] == "none"


def _kept_call() -> AssistantChatMessage:
    """The pairing carrier ``rollback_tail`` keeps for a call that ran."""
    return AssistantChatMessage(
        content=None,
        tool_calls=[
            AssistantFunctionToolCall(
                id="call_1",
                function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
            )
        ],
    )


@pytest.mark.asyncio
async def test_tool_call_without_recorded_reasoning_gets_an_empty_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API refuses a call whose ``reasoning_content`` never came back.

    A rolled-back turn keeps its executed calls without one, and a synthetic
    ``todo_list`` nag is forged the same way; the field only has to be present,
    so an empty string satisfies the check.
    """
    body = await _posted_body(
        monkeypatch,
        messages=[
            UserChatMessage(content="read it"),
            _kept_call(),
            ToolChatMessage(content="file body", tool_call_id="call_1"),
        ],
    )
    assistant = body["messages"][1]
    assert assistant["reasoning_content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call_1"


@pytest.mark.asyncio
async def test_recorded_reasoning_content_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    call = _kept_call()
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="read it"),
        msgspec.structs.replace(call, reasoning_content="Open the file first."),
        ToolChatMessage(content="file body", tool_call_id="call_1"),
    ]
    body = await _posted_body(monkeypatch, messages=messages)
    assert body["messages"][1]["reasoning_content"] == "Open the file first."


@pytest.mark.asyncio
async def test_answer_without_recorded_reasoning_gets_an_empty_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round that called nothing needs the field too.

    The rule is per round, not per call: the API answers "the
    ``reasoning_content`` … must be passed back" for an answer round that never
    reasoned as well, and an empty string satisfies it.
    """
    body = await _posted_body(
        monkeypatch,
        messages=[UserChatMessage(content="hi"), AssistantChatMessage(content="all done")],
    )
    assert body["messages"][1]["reasoning_content"] == ""
    assert body["messages"][1]["content"] == "all done"
