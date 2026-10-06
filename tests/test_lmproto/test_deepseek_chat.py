"""Tests for the DeepSeek chat-completions client's ``thinking`` flag."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from msgspec import UNSET

from plyngent.lmproto.deepseek import DeepseekOpenAIClient
from plyngent.lmproto.openai_compatible.config import OpenAIConfig
from plyngent.lmproto.openai_compatible.model import ChatCompletionsParam, UserChatMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.lmproto.openai_compatible.model import ReasoningEffort
    from plyngent.typedef import Unset


async def _posted_body(
    monkeypatch: pytest.MonkeyPatch,
    effort: ReasoningEffort | Unset = UNSET,
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
        messages=[UserChatMessage(content="hi")],
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
