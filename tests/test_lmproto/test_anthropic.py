from __future__ import annotations

import msgspec
import pytest

from plyngent.lmproto.anthropic import AnthropicClient
from plyngent.lmproto.anthropic.config import AnthropicConfig
from plyngent.lmproto.anthropic.model import (
    AnthropicContentBlockDelta,
    AnthropicMessageResponse,
    AnthropicMessagesParam,
    AnthropicMessageStop,
    AnthropicResponseText,
    AnthropicUsage,
    AnthropicUserMessage,
)


def _sample_message_body() -> bytes:
    return msgspec.json.encode(
        AnthropicMessageResponse(
            id="msg_1",
            model="claude-test",
            content=[AnthropicResponseText(text="hello world")],
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=9, output_tokens=3),
        )
    )


def test_messages_param_encode_omits_defaults() -> None:
    param = AnthropicMessagesParam(
        model="claude-test",
        max_tokens=2048,
        messages=[AnthropicUserMessage(content="hi")],
    )
    raw = msgspec.json.encode(param)
    data = msgspec.json.decode(raw)
    assert data["model"] == "claude-test"
    # omit_defaults: default stream=False is not encoded until client sets stream
    assert "stream" not in data
    # Non-default fields ARE encoded
    assert data["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_client_messages_create(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(AnthropicConfig(api_key="sk-test", base_url="https://example/v1"))

    class _Resp:
        status_code = 200

        @property
        def content(self) -> bytes:
            return _sample_message_body()

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        assert path == "/messages"
        assert kwargs.get("stream") is False
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    result = await client.messages(AnthropicMessagesParam(model="claude-test", messages=[]))
    assert isinstance(result, AnthropicMessageResponse)
    assert result.content[0].text == "hello world"
    assert result.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_client_messages_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(AnthropicConfig(api_key="sk-test", base_url="https://example/v1"))

    class _Resp:
        status_code = 200

        async def iter_lines(self):
            yield b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}'
            yield b'data: {"type":"message_stop"}'

        def close(self) -> None:
            return None

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        assert path == "/messages"
        assert kwargs.get("stream") is True
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    stream = await client.messages(AnthropicMessagesParam(model="claude-test", messages=[]), stream=True)
    events = [event async for event in stream]
    assert len(events) == 2
    assert isinstance(events[0], AnthropicContentBlockDelta)
    assert events[0].delta.text == "hel"
    assert isinstance(events[1], AnthropicMessageStop)
