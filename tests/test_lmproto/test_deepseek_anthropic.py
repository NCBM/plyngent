from __future__ import annotations

import msgspec
import pytest

from plyngent.lmproto.anthropic.config import AnthropicConfig
from plyngent.lmproto.anthropic.model import (
    AnthropicMessageResponse,
    AnthropicMessagesParam,
    AnthropicResponseText,
    AnthropicUsage,
)
from plyngent.lmproto.deepseek import DeepseekAnthropicClient


def _sample_message_body() -> bytes:
    return msgspec.json.encode(
        AnthropicMessageResponse(
            id="msg_1",
            model="deepseek-v4-flash",
            content=[AnthropicResponseText(text="hello world")],
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=9, output_tokens=3),
        )
    )


def test_client_kind_and_base_url() -> None:
    client = DeepseekAnthropicClient(AnthropicConfig(api_key="sk-test", base_url="https://api.deepseek.com/anthropic"))
    assert client.kind == "messages"
    assert client.session.base_url == "https://api.deepseek.com/anthropic"


@pytest.mark.asyncio
async def test_client_messages_create(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepseekAnthropicClient(AnthropicConfig(api_key="sk-test", base_url="https://api.deepseek.com/anthropic"))

    class _Resp:
        status_code = 200

        @property
        def content(self) -> bytes:
            return _sample_message_body()

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        assert path == "/messages"
        assert kwargs.get("stream") is False
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        assert headers.get("x-api-key") == "sk-test"
        assert "anthropic-version" in headers
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    result = await client.messages(AnthropicMessagesParam(model="deepseek-v4-flash", messages=[]))
    assert isinstance(result, AnthropicMessageResponse)
    assert result.content[0].text == "hello world"
    assert result.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_client_messages_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepseekAnthropicClient(AnthropicConfig(api_key="sk-test", base_url="https://api.deepseek.com/anthropic"))

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
    stream = await client.messages(AnthropicMessagesParam(model="deepseek-v4-flash", messages=[]), stream=True)
    events = [event async for event in stream]
    assert len(events) == 2


@pytest.mark.asyncio
async def test_client_models_empty_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepseekAnthropicClient(AnthropicConfig(api_key="sk-test", base_url="https://api.deepseek.com/anthropic"))

    async def fail_get(*_args: object, **_kwargs: object) -> object:
        msg = "GET /models must not be called on the /anthropic base"
        raise AssertionError(msg)

    monkeypatch.setattr(client.session, "get", fail_get)
    assert await client.models() == []
