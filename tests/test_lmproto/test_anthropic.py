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
    AnthropicWebSearchToolResultContent,
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
async def test_client_decodes_a_server_side_search_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek's Anthropic compat answers a search turn with server tool blocks.

    A live turn reads ``thinking, server_tool_use, web_search_tool_result, text``
    (the results carry an ``encrypted_content`` the API demands back unchanged);
    the response must decode, even though none of those blocks is a local call.
    """
    body = {
        "id": "msg_search",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-flash",
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "Let me search.", "signature": "b032a4ad-87fe-42ce-801d-e33dd743d8b9"},
            {
                "type": "server_tool_use",
                "id": "call_00_joE6PEEC1otU8pvSe4zY3017",
                "name": "web_search",
                "input": {"query": "DeepSeek latest news"},
                "caller": {"type": "direct"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "call_00_joE6PEEC1otU8pvSe4zY3017",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Change Log | DeepSeek API Docs",
                        "url": "https://api-docs.deepseek.com/updates/",
                        "encrypted_content": "sd918tZXSvVp6j7ermvF4cl/lspbqHzths9SMQw6ziEw",
                    }
                ],
            },
            {"type": "text", "text": "DeepSeek shipped a new model."},
        ],
        "usage": {"input_tokens": 16592, "output_tokens": 1419, "server_tool_use": {"web_search_requests": 4}},
    }
    client = AnthropicClient(AnthropicConfig(api_key="sk-test", base_url="https://example/v1"))

    class _Resp:
        status_code = 200
        content = msgspec.json.encode(body)

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    result = await client.messages(AnthropicMessagesParam(model="deepseek-flash", messages=[]))
    kinds = [block.__struct_config__.tag for block in result.content]
    assert kinds == ["thinking", "server_tool_use", "web_search_tool_result", "text"]
    search = result.content[2]
    assert isinstance(search, AnthropicWebSearchToolResultContent)
    assert search.tool_use_id == "call_00_joE6PEEC1otU8pvSe4zY3017"
    assert search.content[0]["encrypted_content"].startswith("sd918tZXSvVp")
    assert result.content[3].text == "DeepSeek shipped a new model."


def test_messages_param_encodes_server_tool_dicts() -> None:
    """A hosted tool goes on the wire as the server-tool dict the surface wants."""
    param = AnthropicMessagesParam(
        model="deepseek-flash",
        messages=[AnthropicUserMessage(content="hi")],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    )
    data = msgspec.json.decode(msgspec.json.encode(param))
    assert data["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]


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
