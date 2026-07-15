from __future__ import annotations

import msgspec
import pytest

from plyngent.lmproto.openai_compatible.client import OpenAIClient
from plyngent.lmproto.openai_compatible.config import OpenAIConfig
from plyngent.lmproto.openai_compatible.responses_model import (
    Response,
    ResponseDeleted,
    ResponseFunctionTool,
    ResponseFunctionToolCall,
    ResponsesCreateParam,
    ResponseStreamEvent,
    response_function_calls,
    response_output_text,
)


def _sample_response_body() -> bytes:
    return (
        b'{"id":"resp_1","object":"response","created_at":1,"model":"gpt-test",'
        b'"status":"completed","output":['
        b'{"id":"rs_1","type":"reasoning","content":[],"summary":[]},'
        b'{"id":"msg_1","type":"message","role":"assistant","status":"completed",'
        b'"content":[{"type":"output_text","text":"hello world","annotations":[]}]},'
        b'{"type":"function_call","call_id":"call_1","name":"add",'
        b'"arguments":"{\\"a\\":1}","status":"completed"}'
        b"]}"
    )


def test_response_decode_and_helpers() -> None:
    parsed = msgspec.json.decode(_sample_response_body(), type=Response)
    assert parsed.id == "resp_1"
    assert parsed.object == "response"
    assert response_output_text(parsed) == "hello world"
    calls = response_function_calls(parsed)
    assert len(calls) == 1
    assert isinstance(calls[0], ResponseFunctionToolCall)
    assert calls[0].name == "add"
    assert calls[0].call_id == "call_1"


def test_create_param_encode_omits_defaults() -> None:
    param = ResponsesCreateParam(
        model="gpt-test",
        input="hi",
        tools=[ResponseFunctionTool(name="add", parameters={"type": "object"})],
    )
    raw = msgspec.json.encode(param)
    data = msgspec.json.decode(raw)
    assert data["model"] == "gpt-test"
    assert data["input"] == "hi"
    # omit_defaults: default stream=False is not encoded until client sets stream
    assert "stream" not in data
    assert data["tools"][0]["type"] == "function"
    assert data["tools"][0]["name"] == "add"


def test_stream_event_decode() -> None:
    raw = (
        b'{"type":"response.output_text.delta","item_id":"msg_1",'
        b'"output_index":0,"content_index":0,"delta":"hel"}'
    )
    event = msgspec.json.decode(raw, type=ResponseStreamEvent)
    assert event.type == "response.output_text.delta"
    assert event.delta == "hel"


@pytest.mark.asyncio
async def test_client_responses_create(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIClient(OpenAIConfig(access_key_or_token="sk", base_url="https://example/v1"))

    class _Resp:
        status_code = 200

        @property
        def content(self) -> bytes:
            return _sample_response_body()

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        assert path == "/responses"
        assert kwargs.get("stream") is False
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    result = await client.responses(ResponsesCreateParam(model="gpt-test", input="hi"))
    assert isinstance(result, Response)
    assert response_output_text(result) == "hello world"


@pytest.mark.asyncio
async def test_client_responses_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIClient(OpenAIConfig(access_key_or_token="sk", base_url="https://example/v1"))

    class _Resp:
        status_code = 200

        async def iter_lines(self):
            yield (
                b'data: {"type":"response.output_text.delta","delta":"a",'
                b'"item_id":"m","output_index":0,"content_index":0}'
            )
            yield b"data: [DONE]"

        def close(self) -> None:
            return None

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        assert path == "/responses"
        assert kwargs.get("stream") is True
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    stream = await client.responses(
        ResponsesCreateParam(model="gpt-test", input="hi"),
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) == 1
    assert events[0].type == "response.output_text.delta"
    assert events[0].delta == "a"


@pytest.mark.asyncio
async def test_client_get_delete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIClient(OpenAIConfig(access_key_or_token="sk", base_url="https://example/v1"))

    class _Get:
        status_code = 200

        @property
        def content(self) -> bytes:
            return _sample_response_body()

    class _Del:
        status_code = 200

        @property
        def content(self) -> bytes:
            return b'{"id":"resp_1","object":"response.deleted","deleted":true}'

    async def fake_get(path: str, **kwargs: object) -> _Get:
        assert path == "/responses/resp_1"
        return _Get()

    async def fake_delete(path: str, **kwargs: object) -> _Del:
        assert path == "/responses/resp_1"
        return _Del()

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr(client.session, "delete", fake_delete)
    got = await client.get_response("resp_1")
    assert got.id == "resp_1"
    deleted = await client.delete_response("resp_1")
    assert isinstance(deleted, ResponseDeleted)
    assert deleted.deleted is True
