from __future__ import annotations

import pytest

from plyngent.lmproto.deepseek import DeepseekResponsesClient
from plyngent.lmproto.openai import Response, ResponsesCreateParam, response_output_text
from plyngent.lmproto.openai_compatible.config import OpenAIConfig


def _sample_response_body() -> bytes:
    return (
        b'{"id":"resp_1","object":"response","created_at":1,"model":"deepseek-v4-flash",'
        b'"status":"completed","output":['
        b'{"id":"msg_1","type":"message","role":"assistant","status":"completed",'
        b'"content":[{"type":"output_text","text":"hello from deepseek","annotations":[]}]}'
        b"]}"
    )


def test_client_kind_and_base_url() -> None:
    client = DeepseekResponsesClient(OpenAIConfig(access_key_or_token="sk", base_url="https://api.deepseek.com"))
    assert client.kind == "responses"
    assert client.session.base_url == "https://api.deepseek.com"


@pytest.mark.asyncio
async def test_client_responses_create(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepseekResponsesClient(OpenAIConfig(access_key_or_token="sk", base_url="https://api.deepseek.com"))

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
    result = await client.responses(ResponsesCreateParam(model="deepseek-v4-flash", input="hi"))
    assert isinstance(result, Response)
    assert response_output_text(result) == "hello from deepseek"


@pytest.mark.asyncio
async def test_client_responses_stream_stops_at_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek sends no ``[DONE]``: the stream must stop at a terminal event."""
    client = DeepseekResponsesClient(OpenAIConfig(access_key_or_token="sk", base_url="https://api.deepseek.com"))

    class _Resp:
        status_code = 200

        async def iter_lines(self):
            yield (
                b'data: {"type":"response.output_text.delta","delta":"hel",'
                b'"item_id":"m","output_index":0,"content_index":0,"sequence_number":1}'
            )
            # Terminal event carries the full response (no [DONE] follows).
            yield (
                b'data: {"type":"response.completed","sequence_number":2,'
                b'"response":{"id":"resp_1","object":"response","created_at":1,'
                b'"model":"deepseek-v4-flash","status":"completed","output":[]}}'
            )
            # Anything after the terminal event must not be consumed.
            yield b'data: {"type":"response.output_text.delta","delta":"ignored"}'

        def close(self) -> None:
            return None

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        assert path == "/responses"
        assert kwargs.get("stream") is True
        return _Resp()

    monkeypatch.setattr(client.session, "post", fake_post)
    stream = await client.responses(
        ResponsesCreateParam(model="deepseek-v4-flash", input="hi"),
        stream=True,
    )
    events = [event async for event in stream]
    assert [event.type for event in events] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert isinstance(events[1].response, dict)
    assert events[1].response["status"] == "completed"
