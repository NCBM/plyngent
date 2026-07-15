from __future__ import annotations

from plyngent.lmproto.openai_compatible.client import http_error_message, sse_data_payload
from plyngent.lmproto.openai_compatible.model import ChatCompletionChunk


def test_sse_data_payload_done_variants() -> None:
    assert sse_data_payload(b"data: [DONE]") is False
    assert sse_data_payload(b"[DONE]") is False
    assert sse_data_payload(b"data: [DONE]\n") is False
    assert sse_data_payload(b"data:  [DONE]  ") is False


def test_sse_data_payload_skip() -> None:
    assert sse_data_payload(b"") is None
    assert sse_data_payload(b": comment") is None
    assert sse_data_payload(b"event: message") is None
    assert sse_data_payload(b"data: ") is None


def test_sse_data_payload_json() -> None:
    line = (
        b'data: {"id":"1","object":"chat.completion.chunk","created":0,'
        b'"model":"m","choices":[{"index":0,"delta":{"content":"hi"},'
        b'"finish_reason":null}]}'
    )
    payload = sse_data_payload(line)
    assert isinstance(payload, bytes)
    chunk = msgspec_decode_chunk(payload)
    assert chunk.choices[0].delta.content == "hi"


def msgspec_decode_chunk(payload: bytes) -> ChatCompletionChunk:
    import msgspec

    return msgspec.json.decode(payload, type=ChatCompletionChunk)


def test_sse_stream_stops_at_done() -> None:
    """Simulate a line iterator: only pre-DONE data lines are decoded."""
    lines = [
        b'data: {"id":"1","object":"chat.completion.chunk","created":0,'
        b'"model":"m","choices":[{"index":0,"delta":{"content":"a"},'
        b'"finish_reason":null}]}',
        b"",
        b"data: [DONE]",
        b'data: {"id":"1","choices":[{"delta":{"content":"never"}}]}',
    ]
    payloads: list[bytes] = []
    for line in lines:
        parsed = sse_data_payload(line)
        if parsed is None:
            continue
        if parsed is False:
            break
        payloads.append(parsed)
    assert len(payloads) == 1
    assert msgspec_decode_chunk(payloads[0]).choices[0].delta.content == "a"


def test_http_error_message() -> None:
    assert http_error_message(200) is None
    assert http_error_message(399) is None
    err = http_error_message(500, b'{"error":"boom"}')
    assert err is not None
    assert "500" in err
    assert "boom" in err
    err503 = http_error_message(503, "unavailable")
    assert err503 is not None
    assert "503" in err503
    assert "unavailable" in err503
