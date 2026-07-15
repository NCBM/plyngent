from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.lmproto.openai_compatible.client import BaseOpenAIClient
from plyngent.lmproto.openai_compatible.config import OpenAIConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from niquests.models import AsyncResponse


class _FakeResp:
    status_code: int
    closed: bool
    content: bytes
    _lines: list[bytes]

    def __init__(self, lines: list[bytes], *, status_code: int = 200) -> None:
        self.status_code = status_code
        self.closed = False
        self._lines = lines
        self.content = b""

    async def iter_lines(self) -> AsyncIterator[bytes]:
        for line in self._lines:
            yield line

    def close(self) -> None:
        self.closed = True


async def test_parse_sse_stops_at_done() -> None:
    client = BaseOpenAIClient(OpenAIConfig(access_key_or_token="t", base_url="http://x"))
    # Minimal OpenAI-style chunk payload
    chunk = (
        b'data: {"id":"1","object":"chat.completion.chunk","created":0,'
        b'"model":"m","choices":[{"index":0,"delta":{"content":"hi"},'
        b'"finish_reason":null}]}'
    )
    resp = _FakeResp([chunk, b"", b"data: [DONE]", b"data: should-not-parse"])
    out = [c async for c in client._parse_sse(cast("AsyncResponse", cast("object", resp)))]
    assert len(out) == 1
    assert out[0].choices[0].delta.content == "hi"
    assert resp.closed is True


async def test_parse_sse_http_error() -> None:
    import pytest

    client = BaseOpenAIClient(OpenAIConfig(access_key_or_token="t", base_url="http://x"))
    resp = _FakeResp([], status_code=500)
    resp.content = b'{"error":"boom"}'
    with pytest.raises(RuntimeError, match="500"):
        _ = [c async for c in client._parse_sse(cast("AsyncResponse", cast("object", resp)))]
    assert resp.closed is True
