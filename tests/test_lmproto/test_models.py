from __future__ import annotations

import msgspec
import pytest

from plyngent.lmproto.openai_compatible.client import OpenAIClient
from plyngent.lmproto.openai_compatible.config import OpenAIConfig
from plyngent.lmproto.openai_compatible.model import ModelObject, ModelsResponse


def test_models_response_decode() -> None:
    raw = b'{"object":"list","data":[{"id":"b","object":"model"},{"id":"a","owned_by":"x"}]}'
    parsed = msgspec.json.decode(raw, type=ModelsResponse)
    assert [m.id for m in parsed.data] == ["b", "a"]
    assert isinstance(parsed.data[0], ModelObject)


@pytest.mark.asyncio
async def test_client_models_sorted_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIClient(OpenAIConfig(access_key_or_token="sk", base_url="https://example/v1"))

    class _Resp:
        status_code = 200

        @property
        def content(self) -> bytes:
            return (
                b'{"object":"list","data":['
                b'{"id":"m2"},{"id":"m1"},{"id":"m2"},{"id":""}'
                b"]}"
            )

    async def fake_get(path: str, **kwargs: object) -> _Resp:
        assert path == "/models"
        return _Resp()

    monkeypatch.setattr(client.session, "get", fake_get)
    ids = await client.models()
    assert ids == ["m1", "m2"]


@pytest.mark.asyncio
async def test_client_models_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAIClient(OpenAIConfig(access_key_or_token="sk", base_url="https://example/v1"))

    class _Resp:
        status_code = 401

        @property
        def content(self) -> bytes:
            return b'{"error":"nope"}'

        def close(self) -> None:
            return None

    async def fake_get(path: str, **kwargs: object) -> _Resp:
        del path, kwargs
        return _Resp()

    monkeypatch.setattr(client.session, "get", fake_get)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        _ = await client.models()
