"""DeepSeek server-side search backend (Anthropic-compatible endpoint)."""

from __future__ import annotations

import msgspec
import pytest

from plyngent.lmproto.anthropic.config import AnthropicConfig
from plyngent.lmproto.deepseek.anthropic.search import (
    DeepseekSearchClient,
    SearchHit,
    hits_from_content,
)


def test_hits_from_content_dedupes_and_skips_noise() -> None:
    content = [
        {"type": "thinking", "thinking": "search"},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "call_1",
            "content": [
                {"type": "web_search_result", "title": "First", "url": "https://a.example/x"},
                {"type": "web_search_result", "title": "First again", "url": "https://a.example/x"},
                {"type": "web_search_result", "url": "https://b.example/y"},
                {"type": "web_search_result", "title": "no url"},
            ],
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "call_2",
            "content": [{"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}],
        },
    ]
    hits = hits_from_content(content)
    assert hits == [
        SearchHit(title="First", url="https://a.example/x"),
        SearchHit(title="", url="https://b.example/y"),
    ]
    assert hits_from_content("not a list") == []


class _Resp:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.content = body


def _search_body() -> bytes:
    return msgspec.json.encode(
        {
            "id": "msg_1",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "call_00_1",
                    "name": "web_search",
                    "input": {"query": "DeepSeek-V4"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "call_00_1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "DeepSeek-V4 preview",
                            "url": "https://api-docs.deepseek.com/news/news260424/",
                            "encrypted_content": "opaque",
                        }
                    ],
                },
            ],
            "usage": {"input_tokens": 9932, "output_tokens": 377},
        }
    )


@pytest.mark.asyncio
async def test_search_sends_the_hosted_tool_and_reads_its_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepseekSearchClient(AnthropicConfig(api_key="sk-test", base_url="https://api.deepseek.com/anthropic"))
    sent: dict[str, object] = {}

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        sent["path"] = path
        sent.update(kwargs)
        return _Resp(200, _search_body())

    monkeypatch.setattr(client.session, "post", fake_post)
    hits = await client.search("DeepSeek-V4", model="deepseek-flash")

    assert hits == [SearchHit(title="DeepSeek-V4 preview", url="https://api-docs.deepseek.com/news/news260424/")]
    assert sent["path"] == "/messages"
    assert sent["stream"] is False
    data = sent["data"]
    assert isinstance(data, bytes)
    payload = msgspec.json.decode(data, type=dict[str, object])
    assert payload["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    # A search turn stops as soon as it has searched; the answer is not needed.
    assert payload["max_tokens"] == 512
    assert payload["messages"] == [{"role": "user", "content": "DeepSeek-V4"}]
    headers = sent["headers"]
    assert isinstance(headers, dict)
    assert headers["x-api-key"] == "sk-test"


@pytest.mark.asyncio
async def test_search_raises_with_the_endpoint_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepseekSearchClient(AnthropicConfig(api_key="sk-test", base_url="https://api.deepseek.com/anthropic"))

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        return _Resp(422, b'{"error": {"message": "tools[0]: unknown variant"}}')

    monkeypatch.setattr(client.session, "post", fake_post)
    with pytest.raises(RuntimeError, match="unknown variant"):
        _ = await client.search("DeepSeek-V4", model="deepseek-flash")
