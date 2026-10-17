"""DeepSeek's Search API client (``POST /v1/tools/search``)."""

from __future__ import annotations

import msgspec
import pytest

from plyngent.lmproto.deepseek.search import (
    DeepseekSearchClient,
    SearchRequest,
)
from plyngent.lmproto.openai_compatible.config import OpenAIConfig
from plyngent.lmproto.search import SearchHit


class _Resp:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.content = body


def _body(*results: dict[str, object]) -> bytes:
    return msgspec.json.encode({"results": list(results)})


def _client() -> DeepseekSearchClient:
    return DeepseekSearchClient(
        OpenAIConfig(access_key_or_token="sk-test", base_url="https://api.deepseek.com"),
    )


def _decode_request(data: object) -> dict[str, object]:
    assert isinstance(data, bytes)
    return msgspec.json.decode(data, type=dict[str, object])


@pytest.mark.asyncio
async def test_search_posts_the_query_and_maps_the_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    sent: dict[str, object] = {}

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        sent["path"] = path
        sent.update(kwargs)
        return _Resp(
            200,
            _body(
                {
                    "rerank_score": 2.4,
                    "title": " DeepSeek V4 released ",
                    "url": " https://example.com/v4 ",
                    "snippet": "Tool calling and long context.",
                    "summary": "A longer summary.",
                    "authority": 3,
                },
                {"title": "Unusable", "url": ""},
            ),
        )

    monkeypatch.setattr(client.session, "post", fake_post)
    hits = await client.search("DeepSeek V4", count=5)

    assert hits == [
        SearchHit(
            title="DeepSeek V4 released",
            url="https://example.com/v4",
            snippet="Tool calling and long context.",
        )
    ]
    assert sent["path"] == "/v1/tools/search"
    assert sent["stream"] is False
    payload = _decode_request(sent["data"])
    assert payload == {"query": "DeepSeek V4", "count": 5}


@pytest.mark.asyncio
async def test_summary_stands_in_when_the_index_has_no_snippet(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        return _Resp(
            200,
            _body(
                {
                    "title": "T",
                    "url": "https://example.com/1",
                    "snippet": "   ",
                    "summary": "  A 600 character\tsummary\nspread over lines. ",
                }
            ),
        )

    monkeypatch.setattr(client.session, "post", fake_post)
    hits = await client.search("q")
    assert hits == [
        SearchHit(title="T", url="https://example.com/1", snippet="A 600 character summary spread over lines.")
    ]


@pytest.mark.asyncio
async def test_long_summaries_are_cut_for_the_tool_line(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        return _Resp(200, _body({"title": "T", "url": "https://example.com/1", "summary": "x" * 500}))

    monkeypatch.setattr(client.session, "post", fake_post)
    hits = await client.search("q")
    assert len(hits[0].snippet) == 200
    assert hits[0].snippet.endswith("…")


@pytest.mark.asyncio
async def test_empty_results_are_a_normal_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        return _Resp(200, _body())

    monkeypatch.setattr(client.session, "post", fake_post)
    assert await client.search("nothing matches this") == []


@pytest.mark.asyncio
async def test_count_is_clamped_to_the_documented_range(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    seen: list[dict[str, object]] = []

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        seen.append(_decode_request(kwargs["data"]))
        return _Resp(200, _body())

    monkeypatch.setattr(client.session, "post", fake_post)
    _ = await client.search("q", count=0)
    _ = await client.search("q", count=99)
    assert [payload["count"] for payload in seen] == [1, 50]


@pytest.mark.asyncio
async def test_an_oversized_query_is_refused_before_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    calls: list[object] = []

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        calls.append(path)
        return _Resp(200, _body())

    monkeypatch.setattr(client.session, "post", fake_post)
    with pytest.raises(ValueError, match="at most 128 characters"):
        _ = await client.search("x" * 129)
    assert calls == []


@pytest.mark.asyncio
async def test_the_endpoint_error_detail_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        return _Resp(400, b'{"error": {"message": "\'count\' must be between 1 and 50"}}')

    monkeypatch.setattr(client.session, "post", fake_post)
    with pytest.raises(RuntimeError, match="must be between 1 and 50"):
        _ = await client.search("q")


@pytest.mark.asyncio
async def test_a_retryable_status_is_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr("plyngent.lmproto.deepseek.search.client.RETRY_DELAYS", (0.0,))
    statuses = [429, 200]

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        return _Resp(statuses.pop(0), _body({"title": "T", "url": "https://example.com/1"}))

    monkeypatch.setattr(client.session, "post", fake_post)
    hits = await client.search("q")
    assert [hit.url for hit in hits] == ["https://example.com/1"]
    assert statuses == []


@pytest.mark.asyncio
async def test_a_balance_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr("plyngent.lmproto.deepseek.search.client.RETRY_DELAYS", (0.0,))
    calls: list[int] = []

    async def fake_post(path: str, **kwargs: object) -> _Resp:
        calls.append(1)
        return _Resp(402, b'{"error": {"message": "Insufficient Balance"}}')

    monkeypatch.setattr(client.session, "post", fake_post)
    with pytest.raises(RuntimeError, match="Insufficient Balance"):
        _ = await client.search("q")
    assert calls == [1]


def test_the_request_model_omits_what_was_not_set() -> None:
    encoded = msgspec.json.encode(SearchRequest(query="q"))
    assert msgspec.json.decode(encoded, type=dict[str, object]) == {"query": "q"}
