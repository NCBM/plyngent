from __future__ import annotations

import importlib
from pathlib import Path

from plyngent.tools.truncate_token import TruncateToken, decode_truncate_token, encode_truncate_token
from tests.test_tools.helpers import call_async

get_truncated_module = importlib.import_module("plyngent.tools.file.get_truncated")
get_truncated = get_truncated_module.get_truncated


def _extract_token(out: str) -> str | None:
    for line in out.splitlines():
        if "truncate_token=" in line:
            return line.split("truncate_token=", 1)[1].rstrip("]")
    return None


async def test_get_truncated_invalid_token() -> None:
    out = await call_async(get_truncated, "garbage-token")
    assert out.startswith("error:")
    assert "invalid" in out


async def test_get_truncated_file_chains_without_gap(workspace: Path) -> None:
    (workspace / "big.txt").write_text("a" * 1000, encoding="utf-8")
    token = encode_truncate_token(TruncateToken(kind="file", location="big.txt", offset=0, limit=200))
    total_read = 0
    for _ in range(20):
        out = await call_async(get_truncated, token, max_chars=200)
        body, _, _ = out.partition("\n[Truncated")
        header, _, content = body.partition("\n")
        assert header.startswith("L") and "-" in header
        assert content == "a" * len(content)
        total_read += len(content)
        next_token = _extract_token(out)
        if next_token is None:
            break
        parsed = decode_truncate_token(next_token)
        assert parsed is not None
        assert parsed.offset == total_read  # no gap, no overlap
        token = next_token
    assert total_read == 1000


async def test_get_truncated_file_missing(workspace: Path) -> None:
    token = encode_truncate_token(TruncateToken(kind="file", location="nope.txt", offset=0, limit=200))
    out = await call_async(get_truncated, token, max_chars=200)
    assert out.startswith("error:")


async def test_get_truncated_http_passthrough(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []

    async def fake_fetch(url: str, *, offset: int, max_chars: int) -> str:
        calls.append((url, offset, max_chars))
        return f"body-from-{offset}"

    monkeypatch.setattr(get_truncated_module.fetch, "handler", fake_fetch)
    token = encode_truncate_token(TruncateToken(kind="http", location="https://example.com/x", offset=500, limit=400))
    out = await call_async(get_truncated, token, max_chars=200)
    assert out == "body-from-500"
    assert calls == [("https://example.com/x", 500, 400)]


async def test_get_truncated_memory_chains() -> None:
    from plyngent.tools.truncate_token import truncate_generic

    bounded = truncate_generic("m" * 1000, 200)
    assert "truncate_token=" in bounded
    token = bounded.split("truncate_token=", 1)[1].rstrip("]")
    parsed = decode_truncate_token(token)
    assert parsed is not None
    assert parsed.kind == "memory"
    cursor = parsed.offset  # chars already delivered by truncate_generic
    delivered = cursor
    for _ in range(20):
        out = await call_async(get_truncated, token, max_chars=200)
        content, _, _ = out.partition("\n[Truncated")
        assert content == "m" * len(content)
        delivered += len(content)
        next_token = _extract_token(out)
        if next_token is None:
            break  # source exhausted → no token, no marker
        parsed = decode_truncate_token(next_token)
        assert parsed is not None
        assert parsed.kind == "memory"
        # No gap: the next cursor is exactly where this chunk ended.
        assert parsed.offset == delivered
        token = next_token
    assert delivered == 1000


async def test_get_truncated_memory_expired() -> None:
    from plyngent.tools.truncate_token import _MEMORY_REMAINDERS, store_remainder

    key = store_remainder("x" * 500)
    del _MEMORY_REMAINDERS[key]
    token = encode_truncate_token(TruncateToken(kind="memory", location=key, offset=0, limit=200))
    out = await call_async(get_truncated, token, max_chars=200)
    assert out.startswith("error:")
    assert "expired" in out
