from __future__ import annotations

from plyngent.tools.truncate_token import (
    TruncateToken,
    decode_truncate_token,
    encode_truncate_token,
    truncate_with_token,
)


def test_token_roundtrip() -> None:
    token = TruncateToken(kind="file", location="src/a.py", offset=100, limit=32_000)
    encoded = encode_truncate_token(token)
    assert decode_truncate_token(encoded) == token


def test_decode_garbage_returns_none() -> None:
    assert decode_truncate_token("not-a-token") is None
    assert decode_truncate_token("") is None
    assert decode_truncate_token("!!!") is None


def test_decode_rejects_invalid_fields() -> None:
    bad = TruncateToken(kind="file", location="x", offset=-1, limit=10)
    assert decode_truncate_token(encode_truncate_token(bad)) is None
    bad_limit = TruncateToken(kind="http", location="u", offset=0, limit=0)
    assert decode_truncate_token(encode_truncate_token(bad_limit)) is None


def test_truncate_with_token_short_text_untouched() -> None:
    out, token = truncate_with_token(
        "hello",
        32_000,
        kind="file",
        location="a.txt",
        offset=0,
        limit=32_000,
        total_len=5,
    )
    assert out == "hello"
    assert token is None


def test_truncate_with_token_cuts_and_advances() -> None:
    text = "x" * 500
    out, token = truncate_with_token(
        text,
        200,
        kind="file",
        location="a.txt",
        offset=0,
        limit=200,
        total_len=500,
    )
    assert token is not None
    assert "truncated 300 chars" in out
    assert f"[TRUNCATE_TOKEN: {encode_truncate_token(token)}]" in out
    assert token.offset == 200
    assert token.limit == 200
    assert len(out) <= 200  # marker fits inside the char budget


def test_truncate_with_token_more_after_nonoverflow() -> None:
    # Segment fits the chunk, but the source continues: still emit a token.
    text = "y" * 30
    out, token = truncate_with_token(
        text,
        40,
        kind="file",
        location="a.txt",
        offset=0,
        limit=40,
        total_len=200,
    )
    assert token is not None
    assert token.offset == 40
    assert "more content available" in out
    assert out.startswith("y" * 30)


def test_truncate_with_token_end_reached_no_token() -> None:
    text = "z" * 30
    out, token = truncate_with_token(
        text,
        40,
        kind="http",
        location="https://example.com/x",
        offset=0,
        limit=40,
        total_len=30,
    )
    assert token is None
    assert out == text


def test_truncate_with_token_nonzero_start() -> None:
    text = "y" * 500
    out, token = truncate_with_token(
        text,
        200,
        kind="http",
        location="https://example.com/x",
        offset=150,
        limit=200,
        total_len=700,
    )
    assert token is not None
    assert token.kind == "http"
    assert token.offset == 350  # 150 + 200
    assert "truncated 300 chars" in out
    assert len(out) <= 200
