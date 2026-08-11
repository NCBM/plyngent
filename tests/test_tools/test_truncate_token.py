from __future__ import annotations

from plyngent.tools.truncate_token import (
    TruncateToken,
    decode_truncate_token,
    encode_truncate_token,
    get_remainder,
    truncate_generic,
    truncate_with_token,
    truncation_marker,
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


def test_marker_format() -> None:
    token = TruncateToken(kind="file", location="a.txt", offset=100, limit=32_000)
    marker = truncation_marker(32_000, 500, token)
    assert marker.startswith("\n[Truncated (32000 chars max.; 500 omitted). truncate_token=")
    assert marker.endswith("]")
    assert encode_truncate_token(token) in marker


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
    assert "Truncated" not in out  # no hint when no truncation


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
    content, _, _ = out.partition("\n[Truncated")
    assert content.startswith("x")
    assert "[Truncated (200 chars max.; 300 omitted). truncate_token=" in out
    assert token.offset == len(content)  # cursor lands exactly where content stops
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
    assert token.offset == 30  # advanced by returned length only
    assert "[Truncated (40 chars max.; 170 omitted). truncate_token=" in out
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
    assert "Truncated" not in out  # no hint when the source ends


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
    content, _, _ = out.partition("\n[Truncated")
    assert token.offset == 150 + len(content)
    assert "[Truncated (200 chars max.; 350 omitted). truncate_token=" in out
    assert len(out) <= 200


def test_truncate_generic_short_untouched() -> None:
    assert truncate_generic("tiny", 200) == "tiny"


def test_truncate_generic_embeds_memory_token() -> None:
    text = "a" * 500
    out = truncate_generic(text, 200)
    assert len(out) <= 200
    assert "[Truncated (200 chars max." in out
    token_str = out.split("truncate_token=", 1)[1].rstrip("]")
    token = decode_truncate_token(token_str)
    assert token is not None
    assert token.kind == "memory"
    # Remainder is the full original text; token offset points into it.
    remainder = get_remainder(token.location)
    assert remainder == text
    assert token.offset < len(text)


def test_truncate_generic_no_marker_when_fits() -> None:
    out = truncate_generic("a" * 50, 200)
    assert out == "a" * 50
    assert "Truncated" not in out
