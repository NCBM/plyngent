from __future__ import annotations

from plyngent.cli.input_text import (
    assemble_multiline,
    finish_triple_quote_line,
    parse_triple_quote_line,
    read_repl_entry,
)


def test_parse_same_line_block() -> None:
    assert parse_triple_quote_line('"""hello"""') == ("hello", True)
    assert parse_triple_quote_line('  """hi"""  ') == ("hi", True)


def test_parse_open_block() -> None:
    assert parse_triple_quote_line('"""first') == ("first", False)
    assert parse_triple_quote_line('"""') == ("", False)


def test_parse_not_block() -> None:
    assert parse_triple_quote_line("hello") is None
    assert parse_triple_quote_line("/help") is None


def test_finish_closer() -> None:
    assert finish_triple_quote_line('"""') == ("", True)
    assert finish_triple_quote_line('last line"""') == ("last line", True)
    assert finish_triple_quote_line("middle") == ("middle", False)


def test_assemble_multiline() -> None:
    lines = iter(["line two", '"""'])
    text = assemble_multiline(
        '"""line one',
        read_line=lambda: next(lines),
    )
    assert text == "line one\nline two"


def test_assemble_empty_cancels() -> None:
    lines = iter(['"""'])
    assert assemble_multiline('"""', read_line=lambda: next(lines)) is None


def test_read_repl_entry_slash() -> None:
    assert read_repl_entry(read_line=lambda: "/status") == "/status"


def test_read_repl_entry_simple() -> None:
    assert read_repl_entry(read_line=lambda: "  hi  ") == "hi"


def test_read_repl_entry_empty() -> None:
    assert read_repl_entry(read_line=lambda: "   ") is None


def test_read_repl_entry_multiline() -> None:
    seq = iter(['"""a', "b", '"""'])
    text = read_repl_entry(read_line=lambda: next(seq), echo=lambda _s: None)
    assert text == "a\nb"
