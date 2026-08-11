from __future__ import annotations

import importlib
from pathlib import Path

from plyngent.tools.file import edit_replace, listdir, read_file, write_file
from tests.test_tools.helpers import call_sync

get_truncated_module = importlib.import_module("plyngent.tools.file.get_truncated")


def _extract_token(out: str) -> str | None:
    for line in out.splitlines():
        if "TRUNCATE_TOKEN:" in line:
            return line.split("[TRUNCATE_TOKEN: ", 1)[1].rstrip("]")
    return None


def test_write_read_listdir_edit(workspace: object) -> None:
    del workspace
    assert "wrote" in call_sync(write_file, "notes/a.txt", "hello world")
    assert call_sync(read_file, "notes/a.txt") == "L1-1\nhello world"
    listing = call_sync(listdir, "notes")
    assert "a.txt" in listing
    assert "file" in listing
    result = call_sync(edit_replace, "notes/a.txt", "world", "there")
    assert "replaced" in result
    assert call_sync(read_file, "notes/a.txt") == "L1-1\nhello there"


def test_read_file_max_chars_embeds_token(workspace: Path) -> None:
    (workspace / "big.txt").write_text("a" * 1000, encoding="utf-8")
    out = call_sync(read_file, "big.txt", max_chars=200)
    header, _, rest = out.partition("\n")
    assert header.startswith("L1-")  # 1-based range of the raw slice read
    content, _, marker = rest.partition("\n...[")
    assert content == "a" * len(content)
    assert len(content) < 200
    assert "TRUNCATE_TOKEN:" in marker
    token = _extract_token(out)
    assert token is not None
    parsed = get_truncated_module.decode_truncate_token(token)
    assert parsed is not None
    assert parsed.kind == "file"
    assert parsed.offset == len(content)  # cursor lands exactly where content stops


def test_read_file_max_chars_small_file_untouched(workspace: Path) -> None:
    (workspace / "small.txt").write_text("tiny", encoding="utf-8")
    out = call_sync(read_file, "small.txt", max_chars=200)
    assert out == "L1-1\ntiny"
    assert "TRUNCATE_TOKEN:" not in out


def test_edit_replace_first_only(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "aa aa")
    result = call_sync(edit_replace, "t.txt", "aa", "bb")
    assert call_sync(read_file, "t.txt") == "L1-1\nbb aa"
    assert "1 of 2" in result or "1 of 2 matches" in result
    assert "remain" in result


def test_edit_replace_max_replaces(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "aa aa aa")
    result = call_sync(edit_replace, "t.txt", "aa", "bb", max_replaces=2)
    assert call_sync(read_file, "t.txt") == "L1-1\nbb bb aa"
    assert "2 of 3" in result
    assert "1 remain" in result


def test_edit_replace_all_matches(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "aa aa")
    result = call_sync(edit_replace, "t.txt", "aa", "bb", max_replaces=10)
    assert call_sync(read_file, "t.txt") == "L1-1\nbb bb"
    assert "all 2 matches" in result


def test_edit_replace_max_replaces_invalid(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "aa")
    assert "max_replaces" in call_sync(edit_replace, "t.txt", "aa", "bb", max_replaces=0)


def test_edit_missing_old_string(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "x")
    assert "not found" in call_sync(edit_replace, "t.txt", "missing", "y")


def test_read_offset_limit(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "lines.txt", "a\nb\nc\nd\n")
    # offset is 0-based: offset=1 skips line 1, so the read starts at 1-based line 2.
    assert call_sync(read_file, "lines.txt", offset=1, limit=2) == "L2-3\nb\nc\n"


def test_read_file_missing_and_directory(workspace: Path) -> None:
    (workspace / "sub").mkdir()
    assert call_sync(read_file, "nope.txt") == "error: file not found: nope.txt"
    assert call_sync(read_file, "sub") == "error: not a file: sub"


def test_read_with_lineno(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "num.txt", "a\nb\nc\n")
    out = call_sync(read_file, "num.txt", with_lineno=True)
    assert "     1|a\n" in out
    assert "     2|b\n" in out
    assert "     3|c\n" in out
    # offset is 0-based; line numbers stay absolute 1-based file lines
    mid = call_sync(read_file, "num.txt", offset=1, limit=1, with_lineno=True)
    assert mid == "     2|b\n"


def test_listdir_missing(workspace: object) -> None:
    del workspace
    assert "error" in call_sync(listdir, "nope")
