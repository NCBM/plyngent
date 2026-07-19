from __future__ import annotations

from plyngent.tools.file import edit_replace, listdir, read_file, write_file
from tests.test_tools.helpers import call_sync


def test_write_read_listdir_edit(workspace: object) -> None:
    del workspace
    assert "wrote" in call_sync(write_file, "notes/a.txt", "hello world")
    assert call_sync(read_file, "notes/a.txt") == "hello world"
    listing = call_sync(listdir, "notes")
    assert "a.txt" in listing
    assert "file" in listing
    result = call_sync(edit_replace, "notes/a.txt", "world", "there")
    assert "replaced" in result
    assert call_sync(read_file, "notes/a.txt") == "hello there"


def test_edit_replace_first_only(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "aa aa")
    result = call_sync(edit_replace, "t.txt", "aa", "bb")
    assert call_sync(read_file, "t.txt") == "bb aa"
    assert "1 of 2" in result or "1 of 2 matches" in result
    assert "remain" in result


def test_edit_replace_max_replaces(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "aa aa aa")
    result = call_sync(edit_replace, "t.txt", "aa", "bb", max_replaces=2)
    assert call_sync(read_file, "t.txt") == "bb bb aa"
    assert "2 of 3" in result
    assert "1 remain" in result


def test_edit_replace_all_matches(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "t.txt", "aa aa")
    result = call_sync(edit_replace, "t.txt", "aa", "bb", max_replaces=10)
    assert call_sync(read_file, "t.txt") == "bb bb"
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
    assert call_sync(read_file, "lines.txt", offset=1, limit=2) == "b\nc\n"


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
