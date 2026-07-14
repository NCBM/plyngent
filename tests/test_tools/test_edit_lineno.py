from __future__ import annotations

from pathlib import Path

from plyngent.tools.file import edit_lineno, read_file, write_file
from tests.test_tools.helpers import call_sync


def test_edit_lineno_replace_middle(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "a.txt", "one\ntwo\nthree\nfour\n")
    out = call_sync(edit_lineno, "a.txt", 2, 3, "TWO\nTHREE\n")
    assert "replaced lines 2-3" in out
    assert call_sync(read_file, "a.txt") == "one\nTWO\nTHREE\nfour\n"


def test_edit_lineno_delete_range(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "b.txt", "a\nb\nc\n")
    out = call_sync(edit_lineno, "b.txt", 2, 2, "")
    assert "replaced" in out
    assert call_sync(read_file, "b.txt") == "a\nc\n"


def test_edit_lineno_append(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "c.txt", "only\n")
    out = call_sync(edit_lineno, "c.txt", 2, 2, "more\n")
    assert "appended" in out
    assert call_sync(read_file, "c.txt") == "only\nmore\n"


def test_edit_lineno_invalid(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "d.txt", "x\n")
    assert "start_line" in call_sync(edit_lineno, "d.txt", 0, 1, "y")
    assert "end_line" in call_sync(edit_lineno, "d.txt", 2, 1, "y")
