from __future__ import annotations

from pathlib import Path

from plyngent.agent import reset_lineno_tracker
from plyngent.tools.file import edit_lineno, read_file, write_file
from tests.test_tools.helpers import call_sync


def _read_lineno(path: str) -> str:
    """Read file with line numbers so edit_lineno allows the edit."""
    _ = reset_lineno_tracker()
    return call_sync(read_file, path, with_lineno=True)


def test_edit_lineno_replace_middle(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "a.txt", "one\ntwo\nthree\nfour\n")
    _ = _read_lineno("a.txt")
    out = call_sync(edit_lineno, "a.txt", 2, 3, "TWO\nTHREE\n")
    assert "replaced lines 2-3" in out
    assert call_sync(read_file, "a.txt") == "one\nTWO\nTHREE\nfour\n"


def test_edit_lineno_requires_all_lines_read(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "r.txt", "a\nb\nc\nd\ne\n")
    _ = reset_lineno_tracker()
    # Only lines 2-4 are read; editing line 1 or 5 must be rejected.
    _ = call_sync(read_file, "r.txt", offset=1, limit=3, with_lineno=True)
    out = call_sync(edit_lineno, "r.txt", 1, 2, "X\n")
    assert "not read" in out and "1" in out
    out = call_sync(edit_lineno, "r.txt", 4, 5, "Y\n")
    assert "not read" in out and "5" in out
    # Fully read range still edits.
    out = call_sync(edit_lineno, "r.txt", 2, 3, "B\nC\n")
    assert "replaced lines 2-3" in out


def test_edit_lineno_append_requires_last_line_read(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "ap.txt", "a\nb\nc\n")
    _ = reset_lineno_tracker()
    # Only lines 1-2 read; appending after line 3 requires line 3 read.
    _ = call_sync(read_file, "ap.txt", offset=0, limit=2, with_lineno=True)
    out = call_sync(edit_lineno, "ap.txt", 4, 4, "d\n")
    assert "not read" in out and "3" in out
    # Read the last line, then append succeeds.
    _ = call_sync(read_file, "ap.txt", offset=2, limit=1, with_lineno=True)
    out = call_sync(edit_lineno, "ap.txt", 4, 4, "d\n")
    assert "appended" in out
    assert call_sync(read_file, "ap.txt") == "a\nb\nc\nd\n"


def test_edit_lineno_delete_range(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "b.txt", "a\nb\nc\n")
    _ = _read_lineno("b.txt")
    out = call_sync(edit_lineno, "b.txt", 2, 2, "")
    assert "replaced" in out
    assert call_sync(read_file, "b.txt") == "a\nc\n"


def test_edit_lineno_append(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "c.txt", "only\n")
    _ = _read_lineno("c.txt")
    out = call_sync(edit_lineno, "c.txt", 2, 2, "more\n")
    assert "appended" in out
    assert call_sync(read_file, "c.txt") == "only\nmore\n"


def test_edit_lineno_invalid(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "d.txt", "x\n")
    _ = _read_lineno("d.txt")
    assert "start_line" in call_sync(edit_lineno, "d.txt", 0, 1, "y")
    assert "end_line" in call_sync(edit_lineno, "d.txt", 2, 1, "y")


def test_edit_lineno_rejects_without_prior_lineno_read(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "e.txt", "hello\nworld\n")
    _ = reset_lineno_tracker()
    out = call_sync(edit_lineno, "e.txt", 1, 1, "hi\n")
    assert "read" in out and "with_lineno=true" in out
