from __future__ import annotations

from pathlib import Path

import pytest

from plyngent.tools.file import read_file, write_file
from plyngent.tools.temp_workspace import cleanup_temporary_workspaces, new_temporary_workspace
from plyngent.tools.workspace import (
    WorkspaceError,
    clear_workspace_allowlist,
    list_workspace_allowlist,
    resolve_path,
)
from tests.test_tools.helpers import call_sync


def _temp_path_from_tool_output(out: str) -> Path:
    line = next(part for part in out.splitlines() if part.startswith("temporary_workspace="))
    return Path(line.split("=", 1)[1].strip())


def test_new_temporary_workspace_allowlist(workspace: object) -> None:
    assert isinstance(workspace, Path)
    out = call_sync(new_temporary_workspace, "unit")
    assert "temporary_workspace=" in out
    assert "project workspace unchanged" in out
    temp = _temp_path_from_tool_output(out)
    assert temp.is_dir()
    assert temp in list_workspace_allowlist()

    # Absolute path under temp is allowed; project relative still works.
    target = temp / "scratch.txt"
    _ = target.write_text("hello-temp", encoding="utf-8")
    assert resolve_path(str(target)) == target.resolve()
    assert call_sync(read_file, str(target)) == "hello-temp"

    _ = call_sync(write_file, "project.txt", "proj")
    assert call_sync(read_file, "project.txt") == "proj"

    # Sibling under system temp that we did not allowlist still fails.
    outside = temp.parent / "not-ours-should-fail"
    with pytest.raises(WorkspaceError, match="escapes"):
        _ = resolve_path(str(outside))


def test_cleanup_removes_owned_temps(workspace: object) -> None:
    assert isinstance(workspace, Path)
    out = call_sync(new_temporary_workspace)
    temp = _temp_path_from_tool_output(out)
    assert temp.is_dir()
    n = cleanup_temporary_workspaces()
    assert n >= 1
    assert not temp.exists()
    assert list_workspace_allowlist() == []


def test_prefix_sanitized(workspace: object) -> None:
    del workspace
    out = call_sync(new_temporary_workspace, "bad/../x y!")
    assert "temporary_workspace=" in out
    assert not out.startswith("error")
    _ = cleanup_temporary_workspaces()
    clear_workspace_allowlist()
