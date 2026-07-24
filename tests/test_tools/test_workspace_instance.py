"""Workspace root prefers bound InstanceState when set."""

from __future__ import annotations

from pathlib import Path

import pytest

from plyngent.tools.context import InstanceState, bind_instance
from plyngent.tools.workspace import (
    WorkspaceError,
    clear_workspace_root,
    get_workspace_root,
    resolve_path,
    set_workspace_root,
)


def test_get_workspace_prefers_instance(tmp_path: Path) -> None:
    clear_workspace_root()
    other = tmp_path / "other"
    other.mkdir()
    _ = set_workspace_root(tmp_path)
    instance = InstanceState(workspace_root=other)
    with bind_instance(instance):
        assert get_workspace_root() == other.resolve()
    assert get_workspace_root() == tmp_path.resolve()
    clear_workspace_root()


def test_set_workspace_mirrors_to_bound_instance(tmp_path: Path) -> None:
    clear_workspace_root()
    instance = InstanceState()
    with bind_instance(instance):
        path = set_workspace_root(tmp_path)
        assert instance.workspace_root == path
        assert get_workspace_root() == path
    clear_workspace_root()


def test_clear_clears_bound_instance(tmp_path: Path) -> None:
    instance = InstanceState()
    with bind_instance(instance):
        _ = set_workspace_root(tmp_path)
        clear_workspace_root()
        assert instance.workspace_root is None
        with pytest.raises(WorkspaceError):
            _ = get_workspace_root()


def test_resolve_path_accepts_instance_root_without_process_global(tmp_path: Path) -> None:
    """Instance-only workspace must be a valid resolve root (not only process global)."""
    clear_workspace_root()
    target = tmp_path / "note.txt"
    _ = target.write_text("hi", encoding="utf-8")
    instance = InstanceState(workspace_root=tmp_path.resolve())
    with bind_instance(instance):
        resolved = resolve_path("note.txt")
        assert resolved == target.resolve()
    clear_workspace_root()


def test_path_denylist_uses_instance_policy(tmp_path: Path) -> None:
    """Path denylist is read from the bound instance policy bag."""
    clear_workspace_root()
    secret = tmp_path / "secrets"
    secret.mkdir()
    _ = (secret / "x.txt").write_text("no", encoding="utf-8")
    instance = InstanceState(workspace_root=tmp_path.resolve())
    instance.workspace.path_denylist = ("/secrets/",)
    with bind_instance(instance), pytest.raises(WorkspaceError, match="denied by policy"):
        _ = resolve_path("secrets/x.txt")
    clear_workspace_root()
