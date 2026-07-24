"""Workspace policy requires bound InstanceState."""

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


def test_unbound_get_workspace_errors() -> None:
    with pytest.raises(WorkspaceError, match="instance state is not bound"):
        _ = get_workspace_root()


def test_set_workspace_on_bound_instance(tmp_path: Path) -> None:
    instance = InstanceState()
    with bind_instance(instance):
        path = set_workspace_root(tmp_path)
        assert instance.workspace_root == path
        assert instance.workspace.root == path
        assert get_workspace_root() == path


def test_clear_clears_bound_instance(tmp_path: Path) -> None:
    instance = InstanceState()
    with bind_instance(instance):
        _ = set_workspace_root(tmp_path)
        clear_workspace_root()
        assert instance.workspace_root is None
        with pytest.raises(WorkspaceError, match="workspace root is not set"):
            _ = get_workspace_root()


def test_resolve_path_uses_instance_root(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    _ = target.write_text("hi", encoding="utf-8")
    instance = InstanceState(workspace_root=tmp_path.resolve())
    with bind_instance(instance):
        resolved = resolve_path("note.txt")
        assert resolved == target.resolve()


def test_path_denylist_uses_instance_policy(tmp_path: Path) -> None:
    secret = tmp_path / "secrets"
    secret.mkdir()
    _ = (secret / "x.txt").write_text("no", encoding="utf-8")
    instance = InstanceState(workspace_root=tmp_path.resolve())
    instance.workspace.path_denylist = ("/secrets/",)
    with bind_instance(instance), pytest.raises(WorkspaceError, match="denied by policy"):
        _ = resolve_path("secrets/x.txt")
