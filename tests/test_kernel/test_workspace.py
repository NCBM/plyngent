"""A workspace resolves paths under its own root, or refuses them with the reason attached."""

from __future__ import annotations

import pytest

from plyngent.kernel.access import AccessMode, mode_covers, parse_access_mode
from plyngent.kernel.workspace import (
    Denied,
    OutsideWorkspace,
    Workspace,
    WorkspaceConfig,
)


def test_a_relative_path_resolves_against_the_root(tmp_path) -> None:
    workspace = Workspace(tmp_path)

    assert workspace.resolve("notes.txt") == tmp_path.resolve() / "notes.txt"
    assert workspace.resolve(tmp_path / "deep" / "x") == tmp_path.resolve() / "deep" / "x"
    assert workspace.under_roots(tmp_path.resolve())


def test_a_path_no_root_covers_is_refused_with_what_it_would_take(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "other" / "file.txt"
    workspace = Workspace(root)

    with pytest.raises(OutsideWorkspace) as caught:
        _ = workspace.resolve(outside)

    refusal = caught.value
    assert refusal.root == root
    assert refusal.need is AccessMode.WRITE, "the conservative default is a write"
    assert refusal.resolved == outside
    assert refusal.related_tools == (), "the kernel names no tools itself"
    assert "escapes workspace root" in str(refusal)


def test_the_same_refusal_can_be_given_the_tools_that_lift_it(tmp_path) -> None:
    workspace = Workspace(tmp_path / "ws")
    with pytest.raises(OutsideWorkspace) as caught:
        _ = workspace.resolve("../elsewhere")

    named = caught.value.with_tools("request_directory_access")
    assert named.related_tools == ("request_directory_access",)
    assert caught.value.related_tools == (), "the refusal raised is not modified in place"
    assert named.resolved == caught.value.resolved


def test_a_grant_is_what_lifts_a_refusal_and_it_has_to_cover_the_need(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    workspace = Workspace(root)

    assert workspace.resolve(shared / "a.txt", grants={shared: AccessMode.WRITE}) == shared / "a.txt"
    assert workspace.resolve(shared / "a.txt", AccessMode.READ, grants={shared: AccessMode.WRITE}) == shared / "a.txt"
    with pytest.raises(OutsideWorkspace):
        _ = workspace.resolve(shared / "a.txt", AccessMode.EXEC, grants={shared: AccessMode.WRITE})


def test_the_widest_grant_wins(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    shared = tmp_path / "shared"
    nested = shared / "nested"
    nested.mkdir(parents=True)
    workspace = Workspace(root)

    grants = {shared: AccessMode.READ, nested: AccessMode.EXEC}
    assert workspace.resolve(nested / "run.sh", AccessMode.EXEC, grants=grants) == nested / "run.sh"


def test_the_denylist_matches_a_substring_and_lifts_nothing(tmp_path) -> None:
    root = tmp_path / "ws"
    secret = root / "secrets"
    secret.mkdir(parents=True)
    workspace = Workspace(root, WorkspaceConfig(path_denylist=("/secrets/",)))

    assert workspace.denied_by(secret / "key.pem") == "/secrets/"
    with pytest.raises(Denied) as caught:
        _ = workspace.resolve("secrets/key.pem")
    assert caught.value.pattern == "/secrets/"

    # A trailing-separator pattern is about what is *in* the directory, so the directory itself only
    # matches when the caller says it means the directory.
    assert workspace.denied_by(secret) is None
    assert workspace.denied_by(secret, directory=True) == "/secrets/"


def test_a_denial_wins_over_a_grant(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    workspace = Workspace(root, WorkspaceConfig(path_denylist=("private",)))

    with pytest.raises(Denied):
        _ = workspace.resolve("private/x", grants={root: AccessMode.EXEC})


def test_a_new_config_value_replaces_the_old_one(tmp_path) -> None:
    workspace = Workspace(tmp_path, WorkspaceConfig(path_denylist=("zzz-old-secret",)))
    workspace.configure(WorkspaceConfig(path_denylist=("zzz-new-secret",)))

    assert workspace.config.path_denylist == ("zzz-new-secret",)
    assert workspace.denied_by(tmp_path / "zzz-old-secret") is None
    assert workspace.denied_by(tmp_path / "zzz-new-secret") == "zzz-new-secret"


def test_two_workspaces_resolve_the_same_text_differently(tmp_path) -> None:
    a = Workspace(tmp_path / "a")
    b = Workspace(tmp_path / "b")

    assert a.resolve("x") != b.resolve("x")
    assert a.resolve("x") == tmp_path.resolve() / "a" / "x"
    with pytest.raises(OutsideWorkspace):
        _ = b.resolve(a.resolve("x"))


def test_modes_are_ordered_and_parse_from_what_a_human_writes() -> None:
    assert parse_access_mode(" READ ") is AccessMode.READ
    assert parse_access_mode("exec") is AccessMode.EXEC
    assert parse_access_mode("admin") is None
    assert mode_covers(AccessMode.EXEC, AccessMode.READ)
    assert not mode_covers(AccessMode.READ, AccessMode.WRITE)
