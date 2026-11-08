"""Grants are three tables that answer together, and the source of each answer is part of it."""

from __future__ import annotations

import pytest

from plyngent.kernel.access import MAX_GRANTS, Access, AccessMode, GrantRow
from plyngent.kernel.workspace import OutsideWorkspace, Workspace, WorkspaceConfig


def test_a_grant_covers_what_is_under_it_and_nothing_else(tmp_path) -> None:
    access = Access()
    shared = tmp_path / "shared"
    _ = access.grant(shared, AccessMode.READ, source="config")

    assert access.mode(shared / "a.txt") is AccessMode.READ
    assert access.mode(shared) is AccessMode.READ
    assert access.mode(tmp_path / "elsewhere") is None
    assert access.covers(shared / "a.txt", AccessMode.READ)
    assert not access.covers(shared / "a.txt", AccessMode.WRITE)


def test_the_widest_of_the_three_tables_wins(tmp_path) -> None:
    access = Access()
    shared = tmp_path / "shared"
    nested = shared / "nested"
    _ = access.grant(shared, AccessMode.READ, source="config")
    _ = access.grant(nested, AccessMode.EXEC, source="process")
    session = {shared: AccessMode.WRITE}

    assert access.mode(shared / "a", extra=session) is AccessMode.WRITE, "the session grant beats the config one"
    assert access.mode(shared / "a") is AccessMode.READ, "without the session table only the config answers"
    assert access.mode(nested / "run.sh", extra=session) is AccessMode.EXEC
    assert access.mode(nested / "run.sh") is AccessMode.EXEC


def test_a_read_only_root_is_readable_and_is_not_a_grant(tmp_path) -> None:
    skills = tmp_path / "skills"
    skills.mkdir()
    access = Access()
    _ = access.use_read_only([skills, tmp_path / "gone"])

    assert access.read_only == (skills.resolve(),), "a directory that is not there holds nothing to read"
    assert access.mode(skills / "SKILL.md") is AccessMode.READ
    assert not access.covers(skills / "SKILL.md", AccessMode.WRITE)
    assert access.rows() == (), "a read-only root is not something a human granted"
    assert access.count() == 0


def test_rows_name_the_source_and_the_process_comes_first(tmp_path) -> None:
    access = Access()
    config_path = access.grant(tmp_path / "config-dir", AccessMode.READ, source="config")
    process_path = access.grant(tmp_path / "process-dir", AccessMode.WRITE, source="process")

    assert access.rows() == (
        GrantRow(process_path, AccessMode.WRITE, "process"),
        GrantRow(config_path, AccessMode.READ, "config"),
    )
    assert access.count() == 2
    assert MAX_GRANTS == 32


def test_a_process_grant_can_be_taken_back_and_a_config_grant_cannot(tmp_path) -> None:
    access = Access()
    path = access.grant(tmp_path, AccessMode.WRITE, source="config")

    assert access.revoke(path, source="config") is False, "the config file owns that answer"
    assert access.mode(path) is AccessMode.WRITE

    _ = access.grant(path, AccessMode.WRITE, source="process")
    assert access.revoke(path, source="process") is True
    assert access.revoke(path, source="process") is False
    assert access.mode(path) is AccessMode.WRITE, "the config grant remains"

    access.clear(source="process")
    assert access.count() == 1


def test_config_entries_install_and_the_unusable_ones_are_reported(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    access = Access()

    applied, skipped = access.install_config(
        {
            str(outside): "read",
            str(tmp_path / "missing"): "write",
            str(tmp_path / "bogus-mode"): "admin",
        }
    )

    assert applied == (outside.resolve(),)
    assert sorted(skipped) == sorted([str(tmp_path / "missing"), str(tmp_path / "bogus-mode")])
    assert access.mode(outside / "a") is AccessMode.READ
    assert access.count() == 1
    assert access.mode(outside / "a") is AccessMode.READ
    assert access.count() == 1


def test_a_new_config_value_replaces_the_grants_and_the_paths(tmp_path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    workspace = Workspace(tmp_path / "ws", WorkspaceConfig(allow_paths=((str(shared), "write"),)))

    assert workspace.resolve(shared / "a.txt", AccessMode.WRITE) == shared / "a.txt"

    workspace.configure(WorkspaceConfig())

    assert workspace.access.count() == 0
    with pytest.raises(OutsideWorkspace):
        _ = workspace.resolve(shared / "a.txt")


def test_a_session_table_arrives_as_an_argument(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    workspace = Workspace(root)
    session_grants = {shared.resolve(): AccessMode.READ}

    with pytest.raises(OutsideWorkspace):
        _ = workspace.resolve(shared / "a.txt", AccessMode.READ)
    assert workspace.resolve(shared / "a.txt", AccessMode.READ, grants=session_grants) == shared / "a.txt"
