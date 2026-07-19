from __future__ import annotations

import pytest

from plyngent.tools import (
    WorkspaceError,
    check_command_allowed,
    clear_workspace_root,
    get_workspace_root,
    resolve_path,
    set_command_denylist,
    set_path_denylist,
)


def test_resolve_relative_and_absolute(workspace: object) -> None:
    from pathlib import Path

    assert isinstance(workspace, Path)
    _ = (workspace / "a.txt").write_text("x", encoding="utf-8")
    assert resolve_path("a.txt") == workspace / "a.txt"
    assert resolve_path(workspace / "a.txt") == workspace / "a.txt"


def test_escape_rejected(workspace: object) -> None:
    del workspace
    with pytest.raises(WorkspaceError, match="escapes"):
        _ = resolve_path("../outside")


def test_path_denylist(workspace: object) -> None:
    from pathlib import Path

    assert isinstance(workspace, Path)
    secrets = workspace / "secrets"
    secrets.mkdir()
    _ = (secrets / "key").write_text("k", encoding="utf-8")
    set_path_denylist(["/secrets/"])
    with pytest.raises(WorkspaceError, match="matched '/secrets/'"):
        _ = resolve_path("secrets/key")
    set_path_denylist(None)


def test_command_denylist(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import (
        clear_policy_allowed_commands,
        set_policy_confirm_hook,
    )

    set_policy_confirm_hook(None)
    clear_policy_allowed_commands()
    with pytest.raises(WorkspaceError, match="basename 'rm' is blocked"):
        check_command_allowed(["rm", "-rf", "/"])
    check_command_allowed(["echo", "ok"])
    set_command_denylist(None)


def test_command_denylist_policy_confirm_allow(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import (
        clear_policy_allowed_commands,
        set_policy_confirm_hook,
    )

    calls: list[tuple[str, float]] = []

    def hook(basename: str, argv: object, timeout: float) -> bool:
        del argv
        calls.append((basename, timeout))
        return basename == "sudo"

    set_policy_confirm_hook(hook)
    clear_policy_allowed_commands()
    try:
        check_command_allowed(["sudo", "echo", "test"])
        assert calls == [("sudo", 30.0)]
        # Session grant: second call does not re-prompt.
        check_command_allowed(["sudo", "id"])
        assert len(calls) == 1
    finally:
        set_policy_confirm_hook(None)
        clear_policy_allowed_commands()


def test_command_denylist_policy_confirm_deny(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import (
        clear_policy_allowed_commands,
        set_policy_confirm_hook,
    )

    set_policy_confirm_hook(lambda *_a: False)
    clear_policy_allowed_commands()
    try:
        with pytest.raises(WorkspaceError, match="declined or timed out"):
            check_command_allowed(["sudo", "echo", "x"])
    finally:
        set_policy_confirm_hook(None)
        clear_policy_allowed_commands()


def test_command_denylist_policy_confirm_timeout_value(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import (
        clear_policy_allowed_commands,
        set_policy_confirm_hook,
        set_policy_confirm_timeout,
    )

    seen: list[float] = []

    def hook(basename: str, argv: object, timeout: float) -> bool:
        del basename, argv
        seen.append(timeout)
        return False

    set_policy_confirm_timeout(5.0)
    set_policy_confirm_hook(hook)
    clear_policy_allowed_commands()
    try:
        with pytest.raises(WorkspaceError):
            check_command_allowed(["rm", "x"])
        assert seen == [5.0]
    finally:
        set_policy_confirm_timeout(30.0)
        set_policy_confirm_hook(None)
        clear_policy_allowed_commands()


def test_root_required() -> None:
    clear_workspace_root()
    with pytest.raises(WorkspaceError, match="not set"):
        _ = get_workspace_root()
