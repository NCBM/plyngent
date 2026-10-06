from __future__ import annotations

import pytest

from plyngent.tools import (
    AccessMode,
    WorkspaceError,
    active_workspace_policy,
    check_command_allowed,
    get_workspace_root,
    mode_covers,
    parse_access_mode,
    resolve_path,
    set_command_denylist,
    set_path_denylist,
)
from plyngent.tools.context import SessionState, bind_session


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


def test_escape_error_hints_request_tool(workspace: object) -> None:
    del workspace
    with pytest.raises(WorkspaceError, match="request_directory_access"):
        _ = resolve_path("../outside")


def test_access_mode_parse_and_covers() -> None:
    assert parse_access_mode(" Read ") is AccessMode.READ
    assert parse_access_mode("EXEC") is AccessMode.EXEC
    assert parse_access_mode("bogus") is None
    assert mode_covers(AccessMode.EXEC, AccessMode.WRITE)
    assert mode_covers(AccessMode.WRITE, AccessMode.WRITE)
    assert not mode_covers(AccessMode.READ, AccessMode.WRITE)


def test_config_grant_modes(workspace: object, tmp_path_factory: pytest.TempPathFactory) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("granted")
    target = outside / "data.txt"
    _ = target.write_text("x", encoding="utf-8")
    policy = active_workspace_policy()
    policy.config_allow[outside] = AccessMode.READ
    try:
        assert resolve_path(str(target), required=AccessMode.READ) == target
        with pytest.raises(WorkspaceError, match="escapes"):
            _ = resolve_path(str(target), required=AccessMode.WRITE)
        policy.config_allow[outside] = AccessMode.WRITE
        assert resolve_path(str(target), required=AccessMode.WRITE) == target
        with pytest.raises(WorkspaceError, match="escapes"):
            _ = resolve_path(str(target), required=AccessMode.EXEC)
    finally:
        policy.config_allow.clear()


def test_session_grant_covers_subtree(workspace: object, tmp_path_factory: pytest.TempPathFactory) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("session")
    nested = outside / "a" / "b"
    nested.mkdir(parents=True)
    with bind_session(SessionState(access_grants={outside: AccessMode.EXEC})):
        assert resolve_path(str(nested), required=AccessMode.EXEC) == nested
    with pytest.raises(WorkspaceError, match="escapes"):
        _ = resolve_path(str(nested), required=AccessMode.READ)


def test_file_grant_is_exact_path(workspace: object, tmp_path_factory: pytest.TempPathFactory) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("file-grant")
    granted = outside / "one.txt"
    sibling = outside / "two.txt"
    _ = granted.write_text("1", encoding="utf-8")
    _ = sibling.write_text("2", encoding="utf-8")
    policy = active_workspace_policy()
    policy.config_allow[granted] = AccessMode.READ
    try:
        assert resolve_path(str(granted), required=AccessMode.READ) == granted
        with pytest.raises(WorkspaceError, match="escapes"):
            _ = resolve_path(str(sibling), required=AccessMode.READ)
        # A file grant does not cover its parent directory.
        with pytest.raises(WorkspaceError, match="escapes"):
            _ = resolve_path(str(outside), required=AccessMode.READ)
    finally:
        policy.config_allow.clear()


def test_denylist_wins_over_grant(workspace: object, tmp_path_factory: pytest.TempPathFactory) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("denied")
    secret = outside / "secret"
    secret.mkdir()
    key = secret / "key"
    _ = key.write_text("k", encoding="utf-8")
    policy = active_workspace_policy()
    policy.config_allow[outside] = AccessMode.WRITE
    set_path_denylist(["/secret/"])
    try:
        with pytest.raises(WorkspaceError, match="matched '/secret/'"):
            _ = resolve_path(str(key), required=AccessMode.READ)
    finally:
        set_path_denylist(None)
        policy.config_allow.clear()


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


def test_command_denylist_sees_the_program_behind_value_free_flags(workspace: object) -> None:
    """`exec -c` / `command -p` take no value, so the program stays visible."""
    del workspace
    from plyngent.tools.workspace import clear_policy_allowed_commands, set_policy_confirm_hook

    set_policy_confirm_hook(None)
    clear_policy_allowed_commands()
    set_command_denylist(None)
    try:
        for argv in (
            ["exec", "-c", "rm", "-rf", "/"],
            ["command", "-p", "rm", "-rf", "/"],
            ["command", "-v", "rm"],
        ):
            with pytest.raises(WorkspaceError, match="basename 'rm' is blocked"):
                check_command_allowed(list(argv))
        # ``init`` here is a git subcommand, not the denylisted program: the
        # command resolves to ``git`` and stays allowed.
        check_command_allowed(["exec", "-c", "git", "-c", "init.defaultBranch=main", "init"])
    finally:
        clear_policy_allowed_commands()
        set_command_denylist(None)


def test_command_denylist_reaches_through_wrappers(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import clear_policy_allowed_commands, set_policy_confirm_hook

    set_policy_confirm_hook(None)
    clear_policy_allowed_commands()
    try:
        with pytest.raises(WorkspaceError, match="basename 'rm' is blocked"):
            check_command_allowed(["env", "FOO=1", "rm", "-rf", "/"])
        with pytest.raises(WorkspaceError, match="basename 'rm' is blocked"):
            check_command_allowed(["pdm", "run", "rm", "-rf", "/"])
        with pytest.raises(WorkspaceError, match="basename 'rm' is blocked"):
            check_command_allowed(["timeout", "5", "rm", "-rf", "/"])
        # The outermost denylisted program is the one named (``sudo`` here).
        with pytest.raises(WorkspaceError, match="basename 'sudo' is blocked"):
            check_command_allowed(["sudo", "-u", "root", "rm", "-rf", "/"])
        # Wrapped programs that are not denylisted stay allowed.
        check_command_allowed(["pdm", "run", "pytest", "-x"])
        check_command_allowed(["env", "FOO=1", "echo", "ok"])
    finally:
        clear_policy_allowed_commands()


def test_command_denylist_matches_lowercased_basename(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import clear_policy_allowed_commands, set_policy_confirm_hook

    set_policy_confirm_hook(None)
    clear_policy_allowed_commands()
    try:
        with pytest.raises(WorkspaceError, match="basename 'rm' is blocked"):
            check_command_allowed(["RM", "-rf", "/"])
        with pytest.raises(WorkspaceError, match="basename 'rm' is blocked"):
            check_command_allowed(["C:\\Tools\\rm.exe", "-rf", "/"])
    finally:
        clear_policy_allowed_commands()


def test_command_denylist_prompts_for_the_wrapped_program(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import clear_policy_allowed_commands, set_policy_confirm_hook

    asked: list[str] = []

    def hook(basename: str, argv: object, timeout: float) -> bool:
        del argv, timeout
        asked.append(basename)
        return True

    set_policy_confirm_hook(hook)
    clear_policy_allowed_commands()
    try:
        check_command_allowed(["pdm", "run", "rm", "-rf", "build"])
        assert asked == ["rm"]
        # Session grant: the same wrapped program does not re-prompt.
        check_command_allowed(["pdm", "run", "rm", "-rf", "dist"])
        assert asked == ["rm"]
    finally:
        set_policy_confirm_hook(None)
        clear_policy_allowed_commands()


def test_match_command_denylist_entries() -> None:
    from plyngent.tools import match_command_denylist

    denylist = frozenset({"rm", "systemctl reboot poweroff", "loginctl kill-session"})
    hits = match_command_denylist(["sudo", "rm", "-rf", "/"], denylist)
    assert [(hit.program, hit.keyword) for hit in hits] == [("rm", None)]
    assert [hit.program for hit in match_command_denylist(["env", "FOO=1", "rm", "x"], denylist)] == ["rm"]
    hits = match_command_denylist(["sudo", "sudo", "rm", "x"], frozenset({"sudo", "rm"}))
    assert [hit.program for hit in hits] == ["sudo", "rm"]
    # Subcommand entries match the program's own arguments, targets included.
    hits = match_command_denylist(["systemctl", "reboot"], denylist)
    assert [(hit.program, hit.keyword, hit.entry) for hit in hits] == [
        ("systemctl", "reboot", "systemctl reboot poweroff")
    ]
    assert match_command_denylist(["systemctl", "start", "poweroff.target"], denylist)[0].keyword == "poweroff"
    assert match_command_denylist(["pdm", "run", "loginctl", "kill-session"], denylist)[0].program == "loginctl"
    # Read-only and unrelated calls stay clean.
    assert match_command_denylist(["systemctl", "status", "sshd"], denylist) == []
    assert match_command_denylist(["loginctl", "list-sessions"], denylist) == []


def test_command_denylist_power_subcommands(workspace: object) -> None:
    del workspace
    from plyngent.tools.workspace import clear_policy_allowed_commands, set_policy_confirm_hook

    set_policy_confirm_hook(None)
    clear_policy_allowed_commands()
    try:
        with pytest.raises(WorkspaceError, match="basename 'systemctl' is blocked: 'reboot'"):
            check_command_allowed(["systemctl", "reboot"])
        with pytest.raises(WorkspaceError, match="basename 'systemctl' is blocked: 'reboot'"):
            check_command_allowed(["systemctl", "start", "reboot.target"])
        with pytest.raises(WorkspaceError, match="basename 'systemctl' is blocked: 'poweroff'"):
            check_command_allowed(["systemctl", "--user", "poweroff"])
        with pytest.raises(WorkspaceError, match="basename 'systemctl' is blocked: 'emergency'"):
            check_command_allowed(["timeout", "3", "systemctl", "emergency"])
        with pytest.raises(WorkspaceError, match="basename 'loginctl' is blocked: 'kill-session'"):
            check_command_allowed(["loginctl", "kill-session", "3"])
        with pytest.raises(WorkspaceError, match="basename 'pm-suspend' is blocked"):
            check_command_allowed(["pm-suspend"])
        with pytest.raises(WorkspaceError, match="basename 'run0' is blocked"):
            check_command_allowed(["run0", "id"])
        with pytest.raises(WorkspaceError, match="basename 'systemd-run' is blocked"):
            check_command_allowed(["systemd-run", "--user", "echo", "hi"])
        # The outermost denylisted program is reported first.
        with pytest.raises(WorkspaceError, match="basename 'sudo' is blocked"):
            check_command_allowed(["sudo", "systemctl", "reboot"])
        # Read-only systemd calls stay allowed.
        check_command_allowed(["systemctl", "status", "sshd"])
        check_command_allowed(["systemctl", "list-units", "--type=service"])
        check_command_allowed(["systemctl", "show", "-p", "CanReboot"])
        check_command_allowed(["loginctl", "list-sessions"])
    finally:
        clear_policy_allowed_commands()


def test_root_required() -> None:
    from plyngent.tools.context import InstanceState, bind_instance

    with bind_instance(InstanceState()), pytest.raises(WorkspaceError, match="workspace root is not set"):
        _ = get_workspace_root()


def test_unbound_instance_errors() -> None:
    with pytest.raises(WorkspaceError, match="instance state is not bound"):
        _ = get_workspace_root()
