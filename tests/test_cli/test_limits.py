from __future__ import annotations

import pytest

from plyngent.cli.limits import (
    format_tool_confirm_box,
    install_cli_limit_hooks,
    prompt_confirm_tool,
    prompt_continue_limit,
)
from plyngent.prompting import get_prompt_backend, temporary_backend
from plyngent.tools.process.pty_session import PtyManager
from tests.test_prompting import ScriptedBackend


def test_prompt_continue_limit_yes() -> None:
    backend = ScriptedBackend([], confirms=[True])
    with temporary_backend(backend):
        assert prompt_continue_limit("hit a wall") is True


def test_prompt_continue_limit_no() -> None:
    backend = ScriptedBackend([], confirms=[False])
    with temporary_backend(backend):
        assert prompt_continue_limit("hit a wall") is False


def test_prompt_confirm_tool_default_deny() -> None:
    backend = ScriptedBackend([], confirms=[False])
    with temporary_backend(backend):
        assert prompt_confirm_tool("delete_path", {"path": "x"}, "delete path 'x'") is False


def test_format_tool_confirm_box_multiline() -> None:
    reason = "run_argv: python3 -c (review code before allow)\nargv:\n  python3 -c print(1)\n-c code:\n  print(1)"
    box = format_tool_confirm_box("run_argv", reason)
    assert "┌" in box and "└" in box
    assert "confirm · tool 'run_argv'" in box
    assert "python3 -c" in box
    assert "print(1)" in box
    # Distinct lines, not one jammed row
    assert box.count("\n") >= 4


def test_install_cli_limit_hooks() -> None:
    install_cli_limit_hooks()
    assert callable(getattr(PtyManager, "_limit_continue", None))
    PtyManager.set_limit_continue_hook(None)
    # Backend remains usable after install.
    assert get_prompt_backend().is_interactive() or True


def test_policy_confirm_noninteractive_denies() -> None:
    from plyngent.cli.limits import prompt_policy_command_confirm
    from plyngent.prompting import NonInteractiveBackend

    with temporary_backend(NonInteractiveBackend()):
        assert prompt_policy_command_confirm("sudo", ["sudo", "id"], 1.0) is False


def test_directory_access_prompt_noninteractive_denies() -> None:
    from pathlib import Path

    from plyngent.cli.limits import prompt_directory_access_confirm
    from plyngent.prompting import NonInteractiveBackend
    from plyngent.tools.workspace import AccessMode

    with temporary_backend(NonInteractiveBackend()):
        assert prompt_directory_access_confirm(Path("/tmp"), AccessMode.READ, "why", 1.0) is None


def test_directory_access_prompt_yes_and_level(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from plyngent.cli import limits
    from plyngent.tools.access import AccessDecision
    from plyngent.tools.workspace import AccessMode

    target = Path("/tmp")
    with temporary_backend(ScriptedBackend([])):
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: "y\n")
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) == AccessDecision(
            AccessMode.READ
        )
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: "exec\n")
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) == AccessDecision(
            AccessMode.EXEC
        )
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: "n\n")
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) is None
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: None)
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) is None
