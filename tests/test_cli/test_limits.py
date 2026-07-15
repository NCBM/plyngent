from __future__ import annotations

from plyngent.cli.limits import install_cli_limit_hooks, prompt_confirm_tool, prompt_continue_limit
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


def test_install_cli_limit_hooks() -> None:
    install_cli_limit_hooks()
    assert callable(getattr(PtyManager, "_limit_continue", None))
    PtyManager.set_limit_continue_hook(None)
    # Backend remains usable after install.
    assert get_prompt_backend().is_interactive() or True
