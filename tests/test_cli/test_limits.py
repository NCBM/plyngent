from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.cli.limits import install_cli_limit_hooks, prompt_confirm_tool, prompt_continue_limit
from plyngent.tools.process.pty_session import PtyManager

if TYPE_CHECKING:
    import pytest


def test_prompt_continue_limit_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _confirm(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr("click.confirm", _confirm)
    assert prompt_continue_limit("hit a wall") is True


def test_prompt_continue_limit_no(monkeypatch: pytest.MonkeyPatch) -> None:
    def _confirm(*_a: object, **_k: object) -> bool:
        return False

    monkeypatch.setattr("click.confirm", _confirm)
    assert prompt_continue_limit("hit a wall") is False


def test_prompt_confirm_tool_default_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _confirm(message: str, **kwargs: object) -> bool:
        seen["message"] = message
        seen["default"] = kwargs.get("default")
        return False

    monkeypatch.setattr("click.confirm", _confirm)
    assert prompt_confirm_tool("delete_path", {"path": "x"}, "delete path 'x'") is False
    assert seen["default"] is False


def test_install_cli_limit_hooks() -> None:
    install_cli_limit_hooks()
    # Hook is installed process-wide for the CLI session.
    assert callable(getattr(PtyManager, "_limit_continue", None))
    PtyManager.set_limit_continue_hook(None)
