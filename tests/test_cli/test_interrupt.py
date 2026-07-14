from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.cli.interrupt import allow_task_cancel, pause_task_cancel_for_prompt
from plyngent.cli.limits import prompt_continue_limit

if TYPE_CHECKING:
    import pytest


def test_pause_task_cancel_for_prompt() -> None:
    assert allow_task_cancel() is True
    with pause_task_cancel_for_prompt():
        assert allow_task_cancel() is False
    assert allow_task_cancel() is True


def test_prompt_continue_limit_under_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    def _confirm(*_a: object, **_k: object) -> bool:
        assert allow_task_cancel() is False
        return True

    monkeypatch.setattr("click.confirm", _confirm)
    assert prompt_continue_limit("too many rounds") is True
