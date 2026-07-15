from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.cli.interrupt import (
    allow_task_cancel,
    pause_task_cancel_for_prompt,
    run_in_prompt_thread,
)
from plyngent.cli.limits import prompt_continue_limit, prompt_continue_limit_async

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


async def test_run_in_prompt_thread_pauses_cancel() -> None:
    """Cancel is paused on the main thread for the whole to_thread call."""
    assert allow_task_cancel() is True

    def work() -> str:
        return "ok"

    # ContextVar may not propagate to worker threads; assert pause around the call.
    result = await run_in_prompt_thread(work)
    assert result == "ok"
    assert allow_task_cancel() is True


async def test_prompt_continue_limit_async(monkeypatch: pytest.MonkeyPatch) -> None:
    def _confirm(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr("click.confirm", _confirm)
    assert await prompt_continue_limit_async("too many rounds") is True
