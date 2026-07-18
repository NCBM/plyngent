from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING

import pytest

from plyngent.cli.interrupt import (
    allow_task_cancel,
    pause_task_cancel_for_prompt,
    run_in_prompt_thread,
    set_sigint_reinstall,
)
from plyngent.cli.limits import prompt_continue_limit, prompt_continue_limit_async
from plyngent.cli.retry import run_cancellable

if TYPE_CHECKING:
    pass


def test_pause_task_cancel_for_prompt() -> None:
    assert allow_task_cancel() is True
    with pause_task_cancel_for_prompt():
        assert allow_task_cancel() is False
    assert allow_task_cancel() is True


def test_nested_pause_depth() -> None:
    assert allow_task_cancel() is True
    with pause_task_cancel_for_prompt():
        assert allow_task_cancel() is False
        with pause_task_cancel_for_prompt():
            assert allow_task_cancel() is False
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

    result = await run_in_prompt_thread(work)
    assert result == "ok"
    assert allow_task_cancel() is True


async def test_prompt_continue_limit_async(monkeypatch: pytest.MonkeyPatch) -> None:
    def _confirm(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr("click.confirm", _confirm)
    assert await prompt_continue_limit_async("too many rounds") is True


async def test_sigint_cancels_after_prompt_pause() -> None:
    """Regression: after a mid-turn prompt, SIGINT must still cancel the turn.

    Previously, reinstalling the asyncio SIGINT handler while
    allow_task_cancel was still False froze that value into the callback
    forever (ContextVar snapshot at add_signal_handler).
    """
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: None)
        loop.remove_signal_handler(signal.SIGINT)
    except NotImplementedError, RuntimeError, ValueError:
        pytest.skip("asyncio signal handlers not available on this platform")

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang() -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def turn() -> None:
        # Simulate soft-confirm pause mid-turn, then keep streaming.
        with pause_task_cancel_for_prompt():
            assert allow_task_cancel() is False
        assert allow_task_cancel() is True
        await hang()

    task = asyncio.create_task(run_cancellable(turn()))
    await started.wait()
    # Give run_cancellable a tick to install the handler after the pause reinstall.
    await asyncio.sleep(0)
    assert allow_task_cancel() is True
    # Deliver SIGINT the same way Ctrl+C does under asyncio.
    loop.call_soon(lambda: None)  # ensure loop is processing
    # Invoke the process signal: raise SIGINT to this process.
    import os

    os.kill(os.getpid(), signal.SIGINT)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    # Cleanup any leftover reinstall hook from run_cancellable
    set_sigint_reinstall(None)
