from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING

import pytest

from plyngent.cli.interrupt import (
    allow_task_cancel,
    install_keyboard_interrupt_sigint,
    pause_task_cancel_for_prompt,
    raise_keyboard_interrupt_sigint,
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


def test_install_keyboard_interrupt_sigint() -> None:
    """Installs a non-default SIGINT handler that raises KeyboardInterrupt.

    The handler must be a distinct function object: ``asyncio.run`` only skips
    installing its own silent-cancel handler when the current handler is not
    ``signal.default_int_handler``.
    """
    install_keyboard_interrupt_sigint()
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not signal.default_int_handler
        assert handler is not signal.SIG_DFL
        assert callable(handler)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)  # type: ignore[call-arg]
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def test_asyncio_run_keeps_keyboard_interrupt_sigint() -> None:
    """Regression: Runner must not replace our handler with its own.

    Previously the first Ctrl+C under ``asyncio.run`` silently cancelled the
    main task (invisible at the REPL prompt) and the second raised
    KeyboardInterrupt, which also cancelled ``memory.close()`` on a Ctrl+D exit.
    With a non-default handler installed first, Runner leaves SIGINT alone.
    """
    import asyncio

    install_keyboard_interrupt_sigint()
    try:

        async def quick() -> int:
            return 7

        assert asyncio.run(quick()) == 7
        assert signal.getsignal(signal.SIGINT) is raise_keyboard_interrupt_sigint
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


async def test_run_cancellable_restores_previous_sigint() -> None:
    """Regression: after a turn SIGINT is restored, not left as SIG_DFL.

    ``loop.remove_signal_handler`` leaves SIG_DFL, which would terminate the
    process on a stray Ctrl+C between turns instead of letting the REPL catch
    a KeyboardInterrupt.
    """
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: None)
        loop.remove_signal_handler(signal.SIGINT)
    except NotImplementedError, RuntimeError, ValueError:
        pytest.skip("asyncio signal handlers not available on this platform")

    previous = signal.getsignal(signal.SIGINT)

    async def quick() -> None:
        return None

    try:
        await run_cancellable(quick())
        assert signal.getsignal(signal.SIGINT) is previous
        assert signal.getsignal(signal.SIGINT) is not signal.SIG_DFL
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
