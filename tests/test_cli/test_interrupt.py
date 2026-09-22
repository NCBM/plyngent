from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from typing import TYPE_CHECKING

import pytest

from plyngent.cli.interrupt import (
    install_keyboard_interrupt_sigint,
    install_sigint_router,
    off_loop_prompt,
    pause_task_cancel_for_prompt,
    raise_keyboard_interrupt_sigint,
    sigint_cancels,
    sigint_routed,
    uninstall_sigint_router,
)
from plyngent.cli.limits import prompt_continue_limit
from plyngent.cli.retry import run_cancellable, sleep_cancellable
from plyngent.prompting import temporary_backend
from tests.test_prompting import ScriptedBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest as pytest_module


class FakeTarget:
    """Minimal ``InterruptTarget`` recording its cancels."""

    def __init__(self, log: list[str], name: str = "cancel") -> None:
        self.log = log
        self.name = name

    def cancel(self) -> None:
        self.log.append(self.name)


@pytest.fixture
async def router() -> AsyncIterator[None]:
    """Install the session SIGINT router; restore the CLI handler afterwards."""
    install_keyboard_interrupt_sigint()
    _ = install_sigint_router()
    try:
        yield
    finally:
        uninstall_sigint_router()
        signal.signal(signal.SIGINT, signal.default_int_handler)


async def _deliver_sigint() -> None:
    """Send SIGINT to this process and let the loop run its signal callback.

    asyncio's signal wakeup is only noticed while the loop parks in its
    selector, so a real (short) wait is needed — ``sleep(0)`` never parks and
    the callback would not run at all.
    """
    os.kill(os.getpid(), signal.SIGINT)
    await asyncio.sleep(0.05)


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
    install_keyboard_interrupt_sigint()
    try:

        async def quick() -> int:
            return 7

        assert asyncio.run(quick()) == 7
        assert signal.getsignal(signal.SIGINT) is raise_keyboard_interrupt_sigint
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


async def test_sigint_cancels_registered_target(router: None) -> None:
    """SIGINT cancels the target registered around the awaited turn task."""
    del router
    cancelled = asyncio.Event()

    async def hang() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(hang())
    await asyncio.sleep(0)
    with sigint_cancels(task):
        os.kill(os.getpid(), signal.SIGINT)
        # Waiting on the effect (not on a fixed sleep) keeps this deterministic.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    assert cancelled.is_set()


async def test_sigint_while_loop_is_idle_does_not_escape(router: None) -> None:
    """Regression: a stray Ctrl+C while the loop is parked in its selector.

    The interrupt used to be raised in the loop's own frame (not inside any
    coroutine), escaping ``run_until_complete``: the Runner cancelled the main
    task and the REPL was left half-shut-down, so the next Ctrl+C killed the
    process instead of returning to the prompt.
    """
    del router

    def _send() -> None:
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    sender = threading.Thread(target=_send)
    sender.start()
    try:
        await asyncio.sleep(0.2)  # parked in the selector when SIGINT lands
    finally:
        sender.join()
    task = asyncio.current_task()
    assert task is not None
    assert task.cancelling() == 0
    assert not task.cancelled()


async def test_sigint_cancels_innermost_target_only(router: None) -> None:
    """Nested targets: the innermost (the active turn or prompt) wins."""
    del router
    outer: list[str] = []
    inner: list[str] = []
    with sigint_cancels(FakeTarget(outer, "outer")):
        await _deliver_sigint()
        assert outer == ["outer"]
        with sigint_cancels(FakeTarget(inner, "inner")):
            await _deliver_sigint()
            assert inner == ["inner"]
        assert outer == ["outer"]  # the nested signal never reached it


async def test_sigint_target_stops_after_context_exit(router: None) -> None:
    """A popped target no longer sees SIGINT (and an empty stack ignores it)."""
    del router
    calls: list[str] = []
    with sigint_cancels(FakeTarget(calls)):
        await _deliver_sigint()
    assert calls == ["cancel"]
    await _deliver_sigint()
    assert calls == ["cancel"]


async def test_off_loop_prompt_keeps_turn_alive(
    router: None,
    capsys: pytest_module.CaptureFixture[str],
) -> None:
    """A prompt's SIGINT target shields the turn; afterwards the turn cancels.

    Regression (badcb3c, revisited): the router resolves the innermost target
    per signal instead of freezing cancel state into the handler, so a mid-turn
    prompt neither aborts the turn nor freezes turn-cancel.
    """
    del router
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang() -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(hang())
    await started.wait()
    with sigint_cancels(task):
        with off_loop_prompt():
            await _deliver_sigint()
            assert not task.cancelling()
        err = capsys.readouterr().err
        assert "Ctrl+C ignored while a prompt is waiting" in err
        # The prompt returned (outer context still active): SIGINT cancels now.
        os.kill(os.getpid(), signal.SIGINT)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    assert cancelled.is_set()


async def test_off_loop_prompt_cancel_hook_is_preferred(router: None) -> None:
    """A poll-based prompt read gets cancelled instead of the enclosing turn."""
    del router
    hooks: list[str] = []
    turn_calls: list[str] = []
    with sigint_cancels(FakeTarget(turn_calls, "turn")), off_loop_prompt(lambda: hooks.append("read")):
        await _deliver_sigint()
    assert hooks == ["read"]
    assert turn_calls == []


async def test_pause_task_cancel_for_prompt_swaps_sigint(router: None) -> None:
    """A main-thread blocking read takes SIGINT as a KeyboardInterrupt."""
    del router
    routed = signal.getsignal(signal.SIGINT)
    assert routed is not raise_keyboard_interrupt_sigint
    with pause_task_cancel_for_prompt():
        assert signal.getsignal(signal.SIGINT) is raise_keyboard_interrupt_sigint
        with pause_task_cancel_for_prompt():
            assert signal.getsignal(signal.SIGINT) is raise_keyboard_interrupt_sigint
        # Inner exit must not resume the router while the outer read blocks.
        assert signal.getsignal(signal.SIGINT) is raise_keyboard_interrupt_sigint
    assert signal.getsignal(signal.SIGINT) is routed


async def test_prompt_continue_limit_under_pause(
    router: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synchronous prompts read on the main thread, where SIGINT is a KI."""
    del router
    backend = ScriptedBackend(["y"])

    def _read_line(prompt: str, *, default: str | None = None, completions: object = None) -> str:
        del prompt, default, completions
        assert signal.getsignal(signal.SIGINT) is raise_keyboard_interrupt_sigint
        return "y"

    monkeypatch.setattr(backend, "read_line", _read_line)
    with temporary_backend(backend):
        assert prompt_continue_limit("too many rounds") is True


async def test_run_cancellable_keeps_router_installed(router: None) -> None:
    """Regression: after a turn SIGINT must not be left as SIG_DFL.

    ``loop.remove_signal_handler`` leaves SIG_DFL, which would terminate the
    process on a stray Ctrl+C between turns; the session router stays installed.
    """
    del router

    async def quick() -> None:
        return None

    await run_cancellable(quick())
    assert sigint_routed()
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_DFL
    assert signal.getsignal(signal.SIGINT) is not signal.default_int_handler


async def test_sleep_cancellable_cancelled_by_sigint(router: None) -> None:
    """The auto-retry countdown is cancelled by Ctrl+C (was a teardown window)."""
    del router
    waiter = asyncio.create_task(sleep_cancellable(30))
    await asyncio.sleep(0)
    os.kill(os.getpid(), signal.SIGINT)
    assert await asyncio.wait_for(waiter, timeout=5) is False
