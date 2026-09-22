from __future__ import annotations

import asyncio
import contextlib
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import click

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import FrameType

    type SigHandler = Callable[[int, FrameType | None], None] | int | None


class InterruptTarget(Protocol):
    """Something a pending SIGINT can cancel (a task, a prompt read, a hint)."""

    def cancel(self) -> object:
        """Cancel the pending work; the return value is ignored.

        Declared loosely (``object``) so ``asyncio.Task.cancel`` — which returns
        ``bool`` and takes an optional message — satisfies it.
        """
        ...


def raise_keyboard_interrupt_sigint(signum: int, frame: FrameType | None) -> None:
    """SIGINT handler: raise KeyboardInterrupt (same effect as the default).

    A distinct function object — deliberately *not*
    ``signal.default_int_handler`` — so that :func:`asyncio.run`'s Runner does
    not recognise it and install its own silent-cancel handler on top.
    """
    del signum, frame
    raise KeyboardInterrupt


def install_keyboard_interrupt_sigint() -> None:
    """Install :func:`raise_keyboard_interrupt_sigint` as the SIGINT handler.

    ``asyncio.run`` (Runner) replaces the SIGINT handler with its own when the
    current one *is* ``signal.default_int_handler``: the first Ctrl+C then
    silently cancels the main task (invisible at the REPL prompt, leaving the
    task marked cancelled) and only the second raises KeyboardInterrupt. A
    later Ctrl+D (EOF) is then cancelled mid-``memory.close()`` and the CLI
    exits with ``Aborted!``. Installing a plain KeyboardInterrupt-raising
    handler first keeps every Ctrl+C a normal KeyboardInterrupt that the CLI
    (prompt, turn-retry, oneshot) already handles deliberately.
    """
    with contextlib.suppress(ValueError):
        _ = signal.signal(signal.SIGINT, raise_keyboard_interrupt_sigint)


# --------------------------------------------------------------------------- #
# SIGINT routing (session scope)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _RouterState:
    """Session SIGINT routing state (one instance, kept in ``_router``)."""

    # Cancel targets, innermost last: a SIGINT cancels the innermost one.
    targets: list[InterruptTarget] = field(default_factory=list)
    installed: bool = False
    # Nesting depth of main-thread blocking reads. A plain counter (not a
    # ContextVar): asyncio.add_signal_handler freezes the ContextVar snapshot at
    # install time, so a ContextVar reset after reinstall would leave SIGINT
    # permanently routed to the wrong place.
    blocking_reads: int = 0


_router = _RouterState()


def sigint_routed() -> bool:
    """Whether the session SIGINT router currently owns SIGINT."""
    return _router.installed


def install_sigint_router() -> bool:
    """Install the session SIGINT handler; False when the loop cannot take it.

    A SIGINT delivered while the event loop is parked in its selector is raised
    in the *loop's* frame: it escapes ``run_until_complete``, the Runner cancels
    the main task and the REPL (or a half-finished shutdown) is left broken —
    the next Ctrl+C then kills the process. Keeps an asyncio-level SIGINT
    handler installed for the whole session instead: it cancels the innermost
    target registered with :func:`sigint_cancels` and never raises.

    Blocking main-thread reads suspend the router via
    :func:`pause_task_cancel_for_prompt`. Platforms without asyncio signal
    handlers (Windows proactor) return False and keep the plain
    KeyboardInterrupt behaviour. Idempotent.
    """
    if _router.installed:
        return True
    if not _add_router_handler():
        return False
    _router.installed = True
    return True


def uninstall_sigint_router() -> None:
    """Drop the session SIGINT handler and restore the KeyboardInterrupt one."""
    if not _router.installed:
        return
    _router.installed = False
    _remove_router_handler()
    _router.targets.clear()
    install_keyboard_interrupt_sigint()


def _add_router_handler() -> bool:
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _route_sigint)
    except RuntimeError, NotImplementedError, ValueError:
        return False
    return True


def _remove_router_handler() -> None:
    with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
        _ = asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)


def _route_sigint() -> None:
    """Cancel the innermost target (none → ignore: never escape the loop)."""
    if _router.targets:
        _ = _router.targets[-1].cancel()


@contextmanager
def sigint_cancels(target: InterruptTarget) -> Generator[None]:
    """Route SIGINT to ``target.cancel()`` while inside (innermost wins)."""
    _router.targets.append(target)
    try:
        yield
    finally:
        for index in range(len(_router.targets) - 1, -1, -1):
            if _router.targets[index] is target:
                del _router.targets[index]
                break


@contextmanager
def pause_task_cancel_for_prompt() -> Generator[None]:
    """Give SIGINT to a blocking *main-thread* read instead of routing it.

    The read (REPL entry prompt, synchronous click confirm/prompt) runs on the
    main thread inside a coroutine, so the resulting KeyboardInterrupt is raised
    in the reader's own frame: its handlers see it, and neither the in-flight
    turn nor the event loop is cancelled. Use it only around blocking reads — an
    ``await`` that parks the loop in its selector must keep the router installed
    (:func:`install_sigint_router`).

    Suspends the router; nested calls restore it when the outermost exits.
    """
    _router.blocking_reads += 1
    resuming = _router.blocking_reads == 1 and _router.installed
    if resuming:
        _remove_router_handler()
    previous: SigHandler = signal.SIG_DFL
    try:
        previous = signal.getsignal(signal.SIGINT)
        _ = signal.signal(signal.SIGINT, raise_keyboard_interrupt_sigint)
    except ValueError:
        # Not on the main thread — signal handlers cannot be rebound here.
        previous = signal.SIG_DFL
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            _ = signal.signal(signal.SIGINT, previous)
        _router.blocking_reads = max(0, _router.blocking_reads - 1)
        if resuming:
            _ = _add_router_handler()


@dataclass(frozen=True, slots=True)
class _CancelHook:
    """SIGINT target for a poll-based prompt read (abort that read)."""

    cancel_read: Callable[[], None]

    def cancel(self) -> None:
        self.cancel_read()


@dataclass(slots=True)
class _WaitingPrompt:
    """SIGINT target for a prompt read that cannot be interrupted.

    A readline read blocks inside the C library on a worker thread; aborting it
    would abandon that thread on stdin, where it would eat the next line the
    human types. Explaining itself once is the best available answer.
    """

    _hinted: bool = False

    def cancel(self) -> None:
        if self._hinted:
            return
        self._hinted = True
        click.secho(
            "Ctrl+C ignored while a prompt is waiting — answer it to continue",
            fg="yellow",
            err=True,
        )


@contextmanager
def off_loop_prompt(cancel: Callable[[], None] | None = None) -> Generator[None]:
    """A human prompt is waiting in a worker thread.

    While inside, SIGINT must not cancel the enclosing turn: poll-based reads
    (``cancel`` hook) abort just that read, readline-based ones ignore it with a
    one-line hint (see :class:`_WaitingPrompt`).
    """
    target: InterruptTarget = _WaitingPrompt() if cancel is None else _CancelHook(cancel)
    with sigint_cancels(target):
        yield
