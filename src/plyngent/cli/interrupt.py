from __future__ import annotations

import asyncio
import contextlib
import signal
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import FrameType

    type SigHandler = Callable[[int, FrameType | None], None] | int | None

_allow_task_cancel: ContextVar[bool] = ContextVar("allow_task_cancel", default=True)
_reinstall_holder: list[Callable[[], None] | None] = [None]


def allow_task_cancel() -> bool:
    """Whether the CLI SIGINT handler should cancel the in-flight turn task."""
    return _allow_task_cancel.get()


def set_sigint_reinstall(callback: Callable[[], None] | None) -> None:
    """Register how to re-bind SIGINT after a TTY prompt (set by run_cancellable)."""
    _reinstall_holder[0] = callback


@contextmanager
def pause_task_cancel_for_prompt() -> Generator[None]:
    """Disable turn-task cancel during blocking TTY prompts (confirm, etc.).

    Must run on the main thread (signal handlers are main-thread only).
    Restores the default SIGINT handler so ``click.confirm`` can receive
    KeyboardInterrupt / Abort instead of the asyncio turn being cancelled.
    """
    token = _allow_task_cancel.set(False)
    loop_handler_removed = False
    previous: SigHandler = signal.SIG_DFL
    try:
        try:
            loop = asyncio.get_running_loop()
            _ = loop.remove_signal_handler(signal.SIGINT)
            loop_handler_removed = True
        except RuntimeError, NotImplementedError, ValueError:
            loop_handler_removed = False

        try:
            previous = signal.getsignal(signal.SIGINT)
            _ = signal.signal(signal.SIGINT, signal.default_int_handler)
        except ValueError:
            # Not on the main thread — skip OS signal rebinding.
            previous = signal.SIG_DFL
        yield
    finally:
        with contextlib.suppress(ValueError):
            _ = signal.signal(signal.SIGINT, previous)
        reinstall = _reinstall_holder[0]
        if loop_handler_removed and reinstall is not None:
            with contextlib.suppress(RuntimeError, NotImplementedError, ValueError):
                reinstall()
        _allow_task_cancel.reset(token)


async def run_in_prompt_thread[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a blocking TTY prompt off the event loop with cancel paused.

    Pause/SIGINT rebinding happens on the main thread; only the prompt body
    runs in a worker so the asyncio loop stays free and turn cancel is disabled.
    """
    with pause_task_cancel_for_prompt():
        return await asyncio.to_thread(func, *args, **kwargs)
