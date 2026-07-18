from __future__ import annotations

import asyncio
import contextlib
import signal
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import FrameType

    type SigHandler = Callable[[int, FrameType | None], None] | int | None

# Depth of nested pause_task_cancel_for_prompt. Must be a plain int (not ContextVar):
# asyncio.add_signal_handler freezes the ContextVar snapshot at install time, so a
# ContextVar reset after reinstall would leave SIGINT permanently non-cancelling.
_prompt_depth: int = 0
_reinstall_holder: list[Callable[[], None] | None] = [None]


def allow_task_cancel() -> bool:
    """Whether the CLI SIGINT handler should cancel the in-flight turn task."""
    return _prompt_depth == 0


def set_sigint_reinstall(callback: Callable[[], None] | None) -> None:
    """Register how to re-bind SIGINT after a TTY prompt (set by run_cancellable)."""
    _reinstall_holder[0] = callback


@contextmanager
def pause_task_cancel_for_prompt() -> Generator[None]:
    """Disable turn-task cancel during blocking TTY prompts (confirm, etc.).

    Must run on the main thread (signal handlers are main-thread only).
    Restores the default SIGINT handler so ``click.confirm`` can receive
    KeyboardInterrupt / Abort instead of the asyncio turn being cancelled.

    Nested prompts are supported via a depth counter. The asyncio SIGINT
    handler is reinstalled only after depth returns to 0, and only after
    cancel is re-enabled, so the handler never freezes ``allow=False``.
    """

    global _prompt_depth  # noqa: PLW0603
    _prompt_depth += 1
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
        _prompt_depth = max(0, _prompt_depth - 1)
        # Reinstall only when fully out of prompts and cancel is allowed again.
        # Order matters: decrement depth first so allow_task_cancel() is True
        # before reinstall (and handlers never freeze allow=False).
        if _prompt_depth == 0 and loop_handler_removed:
            reinstall = _reinstall_holder[0]
            if reinstall is not None:
                with contextlib.suppress(RuntimeError, NotImplementedError, ValueError):
                    reinstall()


async def run_in_prompt_thread[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a blocking TTY prompt off the event loop with cancel paused.

    Pause/SIGINT rebinding happens on the main thread; only the prompt body
    runs in a worker so the asyncio loop stays free and turn cancel is disabled.
    """
    with pause_task_cancel_for_prompt():
        return await asyncio.to_thread(func, *args, **kwargs)
