"""What an interrupting signal means to the loop: the stop a keyboard's Ctrl+C is, and never a goodbye.

A person pressing Ctrl+C on a terminal answers a prompt, and the prompt hands the key over as a value —
so a run with a keyboard in front of it never sees a signal. A run reading a pipe, or one somebody
signals from another process, is the same intention arriving the other way, and the only place that can
be seen from is a test that raises the signal at itself. What these tests keep is the one thing no story
can: that the handler is installed while the loop runs and taken away when it stops, so a signal after
the frontend is done is the process's own again.
"""

from __future__ import annotations

import asyncio
import os
import signal

from plyngent.cli.app import _a_signal_is_a_stop


def _sent() -> int:
    """Raise the signal a keyboard raises, at this very process."""
    _ = os.kill(os.getpid(), signal.SIGINT)
    return 0


def test_a_signal_is_the_stop_a_keyboard_makes() -> None:
    """One signal, one call: the loop is asked to do what Ctrl+C as a line asks it to do."""
    stopped: list[int] = []

    async def run() -> None:
        with _a_signal_is_a_stop(lambda: stopped.append(1)) as told:
            assert told, "a loop on this platform can be told what SIGINT is"
            _ = await asyncio.to_thread(_sent)
            for _ in range(50):
                if stopped:
                    return
                await asyncio.sleep(0.01)

    was = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        asyncio.run(run())
    finally:
        _ = signal.signal(signal.SIGINT, was)

    assert stopped == [1], "the signal reached the loop, once"


def test_the_signal_is_the_processes_own_again_afterwards() -> None:
    """A handler that outlived the loop would still be answering signals nothing is waiting for."""

    async def run() -> None:
        with _a_signal_is_a_stop(lambda: None):
            pass

    asyncio.run(run())

    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, (
        "the loop's handler is taken away with the loop it was installed on"
    )


def test_a_loop_that_cannot_be_told_says_so() -> None:
    """The answer is a value, so the caller keeps an exit code for the case nothing was installed.

    A caller that is not on a loop at all is the same answer: there is nowhere to install a handler, and
    a signal raised here is the process's own rather than a stop the loop can be asked for.
    """
    with _a_signal_is_a_stop(lambda: None) as told:
        assert told is False
