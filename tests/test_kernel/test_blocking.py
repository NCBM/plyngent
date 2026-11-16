"""What a cancellation can and cannot reach: a thread is on the books even though it cannot be stopped.

No story can say this. From outside, a turn that was stopped looks the same whether the file read under
it was joined or abandoned — the difference is a thread still running after the history was written, and
the only place that is visible is here, where the test can hold the thread open and look.

The rule being pinned is §0.13's: what a stop ends is the *waiting*, and a call that was mid-flight still
owes its result to the history. A blocking call therefore registers the thread it started, so
:meth:`~plyngent.kernel.cancel.CancelToken.settle` waits for the work rather than for the task.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from plyngent.kernel.cancel import Cancelled, CancelToken
from plyngent.wire.scope import CancelReason

#: How long the call under test pretends to be reading something. Long enough that a stop lands in the
#: middle of it, short enough that waiting for it costs nothing worth counting.
SLOW_SECONDS = 0.2


async def test_a_stopped_call_waits_for_the_thread_it_started() -> None:
    """The thread cannot be stopped, so what a stop owes is waiting for it — and that is ``settle``."""
    scope = CancelToken().at("turn", "call")
    started = threading.Event()
    finished = threading.Event()

    def reader() -> str:
        started.set()
        time.sleep(SLOW_SECONDS)
        finished.set()
        return "read"

    running = asyncio.create_task(scope.guard(scope.blocking(reader)))
    _ = await asyncio.to_thread(started.wait, 5.0)
    _ = scope.fire(CancelReason.USER)

    with pytest.raises(Cancelled):
        await running
    assert not finished.is_set(), "a thread is not something a cancellation reaches"

    assert await scope.settle() == 1, "the call it started is what settling waits for"
    assert finished.is_set(), "and settling is over only once the thread is"


async def test_a_stopped_call_that_has_not_begun_never_starts_a_thread() -> None:
    """A call under a scope that is already gone has not begun, so there is nothing to wait for."""
    scope = CancelToken().at("turn", "call")
    ran = threading.Event()
    _ = scope.fire(CancelReason.USER)

    with pytest.raises(Cancelled):
        await scope.blocking(ran.set)

    assert not ran.is_set(), "the work was never handed to a thread at all"


async def test_a_call_that_finished_is_not_waited_for_again() -> None:
    """Settling is a question about what is *still* running, so a call that answered is not in it."""
    scope = CancelToken().at("turn", "call")

    assert await scope.blocking(lambda: "read") == "read"

    assert await scope.settle() == 0, "nothing is running under a call that has answered"
