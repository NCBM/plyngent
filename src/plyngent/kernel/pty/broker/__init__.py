"""The PTY broker: one process per kernel that owns every terminal, and the kernel's end of it.

The kernel does not fork and does not park a thread on a read; the process that does both is this one,
started by ``python -m plyngent.kernel.pty.broker`` and reached over a private channel. The parts:

``frames``
    the vocabulary — one tagged union each way, and the length-prefixed stream that carries them.
``link``
    the kernel's end of the channel: ``socketpair`` on POSIX, a named pipe on Windows.
``component``
    :class:`PtyBroker` — the component, its lifecycle, and the epoch that says which generation it is.
``child``
    the broker process itself: the session table, the fork, ``select``, ``waitpid``, the kill.
"""

from plyngent.kernel.pty.broker.component import BrokerSession, PtyBroker
from plyngent.kernel.pty.broker.frames import PROTOCOL

__all__ = ["PROTOCOL", "BrokerSession", "PtyBroker"]
