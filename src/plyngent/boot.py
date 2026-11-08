"""The composition root: the one module allowed to know both halves of the boundary.

Every other module is on one side or the other. The kernel does not import a terminal library, and the
CLI imports ``plyngent.wire`` and nothing inward — but a program still has to *hand* a client a link,
and whoever does that has both in its hands. That is this file, and it does nothing else: it builds a
host, brings it up, opens a connection, starts serving on it, and closes the same three things in
reverse. It draws nothing and parses no arguments, which is why the rule "the CLI may not import the
kernel" and the fact that a CLI has to be given a connection can both be true.

The workspace, the host's qualifier and the connection are all decided here, once, and every other
place reads them from what they were handed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from composat import Assembly, ComponentKey, ComponentRegistry, Declaration
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel.events import HumanChannel
from plyngent.kernel.host import Host
from plyngent.kernel.link import InProcLink
from plyngent.kernel.pty.registry import PtyRegistry
from plyngent.kernel.serving import Serving
from plyngent.wire.manifest import KernelManifest, command_specs

if TYPE_CHECKING:
    from pathlib import Path

    from composat import Component, ComponentKey

    from plyngent.wire.link import KernelLink

#: The host's qualifier when a program does not name one. It is part of every session's address
#: (``ws/s1``), so it is spelled once, here, and never guessed elsewhere.
DEFAULT_HOST = "ws"

#: The transports a kernel can be reached over. ``inproc`` is the only one that exists; ``local``
#: arrives with the socket/pipe transport, and because the story suite is already parameterized over
#: this tuple, every story then runs over both without a story changing (R2.1 D29).
KERNEL_TRANSPORTS = ("inproc",)


class Installed:
    """A host brought up with one connection served on it, and the one call that ends both.

    ``client`` is what a frontend gets: the link face, never this object, and never the assembly. A
    frontend that held the assembly could admit a session without going through a command, which is
    the in-process shortcut the boundary exists to forbid.
    """

    assembly: Assembly
    serving: Serving
    _link: InProcLink
    _task: asyncio.Task[None]

    def __init__(self, assembly: Assembly, serving: Serving, link: InProcLink, task: asyncio.Task[None]) -> None:
        self.assembly = assembly
        self.serving = serving
        self._link = link
        self._task = task

    @property
    def client(self) -> KernelLink:
        """The connection as a frontend sees it."""
        return self._link

    async def settle(self) -> None:
        """Wait until the connection's loop has finished — the client left, or the kernel did.

        It is the handle a supervisor needs after telling the client to go away: the loop's last act
        is to put the human back, and that has to have happened before anything else moves.
        """
        await self._task

    async def close(self) -> None:
        """Leave in the order things depend on: stop serving, let the loop end, then take the host down.

        The loop is awaited rather than merely told to stop, because its last act is to put the human
        back — and a ``replace`` on an assembly that is already going down has nowhere to land.
        """
        await self.serving.stop("the kernel is leaving")
        await self._task
        _ = await self.assembly.stop(*self.assembly.keys())


async def install(root: Path | str, *, host: str = DEFAULT_HOST, transport: str = "inproc") -> Installed:
    """Bring up one workspace and one connection, serving in the background.

    The declarations are the whole kernel: a PTY registry, the channel a person answers through, and
    the host they belong to. Sessions are *not* declared here — a session arrives with ``Open``, which
    is what makes it a command rather than a startup flag.

    ``transport`` is named rather than assumed: it is what a client reaches the kernel over, and a
    name this build cannot open is refused here instead of half-working somewhere further in.
    """
    if transport not in KERNEL_TRANSPORTS:
        msg = f"{transport!r} is not a transport this kernel can open; it knows {KERNEL_TRANSPORTS}"
        raise ValueError(msg)
    registry = ComponentRegistry()
    up: dict[ComponentKey, Component] = {}
    _ = registry.declare_all(
        [
            Declaration(PtyRegistry, instance=host),
            Declaration(HumanChannel, instance=host),
            Declaration(Host, instance=host, args=(root,)),
        ]
    )
    rule = answering_from_the_root(registry, up, only=set(), below=ancestors)
    assembly = Assembly(registry, up=up, rule=rule)

    started = await assembly.start(Declaration(Host, instance=host).key)
    if not started.ok:
        msg = f"the host at {root!r} did not come up: {started.failed}"
        raise RuntimeError(msg)

    link = InProcLink(manifest=KernelManifest(commands=command_specs()))
    serving = Serving(assembly, host=host, link=link)
    return Installed(assembly, serving, link, asyncio.create_task(serving.run()))
