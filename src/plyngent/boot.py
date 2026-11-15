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
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from composat import Assembly, ComponentKey, ComponentRegistry, Declaration, ServiceNotUpError
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel.catalog import BUILTIN
from plyngent.kernel.client import Offers
from plyngent.kernel.documents import Documents, documents_path
from plyngent.kernel.events import HumanChannel
from plyngent.kernel.host import Host
from plyngent.kernel.journal import MEMORY_JOURNAL, Journal, SqliteTrail, journal_path
from plyngent.kernel.link import InProcLink
from plyngent.kernel.model import Model, ModelChannel
from plyngent.kernel.net.web import Web, WebChannel
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.pty.registry import PtyRegistry
from plyngent.kernel.serving import Serving
from plyngent.kernel.tools.net import net_specs
from plyngent.kernel.tools.skills import skill_specs
from plyngent.wire.ask import DEFAULT_ASK_TIMEOUT_SECONDS
from plyngent.wire.manifest import CLIENT_OFFER, KernelManifest, command_specs

if TYPE_CHECKING:
    from composat import Component, ComponentKey

    from plyngent.kernel.skills import Skills
    from plyngent.wire.link import KernelLink

#: The host's qualifier when a program does not name one. It is part of every session's address
#: (``ws/s1``), so it is spelled once, here, and never guessed elsewhere.
DEFAULT_HOST = "ws"

#: The transports a kernel can be reached over. ``inproc`` is the only one that exists; ``local``
#: arrives with the socket/pipe transport, and because the story suite is already parameterized over
#: this tuple, every story then runs over both without a story changing (R2.1 D29).
KERNEL_TRANSPORTS = ("inproc",)

#: How often the composition root looks at the one component it may have to replace. It is a poll
#: rather than a callback because the thing being watched is a child process: nothing tells us it died.
BROKER_WATCH_SECONDS = 0.5


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
    _watch: asyncio.Task[None] | None

    def __init__(
        self,
        assembly: Assembly,
        serving: Serving,
        link: InProcLink,
        task: asyncio.Task[None],
        watch: asyncio.Task[None] | None = None,
    ) -> None:
        self.assembly = assembly
        self.serving = serving
        self._link = link
        self._task = task
        self._watch = watch

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
        """Leave in the order things depend on: stop watching, stop serving, then take the host down.

        The loop is awaited rather than merely told to stop, because its last act is to put the human
        back — and a ``replace`` on an assembly that is already going down has nowhere to land.
        """
        if self._watch is not None:
            _ = self._watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch
            self._watch = None
        await self.serving.stop("the kernel is leaving")
        await self._task
        _ = await self.assembly.stop(*self.assembly.keys())


async def install(
    root: Path | str,
    *,
    host: str = DEFAULT_HOST,
    transport: str = "inproc",
    model: Model | None = None,
    web: Web | None = None,
    skills: Skills | None = None,
    journal: Path | str | None = None,
    deadline: float | None = None,
) -> Installed:
    """Bring up one workspace and one connection, serving in the background.

    The declarations are the whole kernel: the PTY broker every host shares, a PTY registry per host,
    the channel a person answers through, the model a turn asks, the journal a conversation is kept in,
    and the host they belong to. Sessions are *not* declared here — a session arrives with ``Open``,
    which is what makes it a command rather than a startup flag.

    ``model`` is installed rather than reached for: ``None`` is a kernel with nothing to ask, which
    refuses turns instead of pretending a model answered. It is the same move as attaching a client to
    the human channel, and a test's scripted model is installed exactly the way a real one will be.

    ``web`` is the same move for the vendor's web tools: ``None`` is a kernel that has none, and then
    ``web_fetch`` and ``builtin_websearch`` are not in the table at all rather than failing when called.
    That is the one thing about this program that decides *which* tools exist, and it is decided here
    because it is a fact about how the program was started (§0.9, E26).

    ``skills`` is where a skill is looked for, and ``None`` is a kernel with no skill tools at all — the
    same move again, for the five of them (§0.10, E36). Which directories those are is
    :meth:`~plyngent.kernel.skills.Skills.standard`'s answer and it is given by whoever starts the
    program, because a list of places to look is a fact about the machine rather than about the kernel.
    Passing a store built from a temporary directory is what lets a story have skills without reading
    the person's own.

    ``journal`` says where the conversation is kept, and it is a parameter rather than something the
    backend decides for itself. ``None`` is one file per workspace in the user's data directory
    (:func:`~plyngent.kernel.journal.journal_path`), ``":memory:"`` is a conversation nobody meant to
    keep, and a path is that path — which is what lets a story keep its journal beside its workspace
    instead of in the person's own data. The bodies too long to send are kept in their own file beside
    whichever journal answers
    (:func:`~plyngent.kernel.documents.documents_path`), so no second thing has to be named here.

    ``transport`` is named rather than assumed: it is what a client reaches the kernel over, and a
    name this build cannot open is refused here instead of half-working somewhere further in.

    ``deadline`` is how long a question to a person, or a call to one of the client's own tools, may
    wait before it counts as unanswered. It belongs to the connection and not to a session, so it is
    set where the connection is built, and ``None`` is the transport's own default.
    """
    if transport not in KERNEL_TRANSPORTS:
        msg = f"{transport!r} is not a transport this kernel can open; it knows {KERNEL_TRANSPORTS}"
        raise ValueError(msg)
    registry = ComponentRegistry()
    up: dict[ComponentKey, Component] = {}
    kept = _journal_file(root, journal)
    _ = registry.declare_all(
        [
            Declaration(PtyBroker),
            Declaration(PtyRegistry, instance=host),
            Declaration(HumanChannel, instance=host),
            Declaration(Offers, instance=host),
            Declaration(ModelChannel, instance=host, args=(model,)),
            Declaration(WebChannel, instance=host, args=(web,)),
            Declaration(Journal, instance=host, args=(SqliteTrail(kept),)),
            Declaration(Documents, instance=host, args=(documents_path(kept),)),
            Declaration(Host, instance=host, args=(root,), kwargs={"skills": skills}),
        ]
    )
    # One broker for the whole process: it is the root-qualified component, and it is the only thing
    # here that a host reaches for across the qualifier axis (§9.4).
    rule = answering_from_the_root(registry, up, only={PtyBroker}, below=ancestors)
    assembly = Assembly(registry, up=up, rule=rule)

    started = await assembly.start(Declaration(Host, instance=host).key)
    if not started.ok:
        msg = f"the host at {root!r} did not come up: {started.failed}"
        raise RuntimeError(msg)

    link = InProcLink(
        manifest=KernelManifest(commands=command_specs(), caps=(CLIENT_OFFER,)),
        timeout=DEFAULT_ASK_TIMEOUT_SECONDS if deadline is None else deadline,
    )
    # Which tools exist is this line's answer, and the only one: a backend that was not installed is a
    # tool that is not in the table, so a call naming it is answered as a name nothing has (§10.4). What
    # a *client* offers is not here: that arrives over the connection and is added to this set (§10.8).
    serving = Serving(assembly, host=host, link=link, specs=(*BUILTIN, *net_specs(web), *skill_specs(skills)))
    watch = asyncio.create_task(_replace_the_broker(assembly, Declaration(PtyBroker).key))
    return Installed(assembly, serving, link, asyncio.create_task(serving.run()), watch)


async def _replace_the_broker(assembly: Assembly, key: ComponentKey) -> None:
    """Put a new broker behind ``key`` when the one that is there has died (§9.4).

    The kernel cannot do this for itself, and should not try: a component cannot replace itself, the
    registry only reads its field, and asking the composition root for a new one would be a second seam
    into it. The composition root already holds the assembly, so it is where the noticing goes.

    Only a broker that *was* up is replaced. One that never started — no permission, no platform
    support, no ``pywinpty`` — is a fact about the machine, and starting it again every half second
    would not change it; the registry keeps reporting ``PtyUnavailable`` instead.
    """
    while True:
        await asyncio.sleep(BROKER_WATCH_SECONDS)
        try:
            current = assembly.instance(key)
        except ServiceNotUpError:
            continue
        if not isinstance(current, PtyBroker) or current.available or not current.ever_ready:
            continue
        with contextlib.suppress(Exception):
            _ = await assembly.replace(key, Declaration(PtyBroker))


def _journal_file(root: Path | str, journal: Path | str | None) -> Path | str:
    """Which journal answers this host: a file, the one this workspace has by name, or none at all.

    The choice is made here because it is a fact about how the program was started and nowhere further
    in: everything past this point holds a ``Trail`` and never asks which one it is. Both answers are the
    same backend — a temporary journal is SQLite's in-memory database rather than a second implementation
    of the protocol — and the kept bodies of a conversation live beside whichever file answers it.
    """
    if journal == MEMORY_JOURNAL:
        return MEMORY_JOURNAL
    return journal_path(root) if journal is None else Path(journal)
