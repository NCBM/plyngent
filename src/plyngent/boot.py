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
from typing import TYPE_CHECKING, NamedTuple

from composat import Assembly, ComponentKey, ComponentRegistry, Declaration, ServiceNotUpError
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel import local
from plyngent.kernel.catalog import BUILTIN
from plyngent.kernel.channel import Channel, InProcChannel
from plyngent.kernel.client import Offers
from plyngent.kernel.documents import Documents, documents_path
from plyngent.kernel.events import HumanChannel
from plyngent.kernel.history import History, available, history_path
from plyngent.kernel.host import Host
from plyngent.kernel.journal import MEMORY_JOURNAL, Journal, SqliteTrail, journal_path
from plyngent.kernel.link import Link
from plyngent.kernel.lm.route import Posted, Route, Routes
from plyngent.kernel.model import Model, ModelChannel
from plyngent.kernel.net.web import Web, WebChannel
from plyngent.kernel.prompt import PromptPlan
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.pty.registry import Limits, PtyRegistry
from plyngent.kernel.serving import Serving
from plyngent.kernel.tools.net import net_specs
from plyngent.kernel.tools.skills import skill_specs
from plyngent.wire.ask import DEFAULT_ASK_TIMEOUT_SECONDS, Grades
from plyngent.wire.manifest import CLIENT_OFFER, KernelManifest, command_specs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from composat import Component, ComponentKey

    from plyngent.kernel.skills import Skills
    from plyngent.kernel.toolspec import ToolSpec
    from plyngent.wire.link import KernelLink

#: The host's qualifier when a program does not name one. It is part of every session's address
#: (``ws/s1``), so it is spelled once, here, and never guessed elsewhere.
DEFAULT_HOST = "ws"

#: The transports a kernel can be reached over. ``local`` is the one that leaves this process — a
#: socket at a path — and it is offered only where the platform has ``AF_UNIX``: a transport this build
#: cannot run is not one it pretends to have. The story suite is already parameterized over this tuple,
#: so every story runs over both without a story changing (R2.1 D29).
KERNEL_TRANSPORTS = ("inproc", "local") if local.AVAILABLE else ("inproc",)

#: How often the composition root looks at the one component it may have to replace. It is a poll
#: rather than a callback because the thing being watched is a child process: nothing tells us it died.
BROKER_WATCH_SECONDS = 0.5


class _Held(NamedTuple):
    """One connection this kernel opened for somebody else: its two faces, its loop, and its socket."""

    link: Link
    client: KernelLink
    task: asyncio.Task[None]
    streams: tuple[local.SocketChannel, ...] = ()


class Installed:
    """A host brought up with one connection served on it, and the one call that ends both.

    ``client`` is what a frontend gets: the link face, never this object, and never the assembly. A
    frontend that held the assembly could admit a session without going through a command, which is
    the in-process shortcut the boundary exists to forbid.

    The two faces are one object over ``inproc`` and two over ``local``, where the client's half is the
    far end of a socket. Nothing above this class has to know which it is holding, which is what makes
    the transports one implementation with a different codec.
    """

    assembly: Assembly
    serving: Serving
    host: str
    _link: Link
    _client: KernelLink
    _task: asyncio.Task[None]
    _watch: asyncio.Task[None] | None
    _listening: local.Listening | None
    _streams: tuple[local.SocketChannel, ...]
    _others: list[_Held]

    def __init__(
        self,
        assembly: Assembly,
        serving: Serving,
        link: Link,
        task: asyncio.Task[None],
        *,
        host: str = DEFAULT_HOST,
        client: KernelLink | None = None,
        watch: asyncio.Task[None] | None = None,
        listening: local.Listening | None = None,
        streams: tuple[local.SocketChannel, ...] = (),
    ) -> None:
        self.assembly = assembly
        self.serving = serving
        self.host = host
        self._link = link
        self._client = link if client is None else client
        self._task = task
        self._watch = watch
        self._listening = listening
        self._streams = streams
        self._others = []

    @property
    def client(self) -> KernelLink:
        """The connection as a frontend sees it."""
        return self._client

    async def attach(self, *, caps: tuple[str, ...] = ()) -> KernelLink:
        """Open one more connection to this kernel, over the transport it was installed with.

        What it is for is the rule that a capability is a promise: a connection whose manifest does not
        carry one is refused the command that would need it, and the only honest way to be such a
        connection is to make one the way a client does. Over ``local`` that is a socket to the address
        this kernel is listening on; over ``inproc`` it is a channel, and there is nothing else it could
        be — which is the same statement as "the two transports are one implementation".

        The connection is served like the first one: its own command loop, its own tool table, its own
        offer to attach. What it does not get is the one thing this kernel has one of — it answers no
        questions while the client installed here is attached (D19).
        """
        manifest = KernelManifest(commands=command_specs(), caps=caps)
        if self._listening is None:
            link = _opened(InProcChannel(), manifest)
            _serving, task = _served(self.assembly, self.host, link, BUILTIN)
            self._others.append(_Held(link, link, task))
            return link
        far = await local.connect(self._listening.endpoint)
        near = await self._listening.accept()
        link = _opened(near, manifest)
        _serving, task = _served(self.assembly, self.host, link, BUILTIN)
        self._others.append(_Held(link, Link(far, manifest=manifest), task, (near, far)))
        return self._others[-1].client

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

        Both ends of a socket are closed and waited for, which is what makes the last thing this kernel
        says reach the client that is leaving: a stream ends after what was queued has been written.
        """
        await self._dropped()
        if self._watch is not None:
            _ = self._watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch
            self._watch = None
        await self.serving.stop("the kernel is leaving")
        await self._task
        self._client.close()
        for stream in self._streams:
            await stream.flushed()
        if self._listening is not None:
            await self._listening.stop()
        _ = await self.assembly.stop(*self.assembly.keys())

    async def _dropped(self) -> None:
        """Close every connection this kernel opened for somebody else, and wait for its loop to end."""
        for held in self._others:
            held.link.close()
            held.client.close()
        _ = await asyncio.gather(*(held.task for held in self._others), return_exceptions=True)
        for held in self._others:
            for stream in held.streams:
                await stream.flushed()
        self._others.clear()


async def install(
    root: Path | str,
    *,
    host: str = DEFAULT_HOST,
    transport: str = "inproc",
    endpoint: Path | str | None = None,
    model: Model | None = None,
    routes: Iterable[Route] | None = None,
    route: str = "",
    web: Web | None = None,
    skills: Skills | None = None,
    prompt: PromptPlan | None = None,
    journal: Path | str | None = None,
    limits: Limits | None = None,
    deadline: float | None = None,
    grades: Grades | None = None,
    versions: bool = True,
) -> Installed:
    """Bring up one workspace and one connection, serving in the background.

    The declarations are the whole kernel: the PTY broker every host shares, a PTY registry per host,
    the channel a person answers through, the model a turn asks, the journal a conversation is kept in,
    and the host they belong to. Sessions are *not* declared here — a session arrives with ``Open``,
    which is what makes it a command rather than a startup flag.

    ``model`` is installed rather than reached for: ``None`` is a kernel with nothing to ask, which
    refuses turns instead of pretending a model answered. It is the same move as attaching a client to
    the human channel, and a test's scripted model is installed exactly the way a real one will be.

    ``routes`` and ``route`` are the other half of asking a model: the endpoints this program may use,
    and which of them answers a session that has not chosen. They are handed in for the same reason the
    skills directories are — which services exist and which key they take is a fact about how the program
    was started, and nothing below this line could have guessed it. A program given no routes and no
    model has nothing to ask, which is a program nobody configured; one given routes but no model asks
    the route its session names. Both are what :mod:`plyngent.config` hands over (§0.11).

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

    ``prompt`` is what every request to a model begins with, and it is a parameter for the same reason
    ``skills`` is: the catalog of skills under those directories is read from the machine this program was
    started on, and the plan is read once there rather than per request (a prompt is the front of a
    provider's cached prefix, and a section recomputed on every ask would move it with no counter to say
    so). ``None`` is the plan these skills make (:meth:`~plyngent.kernel.prompt.PromptPlan.of`); a program
    that says who the model is says it here.

    ``journal`` says where the conversation is kept, and it is a parameter rather than something the
    backend decides for itself. ``None`` is one file per workspace in the user's data directory
    (:func:`~plyngent.kernel.journal.journal_path`), ``":memory:"`` is a conversation nobody meant to
    keep, and a path is that path — which is what lets a story keep its journal beside its workspace
    instead of in the person's own data. The bodies too long to send are kept in their own file beside
    whichever journal answers
    (:func:`~plyngent.kernel.documents.documents_path`), so no second thing has to be named here.

    ``limits`` is how much of this host its terminals may take, and it is one parameter because it is one
    value: three numbers that only ever travel together, decided here for the same reason the skills
    directories are. ``None`` is the numbers this tree ships with, so a program with no configuration file
    is the program it always was; a story that wants one terminal's worth of room says so here instead of
    reaching into a registry it did not build (§0.11, E85).

    ``transport`` is named rather than assumed: it is what a client reaches the kernel over, and a
    name this build cannot open is refused here instead of half-working somewhere further in.
    ``local`` is one connection over a socket this kernel accepted, so ``endpoint`` is where it listens:
    a path, or ``None`` for one this process invents because nobody has to find it. An address a *person*
    names arrives from the command line and not from here (§5 B).

    ``deadline`` is how long a question to a person, or a call to one of the client's own tools, may
    wait before it counts as unanswered. It belongs to the connection and not to a session, so it is
    set where the connection is built, and ``None`` is the transport's own default.

    timeout without sleeping says so here (§0.15, E112).

    ``versions`` is whether a call that changed something leaves a version behind. ``True`` is the store
    this build ships with — a bare repository beside the journal
    (:func:`~plyngent.kernel.history.history_path`) — and it is kept only where this machine has a git
    to keep it with, so a machine without one records *what* changed and not the bytes. That degradation
    is decided here rather than discovered at the first write (§10.1, E26), the way a kernel with no
    skills directory has no skill tools.
    """
    if transport not in KERNEL_TRANSPORTS:
        msg = f"{transport!r} is not a transport this kernel can open; it knows {KERNEL_TRANSPORTS}"
        raise ValueError(msg)
    registry = ComponentRegistry()
    up: dict[ComponentKey, Component] = {}
    kept = _journal_file(root, journal)
    table = Routes(routes or (), default=route)
    asked = model if model is not None else (Posted(table) if table.default else None)
    # Read once, here: the skills this program can see are a fact about how it was started, and the prompt
    # they are part of is the front of a cached prefix (§10.5, E65/E66). Nothing in the tree is read for it
    # — a model that wants the workspace's own ``AGENTS.md`` reads it with the `read` call.
    plan = PromptPlan.of(skills) if prompt is None else prompt
    # One store per workspace, beside the journal it belongs with — and none at all where nothing can
    # keep one, which is a fact about the machine rather than about the call that asks for it.
    versions_kept = History(root, history_path(kept)) if versions and available() else None
    _ = registry.declare_all(
        [
            Declaration(PtyBroker),
            Declaration(PtyRegistry, instance=host, kwargs=(Limits() if limits is None else limits).declared()),
            Declaration(HumanChannel, instance=host),
            Declaration(Offers, instance=host),
            Declaration(ModelChannel, instance=host, args=(asked,)),
            Declaration(WebChannel, instance=host, args=(web,)),
            Declaration(Journal, instance=host, args=(SqliteTrail(kept),)),
            Declaration(Documents, instance=host, args=(documents_path(kept),)),
            Declaration(
                Host,
                instance=host,
                args=(root,),
                kwargs={"skills": skills, "routes": table, "prompt": plan, "history": versions_kept},
            ),
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

    # Which tools exist is this line's answer, and the only one: a backend that was not installed is a
    # tool that is not in the table, so a call naming it is answered as a name nothing has (§10.4). What
    # a *client* offers is not here: that arrives over the connection and is added to this set (§10.8).
    specs = (*BUILTIN, *net_specs(web), *skill_specs(skills))
    manifest = KernelManifest(commands=command_specs(), caps=(CLIENT_OFFER,))
    patience = DEFAULT_ASK_TIMEOUT_SECONDS if deadline is None else deadline
    graded = Grades() if grades is None else grades
    listening: local.Listening | None = None
    streams: tuple[local.SocketChannel, ...] = ()
    if transport == "local":
        # A kernel that is reached over a socket still serves the client it was started for, and that
        # client arrives the way any other does: at the address, over a connection this kernel accepted.
        named = local.temporary_endpoint() if endpoint is None else Path(endpoint)
        listening = await local.Listening.at(named)
        channel = await local.connect(named)
        near = await listening.accept()
        link = _opened(near, manifest, timeout=patience, grades=graded)
        client: KernelLink = Link(channel, manifest=manifest, timeout=patience, grades=graded)
        streams = (near, channel)
    else:
        link = _opened(InProcChannel(), manifest, timeout=patience, grades=graded)
        client = link
    serving, task = _served(assembly, host, link, specs)
    watch = asyncio.create_task(_replace_the_broker(assembly, Declaration(PtyBroker).key))
    return Installed(
        assembly,
        serving,
        link,
        task,
        host=host,
        client=client,
        watch=watch,
        listening=listening,
        streams=streams,
    )


def _opened(
    channel: Channel,
    manifest: KernelManifest,
    *,
    timeout: float | None = DEFAULT_ASK_TIMEOUT_SECONDS,
    grades: Grades | None = None,
) -> Link:
    """The kernel's half of one connection: the link it reads, started, ready to be served.

    Starting it here rather than by whichever caller remembered to is what makes "an answer is settled
    where the answer lane is read" true of every connection this program opens, however it got one.
    """
    link = Link(channel, manifest=manifest, timeout=timeout, grades=grades)
    link.start()
    return link


def _served(
    assembly: Assembly, host: str, link: Link, specs: tuple[ToolSpec, ...]
) -> tuple[Serving, asyncio.Task[None]]:
    """The command loop for one connection: the server that reads it, and the task that runs it."""
    serving = Serving(assembly, host=host, link=link, specs=specs)
    return serving, asyncio.create_task(serving.run())


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
