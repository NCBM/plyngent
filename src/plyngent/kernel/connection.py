"""One connection, however many trees it is looking at: which one a command reaches, and what it hears back.

A kernel serves a set of trees that grows while it runs (E157), and a client may work in any of them over
one connection: that is the whole of why this module exists as a thing of its own. What it does is three
answers:

* **Which tree a command or a read means.** A value names one with ``workspace``, spelled as the greeting
  spells a tree's path; the empty name is the tree this connection declared on its first frame. Naming a
  tree nothing has opened is not an error the connection can recover from by guessing — opening is
  ``Open``'s job and only its own — so a command is refused with :class:`~plyngent.wire.event.Refusal`
  and a read is answered with the empty value of its kind.
* **What a client is told.** Events, questions and the two faces of the link are the connection's; what
  happens in a tree is said by that tree's
  :class:`~plyngent.kernel.serving.Serving`, which stamps the tree onto every event.
* **When it is over.** One lane read per thing that arrives, one task for the reads, and one goodbye:
  :meth:`stop` says the kernel is leaving and ends the lanes, which is why it belongs here and to no one
  tree.

The trees are found through a *foundry* — a callable that hands back a tree this process serves, opening
one it does not, or saying which of the three structural refusals applies. It is a callable, because
opening a tree is the composition root's decision: it is the only place that knows how a tree is declared
and what a process may hold at once (``E157``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

from plyngent.kernel.serving import Serving
from plyngent.wire.command import Open
from plyngent.wire.event import KernelLeaving, Refusal, WorkspaceRefused
from plyngent.wire.manifest import PROTOCOL_VERSION

if TYPE_CHECKING:
    from composat import Assembly

    from plyngent.kernel.toolspec import ToolSpec
    from plyngent.wire.bind import Bind
    from plyngent.wire.command import AnyCommand
    from plyngent.wire.link import KernelSide
    from plyngent.wire.query import AnyQuery, AnyReply


class TreeRef(NamedTuple):
    """One tree, as the wire names it and as the assembly does.

    Two spellings because they answer two questions: ``path`` is what a client names a tree by and what
    an event carries, while ``qualifier`` is the prefix every component of that tree is declared under
    (``ws``, ``ws2``) and therefore the prefix a session's address has. They are not derivable from each
    other — the qualifier is a counter, the path is wherever the tree is — so both are passed around.
    """

    path: str
    qualifier: str


class Foundry(Protocol):
    """Where a connection finds a tree, and how one that is not there yet comes to exist."""

    async def find(self, where: str) -> TreeRef | Refusal:
        """The tree at ``where``, opening it if this process can, or the reason it will not be opened."""
        ...


class Connection:
    """One connection to a kernel: the link, the trees it is reading, and the one loop both share.

    The trees are made as they are first named, and they are kept until the process stops: a tree opened
    on a client's behalf is not closed by that client leaving, because the conversations in it are not the
    client's (E157). What *is* the client's ends here — its membership of each tree's audience
    (:meth:`~plyngent.kernel.serving.Serving.end`), and its offer of tools.
    """

    assembly: Assembly
    link: KernelSide
    _foundry: Foundry
    _specs: tuple[ToolSpec, ...]
    _bound: str
    _servings: dict[str, Serving]
    _stopped: bool

    def __init__(
        self,
        assembly: Assembly,
        *,
        link: KernelSide,
        foundry: Foundry,
        specs: tuple[ToolSpec, ...],
    ) -> None:
        self.assembly = assembly
        self.link = link
        self._foundry = foundry
        self._specs = specs
        self._bound = ""
        self._servings = {}
        self._stopped = False

    # -- the loop ------------------------------------------------------------------

    async def run(self) -> None:
        """Be this connection's kernel until the client stops asking, then let every tree go.

        The first thing it does is read what the client declared itself to be, because a connection this
        kernel cannot read is not one it may serve: a client that speaks another version of the contract
        has not sent a command these shapes describe, and a client that says nothing has agreed to
        nothing. Both are answered the same way — the connection ends before a session exists — and
        neither is answered *in words*, because a refusal would have to be written in the contract, and
        this is the one case where the two ends have not agreed about it. The kernel's own version is in
        the greeting, so which of the two is out of date is the client's to say.

        The loop ends for one of two reasons and both mean the same thing here: the client closed the
        connection, or :meth:`stop` did. Either way the questions in flight have already been refused by
        the link, and the only thing left to undo is who was watching what.

        Reads are on their own task, because they are on their own lane and because a read must not wait
        behind a turn: a client asking what it may reach is answered while a model is still thinking, and
        neither question is the other's business.
        """
        declared = await self.link.declared()
        if declared is None or declared.protocol_version != PROTOCOL_VERSION:
            self.link.close()
            return
        await self._declared(declared)
        rebinding = asyncio.create_task(self._rebinding())
        reading = asyncio.create_task(self._reading())
        try:
            async for command in self.link.commands():
                await self._handle(command)
        finally:
            for task in (rebinding, reading):
                _ = task.cancel()
            _ = await asyncio.gather(rebinding, reading, return_exceptions=True)
            await self._leaving()

    async def _rebinding(self) -> None:
        """Take a client's later declarations: the same client, saying which tree it means now.

        A declaration carries no receipt, so nothing here answers one — what changes is only what a value
        that names no tree means from then on. A version this kernel does not speak is ignored: the
        connection already exists, and the first frame is where that question was answered (E155).
        """
        async for bind in self.link.bindings():
            if bind.protocol_version == PROTOCOL_VERSION:
                await self._declared(bind)

    async def _declared(self, bind: Bind) -> None:
        """Take what a client said it is: the tree it means, in both places that need to agree about it.

        Reading the declaration is also when this connection *joins* that tree's audience, before it has
        asked for anything: a frontend that connects is watching, and a second frontend on one tree that
        only heard events after its first command would be a viewer of half the conversation (E154). A
        tree that is not open yet is one nothing can be watched in — joining it happens when ``Open``
        opens it.
        """
        self._bound = _named(bind.workspace)
        self.link.bound = self._bound
        _ = await self._tree(self._bound)

    def stop(self, reason: str = "") -> None:
        """Leave: say so, then close the lanes the loop is reading.

        ``KernelLeaving`` first, because D20 makes "the kernel is going down" an event and not an exception
        to catch: a client that sees its event lane end knows why it ended. One goodbye per connection and
        not one per tree, because a connection is what ended.
        """
        if self._stopped:
            return
        self._stopped = True
        self.link.emit(KernelLeaving(reason=reason))
        self.link.close()

    async def _leaving(self) -> None:
        """Leave every tree's audience: what this connection was watching ends with it."""
        for serving in self._servings.values():
            await serving.end()
        self._servings.clear()

    # -- which tree ----------------------------------------------------------------

    async def _handle(self, command: AnyCommand) -> None:
        """Give one command to the tree it names, or say which tree it could not reach.

        The refusal names the tree the command *meant*: a connection that declared its tree sends values
        with no tree in them, and a sentence about the tree that was refused is the one a person can act
        on (E154).
        """
        found = await self._tree(command.workspace, opening=isinstance(command, Open))
        if isinstance(found, Refusal):
            meant = _named(command.workspace) or self._bound
            self.link.emit(WorkspaceRefused(session=command.session, workspace=meant, reason=found))
            return
        await found.handle(command)

    async def _reading(self) -> None:
        """Answer every read this client asks, until its lane ends."""
        async for query in self.link.queries():
            _ = self.link.replied(await self._answered(query))

    async def _answered(self, query: AnyQuery) -> AnyReply:
        """Give one read to the tree it names; a tree nothing opened answers the empty value of its kind."""
        found = await self._tree(query.workspace)
        if isinstance(found, Refusal):
            return Serving.nothing(query)
        return await found.answered(query)

    async def _tree(self, given: str, *, opening: bool = False) -> Serving | Refusal:
        """The serving for the tree ``given`` names, or the reason there is none.

        Only ``Open`` opens: a command that names a tree nothing has opened is answered with a refusal,
        because opening a tree is a change a client asks for by name and not a side effect of
        asking about a conversation (E157). Everything else reaches a tree that is already up.

        A name that is empty *and* undeclared is not a refusal, because a kernel serving one tree has one
        tree a client that named none can only mean: the foundry answers that one, and answers nothing
        ambiguous when there are several (E154).
        """
        name = _named(given) or self._bound
        found = self._servings.get(name)
        if found is not None:
            return found
        if not opening and name:
            return Refusal.NO_WORKSPACE
        looked = await self._foundry.find(name)
        if not isinstance(looked, TreeRef):
            return looked
        return self._serving(looked)

    def _serving(self, looked: TreeRef) -> Serving:
        """This connection's half of one tree, made the first time it is named and joined once.

        Idempotent, because a tree may be named again by any later command: what a second naming gets is
        the same object, already listening, and not a second membership in the same audience.
        """
        found = self._servings.get(looked.path)
        if found is not None:
            return found
        serving = Serving(
            self.assembly, host=looked.qualifier, workspace=looked.path, link=self.link, specs=self._specs
        )
        serving.begin()
        self._servings[looked.path] = serving
        return serving


def _named(where: str) -> str:
    """One spelling of a tree: a path as the greeting would write it, or the empty name for "no tree".

    Resolving here is what makes "the same tree, spelled two ways" one tree: a client that sends
    ``/a/b`` once and ``/a/b/`` later is looking at one tree, and a
    relative path resolves against this process's directory — which is what makes a single-run kernel's
    own tree reachable by ``.``.
    """
    return "" if not where else str(Path(where).expanduser().resolve())
