"""What one tool call may reach: the resources its tool declared, and its own address.

A handler is handed this and nothing else, and that is the whole design of it. A resource a tool did not
declare must not merely be "not to be touched" — it must not be *reachable*, so that reading it fails at
the top of the handler, with the attribute's own name, instead of failing somewhere inside a call that
had already started changing something. The declaration then answers "what does this tool touch" without
anybody reading its source (§10.1).

Three things are always there, because they are the address rather than a resource: which session the
call belongs to, the cancellation scope a piece of work under it is stopped through, and the place a
question asked from it belongs. Everything else is one property per resource, granted by the flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final, override

from plyngent.kernel.client import NO_CLIENT
from plyngent.kernel.toolspec import Res

if TYPE_CHECKING:
    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.catalog import ToolView
    from plyngent.kernel.client import Client
    from plyngent.kernel.documents import Documents
    from plyngent.kernel.events import Human
    from plyngent.kernel.net.policy import NetPolicy
    from plyngent.kernel.net.web import WebChannel
    from plyngent.kernel.pty.policy import PtyPolicy
    from plyngent.kernel.pty.registry import PtyRegistry
    from plyngent.kernel.reads import Reads
    from plyngent.kernel.session import Session, TurnCtx
    from plyngent.kernel.skills import Skills
    from plyngent.kernel.workspace import Workspace
    from plyngent.wire.ask import AskSite

#: The resources that may put a question to a person, and why each of them does.
#:
#: ``HUMAN`` is the tool that *is* the question. ``EXEC`` is a command whose danger only the command
#: policy can judge, so the tool has to be able to ask about the one program it is about to start.
#: ``PTY`` is a live process, where a write cannot be taken back and asking first is the only way to
#: have agreed to it. ``NET`` is a request that leaves this machine: whether it may be made is a judgement
#: about a destination nobody can see from the arguments alone (§0.9). Those four are not four spellings
#: of one thing: they are the reasons a tool legitimately reaches a human, and a tool with none cannot.
ASKING: Res = Res.HUMAN | Res.EXEC | Res.PTY | Res.NET


@final
class Env:
    """One call's world: its address, plus whatever its declaration granted.

    Built by the executor from the specification, once per call, and never stored: a value that outlived
    the call would be a second way to reach the same resources that no declaration is checked against.
    """

    __slots__ = ("_call", "_ctx", "_tools", "_uses")

    def __init__(self, ctx: TurnCtx, uses: Res, call: str, *, tools: ToolView) -> None:
        self._ctx = ctx
        self._uses = uses
        #: This call's segment of the turn's address — ``tool-1`` in ``3/r0/tool-1/proc-0``.
        self._call = call
        #: What this session can reach, as the round it runs in saw it. Handed in rather than reached
        #: for, because the answer belongs to one round's request and nothing here may ask twice.
        self._tools = tools

    # -- the address, which is not a resource --------------------------------------

    @property
    def session(self) -> str:
        """Which session this call belongs to, for a tool that records its owner (a terminal does)."""
        return self._ctx.session.session_id

    @property
    def reads(self) -> Reads:
        """What this turn has already shown with line numbers.

        Not a resource but a record of the turn: it is here rather than reached for because the turn is
        this call's address, and because a record on the turn cannot outlive it or be shared with
        another one.
        """
        return self._ctx.reads

    def scope(self, *segments: str) -> Scope:
        """The cancellation handle for one piece of work under this call."""
        return self._ctx.scope(self._call, *segments)

    def site(self, *segments: str) -> AskSite:
        """Where a question asked from here belongs: under this call, under this turn."""
        return self._ctx.site(self._call, *segments)

    @override
    def __repr__(self) -> str:
        return f"Env({self._call}, {self._uses})"

    # -- what the declaration grants -----------------------------------------------

    @property
    def workspace(self) -> Workspace:
        """The paths this call resolves and changes. Granted by :attr:`Res.PATHS`."""
        self._granted(Res.PATHS, "workspace")
        return self._ctx.host.workspace

    @property
    def trash_enabled(self) -> bool:
        """Whether a delete goes to the system trash where a human can take it back.

        A path tool's question rather than a policy of its own, so :attr:`Res.PATHS` answers it: the
        only tools that can ask it are the ones that delete paths.
        """
        self._granted(Res.PATHS, "trash_enabled")
        return self._ctx.host.trash_enabled

    @property
    def pty(self) -> PtyRegistry:
        """The terminals this host owns. Granted by :attr:`Res.PTY`."""
        self._granted(Res.PTY, "pty")
        return self._ctx.host.pty

    @property
    def pty_policy(self) -> PtyPolicy:
        """Which live sessions have already been agreed to. Granted by :attr:`Res.PTY`."""
        self._granted(Res.PTY, "pty_policy")
        return self._ctx.host.pty_policy

    @property
    def tools(self) -> ToolView:
        """What this session can reach, and what it has not been shown. Granted by :attr:`Res.TOOLS`.

        A view and not the catalog: a tool may see the set it is part of, and may not reach every tool
        this kernel has — the difference between the two is the whole of §10.4.
        """
        self._granted(Res.TOOLS, "tools")
        return self._tools

    @property
    def web(self) -> WebChannel:
        """The vendor's web tools this host was given, or a channel with nothing installed.

        Granted by :attr:`Res.NET`, because it is the other half of what a network tool reaches: a tool
        that reads a page through somebody else's network is still a tool that touches a network, and one
        resource is what the declaration is about.
        """
        self._granted(Res.NET, "web")
        return self._ctx.host.web

    @property
    def session_state(self) -> Session:
        """What this session knows about itself, and the one way to add to it. Granted by :attr:`Res.HISTORY`.

        The name says "the record", not "the state of a turn": what a tool may reach here is the
        conversation's own memory — which groups it has been shown, what its plan is — and adding to it
        is an append to the journal, which is why it takes the same flag as anything else that reads
        history.
        """
        self._granted(Res.HISTORY, "session_state")
        return self._ctx.session

    @property
    def documents(self) -> Documents:
        """Where a body too long for the conversation is kept. Granted by :attr:`Res.DOCUMENT`."""
        self._granted(Res.DOCUMENT, "documents")
        return self._ctx.host.documents

    @property
    def skills(self) -> Skills:
        """The skills this host can read, and where a new one is written. Granted by :attr:`Res.SKILLS`.

        A host started without a store has none, and that is a fact about the kernel rather than about
        this call: the tools that reach here are the ones a kernel serves only when it was given one, so
        a handler asking anyway is asking an object for something it does not have.
        """
        self._granted(Res.SKILLS, "skills")
        found = self._ctx.host.skills
        if found is None:
            msg = "skills is not reachable from this host: it was started with no skills directory to read"
            raise AttributeError(msg)
        return found

    @property
    def net(self) -> NetPolicy:
        """Which machines have already been agreed to, and what a request has to ask about.

        Granted by :attr:`Res.NET`, because the policy *is* the resource: a tool that may contact a
        network is the only one that can be told a destination is private, and the only one that records
        what a person said about it (§0.9, E27).
        """
        self._granted(Res.NET, "net")
        return self._ctx.host.net

    @property
    def human(self) -> Human:
        """The asking hand. Granted by whichever of :data:`ASKING` this tool declared."""
        if not self._uses & ASKING:
            self._refused("human", ASKING)
        return self._ctx.host.human

    @property
    def client(self) -> Client:
        """The client that offered this tool, and the one way to run one of its tools.

        Granted by :attr:`Res.CLIENT`, which is the resource with the least certainty attached to it:
        the call happens in another process, this one records that it was declared and cannot know
        whether it happened (§10.8). Nobody offering is :data:`~plyngent.kernel.client.NO_CLIENT`
        rather than a missing attribute — a tool that reaches here without a client attached is a
        call that lost its client, which is an answer, not a bug in the declaration.
        """
        self._granted(Res.CLIENT, "client")
        attached = self._ctx.host.offers.attached
        return NO_CLIENT if attached is None else attached

    def _granted(self, resource: Res, name: str) -> None:
        if not self._uses & resource:
            self._refused(name, resource)

    def _refused(self, name: str, resource: Res) -> None:
        """Say which attribute is not reachable here, and which declaration would have made it so.

        An ``AttributeError`` rather than an error of our own, because that is what it is: the handler
        asked an object for something it does not have, and a `getattr` that would have answered ``None``
        must fail the same way.
        """
        msg = f"{name} is not reachable from a tool that declared {self._uses or 'nothing'}; it takes {resource}"
        raise AttributeError(msg)
