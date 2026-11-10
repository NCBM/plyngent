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

from plyngent.kernel.toolspec import Res

if TYPE_CHECKING:
    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.events import Human
    from plyngent.kernel.pty.policy import PtyPolicy
    from plyngent.kernel.pty.registry import PtyRegistry
    from plyngent.kernel.session import TurnCtx
    from plyngent.kernel.workspace import Workspace
    from plyngent.wire.ask import AskSite

#: The resources that may put a question to a person, and why each of them does.
#:
#: ``HUMAN`` is the tool that *is* the question. ``EXEC`` is a command whose danger only the command
#: policy can judge, so the tool has to be able to ask about the one program it is about to start.
#: ``PTY`` is a live process, where a write cannot be taken back and asking first is the only way to
#: have agreed to it. Those three are not three spellings of one thing: they are the three reasons a
#: tool legitimately reaches a human, and a tool with none of them cannot.
ASKING: Res = Res.HUMAN | Res.EXEC | Res.PTY


@final
class Env:
    """One call's world: its address, plus whatever its declaration granted.

    Built by the executor from the specification, once per call, and never stored: a value that outlived
    the call would be a second way to reach the same resources that no declaration is checked against.
    """

    __slots__ = ("_call", "_ctx", "_uses")

    def __init__(self, ctx: TurnCtx, uses: Res, call: str) -> None:
        self._ctx = ctx
        self._uses = uses
        #: This call's segment of the turn's address — ``tool-1`` in ``3/r0/tool-1/proc-0``.
        self._call = call

    # -- the address, which is not a resource --------------------------------------

    @property
    def session(self) -> str:
        """Which session this call belongs to, for a tool that records its owner (a terminal does)."""
        return self._ctx.session.session_id

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
    def human(self) -> Human:
        """The asking hand. Granted by whichever of :data:`ASKING` this tool declared."""
        if not self._uses & ASKING:
            self._refused("human", ASKING)
        return self._ctx.host.human

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
