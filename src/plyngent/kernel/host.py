"""The host: what one workspace owns for as long as the process lives.

A host is not replaced within a process, and that is the reason its fields may hold resources that
have to be withdrawn explicitly — a child on a terminal, an fd, a remote lease. A replacement has to
be able to let go of everything it carried by itself, which is why those live here and not on a
``Session``.

The rest of what a host owns arrives with the parts that need it: the path/command policy, the
network policy, the human, the clock. Each one is a field here, never a module-level bag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from composat import Component, require

# Not type-only imports: ``require()`` resolves each field's annotation from this module's globals.
from plyngent.kernel.approvals import Approvals
from plyngent.kernel.client import Offers  # noqa: TC001
from plyngent.kernel.documents import Documents  # noqa: TC001
from plyngent.kernel.events import HumanChannel  # noqa: TC001
from plyngent.kernel.journal import Journal  # noqa: TC001
from plyngent.kernel.lm.route import Routes
from plyngent.kernel.locks import Locks
from plyngent.kernel.model import ModelChannel  # noqa: TC001
from plyngent.kernel.net.policy import NetPolicy
from plyngent.kernel.net.web import WebChannel  # noqa: TC001
from plyngent.kernel.prompt import PromptPlan
from plyngent.kernel.pty.policy import PtyPolicy
from plyngent.kernel.pty.registry import PtyRegistry  # noqa: TC001
from plyngent.kernel.skills import Skills  # noqa: TC001
from plyngent.kernel.workspace import Workspace

if TYPE_CHECKING:
    from pathlib import Path


class Host(Component):
    """One workspace's share of the process."""

    pty: PtyRegistry = require()
    # Where a question goes. A component rather than a field somebody sets, so that attaching a client
    # and detaching it are the same move (`replace`) and two hosts cannot answer for each other's
    # humans. Nobody attached is a channel that answers "no", not a missing one.
    human: HumanChannel = require()
    # The tools the connection serving this host offers, and the way to call them. A component for the
    # same reason the human channel is one: attaching a client's tools and detaching them are one
    # `replace` each, and a reader never sees half of an offer. Nobody offering is a value, not a
    # missing field (§10.8).
    offers: Offers = require()
    # The model a turn asks. A channel like the human's, so that installing one is a declaration and
    # swapping it is a `replace`; nothing installed is a value (`None`) rather than a missing field, so a
    # turn can be refused before it starts instead of failing at its first request.
    model: ModelChannel = require()
    # The vendor's web tools, when a program was configured with one. A component for the same reason the
    # model is: installing a backend is a declaration and swapping it is a `replace`, and nothing installed
    # is a value — which is what decides whether `web_fetch` and `builtin_websearch` exist at all (§0.9).
    web: WebChannel = require()
    # The conversation's journal. Held as the component that owns a backend rather than as the protocol,
    # because a requirement is answered by the class that was declared; everything that reads or appends
    # takes `Trail`, so which backend answers it is a declaration in `boot` and nothing else.
    trail: Journal = require()
    # Where a body too long for the conversation is kept. A component of its own, in a file of its own,
    # because content has a different lifecycle from history: it can be expired, indexed or moved to a
    # blob store without a migration that has to know what a transcript is.
    documents: Documents = require()
    # Not a component: the root is the host's own identity, so it is stored once, here.
    workspace: Workspace
    # Whether a delete goes to the system trash, where the human can take it back, or straight away.
    # It is the host's field because it decides what one verb does, not what a path may be: a
    # workspace does not stop having a trash when a config value changes.
    trash_enabled: bool
    # Which terminal sessions this host has already been allowed to change. It is the host's, not the
    # registry's: the registry holds children, and a decision about whether a person has to agree is not
    # a fact about a child.
    pty_policy: PtyPolicy
    # What this host has already been told it may do, keyed by what a person means by "don't ask again".
    # A plain field rather than a component: it is a table with no lifecycle, and the host is what lives
    # exactly as long as an approval should.
    approvals: Approvals
    # Which machines this host has already been told it may contact, and which requests have to be agreed
    # to at all. Its own table rather than a row in ``approvals``: an agreement here is about one machine
    # at one port, which is not one of the four things an approval is remembered by (§0.9, E27).
    net: NetPolicy
    # One lock per path, across every session on this host: two turns in two conversations that touch one
    # file take turns. It is the host's because a session comes and goes while a file is still a file.
    locks: Locks
    # The skills this host can read, or ``None`` when it was started without any. A plain field: a skill
    # has no lifecycle, so there is nothing to open or close and nothing that a component would answer.
    # ``None`` is a value rather than a missing field — a kernel with no skills directory has no skill
    # tools at all, which is decided where the tools are assembled (§0.10, E36).
    skills: Skills | None
    # The endpoints this program can ask a model at, and which of them answers a session that has not
    # chosen. A plain field, like the skills directory and for the same reason: a table of routes has no
    # lifecycle. What *does* have one is the model channel, because swapping the model is a `replace`.
    routes: Routes
    # What every request to a model begins with: the sections that do not come from the conversation. A
    # plain field again — a prompt plan is text made once at install, and the one part of it that moves
    # (which groups this session can reach) is handed in per request rather than stored here.
    prompt: PromptPlan

    def __init__(
        self,
        root: Path | str,
        *,
        trash_enabled: bool = True,
        skills: Skills | None = None,
        routes: Routes | None = None,
        prompt: PromptPlan | None = None,
    ) -> None:
        super().__init__()
        self.workspace = Workspace(root)
        self.trash_enabled = trash_enabled
        self.pty_policy = PtyPolicy()
        self.approvals = Approvals()
        self.net = NetPolicy()
        self.locks = Locks()
        self.skills = skills
        self.routes = Routes() if routes is None else routes
        self.prompt = PromptPlan() if prompt is None else prompt

    @override
    async def unload(self) -> None:
        """Give back what the workspace owns: its scratch directories, and nothing else yet."""
        self.workspace.close()
