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
from plyngent.kernel.events import HumanChannel  # noqa: TC001
from plyngent.kernel.pty.policy import PtyPolicy
from plyngent.kernel.pty.registry import PtyRegistry  # noqa: TC001
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

    def __init__(self, root: Path | str, *, trash_enabled: bool = True) -> None:
        super().__init__()
        self.workspace = Workspace(root)
        self.trash_enabled = trash_enabled
        self.pty_policy = PtyPolicy()

    @override
    async def unload(self) -> None:
        """Give back what the workspace owns: its scratch directories, and nothing else yet."""
        self.workspace.close()
