"""What one workspace lets a path be: the root it resolves against, and the dots it refuses.

A path arrives from a model, as text, and leaves as a resolved :class:`~pathlib.Path` or as a
refusal. Everything in between is here, and nothing here asks anybody: a path is refused, not
negotiated — the human who can lift a refusal is reached through ``Host.human``, by the tool that
was refused, never from this module.

Two refusals, and the difference is the whole point of having two:

* :class:`OutsideWorkspace` — no root and no grant covers the path. A grant can lift it, so the tool
  layer that knows those tools says so. The names travel back on the refusal, with the one argument a
  refusal can fill in for each: this module does not compose sentences about tools it cannot see.
* :class:`Denied` — the configured path denylist matches. No grant lifts it, and a tool must not
  offer one.

A temporary workspace is not a root here: it belongs to the conversation that asked for it, so it is the
caller's own scratch that reaches it and nobody else's (E152).

The root is taken as given: a root that is gone is a problem for the tools that
touch it, and a workspace whose directory was unmounted must not fail to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode, covers
from plyngent.kernel.command import DEFAULT_COMMAND_DENYLIST, DEFAULT_CONFIRM_TIMEOUT_SECONDS, CommandPolicy
from plyngent.kernel.ignore import Rules
from plyngent.kernel.paths import normalize
from plyngent.kernel.temps import TempWorkspaces

if TYPE_CHECKING:
    from collections.abc import Mapping


class PathRefused(Exception):  # noqa: N818 — a refusal is the workspace's answer, not a fault
    """A path this workspace will not hand out. The subclasses say whether anything can lift it."""


@dataclass(frozen=True, slots=True)
class Lift:
    """A tool that would let a refused path through, and the arguments a refusal can fill in for it.

    A name alone was not enough: the two ways to lift a refusal take different arguments, and a hint
    that names an argument its tool does not have is a call the model cannot make. What a refusal
    knows is the mode the call needed, so that is what a ``given`` patch may say — the tool that takes
    a ``mode`` is told it, and the tool that takes none is told nothing.
    """

    tool: str
    given: tuple[tuple[str, str], ...] = ()


class OutsideWorkspace(PathRefused):
    """No root and no grant covers the path; a grant or a temporary workspace would.

    ``related_tools`` names the tools that can lift this, with the arguments a refusal knows. The
    kernel raises it empty — a module that spells another module's tool names is a module that knows
    about the tool layer — and the tool layer attaches them with :meth:`with_tools` on the way out.
    """

    path: Path
    resolved: Path
    root: Path
    need: AccessMode
    related_tools: tuple[Lift, ...]

    def __init__(
        self,
        path: Path,
        *,
        resolved: Path,
        root: Path,
        need: AccessMode,
        related_tools: tuple[Lift, ...] = (),
    ) -> None:
        super().__init__(f"path escapes workspace root ({root}): {path}")
        self.path = path
        self.resolved = resolved
        self.root = root
        self.need = need
        self.related_tools = related_tools

    def with_tools(self, *lifters: Lift) -> OutsideWorkspace:
        """The same refusal, carrying the tools that can lift it (names and patches, never sentences)."""
        return OutsideWorkspace(
            self.path,
            resolved=self.resolved,
            root=self.root,
            need=self.need,
            related_tools=(*self.related_tools, *lifters),
        )


class Denied(PathRefused):
    """The path denylist matches; nothing lifts it, so it is not offered as a thing to ask for."""

    path: Path
    pattern: str

    def __init__(self, path: Path, pattern: str) -> None:
        super().__init__(f"path denied by policy (matched {pattern!r}): {path}")
        self.path = path
        self.pattern = pattern


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """What the config file says about this workspace, as one value.

    A config reload replaces the whole value, so there is no moment where
    the denylist is half the old file and half the new one.
    """

    path_denylist: tuple[str, ...] = ()
    command_denylist: frozenset[str] = DEFAULT_COMMAND_DENYLIST
    confirm_timeout: float = DEFAULT_CONFIRM_TIMEOUT_SECONDS
    # What a listing under this root keeps back: names, and the ignore sources it reads.
    listing: Rules = field(default_factory=Rules)


class Workspace:
    """One directory, and the rules for reading, writing and running under it.

    It is a plain field of :class:`~plyngent.kernel.host.Host`, not a component: the root is the
    host's own identity, and a second declaration site would be a second place for it to be written.
    """

    root: Path
    config: WorkspaceConfig
    commands: CommandPolicy
    temps: TempWorkspaces
    listing: Rules

    def __init__(self, root: Path | str, config: WorkspaceConfig | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.config = config if config is not None else WorkspaceConfig()
        self.commands = CommandPolicy(self.config.command_denylist)
        self.temps = TempWorkspaces()
        self.listing = self.config.listing
        self._apply(self.config)

    def configure(self, config: WorkspaceConfig) -> None:
        """Install a whole new config value (a reload never lands field by field)."""
        self.config = config
        self._apply(config)

    def close(self) -> None:
        """Let go of what the workspace owns; the host calls this on its way down.

        Every conversation's scratch goes with it, because the host is what counted them: a session that
        is closed reclaims its own (:meth:`~plyngent.kernel.serving.Serving._close`), and this is the one
        that catches the rest (E152).
        """
        _ = self.temps.clear()

    def _apply(self, config: WorkspaceConfig) -> None:
        """Push a config value into the parts that are judged by it."""
        self.commands.use_denylist(config.command_denylist)
        self.listing = config.listing

    # -- paths ---------------------------------------------------------------------

    def resolve(
        self,
        path: str | Path,
        need: AccessMode = AccessMode.WRITE,
        *,
        grants: Mapping[Path, AccessMode] | None = None,
        scratch: str | None = None,
    ) -> Path:
        """Resolve *path* to an absolute path this workspace allows, or refuse it.

        Relative paths resolve against :attr:`root`; an absolute path is taken as it is. ``need`` is what
        the caller is about to do with the result, and ``grants`` is the table of whoever is asking — the
        conversation's, passed in, because a process holds none of its own
        (E153). ``scratch`` is the caller's own conversation, and it is the only one whose temporary
        workspace this call may reach: a directory somebody else asked for is not a root, it is somebody
        else's (E152).
        """
        resolved = normalize(path, base=self.root)
        if (
            not self.under_roots(resolved)
            and not self.under_scratch(resolved, scratch)
            and not covers(grants, resolved, need)
        ):
            raise OutsideWorkspace(
                Path(path),
                resolved=resolved,
                root=self.root,
                need=need,
            )
        matched = self.denied_by(resolved)
        if matched is not None:
            raise Denied(resolved, matched)
        return resolved

    def under_roots(self, resolved: Path) -> bool:
        """Whether *resolved* is under the root every conversation of this host works in."""
        return resolved == self.root or resolved.is_relative_to(self.root)

    def under_scratch(self, resolved: Path, owner: str | None) -> bool:
        """Whether *resolved* is under one of the temporary workspaces one conversation asked for.

        ``None`` is a caller with no conversation of its own — nothing here is reachable then, which is
        the strict half of "a scratch belongs to whoever asked for it".
        """
        if owner is None:
            return False
        return any(resolved == one or resolved.is_relative_to(one) for one in self.temps.paths(owner))

    def denied_by(self, resolved: Path, *, directory: bool = False) -> str | None:
        """The denylist pattern *resolved* matches, or ``None``.

        Patterns are substrings of the resolved path, with ``/`` as the separator on every platform.
        ``directory`` appends a trailing separator, so a pattern written as ``/secret/`` matches the
        directory itself and not only what is inside it.
        """
        text = str(resolved).replace("\\", "/")
        if directory and not text.endswith("/"):
            text += "/"
        for pattern in self.config.path_denylist:
            if pattern and pattern.replace("\\", "/") in text:
                return pattern
        return None
