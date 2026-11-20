"""Which paths a listing shows, and the one walk that decides it for every tool that lists.

Three questions look alike and answer to different owners, so they are kept apart:

* **May this path be handed out at all?** That is the workspace's
  (:meth:`~plyngent.kernel.workspace.Workspace.resolve`), settled before a tool runs, and no tool
  argument can widen or narrow it. A denied path is not in a listing because it is not reachable —
  it is not counted either, since a count is also something about it.
* **What does this workspace keep back?** The names it hides whatever a call asks for, and the
  ignore sources a checkout carries — :class:`Rules`, fixed for the host and written once.
* **What does this call want to see?** A model that asks for dotfiles, or that names the ignore files
  to read, says so in :class:`Filter`, which can only widen: hiding more than the workspace already
  hides is not a request a tool takes.

The walk is here, because a listing built by two implementations is a listing that disagrees with
itself: the same tree searched by ``glob_paths`` and shown by ``tree`` must show
the same paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from msgspec import Struct

from plyngent.kernel.toolresult import Skips
from plyngent.kernel.toolspec import ToolArgs

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.ignore import Patterns
    from plyngent.kernel.workspace import Workspace


class Filter(Struct, frozen=True, forbid_unknown_fields=True):
    """What one call is willing to leave out, and it can only leave out less.

    ``show_hidden`` is about *directories*: a dot directory, or a name in :attr:`Rules.hidden`. Dot
    *files* are visible either way — ``.env`` and ``.gitignore`` are files a checkout is made of, and
    hiding them only teaches a model to guess. ``ignore_lists`` is which ignore sources to read, by
    file name, in this order (a later one wins): ``None`` means the workspace's own list and ``[]``
    means read none of them. The workspace's own patterns are policy and stay in force in every case.

    Both defaults are fixed, and deliberately not configurable: what a checkout keeps back is
    :class:`~plyngent.kernel.ignore.Rules`, and what one call asks to see is this. A default taken from
    the config would mean a workspace switch rewrote the prompt prefix, which is the one thing
    :mod:`plyngent.kernel.toolspec` refuses to let a schema do.
    """

    show_hidden: bool = False
    ignore_lists: list[str] | None = None

    def __post_init__(self) -> None:
        """Refuse a source that is a path: a name is looked for in every directory of the walk."""
        for name in self.ignore_lists or []:
            if not name or "/" in name or "\\" in name:
                msg = f"{name!r} is a path: an ignore source is named by its file name alone"
                raise ValueError(msg)


class EnumerateArgs(ToolArgs, kw_only=True):
    """The one field every tool that lists a tree shares, and nothing else.

    A tool that enumerates cannot help having an opinion about what to leave out, so the opinion is
    declared once, next to the policy it is answered by, and
    not as a boolean per class of path in each tool's own package.
    """

    filter: Filter = Filter()


@dataclass(frozen=True, slots=True)
class Entry:
    """One path a listing shows: where it is under the walk's base, what it is, and how deep.

    ``relative`` is relative to the base the walk started at, never to the workspace root: a walk
    may start in a scratch directory, and the tool that asked knows which of the two it wants to
    show.
    """

    relative: Path
    is_dir: bool
    depth: int


@dataclass(frozen=True, slots=True)
class Listing:
    """One walk's answer: the entries, in the order they are shown, and how much it kept back.

    ``skipped`` counts what the rules hid, so "why is that file not here" has an answer that is not a
    guess. A listing is taken whole, because every caller already needs all of it — a truncated answer is
    counted as it is truncated.
    """

    entries: tuple[Entry, ...] = ()
    skipped: Skips = field(default_factory=Skips)


def walk(workspace: Workspace, base: Path, *, show: Filter | None = None, depth: int | None = None) -> Listing:
    """The entries inside *base*, depth first, in name order, as *show* and the workspace allow.

    ``depth`` is how many levels below *base* to reach — ``1`` for a single directory listing, ``None``
    for the whole tree. A directory a rule excludes is not descended into unless some rule could
    re-include something below it, which is what keeps ``!keep`` working without walking every cache
    directory in the checkout.
    """
    return _Walk(workspace, show if show is not None else Filter(), depth, base).listing()


@dataclass(frozen=True, slots=True)
class _Layer:
    """One source's rules, and the directory they are relative to."""

    base: Path
    patterns: Patterns


@dataclass(slots=True)
class _Walk:
    """One walk's fixed decisions — whose rules, what the call wants, how far to go — and its tally."""

    workspace: Workspace
    show: Filter
    depth: int | None
    base: Path
    entries: list[Entry] = field(default_factory=list)
    dirs: int = 0
    files: int = 0

    def listing(self) -> Listing:
        """The entries under the base this walk was made for."""
        self._visit(self.base, self._layers_to(), level=0, inherited=False)
        return Listing(tuple(self.entries), Skips(dirs=self.dirs, files=self.files))

    def _visit(self, directory: Path, layers: tuple[_Layer, ...], *, level: int, inherited: bool) -> None:
        """Collect the entries of *directory*, whose own ignore rules are in *layers*."""
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError:
            return
        for child in children:
            is_dir = _is_dir(child)
            if self.workspace.denied_by(child, directory=is_dir) is not None:
                continue
            if self._hidden(child, is_dir=is_dir):
                self._skip(is_dir=is_dir)
                continue
            reachable = self._reaches(child, is_dir=is_dir, level=level)
            deeper = self._into(child, layers) if reachable else layers
            if _ignored(layers, child, is_dir=is_dir, inherited=inherited):
                self._skip(is_dir=is_dir)
                if reachable and _rewinds(deeper):
                    self._visit(child, deeper, level=level + 1, inherited=True)
                continue
            self.entries.append(Entry(relative=child.relative_to(self.base), is_dir=is_dir, depth=level + 1))
            if reachable:
                self._visit(child, deeper, level=level + 1, inherited=False)

    def _skip(self, *, is_dir: bool) -> None:
        """Record one path the rules kept back; what is inside a directory is not counted."""
        if is_dir:
            self.dirs += 1
        else:
            self.files += 1

    def _layers_to(self) -> tuple[_Layer, ...]:
        """The ignore layers in force for entries directly inside the base.

        Sources are read from the root down, because a root source speaks about every path under it;
        the workspace's own patterns come last, so a checkout cannot re-include what the host keeps
        back. A base outside the root — a scratch directory — has no sources, only that policy.
        """
        ignore = self.workspace.listing.ignore
        root = self.workspace.root
        base = self.base
        if not base.is_relative_to(root):
            return _append((), base, ignore.policy)
        layers: tuple[_Layer, ...] = ()
        for directory in _through(root, base):
            layers = _append(layers, directory, ignore.read(directory, self._names()))
            if directory == root:
                layers = _append(layers, root, ignore.policy)
        return layers

    def _into(self, child: Path, layers: tuple[_Layer, ...]) -> tuple[_Layer, ...]:
        """*layers* with the rules *child* itself carries, for the entries inside it.

        Only a directory under the root carries rules: a scratch directory is not a checkout, and
        reading a source out of one would make a listing depend on what was written into it.
        """
        if not child.is_relative_to(self.workspace.root):
            return layers
        return _append(layers, child, self.workspace.listing.ignore.read(child, self._names()))

    def _names(self) -> tuple[str, ...]:
        """Which ignore sources to look for, this call's answer or the workspace's."""
        wanted = self.show.ignore_lists
        return tuple(self.workspace.listing.ignore.files if wanted is None else wanted)

    def _reaches(self, child: Path, *, is_dir: bool, level: int) -> bool:
        """Whether the walk looks inside *child*: a real directory, and not past ``depth``.

        A symlinked directory is shown and not entered, because a walk that followed one could leave
        the root or come back around to where it started.
        """
        if not is_dir or child.is_symlink():
            return False
        return self.depth is None or level + 1 < self.depth

    def _hidden(self, child: Path, *, is_dir: bool) -> bool:
        """Whether a name alone keeps *child* out, before any ignore source is consulted."""
        if not is_dir or self.show.show_hidden:
            return False
        name = child.name
        return name.startswith(".") or name in self.workspace.listing.hidden


def _append(layers: tuple[_Layer, ...], base: Path, patterns: Patterns | None) -> tuple[_Layer, ...]:
    """*layers* with *patterns* added as the last word about paths under *base*."""
    if not patterns:
        return layers
    return (*layers, _Layer(base=base, patterns=patterns))


def _through(root: Path, leaf: Path) -> tuple[Path, ...]:
    """The directories from *root* down to *leaf*, both included."""
    relative = leaf.relative_to(root)
    return (root, *(root.joinpath(*relative.parts[: index + 1]) for index in range(len(relative.parts))))


def _ignored(layers: tuple[_Layer, ...], path: Path, *, is_dir: bool, inherited: bool) -> bool:
    """Whether *path* is ignored, the innermost source that speaks about it having the last word."""
    ignored = inherited
    for layer in layers:
        try:
            relative = path.relative_to(layer.base)
        except ValueError:
            continue
        decision = layer.patterns.decide(relative.as_posix(), is_dir=is_dir)
        if decision is not None:
            ignored = decision
    return ignored


def _rewinds(layers: tuple[_Layer, ...]) -> bool:
    """Whether any layer in force could re-include a path below an excluded directory."""
    return any(layer.patterns.rewinds for layer in layers)


def _is_dir(path: Path) -> bool:
    """Whether *path* is a directory; a path that cannot be asked about is not one."""
    try:
        return path.is_dir()
    except OSError:
        return False
