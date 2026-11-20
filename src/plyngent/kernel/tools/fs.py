"""Tools whose operand is a path, a tree or metadata — never a file's contents.

Which paths a listing shows is not decided here: :func:`plyngent.kernel.listing.walk` answers that
for every tool that enumerates, from the workspace's :class:`~plyngent.kernel.listing.Rules` and the
call's :class:`~plyngent.kernel.listing.Filter`. What is left in this module is the shape of the
answer — a glob match, a tree in one of three renderings, a move's one line — and nothing about
dotfiles, version-control metadata or ignore files.

Every refusal, every "not quite" and every failure comes out of a value, with no sentence of its own: the
kind of failure decides the status and the retry (:mod:`plyngent.kernel.failure`), and a call that could
be made to work by changing one argument says so with a hint naming that argument, leaving the model's own
parameters alone. Every filesystem call is blocking, so every one of them runs off the loop: a host
that stalled its own turns on a directory read would be a host whose PTY reads stopped with it.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from stat import S_IMODE
from typing import TYPE_CHECKING, Annotated, Literal

import msgspec

from plyngent.kernel.access import MAX_GRANTS, AccessMode, covers
from plyngent.kernel.failure import Failure
from plyngent.kernel.listing import Entry, EnumerateArgs, Filter, Listing, walk
from plyngent.kernel.paths import display, normalize
from plyngent.kernel.temps import TempWorkspaceError, TooManyWorkspacesError, under_system_temp
from plyngent.kernel.toolresult import DenialKind, Effects, Hint, Meta, ToolResult
from plyngent.kernel.tools import lifted
from plyngent.kernel.tools.bodies import kept
from plyngent.kernel.toolspec import Group, Mutation, Posture, Remember, Res, Role, ToolArgs, ToolSpec
from plyngent.kernel.trash import Kind, Trouble, to_trash
from plyngent.kernel.words import counted
from plyngent.kernel.workspace import Denied, PathRefused
from plyngent.wire.ask import Confirmation, Severity, as_agreed

if TYPE_CHECKING:
    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.env import Env
    from plyngent.kernel.workspace import Workspace

#: What ``listdir`` can be asked for per entry. A closed set, so the schema says what it accepts.
type Stat = Literal["size", "mtime", "mode", "symlink"]


class GlobArgs(EnumerateArgs, kw_only=True):
    """Find files under the workspace matching a glob pattern (e.g. ``**/*.py``).

    The pattern is matched against paths relative to ``path`` (default the workspace root), and the
    matches are shown relative to the root. Which paths are looked at — dotfiles, version-control
    metadata, whatever the tree's ignore sources leave out — is ``filter``'s business.
    """

    pattern: Annotated[str, msgspec.Meta(min_length=1)]
    path: str = "."
    max_matches: Annotated[int, msgspec.Meta(ge=1)] = 200


class ListdirArgs(EnumerateArgs, kw_only=True):
    """List entries in a directory under the workspace (name and type).

    ``stat`` asks for columns per entry, in the order given: ``size`` (bytes), ``mtime`` (UTC,
    second precision), ``mode`` (the permission bits, octal) and ``symlink`` (where it points, or
    ``-``). Empty by default, because a listing of a directory is usually about its shape, and the
    metadata is the second question — ``tree`` is the tool for shape.
    """

    path: str = "."
    stat: list[Stat] = msgspec.field(default_factory=list)


class TreeArgs(EnumerateArgs, kw_only=True):
    """Show a directory tree under the workspace.

    ``format`` is how it is drawn for the model: ``markdown`` (default, cheapest to read),
    ``flat`` (one usable path per line, directories with a trailing ``/``) or ``decorated`` (the
    box-drawing tree a human expects). ``max_depth`` bounds how deep the walk goes (1 = the
    directory's own entries).

    ``max_entries`` is a budget **per directory** for the two tree renderings, so one large directory
    cannot make the rest of the tree disappear; ``flat`` has no visible levels, so there it bounds the
    total number of lines. ``filter`` says what may be left out, and nothing here can widen what the
    workspace itself refuses.
    """

    path: str = "."
    format: Literal["markdown", "flat", "decorated"] = "markdown"
    max_depth: Annotated[int, msgspec.Meta(ge=1)] = 4
    max_entries: Annotated[int, msgspec.Meta(ge=1)] = 50


class CopyArgs(ToolArgs, kw_only=True):
    """Copy a file or directory under the workspace (``shutil.copy2`` / ``copytree``).

    The destination's parents are created as needed. A file copied onto an existing directory lands
    inside it under its own name. If the destination itself exists, set ``overwrite=true`` to replace
    it — a copy does not silently replace anything.
    """

    src: str
    dst: str
    overwrite: bool = False


class MoveArgs(ToolArgs, kw_only=True):
    """Move or rename a file or directory under the workspace (``shutil.move``).

    What the old path was is gone from the tree afterwards, so this is a tool that has to say why it
    is being called.
    """

    src: str
    dst: str
    overwrite: bool = False


class DeleteArgs(ToolArgs, kw_only=True):
    """Delete a file or directory under the workspace.

    The path goes to the system trash, so the deletion can be undone there; ``permanent=true``
    removes it for good instead. A non-empty directory requires ``recursive=true``, and the workspace
    root cannot be deleted. A path under the system temp directory is scratch space and is removed
    for good either way, as it is when the host has no trash. A delete has to say why it is being
    called.
    """

    path: str
    recursive: bool = False
    permanent: bool = False


async def glob_paths(env: Env, args: GlobArgs) -> ToolResult:
    workspace = env.workspace
    try:
        base = env.resolve(args.path, AccessMode.READ)
    except PathRefused as exc:
        return lifted(exc)
    if not base.is_dir():
        return ToolResult.of(Failure.MISSING, f"not a directory: {args.path}")

    listing = await _listed(workspace, base, show=args.filter, scope=env.scope())
    found = [
        display(base / entry.relative, root=workspace.root)
        for entry in listing.entries
        if entry.relative.full_match(args.pattern)
    ]
    if not found:
        return ToolResult.ok("no matches", meta=_meta(listing))
    shown = found[: args.max_matches]
    text = "\n".join(shown)
    if len(found) > len(shown):
        text += f"\n...[truncated at {args.max_matches} matches]"
    part = await kept(env, text, summary=f"{len(shown)} paths matching {args.pattern} under {args.path}")
    if isinstance(part, ToolResult):
        return part
    return ToolResult.ok(f"{len(shown)} paths", body=part, meta=_meta(listing))


async def listdir(env: Env, args: ListdirArgs) -> ToolResult:
    workspace = env.workspace
    try:
        target = env.resolve(args.path, AccessMode.READ)
    except PathRefused as exc:
        return lifted(exc)
    if not target.is_dir():
        return ToolResult.of(Failure.MISSING, f"not a directory: {args.path}")

    listing = await _listed(workspace, target, show=args.filter, depth=1, scope=env.scope())
    if not listing.entries:
        return ToolResult.ok("empty directory", meta=_meta(listing))
    lines = [_line(target, entry, args.stat) for entry in listing.entries]
    part = await kept(env, "\n".join(lines), summary=f"{counted(len(lines), 'entry')} of {args.path}")
    if isinstance(part, ToolResult):
        return part
    return ToolResult.ok(f"{len(lines)} entries", body=part, meta=_meta(listing))


async def tree(env: Env, args: TreeArgs) -> ToolResult:
    workspace = env.workspace
    try:
        origin = env.resolve(args.path, AccessMode.READ)
    except PathRefused as exc:
        return lifted(exc)
    if not origin.is_dir():
        return ToolResult.of(Failure.MISSING, f"not a directory: {args.path}")

    listing = await _listed(workspace, origin, show=args.filter, depth=args.max_depth, scope=env.scope())
    shown, omitted = _within(listing.entries, args)
    lines = _draw(args.format, shown, label=_label(args.path), more=omitted)
    total = len(shown) + omitted
    summary = f"{len(shown)} entries" if omitted == 0 else f"{len(shown)} of {total} entries"
    part = await kept(env, "\n".join(lines), summary=f"{summary} under {args.path}")
    if isinstance(part, ToolResult):
        return part
    return ToolResult.ok(summary, body=part, meta=_meta(listing))


async def copy_path(env: Env, args: CopyArgs) -> ToolResult:
    workspace = env.workspace
    try:
        source = env.resolve(args.src, AccessMode.READ)
        destination = env.resolve(args.dst, AccessMode.WRITE)
    except PathRefused as exc:
        return lifted(exc)
    if not _present(source):
        return ToolResult.of(Failure.MISSING, f"source does not exist: {args.src}")

    try:
        target = _prepare(env, source, destination, overwrite=args.overwrite)
    except _DestinationTakenError:
        return _taken(COPY.name, args)
    except PathRefused as exc:
        return lifted(exc)

    try:
        if source.is_dir() and not source.is_symlink():
            kind = "directory"
            _ = await env.scope().blocking(shutil.copytree, source, target, symlinks=True)
        else:
            kind = _kind(source)
            _ = await env.scope().blocking(shutil.copy2, source, target, follow_symlinks=False)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"copy failed: {exc}", effects=Effects(changed=(target,)))
    summary = f"copied {kind} {args.src} -> {display(target, root=workspace.root)}"
    return ToolResult.ok(summary, effects=Effects(changed=(target,)))


async def move_path(env: Env, args: MoveArgs) -> ToolResult:
    workspace = env.workspace
    pair = _movable(env, args)
    if isinstance(pair, ToolResult):
        return pair
    source, destination = pair

    try:
        target = _prepare(env, source, destination, overwrite=args.overwrite)
    except _DestinationTakenError:
        return _taken(MOVE.name, args)
    except PathRefused as exc:
        return lifted(exc)

    try:
        moved = await env.scope().blocking(_move, source, target)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"move failed: {exc}", effects=Effects(changed=(target,)))
    summary = f"moved {_kind(moved)} {args.src} -> {display(moved, root=workspace.root)}"
    return ToolResult.ok(summary, effects=Effects(changed=(source, moved)))


async def delete_path(env: Env, args: DeleteArgs) -> ToolResult:
    workspace = env.workspace
    try:
        target = env.resolve(args.path, AccessMode.WRITE)
    except PathRefused as exc:
        return lifted(exc)
    if target == workspace.root:
        return ToolResult.denied(DenialKind.NEVER_ALLOWED, "cannot delete the workspace root")
    if not _present(target):
        return ToolResult.of(Failure.MISSING, f"path does not exist: {args.path}")
    return await _deleted(env, target, args)


# -- asking for a path that is not inside the workspace -------------------------------------------


class Wanted(StrEnum):
    """What a path is needed *for*, in the words the model reads.

    A ``StrEnum`` and not :class:`~plyngent.kernel.access.AccessMode`: that one is an ``IntEnum``, and a
    schema generated from it would offer the model ``1``, ``2`` and ``3``. ``exec`` names a need of its
    own, distinct from a wider ``write``: the directory a command runs in is not a file that may be changed.
    """

    READ = "read"
    WRITE = "write"
    EXEC = "exec"


#: The mode each spelling asks for. One row per value of the closed set, so a new one cannot be added
#: to the schema without a decision about what it means.
_MODES: dict[Wanted, AccessMode] = {
    Wanted.READ: AccessMode.READ,
    Wanted.WRITE: AccessMode.WRITE,
    Wanted.EXEC: AccessMode.EXEC,
}


class AccessArgs(ToolArgs, kw_only=True):
    """Ask to reach a path outside the workspace, for the rest of this conversation.

    ``path`` is the path a call was refused on, and ``mode`` is what that call needed. The grant covers
    that path and everything under it, and it belongs to this conversation, outliving this process —
    so it still holds when the conversation is opened again, and it answers for nobody else. A path the
    workspace's denylist refuses is refused here too: nothing grants one of those.
    """

    path: str
    mode: Wanted = Wanted.READ


async def request_directory_access(env: Env, args: AccessArgs) -> ToolResult:
    """Ask a person to allow *path*, and remember a yes for the rest of this conversation."""
    workspace = env.workspace
    target = normalize(args.path, base=workspace.root)
    denied = workspace.denied_by(target)
    if denied is not None:
        return lifted(Denied(target, denied))
    need = _MODES[args.mode]
    shown = display(target, root=workspace.root)
    if (
        workspace.under_roots(target)
        or workspace.under_scratch(target, env.scratch)
        or covers(env.grants, target, need)
    ):
        return ToolResult.ok(f"already accessible: {shown}")
    if len(env.grants) >= MAX_GRANTS:
        return ToolResult.of(
            Failure.LIMIT,
            f"this conversation already holds {MAX_GRANTS} grants, which is the limit",
        )
    answer = as_agreed(
        await env.ask(
            Confirmation(
                prompt=f"allow {args.mode.value} on {shown} for this conversation?",
                # A grant reaches past the workspace, which is the row of the table that says ``danger``:
                # what it hands out outlives this call, and nothing here can check what is done through it.
                severity=Severity.DANGER,
            )
        )
    )
    if not answer.accepted:
        return ToolResult.unagreed(answer, about=f"{args.mode.value} on {shown}")
    _ = await env.grant(target, need)
    return ToolResult.ok(f"allowed {args.mode.value}: {shown}")


class TemporaryArgs(ToolArgs, kw_only=True):
    """Make a scratch directory outside the workspace, which this host removes when it stops.

    ``prefix`` names it on disk (``plyngent-<prefix>-<random>``), so somebody looking at the system temp
    directory can tell two of them apart. Paths inside it need no grant: a temporary workspace is one of
    the roots every path tool resolves against, and the removal is the host's, not the model's.
    """

    prefix: str = ""


async def new_temporary_workspace(env: Env, args: TemporaryArgs) -> ToolResult:
    """Hand out one scratch directory under the system temp dir, and say where it is."""
    try:
        made = await env.workspace.temps.create(args.prefix, owner=env.scratch, scope=env.scope())
    except TooManyWorkspacesError as exc:
        return ToolResult.of(Failure.LIMIT, str(exc))
    except TempWorkspaceError as exc:
        return ToolResult.of(Failure.LIMIT, str(exc))
    return ToolResult.ok(f"temporary workspace: {made}")


GLOB = ToolSpec(
    name="glob_paths",
    group=Group.FS,
    args=GlobArgs,
    run=glob_paths,
    uses=Res.PATHS | Res.DOCUMENT,
    roles={"path": Role.READ},
)

LISTDIR = ToolSpec(
    name="listdir",
    group=Group.FS,
    args=ListdirArgs,
    run=listdir,
    uses=Res.PATHS | Res.DOCUMENT,
    roles={"path": Role.READ},
)

TREE = ToolSpec(
    name="tree",
    group=Group.FS,
    args=TreeArgs,
    run=tree,
    uses=Res.PATHS | Res.DOCUMENT,
    roles={"path": Role.READ},
)

COPY = ToolSpec(
    name="copy_path",
    group=Group.FS,
    args=CopyArgs,
    run=copy_path,
    uses=Res.PATHS,
    mutates=Mutation.WORKSPACE,
    roles={"src": Role.READ, "dst": Role.WRITE},
    # Landing on something that is already there is not the ordinary copy: it replaces, so it is asked
    # every time and at the level a replacement carries.
    escalates={"overwrite": Severity.CONFIRM},
)

MOVE = ToolSpec(
    name="move_path",
    group=Group.FS,
    args=MoveArgs,
    run=move_path,
    uses=Res.PATHS,
    mutates=Mutation.WORKSPACE | Mutation.DESTRUCTIVE,
    roles={"src": Role.WRITE, "dst": Role.WRITE},
    # The ordinary move is already as heavy as a call gets; the argument only takes away the memory.
    escalates={"overwrite": Severity.DANGER},
)

DELETE_PATH = ToolSpec(
    name="delete_path",
    group=Group.FS,
    args=DeleteArgs,
    run=delete_path,
    uses=Res.PATHS,
    mutates=Mutation.WORKSPACE | Mutation.DESTRUCTIVE,
    roles={"path": Role.WRITE},
    escalates={"permanent": Severity.DANGER},
)

#: Asking for a path outside the workspace is not approved like a write inside it: the question *is*
#: the tool, which is why its posture derives to ``AUTO``, and a yes is not remembered by the approvals
#: table but by the conversation's own grant table.
REQUEST_ACCESS = ToolSpec(
    name="request_directory_access",
    group=Group.FS,
    args=AccessArgs,
    run=request_directory_access,
    uses=Res.PATHS | Res.HUMAN,
)

#: A scratch directory widens what is reachable without touching anything, so the posture is declared
#: outright, with nothing derived: one "yes" for this host is enough, because the directory it hands out
#: is the thing that was agreed to and the host removes it again.
TEMPORARY = ToolSpec(
    name="new_temporary_workspace",
    group=Group.FS,
    args=TemporaryArgs,
    run=new_temporary_workspace,
    uses=Res.PATHS,
    mutates=Mutation.HOST,
    posture=Posture.ONCE,
    remember=Remember.HOST,
)

SPECS: tuple[ToolSpec, ...] = (GLOB, LISTDIR, TREE, COPY, MOVE, DELETE_PATH, REQUEST_ACCESS, TEMPORARY)


# -- rendering ---------------------------------------------------------------------------------


def _line(base: Path, entry: Entry, wanted: list[Stat]) -> str:
    """One entry of a single-level listing, with the columns the call asked for."""
    columns = ["dir" if entry.is_dir else "file", entry.relative.name]
    columns.extend(_column(base / entry.relative, column) for column in wanted)
    return "\t".join(columns)


def _column(path: Path, wanted: Stat) -> str:
    """One column of a listing, or ``-`` where the entry cannot answer the question."""
    try:
        info = path.lstat()
        if wanted == "size":
            return str(info.st_size)
        if wanted == "mtime":
            return datetime.fromtimestamp(info.st_mtime, UTC).isoformat(timespec="seconds")
        if wanted == "mode":
            return format(S_IMODE(info.st_mode), "03o")
        return path.readlink().as_posix() if path.is_symlink() else "-"
    except OSError:
        return "-"


def _draw(format: str, entries: list[Entry], *, label: str, more: int) -> list[str]:  # noqa: A002 — the model's name
    """The lines one of the three renderings of a tree comes to."""
    if format == "flat":
        lines = [f"{entry.relative.as_posix()}/" if entry.is_dir else entry.relative.as_posix() for entry in entries]
    elif format == "decorated":
        lines = [f"{label}/", *_decorated(entries)]
    else:
        lines = [f"{'  ' * (entry.depth - 1)}- {entry.relative.name}{'/' if entry.is_dir else ''}" for entry in entries]
    if more > 0:
        lines.append(f"… ({more} more entries not shown)")
    return lines


def _decorated(entries: list[Entry]) -> list[str]:
    """The box-drawing tree: a branch per entry, and a continuation per ancestor that has more."""
    lines: list[str] = []
    continuation: list[str] = [""]
    for index, entry in enumerate(entries):
        last = _last_sibling(entries, index)
        while len(continuation) <= entry.depth:
            continuation.append("")
        continuation[entry.depth] = "    " if last else "│   "
        prefix = "".join(continuation[1 : entry.depth])
        branch = "└── " if last else "├── "
        lines.append(f"{prefix}{branch}{entry.relative.name}{'/' if entry.is_dir else ''}")
    return lines


def _last_sibling(entries: list[Entry], index: int) -> bool:
    """Whether the entry at *index* is the last of its siblings, its own subtree not counted."""
    depth = entries[index].depth
    following = index + 1
    while following < len(entries) and entries[following].depth > depth:
        following += 1
    return following == len(entries) or entries[following].depth < depth


def _within(entries: tuple[Entry, ...], args: TreeArgs) -> tuple[list[Entry], int]:
    """The entries a tree shows and how many its budget kept back.

    The two tree renderings bound *each directory*, because a budget shared by the whole tree makes a
    project with one large directory look like a project with one directory. ``flat`` has no visible
    levels, so there the budget is over the lines themselves.
    """
    if args.format == "flat":
        return list(entries[: args.max_entries]), max(0, len(entries) - args.max_entries)
    return _per_directory(entries, args.max_entries)


def _per_directory(entries: tuple[Entry, ...], limit: int) -> tuple[list[Entry], int]:
    """The entries left after each directory gives up its own budget, and how many were dropped.

    Entries arrive depth first, so an entry at depth *d* belongs to the closest earlier entry at
    depth *d - 1*; when a directory runs out of budget, its own subtree goes with it, which is why
    the walk is not simply filtered entry by entry.
    """
    shown: list[Entry] = []
    taken: dict[Path, int] = {}
    spent: int | None = None
    omitted = 0
    for entry in entries:
        if spent is not None:
            if entry.depth > spent:
                omitted += 1
                continue
            spent = None
        used = taken.get(entry.relative.parent, 0)
        if used >= limit:
            omitted += 1
            spent = entry.depth
            continue
        taken[entry.relative.parent] = used + 1
        shown.append(entry)
    return shown, omitted


# -- filesystem work that runs off the loop ----------------------------------------------------


async def _listed(
    workspace: Workspace,
    base: Path,
    *,
    show: Filter,
    depth: int | None = None,
    scope: Scope,
) -> Listing:
    """The walk's answer, collected in a thread: reading a tree is a blocking call."""
    return await scope.blocking(walk, workspace, base, show=show, depth=depth)


def _meta(listing: Listing) -> Meta:
    """What a listing left out, so "why is that file not here" has an answer."""
    return Meta(skipped=listing.skipped)


def _prepare(env: Env, source: Path, destination: Path, *, overwrite: bool) -> Path:
    """The path the copy or move writes to, with the destination's parents ready.

    A file named onto a directory lands inside it, and that child is resolved again: the destination
    may be a granted path outside the root, and the child must be allowed for the same reason its
    parent was.
    """
    target = destination
    if source.is_file() and destination.is_dir():
        target = env.resolve(destination / source.name, AccessMode.WRITE)
    if _present(target):
        if not overwrite:
            raise _DestinationTakenError
        _destroy(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _movable(env: Env, args: MoveArgs) -> ToolResult | tuple[Path, Path]:
    """The source and destination of a move, or the result that says why there is no move."""
    workspace = env.workspace
    try:
        source = env.resolve(args.src, AccessMode.WRITE)
        destination = env.resolve(args.dst, AccessMode.WRITE)
    except PathRefused as exc:
        return lifted(exc)
    if not _present(source):
        return ToolResult.of(Failure.MISSING, f"source does not exist: {args.src}")
    if source == workspace.root:
        return ToolResult.denied(DenialKind.NEVER_ALLOWED, "cannot move the workspace root")
    if destination.is_relative_to(source):
        return ToolResult.of(Failure.ARGS, f"cannot move {args.src} inside itself")
    return source, destination


async def _deleted(env: Env, target: Path, args: DeleteArgs) -> ToolResult:
    """The delete itself, once the workspace has said *target* may be touched."""
    permanent = args.permanent or not env.trash_enabled or under_system_temp(target)
    try:
        summary = await env.scope().blocking(_remove, target, recursive=args.recursive, permanent=permanent)
    except _NotEmptyError:
        return ToolResult.of(
            Failure.BLOCKED,
            f"directory not empty: {args.path}",
            hints=(Hint(tool=DELETE_PATH.name, given=(("recursive", "true"),), why="remove it whole"),),
        )
    except _TrashDeclinedError as exc:
        return ToolResult.of(Failure.BLOCKED, exc.trouble.said, hints=_trash_hint(exc.trouble))
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"delete failed: {exc}", effects=Effects(changed=(target,)))
    # A directory this host handed out as scratch stops being one when it is gone, which is what gives
    # the slot back: nothing else ever removed one, so a conversation that deleted its own scratch would
    # otherwise keep the room counted until the host went down (E152).
    env.workspace.temps.release(target)
    return ToolResult.ok(f"deleted {summary} {args.path}", effects=Effects(changed=(target,)))


def _move(source: Path, target: Path) -> Path:
    """``shutil.move`` and the resolved place it landed, off the loop with the rest of the syscalls."""
    return Path(shutil.move(str(source), str(target))).resolve()


def _taken(tool: str, args: CopyArgs | MoveArgs) -> ToolResult:
    """The refusal for an occupied destination, naming the one value that would replace it."""
    hint = Hint(tool=tool, given=(("overwrite", "true"),), why="replace what is already there")
    return ToolResult.of(Failure.BLOCKED, f"destination exists: {args.dst}", hints=(hint,))


def _trash_hint(trouble: Trouble) -> tuple[Hint, ...]:
    """The one value that would go through, and only when the no is about this path.

    A volume that will not take a path leaves one way out, so it is named. An installation with no
    ``send2trash`` is not a fact about the path: the same card would invite an irreversible delete to
    work around a broken install, and no argument the model can pass fixes a missing library (E189).
    """
    if trouble.kind is not Kind.REFUSED:
        return ()
    return (Hint(tool=DELETE_PATH.name, given=(("permanent", "true"),), why="remove it for good"),)


def _remove(target: Path, *, recursive: bool, permanent: bool) -> str:
    """Take *target* away, to the trash or for good; returns what was removed, for the summary."""
    directory = target.is_dir() and not target.is_symlink()
    if directory and not recursive and any(target.iterdir()):
        raise _NotEmptyError
    kind = "directory" if directory else _kind(target)
    if permanent:
        _destroy(target)
        return kind
    trouble = to_trash(target)
    if trouble is not None:
        raise _TrashDeclinedError(trouble)
    return f"{kind} (moved to system trash)"


def _destroy(path: Path) -> None:
    """Remove *path* for good, whatever it is."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif _present(path):
        path.unlink()


def _present(path: Path) -> bool:
    """Whether anything is at *path* — a broken symlink is something, so ``exists`` is not enough."""
    return path.exists() or path.is_symlink()


def _kind(path: Path) -> str:
    """What to call *path* in a sentence, in the order the questions have to be asked."""
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    return "file"


def _label(path: str) -> str:
    """The tree's own label: what the caller asked for, without a trailing separator."""
    return path.rstrip("/\\") or "."


class _DestinationTakenError(Exception):
    """The destination is there and the call did not ask to replace it."""


class _NotEmptyError(Exception):
    """The directory has entries and the call did not ask for a recursive delete."""


class _TrashDeclinedError(Exception):
    """The platform's trash would not take the path, and nothing was destroyed.

    It carries the trouble whole, because the two kinds of no are answered differently: a caller that
    only saw a sentence would have to read it back to know whether an argument exists (E189).
    """

    def __init__(self, trouble: Trouble) -> None:
        super().__init__(trouble.said)
        self.trouble: Trouble = trouble
