"""The workspace's own versions: one bare repository beside the journal, one commit per changing call.

**Why a repository and not a table.** What a call left behind has to be readable back — as a difference
for a person, and as bytes to put back — and those two questions are the two git already answers from a
content-addressed object store. A table of blobs beside the journal would have to grow its own
deduplication, its own compression and its own diff, and would then be a worse copy of the thing git is.
So the store is a bare repository in this program's data directory, beside the journal, named for the
workspace the same way (:func:`history_path`). The workspace's own repository, if it has one, is a
different repository and is not read here: this is the record of what *this kernel's calls* did, and it
exists whether or not anybody ever ran `git init` in that tree.

**One commit per changing call.** A version is not a turn, a file, or a session — it is one call, which
is the unit a result is written for and the unit a person means by "undo that". The commits are a line:
each names the one before it, so the *previous* state of a path is a property of the store rather than
something the caller had to remember to keep. Nothing here rewinds the line — a rollback restores
content and records a new commit — because the journal's own rule is that history is appended to.

**What enters a version is decided here, not by git.** ``record`` is handed the paths a call left
different and hashes exactly those; it does not ask git to sweep the tree, because a sweep would apply
git's own ignore rules and find files this kernel deliberately does not look at (a tree's hidden
directories, a denylisted path). The one caller that cannot name its paths — a program that was started
— enumerates them through the kernel's own walk and hands them over the same way. What this store
therefore cannot show is the same thing the kernel could not see: a path its rules hide, a path outside
the workspace, and a change a program made somewhere the walk never goes.

**Hardened like every other git this build starts.** The person's configuration is taken out of the way
(``GIT_CONFIG_GLOBAL`` and ``GIT_CONFIG_SYSTEM`` point at nothing), a commit is not signed, the
repository never takes optional locks, the locale is fixed, and the index is a file of this store's own
— so nothing here can rewrite the workspace's own ``.git/index`` or turn a read into a program somebody
else's configuration chose.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.kernel.cancel import Scope

#: The branch every version is committed to. One line, and the only ref this store writes.
BRANCH = "workspace"

#: The option every call carries: a repository this build only reads from never takes the lock that
#: would leave a file behind in a directory it does not own (§0.8).
LOCKLESS = ("--no-optional-locks",)

#: Who a version says it was made by — this program, because it was.
MADE_BY = "plyngent"

#: The mode git gives a path that is not there. A removal in ``update-index`` is written as this.
GONE = "0" * 6 + " " + "0" * 40

#: What this build calls a regular file and a file somebody may execute.
FILE_MODE = "100644"
EXEC_MODE = "100755"
#: And a path that is a link: git keeps the target as text and this mode beside it.
SYMLINK_MODE = "120000"

#: The empty tree id: what the very first version continues from, when there is no version yet.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def available() -> bool:
    """Whether this machine has a git for the store to be built on.

    One answer in one place: the composition root decides whether a host keeps versions at all (§10.1
    E26 — what exists is decided at install), and a test asks the same question to know whether it can run.
    """
    return shutil.which("git") is not None


def history_path(journal: Path | str) -> Path:
    """Where one workspace's versions live: a bare repository beside the journal that names it.

    A sibling and not a subdirectory, because the two are the same record at two resolutions — the
    conversation, and the bytes its calls left — and both are kept out of the tree they are about.
    """
    return Path(journal).parent / "history.git"


class Unreadable(Exception):  # noqa: N818 — a store that cannot answer is a state, not a failed call
    """The store could not be read or written: git said no, and said why."""


#: Why a rollback cannot happen at all, and why it cannot happen over what is on disk now.
NO_VERSIONS_KEPT = "this workspace has no versions to go back to"
NOTHING_MOVED = "nothing in that version changed a path, so there is nothing to put back"


def _refused_by_disk(name: str, undo: str) -> str:
    """Why one path was not put back: it is not what that version left, so something else wrote it."""
    return f"{name} is not what version {undo[:8]} left, so it was not put back"


def _outside_version(name: str) -> str:
    """Why a version's own path was refused: a rollback may only land inside the tree it is about."""
    return f"{name!r} is not a path a version of this workspace could hold"


class Unwilling(Exception):  # noqa: N818 — a rollback that would lose somebody's work is a refusal
    """The disk no longer holds what the version left, so putting it back would throw a change away."""


@dataclass(frozen=True, slots=True)
class Undone:
    """What one rollback left: the version it wrote, the one it took content from, and the paths moved."""

    revision: str
    target: str
    restored: tuple[str, ...] = ()
    """The store could not be read or written: git said no, and said why."""


class History:
    """One workspace's versions, in a bare repository beside its journal.

    Built from two absolute paths and nothing else: the tree the versions are *about*, and the store
    they are kept in. Neither has to exist yet — the store is made on the first version recorded, so a
    program that is opened and closed without changing anything leaves the data directory as it was.

    Every call here starts a process, so every call runs off the loop. The three a *turn* awaits —
    :meth:`record`, :meth:`head` and :meth:`differing` — take the call's scope, because a turn that was
    stopped still owes its history what it was in the middle of keeping, and settling waits for it. The
    rest answer commands and listings, which belong to no turn and so have nothing to be joined to.
    """

    root: Path
    path: Path

    def __init__(self, root: Path | str, path: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = Path(path)

    # -- what a caller asks for -----------------------------------------------------

    async def record(self, changed: Sequence[Path], *, says: str = "", scope: Scope) -> str:
        """Keep these paths as one version and answer its id, or ``""`` when there was nothing to keep.

        ``changed`` is absolute and may name a path that is no longer there — that is a removal, and a
        version has to record it or a deleted file looks like one that never existed. A path outside the
        tree is not this store's to keep, and is dropped rather than hashed, which is the same line
        :class:`~plyngent.kernel.toolresult.Effects` already draws between ``changed`` and ``external``.

        Everything below happens in one thread hop: it is file system work and process starts, and the
        loop this kernel runs on has other children waiting.
        """
        # Both sides are compared the same way round inside the thread: a handler's path has been
        # through the workspace's own resolution and the root this store was built with may not have
        # been, so comparing them as given would call a path outside its own tree on a system whose
        # `/tmp` is a symlink.
        if not changed:
            return ""
        return await scope.blocking(self._recorded, tuple(changed), says)

    async def head(self, *, scope: Scope) -> str:
        """The version the last recorded call left, or ``""`` when nothing has been recorded."""
        return await scope.blocking(self._head)

    async def parent(self, revision: str) -> str:
        """The version ``revision`` continues from, or ``""`` when it is the first one."""
        return await asyncio.to_thread(self._parent, revision)

    async def changed(self, revision: str) -> tuple[str, ...]:
        """What one version changed, as repository-relative paths — the answer a listing needs."""
        return await asyncio.to_thread(self._changed, revision)

    async def diff(self, older: str, newer: str, *, paths: Sequence[str] = (), context: int = 3) -> str:
        """The difference between two versions, read out of the store and never off the disk.

        This is what makes the answer stable: a file that has been written twice since is still the file
        these two versions hold, so a difference somebody asks for later is the difference that was
        made — not a second one computed against whatever the tree looks like now.
        """
        return await asyncio.to_thread(self._diff, older, newer, tuple(paths), context)

    async def blob(self, revision: str, path: str) -> bytes | None:
        """What one version holds for one path, in bytes; ``None`` when that version does not hold it."""
        return await asyncio.to_thread(self._blob, revision, path)

    async def differing(self, revision: str, paths: Sequence[Path], *, scope: Scope) -> tuple[Path, ...]:
        """Which of these paths ``revision`` does not already hold, plus the paths it holds that are gone.

        This is for the one caller that cannot say what it changed: a program was started, and its argv
        says nothing about the files it touched (§10.12 — a command's paths are not in the batch graph
        either). So the tree is enumerated by the caller's own rules and handed here, and the answer is a
        content comparison rather than a timestamp: a file written back to the same bytes is not a change,
        which is what keeps a build that touched everything from becoming a version of everything.

        Two lines are drawn and both are the store's: a path outside the tree is not compared at all, and
        a path the caller's rules hide is compared only if ``revision`` holds it *and* it is gone from
        disk — a file merely hidden from a walk keeps the version it had, because a store that deleted it
        would be a rollback that takes away a file nobody asked about.
        """
        return await scope.blocking(self._differing, revision, tuple(paths))

    async def undo(self, *, version: str = "", paths: Sequence[str] = (), force: bool = False) -> Undone:
        """Put the workspace back to the version before ``version``, and record the putting-back.

        The content comes from the store and nowhere else, so what is restored is what the version held,
        not a second guess at it. And it is restored as **a new version**, because the history here is
        appended to: "what it was before the rollback" stays where it was, and undoing a rollback is the
        same command one version further on.

        Three refusals, and the middle one is the reason this is not a one-liner: nothing has been kept
        yet, this version changed no path, and — the interesting one — the disk no longer holds what that
        version left. The last is not pedantry: putting old bytes over somebody's newer ones is exactly
        the quiet loss the approval gate exists to prevent, so it takes ``force`` to mean it.
        """
        return await asyncio.to_thread(self._undone, version, tuple(paths), force=force)

    # -- the pieces, off the loop ---------------------------------------------------

    def _recorded(self, changed: Sequence[Path], says: str) -> str:
        kept = [Path(one).resolve() for one in changed]
        ours = [one for one in kept if one.is_relative_to(self.root)]
        if not ours:
            return ""
        self._made()
        parent = self._head()
        tree = self._treed(parent, ours)
        message = f"{says}\n\npaths: {len(ours)}\n" if says else f"paths: {len(ours)}\n"
        commit = self._say("commit-tree", tree, *(() if not parent else ("-p", parent)), "-m", message)
        _ = self._say("update-ref", f"refs/heads/{BRANCH}", commit, parent)
        return commit

    def _made(self) -> None:
        """Make the store if it is not there, so a workspace nobody changed keeps no directory."""
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _ = self._run("init", "--bare", "-q", str(self.path), at=self.path.parent)

    def _treed(self, parent: str, changed: list[Path]) -> str:
        """The tree this version holds: the one before it, with these paths written over.

        Reading the previous version's tree and patching it means a version costs what a call changed,
        not what the tree holds — and it also means two calls that changed different files cannot
        disagree about the rest of the workspace.
        """
        _ = self._say("read-tree", *(() if parent else ("--empty",)), *((parent,) if parent else ()))
        self._indexed(parent, changed)
        return self._say("write-tree")

    def _indexed(self, parent: str, changed: list[Path]) -> None:
        """Put every one of these paths in the index: its current content, or its absence."""
        existing = [item for item in changed if item.exists() or item.is_symlink()]
        lines: list[str] = []
        if existing:
            hashed = self._hashed(existing)
            for item, (mode, sha) in zip(existing, hashed, strict=True):
                lines.append(f"{mode} {sha}\t{_named(item, self.root)}")
        lines.extend(f"{GONE}\t{_named(item, self.root)}" for item in changed if item not in existing)
        if lines:
            _ = self._say("update-index", "--index-info", stdin="\n".join(lines) + "\n")
        assert parent or lines, "a first version that changed nothing is not a version"

    def _hashed(self, existing: list[Path], *, write: bool = True) -> list[tuple[str, str]]:
        """The blob id and mode of each of these paths, in the order they were given.

        A symlink is the link's target as text, because that is what the path *is*; a file is its bytes.
        Both are asked of git rather than computed here, so the id this build stores is the same id any
        other git would store for the same content. ``write`` keeps the objects in the store; a comparison
        that only wants to know whether something moved computes the same id and keeps nothing.
        """
        files = [item for item in existing if not item.is_symlink()]
        keep = ("-w",) if write else ()
        shas: dict[str, str] = {}
        if files:
            said = self._say("hash-object", *keep, "--stdin-paths", stdin="\n".join(str(item) for item in files) + "\n")
            for item, sha in zip(files, said.split(), strict=True):
                shas[str(item)] = sha
        answer: list[tuple[str, str]] = []
        for item in existing:
            if item.is_symlink():
                sha = self._say("hash-object", *keep, "--stdin", stdin=str(item.readlink()))
                answer.append((SYMLINK_MODE, sha))
            else:
                executable = bool(item.stat().st_mode & 0o111)
                answer.append((EXEC_MODE if executable else FILE_MODE, shas[str(item)]))
        return answer

    def _differing(self, revision: str, paths: Sequence[Path]) -> tuple[Path, ...]:
        """Which of these paths the version does not hold as they are, and which of its own are gone."""
        known = self._held(revision)
        ours: dict[str, Path] = {}
        for one in paths:
            here = Path(one).resolve()
            if here.is_relative_to(self.root):
                ours[_named(here, self.root)] = here
        present = [one for one in ours.values() if one.exists() or one.is_symlink()]
        hashed = self._hashed(present, write=False) if present else []
        now = dict(zip([_named(one, self.root) for one in present], hashed, strict=True))
        named = [name for name, pair in now.items() if known.get(name) != pair]
        named.extend(name for name in known if not (self.root / name).exists())
        return tuple(self.root / name for name in sorted(set(named)))

    def _held(self, revision: str) -> dict[str, tuple[str, str]]:
        """What one version holds, as path to (mode, blob) — the comparison a sweep is made of."""
        if not revision:
            return {}
        said = self._say("ls-tree", "-r", "-z", revision)
        held: dict[str, tuple[str, str]] = {}
        for row in said.split("\0"):
            if not row:
                continue
            meta, _, name = row.partition("\t")
            mode, _kind, sha = meta.split()
            held[name] = (mode, sha)
        return held

    def _head(self) -> str:
        return self._ask("rev-parse", "--verify", "--quiet", f"refs/heads/{BRANCH}")

    def _parent(self, revision: str) -> str:
        return self._ask("rev-parse", "--verify", "--quiet", f"{revision}^")

    def _changed(self, revision: str) -> tuple[str, ...]:
        said = self._say("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", revision)
        return tuple(line for line in said.splitlines() if line)

    def _undone(self, version: str, paths: tuple[str, ...], *, force: bool) -> Undone:
        """The rollback itself: refuse what would be lost, put the bytes back, record that it happened."""
        undo = version or self._head()
        if not undo:
            raise Unreadable(NO_VERSIONS_KEPT)
        target = self._parent(undo)
        moved = paths or self._changed(undo)
        if not moved:
            raise Unreadable(NOTHING_MOVED)
        self._only_if_ours(undo, moved, force=force)
        modes = self._held(target)
        restored: list[str] = []
        for name in moved:
            leaf = self._inside(name)
            mode, sha = modes.get(name, ("", ""))
            if not sha:
                if leaf.exists() or leaf.is_symlink():
                    leaf.unlink()
                    restored.append(name)
                continue
            content = self._bytes("cat-file", "blob", sha)
            leaf.parent.mkdir(parents=True, exist_ok=True)
            if leaf.is_symlink() or leaf.exists():
                leaf.unlink()
            if mode == SYMLINK_MODE:
                leaf.symlink_to(content.decode())
            else:
                _ = leaf.write_bytes(content)
                leaf.chmod(0o755 if mode == EXEC_MODE else 0o644)
            restored.append(name)
        revision = self._recorded([self.root / name for name in restored], f"rollback {undo[:8]}") if restored else ""
        return Undone(revision=revision, target=target, restored=tuple(restored))

    def _only_if_ours(self, undo: str, names: tuple[str, ...], *, force: bool) -> None:
        """Refuse a rollback over bytes this version did not leave, unless the caller meant it."""
        held = self._held(undo)
        for name in names:
            leaf = self._inside(name)
            here = self._hashed([leaf], write=False)[0] if (leaf.exists() or leaf.is_symlink()) else None
            if force or here == held.get(name):
                continue
            raise Unwilling(_refused_by_disk(name, undo))

    def _inside(self, name: str) -> Path:
        """The path a version names, refusing one that would land outside the tree it belongs to."""
        named = Path(name)
        if named.is_absolute() or ".." in named.parts:
            raise Unwilling(_outside_version(name))
        return self.root / named

    def _diff(self, older: str, newer: str, paths: tuple[str, ...], context: int) -> str:
        return self._say(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            f"--unified={context}",
            older or EMPTY_TREE,
            newer,
            *(() if not paths else ("--", *paths)),
        )

    def _blob(self, revision: str, path: str) -> bytes | None:
        try:
            return self._bytes("cat-file", "blob", f"{revision}:{path}")
        except Unreadable:
            return None

    # -- one git, hardened ----------------------------------------------------------

    def _say(self, *args: str, stdin: str | None = None) -> str:
        """One command that has to work, and what it said — an answer that is text."""
        return self._run(*args, stdin=stdin).decode("utf-8", "replace").strip()

    def _ask(self, *args: str) -> str:
        """One command whose refusal is an answer ("there is no such version"), not a failure."""
        try:
            return self._say(*args)
        except Unreadable:
            return ""

    def _bytes(self, *args: str) -> bytes:
        return self._run(*args)

    def _run(self, *args: str, stdin: str | None = None, at: Path | None = None) -> bytes:
        """Start git once, in this store, with the environment this build always gives it."""
        done = subprocess.run(
            _git(self.path, *args),
            input=None if stdin is None else stdin.encode(),
            cwd=str(at if at is not None else self.root),
            env=_environment(self.path),
            capture_output=True,
            check=False,
        )
        if done.returncode != 0:
            says = done.stderr.decode("utf-8", "replace").strip()
            raise Unreadable(says or f"git {args[0]} answered {done.returncode}")
        return done.stdout


def _git(git_dir: Path, *args: str) -> list[str]:
    """One git argv: the program, the options every call carries, this store, and the question."""
    return [
        "git",
        *LOCKLESS,
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.fsmonitor=false",
        "--git-dir",
        str(git_dir),
        *args,
    ]


def _environment(git_dir: Path) -> dict[str, str]:
    """What every call runs under: this process's environment, minus everybody else's configuration.

    ``GIT_INDEX_FILE`` is this store's own file and not the tree's: a workspace that happens to be a
    repository has an index of its own, and nothing here may write it. The identities are this
    program's, because a version made by a call is not a commit anybody authored by hand.
    """
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_INDEX_FILE": str(git_dir / "index"),
        "GIT_AUTHOR_NAME": MADE_BY,
        "GIT_AUTHOR_EMAIL": f"{MADE_BY}@localhost",
        "GIT_COMMITTER_NAME": MADE_BY,
        "GIT_COMMITTER_EMAIL": f"{MADE_BY}@localhost",
        "LC_ALL": "C",
    }


def _named(path: Path, root: Path) -> str:
    """A path the way a version names it: relative to the tree, with forward slashes on every platform."""
    return path.relative_to(root).as_posix()
