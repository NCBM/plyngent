"""Scratch directories, owned by one conversation and removed with it.

A temporary workspace is a real directory under the system temp dir plus a host's promise to hand it to
one conversation's tools: paths inside it resolve like paths inside the workspace root, for the
conversation that asked and for nobody else. Ownership is not decoration — a scratch nobody else may name
is what a scratch *is* — and a directory with no owner is one that three questions cannot answer: who may
reach it, how long it lives, and what a sentence about it disappearing would even be about.

The directories go with their conversation: when it is closed, or when the host goes down. Removal is for
good rather than to the trash, because nobody restores scratch space and a trash on the temp volume would
only collect it (E152).

The limit exists because every temporary workspace is also an entry in a listing a human reads, and a few
of them are useful while a thousand are noise. It is per host, like the counters beside it: two hosts in
one process do not share a directory, do not share a count, and do not clean up each other's scratch.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.kernel.cancel import Scope

# Max concurrent temporary workspaces one host may hand out.
MAX_TEMPORARY_WORKSPACES = 16

_DEFAULT_PREFIX = "ws"
_PREFIX_MAX_LEN = 32
_PREFIX_SAFE = re.compile(rf"^[A-Za-z0-9_-]{{1,{_PREFIX_MAX_LEN}}}$")


class TempWorkspaceError(Exception):
    """A scratch directory that could not be handed out."""


class TooManyWorkspacesError(TempWorkspaceError):
    """The host already has its maximum; nothing new is created."""


class TempWorkspaces:
    """The scratch directories one host owns, in the order they were created, each with its owner.

    The owner is a session id, and nothing here knows what a session is: this is a table of which
    conversation asked for which directory, and the one thing it is for is that a directory can be
    named, counted, and removed one conversation at a time.
    """

    limit: int
    _owned: dict[Path, str]

    def __init__(self, limit: int = MAX_TEMPORARY_WORKSPACES) -> None:
        self.limit = limit
        self._owned = {}

    def paths(self, owner: str) -> tuple[Path, ...]:
        """The directories one conversation may reach."""
        return tuple(path for path, held in self._owned.items() if held == owner)

    def count_for(self, owner: str) -> int:
        """How many directories one conversation holds — what "its scratch went away" is counted in."""
        return sum(1 for held in self._owned.values() if held == owner)

    def owners(self) -> frozenset[str]:
        """The conversations holding scratch space — who "it is out of reach now" is about."""
        return frozenset(self._owned.values())

    def owner_of(self, path: Path | str) -> str | None:
        """Which conversation a directory belongs to, or ``None`` when this host did not make it."""
        return self._owned.get(_resolved(path))

    def __len__(self) -> int:
        return len(self._owned)

    async def create(self, prefix: str = _DEFAULT_PREFIX, *, owner: str, scope: Scope) -> Path:
        """Make a scratch directory under the system temp dir and hand it to one conversation.

        Making a directory is a blocking call, so it runs off the loop: a host that stalled its own
        turns on a filesystem syscall would be a host whose PTY reads stopped too. It runs on *scope*'s
        books for the same reason every other blocking call does — a turn that was stopped still waits
        for the thread it started before its history is written.
        """
        if len(self._owned) >= self.limit:
            msg = f"too many temporary workspaces (max {self.limit})"
            raise TooManyWorkspacesError(msg)
        safe = _sanitize_prefix(prefix)
        try:
            created = await scope.blocking(lambda: Path(tempfile.mkdtemp(prefix=f"plyngent-{safe}-")).resolve())
        except OSError as exc:
            msg = f"failed to create temporary workspace: {exc}"
            raise TempWorkspaceError(msg) from exc
        self._owned[created] = owner
        return created

    def release(self, path: Path | str) -> None:
        """Stop handing out *path*: the directory is gone, or it moved out of scratch space.

        Silent for a path this host did not make, which is the ordinary case rather than an error: every
        delete hands its target here, and almost none of them is a scratch directory.
        """
        _ = self._owned.pop(_resolved(path), None)

    def clear(self, owner: str | None = None) -> int:
        """Remove one conversation's directories — or every one of them — saying how many left the disk.

        Only a path still under the system temp dir is deleted. Anything else is unregistered and left
        alone: a directory that moved out of scratch space is no longer ours to remove, and the honest
        answer there is to forget it, not to delete it.
        """
        removed = 0
        gone: Sequence[Path] = tuple(self._owned)
        for path in gone:
            if owner is not None and self._owned.get(path) != owner:
                continue
            del self._owned[path]
            if not under_system_temp(path):
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed


def _resolved(path: Path | str) -> Path:
    """A path spelled the way this table spells it: the one thing every key here is stored as."""
    return Path(path).expanduser().resolve()


def _sanitize_prefix(prefix: str) -> str:
    """A prefix ``mkdtemp`` takes on every platform: letters, digits, ``_`` and ``-``."""
    token = (prefix or _DEFAULT_PREFIX).strip() or _DEFAULT_PREFIX
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", token)
    cleaned = cleaned.strip("-_") or _DEFAULT_PREFIX
    if len(cleaned) > _PREFIX_MAX_LEN:
        cleaned = cleaned[:_PREFIX_MAX_LEN]
    if not _PREFIX_SAFE.match(cleaned):
        return _DEFAULT_PREFIX
    return cleaned


def under_system_temp(path: Path) -> bool:
    """Whether *path* is under the OS temporary directory, where deleting for good is right."""
    try:
        temp_root = Path(tempfile.gettempdir()).expanduser().resolve()
        resolved = path.expanduser().resolve()
        _ = resolved.relative_to(temp_root)
    except OSError, ValueError:
        return False
    return True
