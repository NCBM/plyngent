"""Scratch directories, owned by one host and removed by it.

A temporary workspace is a real directory under the system temp dir plus the host's promise to hand
it to tools: paths inside it resolve like paths inside the workspace root. When the host goes down
the directories go with it — for good, not to the trash, because nobody restores scratch space and a
trash on the temp volume would only collect it.

The limit exists because every temporary workspace is also an entry in a listing a human reads, and a
few of them are useful while a thousand are noise. It is per host, like everything else here: two
hosts in one process do not share a directory, do not share a count, and do not clean up each
other's scratch space.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    """The scratch directories one host owns, in the order they were created."""

    limit: int
    _owned: list[Path]

    def __init__(self, limit: int = MAX_TEMPORARY_WORKSPACES) -> None:
        self.limit = limit
        self._owned = []

    def paths(self) -> tuple[Path, ...]:
        """The directories tools may reach, for a workspace's root list."""
        return tuple(self._owned)

    def owned(self) -> tuple[Path, ...]:
        """The directories this host created, and so has to remove."""
        return tuple(self._owned)

    def __len__(self) -> int:
        return len(self._owned)

    async def create(self, prefix: str = _DEFAULT_PREFIX, *, scope: Scope) -> Path:
        """Make a scratch directory under the system temp dir and hand it out.

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
        self._owned.append(created)
        return created

    def release(self, path: Path | str) -> None:
        """Stop handing out *path*; the directory itself is left where it is."""
        resolved = Path(path).expanduser().resolve()
        while resolved in self._owned:
            self._owned.remove(resolved)

    def clear(self) -> int:
        """Remove every owned directory; returns how many were removed from disk.

        Only a path still under the system temp dir is deleted. Anything else is unregistered and
        left alone: a directory that moved out of scratch space is no longer ours to remove, and the
        honest answer there is to forget it, not to delete it.
        """
        removed = 0
        owned = tuple(self._owned)
        self._owned.clear()
        for path in owned:
            if not under_system_temp(path):
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed


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
