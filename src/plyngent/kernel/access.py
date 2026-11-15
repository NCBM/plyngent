"""Which directories this host already reaches, and where each answer came from.

A grant is one ``path → mode`` pair, and its *source* is part of the value because the sources differ
in when they stop:

* ``config`` — written in the config file, so it never prompts and this process cannot revoke it;
* ``process`` — the human said yes without letting it be kept (``--yes``), so it dies with the host;
* the session's own table — the third source, which is not here: a session owns it, persists it, and
  hands it in as ``extra`` wherever a lookup happens. Reading it from here would be this module
  reaching sideways into something it does not own.

Two more lists are deliberately not grants. ``read_only`` roots are host-contributed directories a
tool may read (skill directories): they are not asked for, cannot be written, and never show up in a
list of grants. And a grant is a gate in front of path-resolving tools, **not a sandbox**: an ``exec``
grant, or any command at all, reaches the rest of the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class AccessMode(IntEnum):
    """How much of a path a caller may use, ordered so a wider grant covers a narrower need.

    ``read`` < ``write`` < ``exec``, and ``exec`` also implies writing files: a permission to run a
    command in a directory is not narrower than a permission to write in it.
    """

    READ = 1
    WRITE = 2
    EXEC = 3


_MODES: dict[str, AccessMode] = {mode.name.lower(): mode for mode in AccessMode}


def parse_access_mode(value: str) -> AccessMode | None:
    """Read a mode written by a human (``read`` / ``write`` / ``exec``), or ``None``.

    Case and surrounding space are forgiven because the token arrives from a tool argument or a
    config file, and neither is a place to be strict about spelling.
    """
    return _MODES.get(value.strip().lower())


def mode_covers(granted: AccessMode, required: AccessMode) -> bool:
    """Whether a grant at *granted* satisfies a caller that needs *required*."""
    return granted >= required


# Bounds how many grants one table may hold, and so how long a list of them would be if one existed. The
# tool that adds to a conversation's table is the one that enforces it.
MAX_GRANTS = 32

type Source = Literal["config", "process"]


@dataclass(frozen=True, slots=True)
class GrantRow:
    """One grant, as a listing shows it: what path, at which mode, from where."""

    path: Path
    mode: AccessMode
    source: Source


class Access:
    """The host's own grant tables, and the read-only roots nobody grants.

    Nothing here asks anybody: an ``AccessDecision`` is the tool layer's business, and a human is
    reached through ``Host.human``. This answers what is already covered.
    """

    config: dict[Path, AccessMode]
    process: dict[Path, AccessMode]
    read_only: tuple[Path, ...]

    def __init__(self) -> None:
        self.config = {}
        self.process = {}
        self.read_only = ()

    # -- lookup --------------------------------------------------------------------

    def mode(self, path: Path, *, extra: Mapping[Path, AccessMode] | None = None) -> AccessMode | None:
        """The widest mode covering *path*, or ``None`` when nothing does.

        ``extra`` is the caller's own table — a session's grants, typically — passed in rather than
        reached for. A read-only root answers ``read`` unless a grant covers more.
        """
        best: AccessMode | None = None
        for table in (self.config, self.process, extra or {}):
            for root, granted in table.items():
                if not (path == root or path.is_relative_to(root)):
                    continue
                if best is None or granted > best:
                    best = granted
        if best is None and any(path == root or path.is_relative_to(root) for root in self.read_only):
            best = AccessMode.READ
        return best

    def covers(self, path: Path, need: AccessMode, *, extra: Mapping[Path, AccessMode] | None = None) -> bool:
        """Whether something already allows *need* on *path*."""
        granted = self.mode(path, extra=extra)
        return granted is not None and mode_covers(granted, need)

    def count(self) -> int:
        """How many grants this host holds; a session's own table is added by whoever asks."""
        return len(self.config) + len(self.process)

    def rows(self) -> tuple[GrantRow, ...]:
        """The host's grants, process before config — the order a human expects to read."""
        return (
            *(GrantRow(path, mode, "process") for path, mode in self.process.items()),
            *(GrantRow(path, mode, "config") for path, mode in self.config.items()),
        )

    # -- changes -------------------------------------------------------------------

    def grant(self, path: Path | str, mode: AccessMode, *, source: Source) -> Path:
        """Allow *mode* on *path* from *source*; returns the resolved path."""
        resolved = Path(path).expanduser().resolve()
        self._table(source)[resolved] = mode
        return resolved

    def revoke(self, path: Path | str, *, source: Source) -> bool:
        """Drop the process grant on exactly *path*; ``False`` when there was none.

        A config grant is not revocable here: it belongs to the file the human wrote, so the answer
        is ``False`` and the caller points at the config instead.
        """
        if source == "config":
            return False
        resolved = Path(path).expanduser().resolve()
        return self.process.pop(resolved, None) is not None

    def clear(self, *, source: Source) -> None:
        """Drop every grant from *source*."""
        self._table(source).clear()

    def install_config(self, entries: Mapping[str, str]) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        """Replace the config grants with *entries*; returns ``(applied, skipped)`` paths.

        An entry is skipped when its mode is unknown or its path does not exist: a grant on nothing
        is not a grant, and a typo in a mode must not silently become the default.
        """
        self.config.clear()
        applied: list[Path] = []
        skipped: list[str] = []
        for raw_path, raw_mode in entries.items():
            mode = parse_access_mode(raw_mode)
            resolved = _resolve_optional(raw_path)
            if mode is None or resolved is None or not resolved.exists():
                skipped.append(raw_path)
                continue
            self.config[resolved] = mode
            applied.append(resolved)
        return tuple(applied), tuple(skipped)

    def use_read_only(self, roots: Iterable[Path | str]) -> tuple[Path, ...]:
        """Install the read-only roots, keeping only the existing directories.

        A directory that is absent holds nothing to read, and keeping it would turn "nothing here
        yet" into an implicit permission later.
        """
        kept: list[Path] = []
        for root in roots:
            resolved = _resolve_optional(root)
            if resolved is not None and resolved.is_dir():
                kept.append(resolved)
        self.read_only = tuple(kept)
        return self.read_only

    def _table(self, source: Source) -> dict[Path, AccessMode]:
        return self.config if source == "config" else self.process


def _resolve_optional(path: Path | str) -> Path | None:
    """Resolve *path*, or ``None`` when the platform cannot even look at it."""
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return None
