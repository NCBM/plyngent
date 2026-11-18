"""How far into a path a conversation may go, and how two of those answers are compared.

A grant is one ``path → mode`` row, and the table holding them belongs to a conversation: authorization
is the conversation's, not the process's (E153), so there is no host table to consult and nothing here
that outlives the conversation that was granted. What this module owns is the vocabulary — the ordered
modes, what a human's spelling means, and whether a table answers a need — so that a workspace, a
conversation and the tool layer ask one question in one way.

A temporary workspace is not a grant either: it is a directory one conversation asked for, so it is
reachable by that conversation and by nobody else (:class:`~plyngent.kernel.temps.TempWorkspaces`).

And a grant is a gate in front of path-resolving tools, **not a sandbox**: an ``exec`` grant, or any
command at all, reaches the rest of the filesystem.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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

    Case and surrounding space are forgiven because the token arrives from a tool argument or from a
    setting, and neither is a place to be strict about spelling.
    """
    return _MODES.get(value.strip().lower())


def mode_covers(granted: AccessMode, required: AccessMode) -> bool:
    """Whether a grant at *granted* satisfies a caller that needs *required*."""
    return granted >= required


def covers(grants: Mapping[Path, AccessMode] | None, path: Path, need: AccessMode) -> bool:
    """Whether one conversation's table allows *need* on *path*.

    A grant answers for the path it names and for everything under it, and the widest of the rows that
    match is the answer: a narrower second "yes" never makes an earlier wider one smaller. ``None`` is a
    caller with no table at all — a probe before a conversation exists — and answers nothing.
    """
    widest: AccessMode | None = None
    for root, granted in (grants or {}).items():
        if not (path == root or path.is_relative_to(root)):
            continue
        if widest is None or granted > widest:
            widest = granted
    return widest is not None and mode_covers(widest, need)


# Bounds how many grants one conversation's table may hold, and so how long a list of them would be if
# one existed. The tool that adds to that table is the one that enforces it.
MAX_GRANTS = 32
