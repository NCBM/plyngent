"""Which tree a person standing in a member should open: a pointer kept in this program's own directory.

A space is a root tree and the trees it gathered (E193), and the authority is one table in the root's own
journal: a member's directory carries nothing, because this program writes no file inside a workspace. What
a member needs is the other direction — somebody standing in it, with no kernel running, asking which tree
to open — and that is a pointer in the person's data directory, named by the same rule a ledger's directory
is named by.

**A pointer, not a grant.** Nothing here decides what may be reached: a member is reachable because a row in
the root's journal says so, and a pointer that names a tree which does not know this member is a wrong
answer to "where do I go", never a wider reach. That is why a stale one is harmless — a tree cloned onto
another machine carries no pointer at all, and one copied beside its data directory names a root whose table
is the root's to check.

**One answer, and a cheap one.** The file holds the root's resolved path, one line, and reading it is a
small read that happens once per run: the frontend asks it before it decides which tree this run means, so
a person who is standing in a member gets the space rather than a tree of its own.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from platformdirs import user_data_path

from plyngent.kernel.journal import slug

#: Where this program's pointers live, under the person's data directory: a directory of its own, beside
#: ``workspaces``, so that a listing of either is a listing of one kind of thing.
SPACES = "spaces"


def pointer_path(member: Path | str, *, data: Path | str | None = None) -> Path:
    """Where the pointer for *member* lives: named the way a ledger's directory is named (E193).

    A slug and a digest of the resolved path, the same pair and the same rule as
    :func:`~plyngent.kernel.journal.workspace_dir` — one spelling for "which tree is this" in this program,
    so a person looking at the data directory reads the two directories the same way. Pure: it answers a
    path and creates nothing.
    """
    resolved = Path(member).expanduser().resolve()
    digest = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:12]
    inside = user_data_path("plyngent") if data is None else Path(data)
    return Path(inside) / SPACES / f"{slug(resolved.name)}-{digest}"


def root_of(member: Path | str, *, data: Path | str | None = None) -> Path | None:
    """The tree this one was gathered into, or ``None`` when nothing says so (E193).

    ``None`` covers three states that a frontend answers the same way — no pointer, a pointer a person
    removed, and a file this process may not read — because the answer it needs is "open this instead", and
    a tree with no pointer is one this run opens as itself.
    """
    try:
        said = pointer_path(member, data=data).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(said) if said else None


def point(member: Path | str, root: Path | str, *, data: Path | str | None = None) -> bool:
    """Say that *member* belongs to the space at *root*; ``False`` when the file could not be written.

    Never raises: a member whose pointer cannot be written is still a member, because the table in the
    root's journal is what makes it one. What is lost is the one convenience — a run started inside the
    member finds the space — and the caller says so rather than pretending it worked.
    """
    named = pointer_path(member, data=data)
    try:
        named.parent.mkdir(parents=True, exist_ok=True)
        _ = named.write_text(f"{Path(root).expanduser().resolve()}\n", encoding="utf-8")
    except OSError:
        return False
    return True


def unpoint(member: Path | str, *, data: Path | str | None = None) -> bool:
    """Forget that *member* belonged anywhere; ``False`` when there was nothing to forget."""
    try:
        pointer_path(member, data=data).unlink()
    except OSError:
        return False
    return True
