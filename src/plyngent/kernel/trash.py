"""Deleting a path so the human can still get it back: the desktop's own trash.

A delete is the one effect a kernel cannot undo, so the default does not destroy anything: it hands
the path to the platform's trash — the freedesktop trash on Linux, the Finder trash on macOS, the
Recycle Bin on Windows — and the human can take it back out. A trash that will not take the path is
reported and the file is left alone; it is never downgraded to a permanent delete, because what the
human was told was "recoverable".

A path under the system temp directory is the exception, and the reason is not convenience: scratch
space is what the host itself made and will remove, so a trash on the temp volume would only collect
it. :func:`~plyngent.kernel.temps.under_system_temp` is the same predicate the host uses for its own
directories, so "scratch" means one thing in both places.
"""

from __future__ import annotations

from enum import StrEnum
from importlib.util import find_spec
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from pathlib import Path


class Kind(StrEnum):
    """The two ways a trash says no, told apart because they are two different problems.

    A volume that will not take this path is a fact about the path, and one value gets past it: remove
    it for good. An installation with no ``send2trash`` is a fact about the installation, and there is
    nothing a caller can type that changes it — so a caller told the second kind has to answer with
    what it knows, and offering a permanent delete there would invite an irreversible one to work
    around a broken install (E189).
    """

    ABSENT = "absent"
    REFUSED = "refused"


class Trouble(msgspec.Struct, frozen=True, kw_only=True):
    """A path the trash did not take: which kind of no it was, and the sentence a reader is given."""

    kind: Kind
    said: str


def available() -> bool:
    """Whether ``send2trash`` can be imported.

    It is a declared dependency, so ``False`` means a broken install rather than an expected absence;
    the caller reports it instead of destroying the file.
    """
    return find_spec("send2trash") is not None


def to_trash(path: Path) -> Trouble | None:
    """Send *path* to the system trash; ``None`` when it went, else why it did not."""
    if not available():
        return Trouble(kind=Kind.ABSENT, said="cannot move to trash: send2trash is not installed")
    from send2trash import TrashPermissionError, send2trash

    try:
        send2trash(str(path))
    except TrashPermissionError as exc:
        # A volume with no trash directory and no permission to create one.
        return Trouble(kind=Kind.REFUSED, said=f"cannot move to trash: {exc}")
    except OSError as exc:
        return Trouble(kind=Kind.REFUSED, said=f"cannot move to trash: {exc}")
    return None
