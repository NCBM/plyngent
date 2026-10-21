"""Move a path to the system trash instead of destroying it.

``delete_path`` hands every removal to :func:`move_to_trash`, so a delete the
human approved (or YOLO skipped) can still be undone from the desktop's own
trash. The platform work is ``send2trash``: the freedesktop trash specification
on Linux, the Finder trash on macOS, the Recycle Bin on Windows.

Scratch paths under the OS temporary directory are the exception — they are
deleted for good, since nobody restores them and a trash on the ``/tmp`` volume
would only collect them (see :func:`under_system_temp`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def is_available() -> bool:
    """True when ``send2trash`` can be imported.

    It is a required dependency, so ``False`` means a broken install rather than
    an expected absence; the delete path then reports the problem instead of
    destroying the file.
    """
    from importlib.util import find_spec

    return find_spec("send2trash") is not None


def under_system_temp(path: Path) -> bool:
    """True if *path* is under the OS temporary directory (scratch space)."""
    try:
        temp_root = Path(tempfile.gettempdir()).expanduser().resolve()
        resolved = path.expanduser().resolve()
        _ = resolved.relative_to(temp_root)
    except OSError, ValueError:
        return False
    return True


def move_to_trash(path: Path) -> str | None:
    """Send *path* to the system trash; return an error text, or ``None`` on success.

    A failure is always reported, never downgraded to a permanent delete: the
    human was told the path is recoverable, so a trash that is unavailable for
    this path leaves the file untouched and says how to remove it on purpose.
    """
    if not is_available():
        return "error: cannot move to trash: send2trash is not installed (use permanent=true)"

    from send2trash import TrashPermissionError, send2trash

    try:
        send2trash(str(path))
    except TrashPermissionError as exc:
        # A volume with no trash directory and no permission to create one.
        return f"error: cannot move to trash: {exc} (use permanent=true)"
    except OSError as exc:
        return f"error: cannot move to trash: {exc}"
    return None
