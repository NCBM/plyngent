from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from .git_backend import GitBackend, is_git_repo

if TYPE_CHECKING:
    from pathlib import Path

    from .backend import VcsBackend

type Detector = Callable[[Path], VcsBackend | None]


def _detect_git(root: Path) -> VcsBackend | None:
    if is_git_repo(root):
        return GitBackend(root)
    return None


# First match wins. Append future detectors (jj, hg, …) without changing tools.
_DETECTORS: list[Detector] = [_detect_git]


def register_detector(detector: Detector, *, prepend: bool = False) -> None:
    """Register a VCS detector (for tests or future backends)."""
    if prepend:
        _DETECTORS.insert(0, detector)
    else:
        _DETECTORS.append(detector)


def clear_extra_detectors() -> None:
    """Reset detectors to built-ins only (tests)."""
    _DETECTORS.clear()
    _DETECTORS.append(_detect_git)


def detectors() -> Sequence[Detector]:
    return tuple(_DETECTORS)


def detect_vcs(root: Path) -> VcsBackend | None:
    """Return a backend for ``root`` if a supported VCS is present."""
    for detector in _DETECTORS:
        backend = detector(root)
        if backend is not None:
            return backend
    return None
