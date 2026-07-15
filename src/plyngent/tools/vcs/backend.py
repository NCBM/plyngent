from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VcsBackend(Protocol):
    """Read-only VCS operations shared across backends (git, future jj/hg/…)."""

    @property
    def kind(self) -> str:
        """Short backend id, e.g. ``git``."""
        ...

    def status(self) -> str:
        """Working-tree status summary."""
        ...

    def diff(self, *, staged: bool = False, path: str | None = None) -> str:
        """Unified diff; ``path`` is relative to workspace when set."""
        ...

    def log(self, *, limit: int = 10) -> str:
        """Recent commit history (newest first)."""
        ...

    def branch(self) -> str:
        """Current branch / bookmark / named head, or detached-state text."""
        ...
