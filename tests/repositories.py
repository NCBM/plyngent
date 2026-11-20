"""A repository on disk, made the way a person makes one.

The vcs group reads a repository by starting git, so a test about it needs a real one: a fake backend
would have to answer every question git answers, and these tests are about what git's own words become
once this build has decided what to ask it — a fake would invent the words as well.

Two happenings are worth knowing about:

* the developer's own git configuration is taken out of the way (``GIT_CONFIG_GLOBAL`` and
  ``GIT_CONFIG_SYSTEM`` point at nothing), because a signing key or a hook somebody has installed is not
  part of what a test means. The repository's own configuration still applies, which is exactly what the
  test about ``diff.external`` is about;
* ``MISSING`` is the skip condition. git is not a package this suite installs, and a machine without one
  can still run everything else here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Whether this machine has a git for the tests that read a repository through one.
MISSING = shutil.which("git") is None

#: The environment a test's own git calls get: this machine's, minus the person's configuration.
_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}


def make(root: Path, *, name: str = "readme.txt", text: str = "hello\n") -> Path:
    """A repository at ``root`` with one commit in it, and answer the root."""
    root.mkdir(parents=True, exist_ok=True)
    run(root, "init", "-q")
    run(root, "config", "user.email", "test@example.com")
    run(root, "config", "user.name", "Test")
    _ = (root / name).write_text(text, encoding="utf-8")
    run(root, "add", name)
    run(root, "commit", "-q", "-m", "initial")
    return root


def empty(root: Path) -> Path:
    """A repository at ``root`` with no commit in it at all — the state git calls unborn."""
    root.mkdir(parents=True, exist_ok=True)
    run(root, "init", "-q")
    return root


def configure(root: Path, *args: str) -> None:
    """Set one thing in this repository's own configuration, as a hook or a driver would be set."""
    run(root, "config", *args)


def run(root: Path, *args: str) -> str:
    """One git command of the test's own, and what it said on stdout.

    It has to work or the test is about nothing, so a non-zero exit is the test failing, not a
    value to look at.
    """
    done = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, env=_ENV)
    return done.stdout
