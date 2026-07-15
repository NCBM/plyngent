from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from plyngent.tools.vcs import (
    detect_vcs,
    vcs_branch,
    vcs_diff,
    vcs_kind,
    vcs_log,
    vcs_status,
)
from plyngent.tools.vcs.detect import clear_extra_detectors, register_detector
from tests.test_tools.helpers import call_sync

if TYPE_CHECKING:
    from plyngent.tools.vcs.backend import VcsBackend

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    del completed


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _ = (root / "readme.txt").write_text("hello\n", encoding="utf-8")
    _git(root, "add", "readme.txt")
    _git(root, "commit", "-m", "initial")


def test_detect_none(workspace: object) -> None:
    assert isinstance(workspace, Path)
    assert detect_vcs(workspace) is None
    assert "no supported VCS" in call_sync(vcs_status)


def test_git_status_and_kind(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _init_repo(workspace)
    assert call_sync(vcs_kind) == "git"
    status = call_sync(vcs_status)
    assert "readme" in status or "main" in status or "master" in status or status == "(clean)"


def test_git_log_and_branch(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _init_repo(workspace)
    log = call_sync(vcs_log, limit=5)
    assert "initial" in log
    branch = call_sync(vcs_branch)
    assert branch  # main/master/detached — non-empty


def test_git_diff(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _init_repo(workspace)
    _ = (workspace / "readme.txt").write_text("hello\nworld\n", encoding="utf-8")
    diff = call_sync(vcs_diff)
    assert "world" in diff or "readme" in diff


def test_register_custom_detector(workspace: object) -> None:
    assert isinstance(workspace, Path)

    class FakeBackend:
        @property
        def kind(self) -> str:
            return "fake"

        def status(self) -> str:
            return "fake-status"

        def diff(self, *, staged: bool = False, path: str | None = None) -> str:
            del staged, path
            return "fake-diff"

        def log(self, *, limit: int = 10) -> str:
            del limit
            return "fake-log"

        def branch(self) -> str:
            return "fake-branch"

    def detect_fake(root: Path) -> VcsBackend | None:
        del root
        return FakeBackend()

    register_detector(detect_fake, prepend=True)
    try:
        assert call_sync(vcs_kind) == "fake"
        assert call_sync(vcs_status) == "fake-status"
        assert call_sync(vcs_branch) == "fake-branch"
    finally:
        clear_extra_detectors()
