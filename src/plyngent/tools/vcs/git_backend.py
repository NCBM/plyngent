from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 64_000


def is_git_repo(root: Path) -> bool:
    """True if ``root`` is inside a git work tree (requires ``git`` on PATH)."""
    if shutil.which("git") is None:
        return False
    result = _run_git(root, ["rev-parse", "--is-inside-work-tree"])
    return result.ok and result.stdout.strip() == "true"


class _GitResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int

    def __init__(self, *, ok: bool, stdout: str, stderr: str, returncode: int) -> None:
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _truncate(text: str) -> str:
    if len(text) <= DEFAULT_MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - DEFAULT_MAX_OUTPUT_CHARS
    return f"{text[:DEFAULT_MAX_OUTPUT_CHARS]}\n...[truncated {omitted} characters]"


def _run_git(root: Path, args: list[str], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> _GitResult:
    argv = ["git", "-c", "color.ui=false", *args]
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _GitResult(ok=False, stdout="", stderr="git executable not found", returncode=127)
    except subprocess.TimeoutExpired:
        return _GitResult(ok=False, stdout="", stderr="git timed out", returncode=124)
    except OSError as exc:
        return _GitResult(ok=False, stdout="", stderr=str(exc), returncode=1)

    stdout = _truncate(completed.stdout or "")
    stderr = _truncate(completed.stderr or "")
    return _GitResult(
        ok=completed.returncode == 0,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
    )


def _format_result(result: _GitResult, *, empty: str = "(empty)") -> str:
    if not result.ok:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return f"error: git failed: {detail}"
    text = result.stdout.rstrip("\n")
    return text or empty


class GitBackend:
    """Read-only git operations via the ``git`` CLI."""

    _root: Path

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def kind(self) -> str:
        return "git"

    def status(self) -> str:
        # porcelain v1 is stable for tooling; short is more human-readable for agents.
        result = _run_git(self._root, ["status", "--short", "--branch"])
        return _format_result(result, empty="(clean)")

    def diff(self, *, staged: bool = False, path: str | None = None) -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        if path:
            args.extend(["--", path])
        result = _run_git(self._root, args)
        return _format_result(result, empty="(no diff)")

    def log(self, *, limit: int = 10) -> str:
        n = max(1, min(limit, 100))
        result = _run_git(
            self._root,
            ["log", f"-n{n}", "--format=%h %ad %an %s", "--date=short"],
        )
        return _format_result(result, empty="(no commits)")

    def branch(self) -> str:
        result = _run_git(self._root, ["branch", "--show-current"])
        if result.ok and result.stdout.strip():
            return result.stdout.strip()
        # Detached HEAD or empty repo
        head = _run_git(self._root, ["rev-parse", "--short", "HEAD"])
        if head.ok and head.stdout.strip():
            return f"(detached {head.stdout.strip()})"
        return _format_result(result, empty="(no branch)")
