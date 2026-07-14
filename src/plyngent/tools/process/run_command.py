from __future__ import annotations

import asyncio
import shlex

from plyngent.agent import tool
from plyngent.tools.workspace import (
    WorkspaceError,
    check_command_allowed,
    get_workspace_root,
    resolve_path,
)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_CHARS = 32_000


def _truncate(text: str, label: str) -> str:
    if len(text) <= DEFAULT_MAX_OUTPUT_CHARS:
        return text
    return text[:DEFAULT_MAX_OUTPUT_CHARS] + f"\n...[{label} truncated]"


def _format_result(
    *,
    returncode: int | None,
    workdir_display: str,
    command: list[str],
    stdout: str,
    stderr: str,
) -> str:
    parts = [
        f"exit_code={returncode}",
        f"cwd={workdir_display}",
        f"cmd={shlex.join(command)}",
    ]
    if stdout:
        parts.append("--- stdout ---\n" + stdout.rstrip("\n"))
    if stderr:
        parts.append("--- stderr ---\n" + stderr.rstrip("\n"))
    return "\n".join(parts)


async def _run_exec(
    command: list[str],
    *,
    workdir: str,
    timeout_seconds: float,
) -> tuple[int | None, str, str] | str:
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return f"error: executable not found: {command[0]}"
    except OSError as exc:
        return f"error: failed to start command: {exc}"

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        proc.kill()
        _ = await proc.communicate()
        return f"error: command timed out after {timeout_seconds}s: {shlex.join(command)}"

    stdout = _truncate(stdout_b.decode(errors="replace"), "stdout")
    stderr = _truncate(stderr_b.decode(errors="replace"), "stderr")
    return proc.returncode, stdout, stderr


@tool
async def run_command(
    command: list[str],
    *,
    cwd: str = ".",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run a command without a shell (argv list) under the workspace.

    ``cwd`` is relative to or under the workspace root. Output is truncated.
    """
    if not command:
        return "error: command must not be empty"
    try:
        check_command_allowed(command)
        workdir = resolve_path(cwd)
    except WorkspaceError as exc:
        return f"error: {exc}"
    if not workdir.is_dir():
        return f"error: cwd is not a directory: {cwd}"
    if timeout_seconds <= 0:
        return "error: timeout_seconds must be > 0"

    result = await _run_exec(command, workdir=str(workdir), timeout_seconds=timeout_seconds)
    if isinstance(result, str):
        return result
    returncode, stdout, stderr = result
    return _format_result(
        returncode=returncode,
        workdir_display=str(workdir.relative_to(get_workspace_root())),
        command=command,
        stdout=stdout,
        stderr=stderr,
    )
