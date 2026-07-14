from __future__ import annotations

import asyncio
import os
import shlex
from typing import TYPE_CHECKING

from plyngent.agent import tool
from plyngent.tools.workspace import (
    WorkspaceError,
    check_command_allowed,
    get_workspace_root,
    resolve_path,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_CHARS = 32_000


def _truncate(text: str, label: str) -> str:
    if len(text) <= DEFAULT_MAX_OUTPUT_CHARS:
        return text
    return text[:DEFAULT_MAX_OUTPUT_CHARS] + f"\n...[{label} truncated]"


def _format_result(  # noqa: PLR0913
    *,
    returncode: int | None,
    workdir_display: str,
    command: list[str],
    stdout: str,
    stderr: str,
    timed_out: bool = False,
) -> str:
    parts = [
        f"exit_code={'' if returncode is None else returncode}",
        f"timed_out={'true' if timed_out else 'false'}",
        f"cwd={workdir_display}",
        f"cmd={shlex.join(command)}",
    ]
    if stdout:
        parts.append("--- stdout ---\n" + stdout.rstrip("\n"))
    if stderr:
        parts.append("--- stderr ---\n" + stderr.rstrip("\n"))
    return "\n".join(parts)


def _validate_env(env: object) -> str | None:
    """Runtime guard for tool-JSON args (may not match static typing at the boundary)."""
    if env is None:
        return None
    if type(env) is not dict:
        return "error: env must be an object of string keys and values"
    # Tool schema should already enforce dict[str, str]; keep a light check.
    return None


def _merge_env(overrides: dict[str, str] | None) -> dict[str, str] | None:
    if overrides is None:
        return None
    return {**os.environ, **overrides}


async def _run_exec(
    command: list[str],
    *,
    workdir: str,
    timeout_seconds: float,
    stdin_data: bytes | None,
    env: dict[str, str] | None,
) -> tuple[int | None, str, str, bool] | str:
    """Return ``(returncode, stdout, stderr, timed_out)`` or an error string."""
    stdin = asyncio.subprocess.PIPE if stdin_data is not None else None
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=workdir,
            env=env,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return f"error: executable not found: {command[0]}"
    except OSError as exc:
        return f"error: failed to start command: {exc}"

    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input=stdin_data),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        timed_out = True
        proc.kill()
        stdout_b, stderr_b = await proc.communicate()

    stdout = _truncate(stdout_b.decode(errors="replace"), "stdout")
    stderr = _truncate(stderr_b.decode(errors="replace"), "stderr")
    return proc.returncode, stdout, stderr, timed_out


def _validate_run_args(
    command: list[str],
    *,
    cwd: str,
    timeout_seconds: float,
    env: dict[str, str] | None,
) -> Path | str:
    """Return resolved workdir, or an error string."""
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
    env_error = _validate_env(env)
    if env_error is not None:
        return env_error
    return workdir


@tool
async def run_command(
    command: list[str],
    *,
    cwd: str = ".",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a command without a shell (argv list) under the workspace.

    ``cwd`` is relative to or under the workspace root. Optional ``stdin`` is
    written to the process stdin. Optional ``env`` overlays process environment
    variables (merged with the current environment). Output is truncated.

    On timeout the process is killed and any partial stdout/stderr is still
    returned with ``timed_out=true``.
    """
    workdir = _validate_run_args(command, cwd=cwd, timeout_seconds=timeout_seconds, env=env)
    if isinstance(workdir, str):
        return workdir

    stdin_data = None if stdin is None else stdin.encode()
    result = await _run_exec(
        command,
        workdir=str(workdir),
        timeout_seconds=timeout_seconds,
        stdin_data=stdin_data,
        env=_merge_env(env),
    )
    if isinstance(result, str):
        return result
    returncode, stdout, stderr, timed_out = result
    return _format_result(
        returncode=returncode,
        workdir_display=str(workdir.relative_to(get_workspace_root())),
        command=command,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )
