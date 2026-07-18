from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from plyngent.tools.workspace import (
    WorkspaceError,
    check_command_allowed,
    get_workspace_root,
    resolve_path,
)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_CHARS = 32_000
DEFAULT_MAX_BATCH_STEPS = 20


def truncate_output(text: str, label: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[{label} truncated]"


def format_command_result(
    *,
    returncode: int | None,
    workdir_display: str,
    command: list[str],
    stdout: str,
    stderr: str,
    timed_out: bool = False,
) -> str:
    parts = [
        f"exit_code={returncode if returncode is not None else ''}",
        f"timed_out={'true' if timed_out else 'false'}",
        f"cwd={workdir_display}",
        f"cmd={shlex.join(command)}",
        "--- stdout ---",
        truncate_output(stdout, "stdout"),
        "--- stderr ---",
        truncate_output(stderr, "stderr"),
    ]
    return "\n".join(parts)


def resolve_workdir(cwd: str) -> tuple["Path", str]:
    """Return (absolute workdir, display path relative to workspace when possible)."""
    workdir = resolve_path(cwd)
    if not workdir.is_dir():
        msg = f"not a directory: {cwd}"
        raise WorkspaceError(msg)
    try:
        workdir_display = str(workdir.relative_to(get_workspace_root()))
    except ValueError:
        workdir_display = str(workdir)
    return workdir, workdir_display


@dataclass
class CommandStepResult:
    command: list[str]
    cwd_display: str
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    mix_stderr: bool
    captured: str  # piped to the next step when provider sets pipe_out


async def execute_argv(
    command: list[str],
    *,
    cwd: str = ".",
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    mix_stderr: bool = False,
) -> CommandStepResult:
    """Run one argv command; no shell. Used by run_command and run_command_batch."""
    if not command:
        msg = "command argv must not be empty"
        raise WorkspaceError(msg)
    check_command_allowed(command)
    workdir, workdir_display = resolve_workdir(cwd)

    stdin_data = None if stdin is None else stdin.encode()
    run_env: dict[str, str] | None = None
    if env is not None:
        run_env = {str(k): str(v) for k, v in {**os.environ, **env}.items()}

    try:
        stderr_arg = asyncio.subprocess.STDOUT if mix_stderr else asyncio.subprocess.PIPE
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(workdir),
            env=run_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_arg,
        )
    except FileNotFoundError:
        msg = f"executable not found: {command[0]!r}"
        raise WorkspaceError(msg) from None
    except OSError as exc:
        msg = f"failed to start command: {exc}"
        raise WorkspaceError(msg) from exc

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

    stdout = (stdout_b or b"").decode(errors="replace")
    if mix_stderr:
        stderr = ""
        captured = stdout
    else:
        stderr = (stderr_b or b"").decode(errors="replace")
        captured = stdout

    return CommandStepResult(
        command=list(command),
        cwd_display=workdir_display,
        exit_code=proc.returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        mix_stderr=mix_stderr,
        captured=captured,
    )
