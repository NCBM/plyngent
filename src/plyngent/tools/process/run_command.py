from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError, argv_shape_error

from .command_exec import (
    DEFAULT_TIMEOUT_SECONDS,
    execute_argv,
    format_command_result,
)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def run_command(
    command: list[str],
    *,
    cwd: str = ".",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run ``command`` (argv, no shell) under the workspace; capture stdout/stderr."""
    shape_error = argv_shape_error(command)
    if shape_error is not None:
        return f"error: `run_command` {shape_error}"
    try:
        result = await execute_argv(
            command,
            cwd=cwd,
            env=env,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            mix_stderr=False,
        )
    except WorkspaceError as exc:
        return f"error: {exc}"

    return format_command_result(
        returncode=result.exit_code,
        workdir_display=result.cwd_display,
        command=result.command,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
    )
