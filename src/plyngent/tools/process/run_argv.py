from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError, argv_shape_error

from .command_exec import (
    DEFAULT_TIMEOUT_SECONDS,
    execute_argv,
    format_command_result,
    timeout_shape_error,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def run_argv_shape(args: Mapping[str, object]) -> str | None:
    """Answer a ``run_argv`` call whose arguments cannot run, before any prompt."""
    shape_error = argv_shape_error(args.get("argv"))
    if shape_error is not None:
        return f"error: `run_argv` {shape_error}"
    timeout_error = timeout_shape_error(args.get("timeout_seconds"))
    if timeout_error is not None:
        return f"error: `run_argv` {timeout_error}"
    return None


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO, validate=run_argv_shape)
async def run_argv(
    argv: list[str],
    *,
    cwd: str = ".",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run ``argv`` (argv, no shell) under the workspace; capture stdout/stderr."""
    shape_error = run_argv_shape({"argv": argv, "timeout_seconds": timeout_seconds})
    if shape_error is not None:
        return shape_error
    try:
        result = await execute_argv(
            argv,
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
