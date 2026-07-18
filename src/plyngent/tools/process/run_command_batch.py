from __future__ import annotations

import json
import shlex
from typing import Any, cast

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError

from .command_exec import (
    DEFAULT_MAX_BATCH_STEPS,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_TIMEOUT_SECONDS,
    CommandStepResult,
    execute_argv,
    truncate_output,
)

BATCH_OUTPUT_SOFT_CAP_FACTOR = 4


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return default


def _as_str_dict(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = "env must be an object of string keys/values"
        raise WorkspaceError(msg)
    return {str(key): str(val) for key, val in cast("dict[object, object]", value).items()}


def _parse_step(raw: object, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        msg = f"commands[{index}] must be an object"
        raise WorkspaceError(msg)
    data = cast("dict[str, object]", raw)
    command = data.get("command")
    if not isinstance(command, list) or not command:
        msg = f"commands[{index}].command must be a non-empty argv list"
        raise WorkspaceError(msg)
    argv = [part for part in command if isinstance(part, str)]
    if len(argv) != len(command):
        msg = f"commands[{index}].command must be a list of strings"
        raise WorkspaceError(msg)
    cwd = data.get("cwd")
    stdin = data.get("stdin")
    stop = data.get("stop_on_error", None)
    return {
        "command": argv,
        "cwd": str(cwd) if isinstance(cwd, str) and cwd else None,
        "env": _as_str_dict(data.get("env")),
        "stdin": None if stdin is None else str(stdin),
        "pipe_out": _as_bool(data.get("pipe_out"), default=False),
        "mix_stderr": _as_bool(data.get("mix_stderr"), default=False),
        "stop_on_error": None if stop is None else _as_bool(stop, default=True),
        "timeout_seconds": _as_float(data.get("timeout_seconds"), default=DEFAULT_TIMEOUT_SECONDS),
    }


def _normalize_commands(commands: list[dict[str, object]] | str) -> list[object]:
    if isinstance(commands, str):
        try:
            loaded: object = json.loads(commands)
        except json.JSONDecodeError as exc:
            msg = f"commands must be a JSON array: {exc}"
            raise WorkspaceError(msg) from exc
        if not isinstance(loaded, list):
            msg = "commands must be a JSON array of step objects"
            raise WorkspaceError(msg)
        return cast("list[object]", loaded)
    return cast("list[object]", commands)


def _format_step(index: int, result: CommandStepResult, *, pipe_out: bool) -> str:
    return "\n".join(
        [
            f"--- step {index} ---",
            f"exit_code={result.exit_code if result.exit_code is not None else ''}",
            f"timed_out={'true' if result.timed_out else 'false'}",
            f"cwd={result.cwd_display}",
            f"cmd={shlex.join(result.command)}",
            f"pipe_out={'true' if pipe_out else 'false'}",
            f"mix_stderr={'true' if result.mix_stderr else 'false'}",
            "--- stdout ---",
            truncate_output(result.stdout, "stdout"),
            "--- stderr ---",
            truncate_output(result.stderr, "stderr"),
        ]
    )


def _merge_env(base_env: dict[str, str] | None, step_env: dict[str, str] | None) -> dict[str, str] | None:
    if base_env is None and step_env is None:
        return None
    return {**(base_env or {}), **(step_env or {})}


async def _run_batch_steps(
    steps: list[dict[str, Any]],
    *,
    cwd: str,
    env: dict[str, str] | None,
    stop_on_error: bool,
) -> tuple[list[tuple[dict[str, Any], CommandStepResult]], bool]:
    results: list[tuple[dict[str, Any], CommandStepResult]] = []
    prev_capture: str | None = None
    prev_piped = False
    stopped_early = False
    total_chars = 0

    for index, step in enumerate(steps):
        stdin_text = prev_capture if prev_piped else step["stdin"]
        result = await execute_argv(
            step["command"],
            cwd=step["cwd"] if step["cwd"] is not None else cwd,
            env=_merge_env(env, step["env"]),
            stdin=stdin_text,
            timeout_seconds=step["timeout_seconds"],
            mix_stderr=step["mix_stderr"],
        )
        results.append((step, result))
        prev_capture = result.captured
        prev_piped = bool(step["pipe_out"])
        total_chars += len(result.stdout) + len(result.stderr)

        failed = bool(result.timed_out) or result.exit_code != 0
        effective_stop = stop_on_error if step["stop_on_error"] is None else bool(step["stop_on_error"])
        if failed and effective_stop:
            return results, True
        if total_chars > DEFAULT_MAX_OUTPUT_CHARS * BATCH_OUTPUT_SOFT_CAP_FACTOR:
            return results, True
        del index  # used for clarity in loop only

    return results, stopped_early


@tool
async def run_command_batch(
    commands: list[dict[str, object]] | str,
    *,
    cwd: str = ".",
    env: dict[str, str] | None = None,
    stop_on_error: bool = True,
) -> str:
    """Run a serial batch of argv commands (no shell).

    Each element of ``commands`` is an object::

        {
          "command": ["git", "status"],   # required argv
          "cwd": ".",                     # optional, else batch cwd
          "env": {"FOO": "1"},            # optional overlay
          "stdin": "...",                 # optional; unused if previous pipe_out
          "pipe_out": false,              # if true, feed capture into *next* stdin
          "mix_stderr": false,            # OS-level merge stderr into capture
          "stop_on_error": null,          # override batch stop_on_error
          "timeout_seconds": 30
        }

    ``pipe_out`` is set on the **provider** step. Last-step ``pipe_out`` is an error.
    Default ``stop_on_error`` is true. Max 20 steps; total output budget 32k chars.
    """
    try:
        raw_steps = _normalize_commands(commands)
        if not raw_steps:
            return "error: commands must not be empty"
        if len(raw_steps) > DEFAULT_MAX_BATCH_STEPS:
            return f"error: at most {DEFAULT_MAX_BATCH_STEPS} commands per batch"

        steps = [_parse_step(item, i) for i, item in enumerate(raw_steps)]
        if steps[-1]["pipe_out"]:
            return "error: last command has pipe_out=true but there is no next step"

        results, stopped_early = await _run_batch_steps(steps, cwd=cwd, env=env, stop_on_error=stop_on_error)
        body = "\n".join(
            _format_step(i, result, pipe_out=bool(step["pipe_out"])) for i, (step, result) in enumerate(results)
        )
        if len(body) > DEFAULT_MAX_OUTPUT_CHARS:
            body = truncate_output(body, "batch")
        last_exit = results[-1][1].exit_code if results else ""
        return "\n".join(
            [
                f"steps={len(steps)} ran={len(results)} "
                f"stop_on_error={'true' if stop_on_error else 'false'} "
                f"stopped_early={'true' if stopped_early else 'false'}",
                body,
                "--- summary ---",
                f"last_exit={last_exit if last_exit is not None else ''}",
            ]
        )
    except WorkspaceError as exc:
        return f"error: {exc}"
    except (TypeError, ValueError) as exc:
        return f"error: invalid batch arguments: {exc}"
