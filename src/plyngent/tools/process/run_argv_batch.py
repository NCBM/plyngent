from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from plyngent.agent import ToolTag, tool
from plyngent.tools.shape import timeout_shape_error
from plyngent.tools.workspace import WorkspaceError

if TYPE_CHECKING:
    from collections.abc import Mapping

from .command_exec import (
    DEFAULT_MAX_BATCH_STEPS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_BATCH_DEPTH,
    CommandStepResult,
    execute_argv,
)

# Execution budget: stop the batch once accumulated output crosses this factor
# of the base budget, so a runaway batch cannot keep spawning huge-output steps.
BATCH_OUTPUT_BUDGET_CHARS = 32_000
BATCH_OUTPUT_SOFT_CAP_FACTOR = 4

# A node is one command plus up to three continuations: ``then`` runs when it
# succeeds, ``else`` when it fails, ``pipe`` takes its stdout as stdin. Only a
# pipe hands data to its child, so a pipe stage may not branch: the stage's
# result belongs to the node that owns the pipe.
_CONTINUATIONS = ("then", "else", "pipe")
_COMMAND_KEYS = frozenset({"argv", "cwd", "env", "stdin", "mix_stderr", "stop_on_error", "timeout_seconds"})
_ACCEPTED_KEYS = "argv, cwd, env, stdin, mix_stderr, stop_on_error, timeout_seconds, then, else, pipe"


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


@dataclass
class _Budget:
    """Parsed node count: every node costs one, however deeply it nests."""

    nodes: int = 0

    def take(self, where: str) -> None:
        self.nodes += 1
        if self.nodes > DEFAULT_MAX_BATCH_STEPS:
            msg = f"more than {DEFAULT_MAX_BATCH_STEPS} nodes in one batch (at {where})"
            raise WorkspaceError(msg)


def _parse_nodes(
    value: object,
    *,
    where: str,
    depth: int,
    budget: _Budget,
    fed: bool,
) -> list[dict[str, Any]]:
    """A continuation: one node or a list of them, in the order they run."""
    if value is None:
        return []
    items = cast("list[object]", value) if isinstance(value, list) else [value]
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        at = f"{where}[{index}]" if isinstance(value, list) else where
        parsed.append(_parse_node(item, where=at, depth=depth, budget=budget, fed=fed))
    return parsed


def _parse_node(raw: object, *, where: str, depth: int, budget: _Budget, fed: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        msg = f"{where} must be an object"
        raise WorkspaceError(msg)
    if depth > MAX_BATCH_DEPTH:
        msg = f"{where}: nodes nest at most {MAX_BATCH_DEPTH} levels deep"
        raise WorkspaceError(msg)
    data = cast("dict[str, object]", raw)
    unknown = sorted(set(data) - _COMMAND_KEYS - set(_CONTINUATIONS))
    if unknown:
        hint = '; write a pipeline as {"pipe": [...]}, not "pipe_out"' if "pipe_out" in unknown else ""
        msg = f"{where} has unknown key(s) {', '.join(unknown)}; accepted keys are {_ACCEPTED_KEYS}{hint}"
        raise WorkspaceError(msg)
    budget.take(where)
    argv = data.get("argv")
    if not isinstance(argv, list) or not argv:
        msg = f"{where}.argv must be a non-empty argv list"
        raise WorkspaceError(msg)
    argv_parts = cast("list[object]", argv)
    argv_str = [part for part in argv_parts if isinstance(part, str)]
    if len(argv_str) != len(argv_parts):
        msg = f"{where}.argv must be a list of strings"
        raise WorkspaceError(msg)
    stdin = data.get("stdin")
    if fed and stdin is not None:
        msg = f"{where}: stdin comes from the pipe; drop stdin or split the command"
        raise WorkspaceError(msg)
    timeout_error = timeout_shape_error(data.get("timeout_seconds"))
    if timeout_error is not None:
        msg = f"{where}.{timeout_error}"
        raise WorkspaceError(msg)
    cwd = data.get("cwd")
    stop = data.get("stop_on_error", None)
    node: dict[str, Any] = {
        "argv": argv_str,
        "cwd": str(cwd) if isinstance(cwd, str) and cwd else None,
        "env": _as_str_dict(data.get("env")),
        "stdin": None if stdin is None else str(stdin),
        "mix_stderr": _as_bool(data.get("mix_stderr"), default=False),
        "stop_on_error": None if stop is None else _as_bool(stop, default=True),
        "timeout_seconds": _as_float(data.get("timeout_seconds"), default=DEFAULT_TIMEOUT_SECONDS),
    }
    node["then"] = _parse_nodes(data.get("then"), where=f"{where}.then", depth=depth + 1, budget=budget, fed=False)
    node["else"] = _parse_nodes(data.get("else"), where=f"{where}.else", depth=depth + 1, budget=budget, fed=False)
    node["pipe"] = _parse_nodes(data.get("pipe"), where=f"{where}.pipe", depth=depth + 1, budget=budget, fed=True)
    if data.get("pipe") is not None and not node["pipe"]:
        msg = f"{where}.pipe must not be empty"
        raise WorkspaceError(msg)
    if fed and (node["then"] or node["else"]):
        msg = f"{where}: a pipe step takes no then/else; branch on the node that owns the pipe"
        raise WorkspaceError(msg)
    return node


def _normalize_steps(steps: object) -> list[object]:
    if isinstance(steps, str):
        try:
            loaded: object = json.loads(steps)
        except json.JSONDecodeError as exc:
            msg = f"steps must be a JSON array: {exc}"
            raise WorkspaceError(msg) from exc
        if not isinstance(loaded, list):
            msg = "steps must be a JSON array of step objects"
            raise WorkspaceError(msg)
        return cast("list[object]", loaded)
    if not isinstance(steps, list):
        msg = "steps must be a JSON array of step objects"
        raise WorkspaceError(msg)
    return cast("list[object]", steps)


def _read_batch(steps: object) -> tuple[list[dict[str, Any]], int]:
    """Parse a batch tree and count its nodes; raises when it cannot run."""
    raw_steps = _normalize_steps(steps)
    if not raw_steps:
        msg = "steps must not be empty"
        raise WorkspaceError(msg)
    budget = _Budget()
    return _parse_nodes(raw_steps, where="steps", depth=0, budget=budget, fed=False), budget.nodes


def run_argv_batch_shape(args: Mapping[str, object]) -> str | None:
    """Answer a batch that cannot run, before it reaches the human."""
    try:
        _ = _read_batch(args.get("steps"))
    except WorkspaceError as exc:
        return f"error: {exc}"
    except (TypeError, ValueError) as exc:
        return f"error: invalid batch arguments: {exc}"
    return None


@dataclass(frozen=True)
class _Step:
    """One executed command, and whether its stdout fed another one."""

    result: CommandStepResult
    depth: int
    via: tuple[str, int] | None
    feeds: bool


@dataclass(frozen=True)
class _Stage:
    """Exit code of a node's command stage — its pipe chain, for a piped node."""

    code: int | None
    failed: bool
    captured: str


def _format_step(index: int, step: _Step) -> str:
    result = step.result
    lines = [f"--- step {index} ---"]
    if step.depth > 0:
        lines.append(f"depth={step.depth}")
    if step.via is not None:
        lines.append(f"via={step.via[0]} of step {step.via[1]}")
    lines.extend(
        [
            f"exit_code={result.exit_code if result.exit_code is not None else ''}",
            f"timed_out={'true' if result.timed_out else 'false'}",
            f"cwd={result.cwd_display}",
            f"cmd={shlex.join(result.command)}",
            f"pipe={'true' if step.feeds else 'false'}",
            f"mix_stderr={'true' if result.mix_stderr else 'false'}",
            "--- stdout ---",
            result.stdout,
            "--- stderr ---",
            result.stderr,
        ]
    )
    return "\n".join(lines)


def _merge_env(base_env: dict[str, str] | None, step_env: dict[str, str] | None) -> dict[str, str] | None:
    if base_env is None and step_env is None:
        return None
    return {**(base_env or {}), **(step_env or {})}


@dataclass
class _Runner:
    """Runs the node tree, remembering each command and when to stop."""

    cwd: str
    env: dict[str, str] | None
    stop_on_error: bool
    steps: list[_Step] = field(default_factory=list)
    chars: int = 0
    stopped: bool = False

    async def run_list(
        self,
        nodes: list[dict[str, Any]],
        *,
        depth: int,
        branch: str | None = None,
        parent: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        via = (branch, parent) if branch is not None and parent is not None else None
        for node in nodes:
            if self.stopped:
                return
            _ = await self.run_node(node, depth=depth, via=via, cwd=cwd, env=env)

    async def run_node(
        self,
        node: dict[str, Any],
        *,
        depth: int,
        via: tuple[str, int] | None,
        stdin: str | None = None,
        also_feeds: bool = False,
        stage_only: bool = False,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _Stage:
        # A continuation runs where its parent ran, unless it names its own
        # directory or environment; the batch's cwd/env are the outermost default.
        node_cwd = node["cwd"] if node["cwd"] is not None else (self.cwd if cwd is None else cwd)
        node_env = _merge_env(self.env if env is None else env, node["env"])
        result = await execute_argv(
            node["argv"],
            cwd=node_cwd,
            env=node_env,
            stdin=stdin if stdin is not None else node["stdin"],
            timeout_seconds=node["timeout_seconds"],
            mix_stderr=node["mix_stderr"],
        )
        index = len(self.steps)
        self.steps.append(_Step(result, depth, via, feeds=bool(node["pipe"]) or also_feeds))
        self.chars += len(result.stdout) + len(result.stderr)
        stage = _Stage(result.exit_code, bool(result.timed_out) or result.exit_code != 0, result.captured)

        chain: list[dict[str, Any]] = node["pipe"]
        piped = result.captured
        for position, child in enumerate(chain):
            if self.stopped:
                break
            stage = await self.run_node(
                child,
                depth=depth + 1,
                via=("pipe", index),
                stdin=piped,
                also_feeds=position + 1 < len(chain),
                stage_only=True,
                cwd=node_cwd,
                env=node_env,
            )
            piped = stage.captured

        if self.chars > BATCH_OUTPUT_BUDGET_CHARS * BATCH_OUTPUT_SOFT_CAP_FACTOR:
            self.stopped = True
            return stage
        # A pipe stage's failure is the pipe owner's to answer: the chain still
        # ran to its last command, so only that result decides the branch.
        if stage_only:
            return stage
        if stage.failed:
            if node["else"]:
                await self.run_list(
                    node["else"],
                    depth=depth + 1,
                    branch="else",
                    parent=index,
                    cwd=node_cwd,
                    env=node_env,
                )
            elif self._effective_stop(node):
                self.stopped = True
            return stage
        if node["then"]:
            await self.run_list(
                node["then"],
                depth=depth + 1,
                branch="then",
                parent=index,
                cwd=node_cwd,
                env=node_env,
            )
        return stage

    def _effective_stop(self, node: dict[str, Any]) -> bool:
        return self.stop_on_error if node["stop_on_error"] is None else bool(node["stop_on_error"])


@tool(
    tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO,
    validate=run_argv_batch_shape,
)
async def run_argv_batch(
    steps: list[dict[str, object]] | str,
    *,
    cwd: str = ".",
    env: dict[str, str] | None = None,
    stop_on_error: bool = True,
) -> str:
    """Run a batch of argv commands (no shell), each with optional continuations.

    A node is an object::

        {
          "argv": ["pytest", "-q"],     # required argv
          "then": {"argv": [...]},      # runs when argv succeeds (node or list)
          "else": {"argv": [...]},      # runs when argv fails (node or list)
          "pipe": {"argv": [...]},      # takes argv's stdout as its stdin
          "cwd": ".",                   # optional, else batch cwd
          "env": {"FOO": "1"},          # optional overlay
          "stdin": "...",               # optional; a pipe step takes it from the pipe
          "mix_stderr": false,          # OS-level merge stderr into capture
          "stop_on_error": null,        # override batch stop_on_error
          "timeout_seconds": 30
        }

    ``steps`` is a list of nodes, and ``then`` / ``else`` / ``pipe`` each take one
    node or a list of them, so the tree nests. A continuation runs where its
    parent ran: it inherits the parent's ``cwd`` and ``env`` (its own win, and
    the batch's are the outermost default). A ``pipe`` chain runs to its last
    command whatever the earlier ones did, and only that last exit code decides
    the branch; a pipe step takes no ``then`` / ``else`` — branch on the node
    that owns the pipe. A failure with no ``else`` stops the batch, unless that
    node sets ``stop_on_error: false``: it then skips its own continuations and
    the next sibling runs. A failure inside a continuation is nobody's to
    answer, so it stops the batch the same way.

    Max 20 nodes and 4 nesting levels; the batch stops early once accumulated
    output passes the run budget (128k chars). Each command's output is kept
    whole for the model (a long result carries a ``document_id`` to read on), in
    a ``--- step N ---`` block that also says how it got there: ``depth=`` when
    nested, ``via=then of step N`` (or ``else`` / ``pipe``) and ``pipe=true``
    when its stdout fed another command.
    """
    try:
        nodes, count = _read_batch(steps)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except (TypeError, ValueError) as exc:
        return f"error: invalid batch arguments: {exc}"

    runner = _Runner(cwd=cwd, env=env, stop_on_error=stop_on_error)
    await runner.run_list(nodes, depth=0)

    body = "\n".join(_format_step(i, step) for i, step in enumerate(runner.steps))
    last_exit = runner.steps[-1].result.exit_code if runner.steps else ""
    return "\n".join(
        [
            f"steps={count} ran={len(runner.steps)} "
            f"stop_on_error={'true' if stop_on_error else 'false'} "
            f"stopped_early={'true' if runner.stopped else 'false'}",
            body,
            "--- summary ---",
            f"last_exit={last_exit if last_exit is not None else ''}",
        ]
    )
