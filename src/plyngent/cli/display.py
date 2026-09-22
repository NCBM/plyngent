from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import click

from plyngent.agent import (
    CancelledEvent,
    ErrorEvent,
    MaxRoundsEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from plyngent.lmproto.openai_compatible.model import AssistantFunctionToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from plyngent.agent import AgentEvent, Notice

_TOOL_RESULT_PREVIEW = 120
_TOOL_ARGS_PREVIEW = 80

# Process/session display flags (set from ReplState / slash).
_verbose_tool_results: ContextVar[bool] = ContextVar("verbose_tool_results", default=False)
_markdown_enabled: ContextVar[bool] = ContextVar("markdown_enabled", default=True)

type StreamSource = Literal["reasoning", "assistant"]


def set_verbose_tool_results(enabled: bool) -> None:  # noqa: FBT001
    """Set whether tool results print in full (True) or as a short preview."""
    _ = _verbose_tool_results.set(enabled)


def get_verbose_tool_results() -> bool:
    return _verbose_tool_results.get()


def set_markdown_enabled(enabled: bool) -> None:  # noqa: FBT001
    """Enable or disable end-of-turn Rich markdown rendering."""
    _ = _markdown_enabled.set(enabled)


def get_markdown_enabled() -> bool:
    return _markdown_enabled.get()


def interactive_terminal() -> bool:
    """True when stdout is a terminal we may write mid-line and erase on."""
    if os.environ.get("PLYNGENT_PLAIN", "").strip() in {"1", "true", "yes", "on"}:
        return False
    try:
        return sys.stdout.isatty()
    except AttributeError, OSError, ValueError:
        return False


def markdown_render_available() -> bool:
    """True when stdout is a TTY and plain mode is not forced via env."""
    return interactive_terminal()


def echo_notice(notice: Notice, *, err: bool = False) -> None:
    """Show a host notice to the user; the model gets the full body."""
    text = f"[notice] {notice.kind}: {notice.summary}" if notice.summary else f"[notice] {notice.kind}"
    click.secho(text, fg="yellow", err=err)


def _preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _preview_result(content: str, limit: int) -> str:
    """Non-verbose tool-result preview: first line + a ``(…N more lines)`` tail."""
    lines = content.splitlines()
    if not lines:
        return "(no output)"
    first = _preview(lines[0], limit)
    more = len(lines) - 1
    if more == 0:
        return first
    unit = "line" if more == 1 else "lines"
    return f"{first} (…{more} more {unit})"


def _json_arg(args_json: str, key: str) -> object | None:
    """Extract a single argument value (any type) from a tool-call args JSON blob."""
    try:
        raw = json.loads(args_json)
    except ValueError:
        return None
    if isinstance(raw, dict):
        return cast("dict[str, object]", raw).get(key)
    return None


def _json_str_arg(args_json: str, key: str) -> str | None:
    value = _json_arg(args_json, key)
    return value if isinstance(value, str) else None


def _json_list_arg(args_json: str, key: str) -> list[str] | None:
    value = _json_arg(args_json, key)
    if isinstance(value, list):
        raw_items = cast("list[object]", value)
        items = [item for item in raw_items if isinstance(item, str)]
        if len(items) == len(raw_items):
            return items
    return None


def _pretty_prefix(text: str) -> str:
    """Style the leading ``* Verb 'target' `` segment of a pretty tool line."""
    return click.style(text, fg="yellow")


def _pretty_segments(*segments: tuple[str, str | None]) -> str:
    """Style pretty-line detail segments.

    ``segments`` are ``(text, fg)`` pairs; ``fg=None`` leaves a segment uncolored
    and ``"dim"`` renders it dim. click.style emits ANSI codes only on a TTY,
    so non-TTY output stays plain.
    """
    parts: list[str] = []
    for text, fg in segments:
        if fg == "dim":
            parts.append(click.style(text, dim=True))
        else:
            parts.append(click.style(text, fg=fg))
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class PrettyLine:
    """Split renderer for one pretty tool call.

    ``prefix`` is written as soon as the call starts — a blocking tool call is
    then visible while it runs — and ``detail`` is appended once its result
    lands. Together they equal the single line the whole-line path prints.
    """

    prefix: Callable[[str], str]
    detail: Callable[[str, str], str]


def _read_file_prefix(args_json: str) -> str:
    """``* Read 'path' ``: known before the result, so it can lead the line."""
    path = _json_str_arg(args_json, "path") or "?"
    return _pretty_prefix(f"* Read '{path}' ")


def _read_file_detail(_args_json: str, result: str) -> str:
    """``read_file`` outcome: range/done, or which failure kind occurred."""
    if result.startswith("error: file not found"):
        return _pretty_segments(("(file not found)", "red"))
    if result.startswith("error: not a file"):
        return _pretty_segments(("(not a file)", "red"))
    if result.startswith("error:"):
        return _pretty_segments(("(error)", "red"))
    lines = result.splitlines()
    first = lines[0] if lines else ""
    if first.startswith("L") and "-" in first:
        return _pretty_segments((f"{first} ", "dim"), ("(done)", "green"))
    return _pretty_segments(("(done)", "green"))


def _todo_line(name: str) -> PrettyLine:
    """``todo_push`` / ``todo_update``: header first, rendered stack as the detail."""
    label = "Todo Push" if name == "todo_push" else "Todo Update"

    def prefix(_args_json: str) -> str:
        return click.style(f"* {label}:", fg="yellow")

    def detail(_args_json: str, result: str) -> str:
        return click.style(f"\n{result}", fg="yellow")

    return PrettyLine(prefix=prefix, detail=detail)


def _tree_line_stats(fmt: str, line: str) -> tuple[str, int] | None:
    """Classify one ``tree`` result line as ``(kind, depth)`` or None (not an entry).

    ``kind`` is ``"dir"`` or ``"file"``; depth is 0 for ``flat`` (no hierarchy).
    """
    stripped = line.strip()
    if not stripped:
        return None
    kind: str | None = None
    depth = 0
    if fmt == "flat":
        if not stripped.startswith("…"):
            kind = "dir" if stripped.endswith("/") else "file"
    elif fmt == "markdown":
        if stripped.startswith("- ") and not stripped.startswith(("- …", "- error")):
            kind = "dir" if stripped[2:].endswith("/") else "file"
            depth = (len(line) - len(line.lstrip(" "))) // 2 + 1
    elif "… (" not in stripped:
        branch = "├── " if "├── " in line else ("└── " if "└── " in line else None)
        if branch is not None:
            name = line[line.index(branch) + len(branch) :].rstrip()
            kind = "dir" if name.endswith("/") else "file"
            depth = line.index(branch) // 4 + 1
    if kind is None:
        return None
    return kind, depth


def _tree_prefix(args_json: str) -> str:
    path = _json_str_arg(args_json, "path") or "."
    return _pretty_prefix(f"* Tree '{path}' ")


def _tree_detail(args_json: str, result: str) -> str:
    """``tree`` outcome: dir/file counts, plus depth when the format has one.

    Flat trees have no depth, so the depth part is omitted for ``format=flat``.
    """
    if result.startswith("error:"):
        return _pretty_segments(("(error)", "red"))
    fmt = _json_str_arg(args_json, "format") or "markdown"
    dirs = 0
    files = 0
    depth = 0
    for line in result.splitlines():
        stats = _tree_line_stats(fmt, line)
        if stats is None:
            continue
        kind, line_depth = stats
        if kind == "dir":
            dirs += 1
        else:
            files += 1
        depth = max(depth, line_depth)
    if depth:
        return _pretty_segments((f"({dirs} dirs, {files} files, depth {depth})", None))
    return _pretty_segments((f"({dirs} dirs, {files} files)", None))


def _listdir_prefix(args_json: str) -> str:
    path = _json_str_arg(args_json, "path") or "."
    return _pretty_prefix(f"* List '{path}' ")


def _listdir_detail(_args_json: str, result: str) -> str:
    """``listdir`` outcome: dir/file counts, or empty/error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if result == "(empty)":
        return _pretty_segments(("(empty)", "dim"))
    dirs = 0
    files = 0
    for line in result.splitlines():
        if line.startswith("dir\t"):
            dirs += 1
        elif line.startswith("file\t"):
            files += 1
    return _pretty_segments((f"({dirs} dirs, {files} files)", None))


def _glob_prefix(args_json: str) -> str:
    pattern = _json_str_arg(args_json, "pattern") or "?"
    path = _json_str_arg(args_json, "path") or "."
    return _pretty_prefix(f"* Glob '{pattern}' in '{path}' ")


def _glob_detail(_args_json: str, result: str) -> str:
    """``glob_paths`` outcome: match count, or none/error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if result == "(no matches)":
        return _pretty_segments(("(no matches)", "dim"))
    count = sum(1 for line in result.splitlines() if line and not line.startswith("...[truncated"))
    unit = "path" if count == 1 else "paths"
    return _pretty_segments((f"({count} {unit})", None))


def _grep_prefix(args_json: str) -> str:
    pattern = _json_str_arg(args_json, "pattern") or "?"
    path = _json_str_arg(args_json, "path") or "."
    return _pretty_prefix(f"* Grep '{pattern}' in '{path}' ")


def _grep_detail(_args_json: str, result: str) -> str:
    """``grep_files`` outcome: matches and file count, or none/error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if result == "(no matches)":
        return _pretty_segments(("(no matches)", "dim"))
    matches = [line for line in result.splitlines() if line and not line.startswith("...[truncated")]
    files = len({line.split(":", 1)[0] for line in matches})
    match_unit = "match" if len(matches) == 1 else "matches"
    file_unit = "file" if files == 1 else "files"
    return _pretty_segments((f"({len(matches)} {match_unit} in {files} {file_unit})", None))


def _run_argv_prefix(args_json: str) -> str:
    argv = _json_list_arg(args_json, "argv")
    cmd = shlex.join(argv) if argv else "?"
    return _pretty_prefix(f"* Run $ {cmd} ")


def _run_argv_detail(_args_json: str, result: str) -> str:
    """``run_argv`` outcome: exit code, timeout, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    fields: dict[str, str] = {}
    for line in result.splitlines():
        if line.startswith("--- "):
            break
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key] = value
    if fields.get("timed_out") == "true":
        return _pretty_segments(("(timed out)", "red"))
    code = fields.get("exit_code", "")
    fg = "green" if code == "0" else "red"
    return _pretty_segments((f"(exit code {code or 'killed'})", fg))


def _run_argv_batch_prefix(_args_json: str) -> str:
    return _pretty_prefix("* Run batch ")


def _run_argv_batch_detail(_args_json: str, result: str) -> str:
    """``run_argv_batch`` outcome: done, stopped early, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    lines = result.splitlines()
    head = lines[0] if lines else ""
    if any(part == "stopped_early=true" for part in head.split()):
        return _pretty_segments(("(stopped early)", "yellow"))
    return _pretty_segments(("(done)", "green"))


def _http_status_fg(status: str) -> str | None:
    """Foreground color by HTTP status class (2xx green, 3xx yellow, 4xx/5xx red)."""
    if status and status[0] in {"2", "3", "4", "5"}:
        return {"2": "green", "3": "yellow", "4": "red", "5": "red"}[status[0]]
    return None


def _fetch_prefix(args_json: str) -> str:
    method = _json_str_arg(args_json, "method") or "GET"
    url = _json_str_arg(args_json, "url") or "?"
    return _pretty_prefix(f"* Fetch {method} {url} ")


def _fetch_detail(_args_json: str, result: str) -> str:
    """``fetch`` outcome: HTTP status, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    status = ""
    for line in result.splitlines():
        if line.startswith("--- "):
            break
        if line.startswith("status="):
            status = line.partition("=")[2]
            break
    return _pretty_segments((f"({status or 'done'})", _http_status_fg(status)))


_MUTATOR_VERBS: dict[str, str] = {
    "edit_replace": "Edit",
    "edit_lineno": "Edit",
    "write_file": "Write",
    "copy_path": "Copy",
    "move_path": "Move",
    "delete_path": "Delete",
}


def _mutator_prefix(name: str, args_json: str) -> str:
    if name in {"copy_path", "move_path"}:
        src = _json_str_arg(args_json, "src") or "?"
        dst = _json_str_arg(args_json, "dst") or "?"
        target = f"'{src}' → '{dst}'"
    else:
        target = f"'{_json_str_arg(args_json, 'path') or '?'}'"
    return _pretty_prefix(f"* {_MUTATOR_VERBS[name]} {target} ")


def _mutator_detail(name: str, result: str) -> str:
    """File mutation outcome: written size, done, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if name == "write_file":
        # write_file keeps its brief detail: ``wrote 120 characters to path``.
        match = re.match(r"wrote (\d+) characters to (.+)$", result)
        if match:
            chars, _path = match.groups()
            return _pretty_segments((f"({chars} chars)", None))
    return _pretty_segments(("(done)", "green"))


def _mutator_line(name: str) -> PrettyLine:
    return PrettyLine(
        prefix=lambda args: _mutator_prefix(name, args),
        detail=lambda _args, result: _mutator_detail(name, result),
    )


def _vcs_status_prefix(_args_json: str) -> str:
    return _pretty_prefix("* VCS Status ")


def _vcs_status_detail(_args_json: str, result: str) -> str:
    """``vcs_status`` outcome: done, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _pretty_segments(("(done)", "green"))


def _vcs_log_prefix(_args_json: str) -> str:
    return _pretty_prefix("* VCS Log ")


def _vcs_log_detail(_args_json: str, result: str) -> str:
    """``vcs_log`` outcome: commit count, none, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if result == "(no commits)":
        return _pretty_segments(("(no commits)", "dim"))
    commits = [line for line in result.splitlines() if line.strip()]
    unit = "commit" if len(commits) == 1 else "commits"
    return _pretty_segments((f"({len(commits)} {unit})", None))


def _wait_prefix(_args_json: str) -> str:
    return _pretty_prefix("* Wait ")


def _wait_detail(_args_json: str, result: str) -> str:
    """``wait`` outcome: slept duration, disturbed, cancelled, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if result.startswith("waited "):
        return _pretty_segments((f"({result.removeprefix('waited ')})", "green"))
    if result.startswith("cancelled by user"):
        return _pretty_segments(("(cancelled)", "yellow"))
    return _pretty_segments(("(disturbed)", "yellow"))


# Pretty tools: ``prefix`` prints as the call starts, ``detail`` when it lands.
_PRETTY_LINES: dict[str, PrettyLine] = {
    "read_file": PrettyLine(prefix=_read_file_prefix, detail=_read_file_detail),
    "tree": PrettyLine(prefix=_tree_prefix, detail=_tree_detail),
    "todo_push": _todo_line("todo_push"),
    "todo_update": _todo_line("todo_update"),
    "listdir": PrettyLine(prefix=_listdir_prefix, detail=_listdir_detail),
    "glob_paths": PrettyLine(prefix=_glob_prefix, detail=_glob_detail),
    "grep_files": PrettyLine(prefix=_grep_prefix, detail=_grep_detail),
    "run_argv": PrettyLine(prefix=_run_argv_prefix, detail=_run_argv_detail),
    "run_argv_batch": PrettyLine(prefix=_run_argv_batch_prefix, detail=_run_argv_batch_detail),
    "fetch": PrettyLine(prefix=_fetch_prefix, detail=_fetch_detail),
    "edit_replace": _mutator_line("edit_replace"),
    "edit_lineno": _mutator_line("edit_lineno"),
    "write_file": _mutator_line("write_file"),
    "copy_path": _mutator_line("copy_path"),
    "move_path": _mutator_line("move_path"),
    "delete_path": _mutator_line("delete_path"),
    "vcs_status": PrettyLine(prefix=_vcs_status_prefix, detail=_vcs_status_detail),
    "vcs_log": PrettyLine(prefix=_vcs_log_prefix, detail=_vcs_log_detail),
    "wait": PrettyLine(prefix=_wait_prefix, detail=_wait_detail),
}

_PRETTY_TOOLS = frozenset(_PRETTY_LINES)


def _pretty_parts(name: str, args_json: str, result: str) -> tuple[str, str] | None:
    """Prefix + detail for a known tool call; None keeps the ``[tool]`` style."""
    line = _PRETTY_LINES.get(name)
    if line is None:
        return None
    return line.prefix(args_json), line.detail(args_json, result)


def _echo_stream(text: str) -> None:
    """Write without newline and flush so assistant text appears token-by-token."""
    click.echo(text, nl=False)
    with contextlib.suppress(OSError):
        _ = sys.stdout.flush()


def _clear_streamed_lines(line_count: int) -> None:
    """Move cursor up and clear the streamed plain-text region (TTY only)."""
    if line_count <= 0:
        return
    # Clear current line, then each previous line of the streamed block.
    for _ in range(line_count):
        _ = sys.stdout.write("\r\033[2K\033[1A")
    _ = sys.stdout.write("\r\033[2K")
    with contextlib.suppress(OSError):
        _ = sys.stdout.flush()


def _line_count_for_clear(label: str, body: str) -> int:
    """Approximate terminal lines used by ``label\\n + body`` for cursor erase."""
    if not body and not label:
        return 0
    # Label is on its own line; body may contain newlines.
    text = f"{label}\n{body}" if label else body
    return text.count("\n") + 1


class _PrettyToolStream:
    """Pretty tool lines: prefix at call time, detail appended when it lands.

    Writing the prefix immediately keeps a blocking tool call visible; appending
    the detail later reproduces the single whole-line write byte for byte, so
    non-interactive output is unchanged. The calls of one batch are yielded
    before their results, so a second call while a prefix is open (parallel
    tools) cannot append its detail in order: that prefix is erased and the
    batch falls back to whole lines.
    """

    _interactive: bool
    _open: bool
    _whole_lines: bool

    def __init__(self, *, interactive: bool) -> None:
        self._interactive = interactive
        self._open = False
        self._whole_lines = False

    def start_call(self, prefix: str) -> None:
        """Show *prefix* for a call that is about to run (cursor left mid-line)."""
        if not (self._interactive and prefix) or self._whole_lines:
            return
        if self._open:
            # Parallel batch: the open prefix could never receive its detail.
            _clear_streamed_lines(1)
            self._open = False
            self._whole_lines = True
            return
        click.echo(f"\n{prefix}", nl=False)
        with contextlib.suppress(OSError):
            _ = sys.stdout.flush()
        self._open = True

    def finish_call(self, detail: str) -> bool:
        """Append *detail* to the open prefix; False when a whole line is needed."""
        if not (self._interactive and self._open):
            return False
        click.echo(f"{detail}\n", nl=False)
        self._open = False
        return True

    def close_line(self) -> None:
        """End a dangling prefix line before other output takes the next line."""
        if not self._open:
            return
        click.echo()
        self._open = False
        self._whole_lines = True

    def end_batch(self) -> None:
        """Reset batch-scoped state once every call of the batch has a result."""
        self.close_line()
        self._whole_lines = False


def print_markdown(text: str, *, label: str = "assistant:") -> None:
    """Render *text* as markdown via Rich; *label* on its own line when set."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text

    console = Console(file=sys.stdout, highlight=False)
    if label:
        console.print(Text(label, style="cyan"))
    console.print(Markdown(text))


def _flush_assistant_markdown(body: str, *, pretty: bool) -> None:
    """Replace the plain assistant stream with markdown when enabled."""
    if not body.strip():
        click.echo()
        return
    if pretty:
        lines = _line_count_for_clear("assistant:", body)
        _clear_streamed_lines(lines)
        print_markdown(body, label="assistant:")
        click.echo()
    else:
        click.echo()


async def render_events(  # noqa: C901, PLR0912, PLR0915
    events: AsyncIterator[AgentEvent],
    *,
    verbose: bool | None = None,
    markdown: bool | None = None,
) -> None:
    """Print agent events to the terminal (text deltas stream as they arrive).

    Assistant and reasoning each start on a new line after their label. When the
    content source changes (reasoning ↔ assistant, or tools/errors), the
    assistant markdown buffer is flushed so streams do not mix and Rich can
    re-render completed assistant segments.

    Pretty tool calls print their ``* Verb 'target' `` prefix as soon as the call
    starts, so a blocking tool is visible while it runs; the outcome is appended
    when the result lands (same bytes as the whole-line fallback).
    """
    show_full = get_verbose_tool_results() if verbose is None else verbose
    use_markdown = get_markdown_enabled() if markdown is None else markdown
    pretty = bool(use_markdown and markdown_render_available())
    tool_lines = _PrettyToolStream(interactive=interactive_terminal() and not show_full)

    source: StreamSource | None = None
    assistant_buf: list[str] = []
    printed_reasoning = False
    printed_assistant = False
    # Tool calls buffer so prettified tools can render once their result lands
    # (the summary line needs the status/range). FIFO matches loop event order.
    pending_tools: list[tuple[str, str, bool]] = []

    def flush_assistant() -> None:
        nonlocal source, assistant_buf, printed_assistant
        if source != "assistant" and not assistant_buf:
            return
        body = "".join(assistant_buf)
        assistant_buf = []
        if printed_assistant:
            _flush_assistant_markdown(body, pretty=pretty)
        printed_assistant = False
        if source == "assistant":
            source = None

    def begin_reasoning() -> None:
        nonlocal source, printed_reasoning
        if source == "reasoning":
            return
        if source == "assistant":
            flush_assistant()
        tool_lines.close_line()
        click.echo()
        click.secho("reasoning:", fg="bright_black")
        source = "reasoning"
        printed_reasoning = True

    def begin_assistant() -> None:
        nonlocal source, printed_assistant
        if source == "assistant":
            return
        if source == "reasoning":
            click.echo()  # end reasoning stream line
            source = None
        tool_lines.close_line()
        click.echo()
        click.secho("assistant:", fg="cyan")
        source = "assistant"
        printed_assistant = True

    async for event in events:
        if isinstance(event, ReasoningDeltaEvent):
            begin_reasoning()
            _echo_stream(event.content)
        elif isinstance(event, TextDeltaEvent):
            begin_assistant()
            assistant_buf.append(event.content)
            _echo_stream(event.content)
        elif isinstance(event, ToolCallEvent):
            flush_assistant()
            call = event.tool_call
            if isinstance(call, AssistantFunctionToolCall):
                name = call.function.name
                args = call.function.arguments
                pretty_tool = name in _PRETTY_TOOLS
                pending_tools.append((name, args, pretty_tool))
                if pretty_tool:
                    tool_lines.start_call(_PRETTY_LINES[name].prefix(args))
                else:
                    tool_lines.close_line()
                    preview = _preview(args, _TOOL_ARGS_PREVIEW)
                    click.secho(f"\n[tool] {name}({preview})", fg="yellow")
            else:
                tool_lines.close_line()
                pending_tools.append(("custom", call.id, False))
                click.secho(f"\n[tool] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            flush_assistant()
            content = event.message.content
            name, args, pretty_tool = pending_tools.pop(0) if pending_tools else ("", "", False)
            parts = _pretty_parts(name, args, content) if pretty_tool else None
            if parts is not None and not show_full:
                prefix, detail = parts
                if not tool_lines.finish_call(detail):
                    click.echo(f"\n{prefix}{detail}")
            elif show_full:
                tool_lines.close_line()
                click.secho(f"[tool ok]\n{content}", fg="magenta")
            else:
                tool_lines.close_line()
                preview = _preview_result(content, _TOOL_RESULT_PREVIEW)
                click.secho(f"[tool ok] {preview}", fg="magenta")
            if not pending_tools:
                tool_lines.end_batch()
        elif isinstance(event, ErrorEvent):
            flush_assistant()
            tool_lines.close_line()
            suffix = ""
            if event.source:
                suffix += f" source={event.source}"
            if not event.retryable:
                suffix += " (fatal)"
            click.secho(f"\n[error]{suffix} {event.message}", fg="bright_red")
        elif isinstance(event, CancelledEvent):
            flush_assistant()
            tool_lines.close_line()
            if event.reason:
                click.secho(f"\n[cancelled] {event.reason}", fg="yellow")
            else:
                click.secho("\n[cancelled]", fg="yellow")
        elif isinstance(event, MaxRoundsEvent):
            flush_assistant()
            tool_lines.close_line()
            if event.continued:
                click.secho(
                    f"\n[max rounds {event.rounds} reached — continuing with a higher allowance]",
                    fg="yellow",
                )
            else:
                click.secho(f"\n[max rounds reached: {event.rounds}]", fg="red")
        elif isinstance(event, UsageEvent):
            _ = event
        else:
            # AssistantMessageEvent — text already shown via TextDeltaEvent.
            _ = event

    # End-of-turn: flush any open assistant segment; close reasoning stream.
    tool_lines.close_line()
    if assistant_buf or printed_assistant:
        flush_assistant()
    elif printed_reasoning:
        click.echo()
    click.echo()
