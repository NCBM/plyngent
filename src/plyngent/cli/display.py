from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import sys
from contextvars import ContextVar
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

    from plyngent.agent import AgentEvent

_TOOL_RESULT_PREVIEW = 120
_TOOL_ARGS_PREVIEW = 80

# Tool calls that render as a single pretty summary line instead of
# ``[tool]`` / ``[tool result]``.
_PRETTY_TOOLS = frozenset(
    {
        "read_file",
        "todo_push",
        "todo_update",
        "tree",
        "listdir",
        "glob_paths",
        "grep_files",
        "run_argv",
        "run_argv_batch",
        "fetch",
        "edit_replace",
        "edit_lineno",
        "write_file",
        "copy_path",
        "move_path",
        "delete_path",
    }
)

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


def markdown_render_available() -> bool:
    """True when stdout is a TTY and plain mode is not forced via env."""
    if os.environ.get("PLYNGENT_PLAIN", "").strip() in {"1", "true", "yes", "on"}:
        return False
    try:
        return sys.stdout.isatty()
    except AttributeError, OSError, ValueError:
        return False


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


def _pretty_line(prefix: str, *segments: tuple[str, str | None]) -> str:
    """Style a pretty tool summary: yellow ``* Verb`` prefix + colored detail segments.

    ``segments`` are ``(text, fg)`` pairs; ``fg=None`` leaves a segment uncolored
    and ``"dim"`` renders it dim. click.style emits ANSI codes only on a TTY,
    so non-TTY output stays plain.
    """
    parts = [click.style(prefix, fg="yellow")]
    for text, fg in segments:
        if fg == "dim":
            parts.append(click.style(text, dim=True))
        else:
            parts.append(click.style(text, fg=fg))
    return "".join(parts)


def _read_file_pretty(args_json: str, result: str) -> str:
    """One-line summary for a ``read_file`` call: ``* Read 'path' (done)``."""
    path = _json_str_arg(args_json, "path") or "?"
    if result.startswith("error: file not found"):
        return _pretty_line(f"* Read '{path}' ", ("(file not found)", "red"))
    if result.startswith("error: not a file"):
        return _pretty_line(f"* Read '{path}' ", ("(not a file)", "red"))
    if result.startswith("error:"):
        return _pretty_line(f"* Read '{path}' ", ("(error)", "red"))
    return _pretty_line(f"* Read '{path}' ", ("(done)", "green"))


def _todo_pretty(name: str, result: str) -> str:
    """Summary for todo push/update: header + the rendered todo stack."""
    label = "Todo Push" if name == "todo_push" else "Todo Update"
    return _pretty_line(f"* {label}:\n{result}")


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


def _tree_pretty(args_json: str, result: str) -> str:
    """One-line summary for a ``tree`` call: ``* Tree 'src' (3 dirs, 24 files, depth 3)``.

    Flat trees have no depth, so the depth part is omitted for ``format=flat``.
    """
    path = _json_str_arg(args_json, "path") or "."
    fmt = _json_str_arg(args_json, "format") or "markdown"
    if result.startswith("error:"):
        return f"* Tree '{path}' (error)"
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
        return _pretty_line(f"* Tree '{path}' ", (f"({dirs} dirs, {files} files, depth {depth})", None))
    return _pretty_line(f"* Tree '{path}' ", (f"({dirs} dirs, {files} files)", None))


def _listdir_pretty(args_json: str, result: str) -> str:
    """One-line summary for a ``listdir`` call: ``* List 'path' (3 dirs, 5 files)``."""
    path = _json_str_arg(args_json, "path") or "."
    if result.startswith("error:"):
        return _pretty_line(f"* List '{path}' ", (f"({result})", "red"))
    if result == "(empty)":
        return _pretty_line(f"* List '{path}' ", ("(empty)", "dim"))
    dirs = 0
    files = 0
    for line in result.splitlines():
        if line.startswith("dir\t"):
            dirs += 1
        elif line.startswith("file\t"):
            files += 1
    return _pretty_line(f"* List '{path}' ", (f"({dirs} dirs, {files} files)", None))


def _glob_pretty(args_json: str, result: str) -> str:
    """One-line summary for a ``glob_paths`` call: ``* Glob '**/*.py' in '.' (12 paths)``."""
    pattern = _json_str_arg(args_json, "pattern") or "?"
    path = _json_str_arg(args_json, "path") or "."
    if result.startswith("error:"):
        return _pretty_line(f"* Glob '{pattern}' in '{path}' ", (f"({result})", "red"))
    if result == "(no matches)":
        return _pretty_line(f"* Glob '{pattern}' in '{path}' ", ("(no matches)", "dim"))
    count = sum(1 for line in result.splitlines() if line and not line.startswith("...[truncated"))
    unit = "path" if count == 1 else "paths"
    return _pretty_line(f"* Glob '{pattern}' in '{path}' ", (f"({count} {unit})", None))


def _grep_pretty(args_json: str, result: str) -> str:
    """One-line summary for a ``grep_files`` call: ``* Grep 'pat' in '.' (12 matches in 3 files)``."""
    pattern = _json_str_arg(args_json, "pattern") or "?"
    path = _json_str_arg(args_json, "path") or "."
    if result.startswith("error:"):
        return _pretty_line(f"* Grep '{pattern}' in '{path}' ", (f"({result})", "red"))
    if result == "(no matches)":
        return _pretty_line(f"* Grep '{pattern}' in '{path}' ", ("(no matches)", "dim"))
    matches = [line for line in result.splitlines() if line and not line.startswith("...[truncated")]
    files = len({line.split(":", 1)[0] for line in matches})
    match_unit = "match" if len(matches) == 1 else "matches"
    file_unit = "file" if files == 1 else "files"
    return _pretty_line(
        f"* Grep '{pattern}' in '{path}' ",
        (f"({len(matches)} {match_unit} in {files} {file_unit})", None),
    )


def _run_argv_pretty(args_json: str, result: str) -> str:
    """One-line summary for a ``run_argv`` call: ``* Run $ git status --short (exit code 0)``."""
    argv = _json_list_arg(args_json, "argv")
    cmd = shlex.join(argv) if argv else "?"
    if result.startswith("error:"):
        return _pretty_line(f"* Run $ {cmd} ", (f"({result})", "red"))
    fields: dict[str, str] = {}
    for line in result.splitlines():
        if line.startswith("--- "):
            break
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key] = value
    if fields.get("timed_out") == "true":
        return _pretty_line(f"* Run $ {cmd} ", ("(timed out)", "red"))
    code = fields.get("exit_code", "")
    fg = "green" if code == "0" else "red"
    return _pretty_line(f"* Run $ {cmd} ", (f"(exit code {code or 'killed'})", fg))


def _run_argv_batch_pretty(_args_json: str, result: str) -> str:
    """One-line summary for a ``run_argv_batch`` call: ``* Run batch (done)``."""
    if result.startswith("error:"):
        return _pretty_line("* Run batch ", (f"({result})", "red"))
    lines = result.splitlines()
    head = lines[0] if lines else ""
    stopped_early = False
    for part in head.split():
        if part == "stopped_early=true":
            stopped_early = True
    if stopped_early:
        return _pretty_line("* Run batch ", ("(stopped early)", "yellow"))
    return _pretty_line("* Run batch ", ("(done)", "green"))


def _http_status_fg(status: str) -> str | None:
    """Foreground color by HTTP status class (2xx green, 3xx yellow, 4xx/5xx red)."""
    if status and status[0] in {"2", "3", "4", "5"}:
        return {"2": "green", "3": "yellow", "4": "red", "5": "red"}[status[0]]
    return None


def _fetch_pretty(args_json: str, result: str) -> str:
    """One-line summary for a ``fetch`` call: ``* Fetch GET https://… (200)``."""
    method = _json_str_arg(args_json, "method") or "GET"
    url = _json_str_arg(args_json, "url") or "?"
    if result.startswith("error:"):
        return _pretty_line(f"* Fetch {method} {url} ", (f"({result})", "red"))
    status = ""
    for line in result.splitlines():
        if line.startswith("--- "):
            break
        if line.startswith("status="):
            status = line.partition("=")[2]
            break
    return _pretty_line(f"* Fetch {method} {url} ", (f"({status or 'done'})", _http_status_fg(status)))


_MUTATOR_VERBS: dict[str, str] = {
    "edit_replace": "Edit",
    "edit_lineno": "Edit",
    "write_file": "Write",
    "copy_path": "Copy",
    "move_path": "Move",
    "delete_path": "Delete",
}


def _mutator_pretty(name: str, args_json: str, result: str) -> str:
    """One-line summary for a file mutation: ``* Wrote 'path' (120 chars)`` / ``* Edit 'path' (done)``."""
    verb = _MUTATOR_VERBS[name]
    if name in {"copy_path", "move_path"}:
        src = _json_str_arg(args_json, "src") or "?"
        dst = _json_str_arg(args_json, "dst") or "?"
        target = f"'{src}' → '{dst}'"
    else:
        path = _json_str_arg(args_json, "path") or "?"
        target = f"'{path}'"
    if result.startswith("error:"):
        return _pretty_line(f"* {verb} {target} ", (f"({result})", "red"))
    if name == "write_file":
        # write_file keeps its brief detail: ``wrote 120 characters to path``.
        match = re.match(r"wrote (\d+) characters to (.+)$", result)
        if match:
            chars, path = match.groups()
            return _pretty_line(f"* Write '{path}' ", (f"({chars} chars)", None))
    return _pretty_line(f"* {verb} {target} ", ("(done)", "green"))


_PRETTY_BUILDERS: dict[str, Callable[[str, str], str]] = {
    "read_file": _read_file_pretty,
    "tree": _tree_pretty,
    "todo_push": lambda _args, result: _todo_pretty("todo_push", result),
    "todo_update": lambda _args, result: _todo_pretty("todo_update", result),
    "listdir": _listdir_pretty,
    "glob_paths": _glob_pretty,
    "grep_files": _grep_pretty,
    "run_argv": _run_argv_pretty,
    "run_argv_batch": _run_argv_batch_pretty,
    "fetch": _fetch_pretty,
    "edit_replace": lambda args, result: _mutator_pretty("edit_replace", args, result),
    "edit_lineno": lambda args, result: _mutator_pretty("edit_lineno", args, result),
    "write_file": lambda args, result: _mutator_pretty("write_file", args, result),
    "copy_path": lambda args, result: _mutator_pretty("copy_path", args, result),
    "move_path": lambda args, result: _mutator_pretty("move_path", args, result),
    "delete_path": lambda args, result: _mutator_pretty("delete_path", args, result),
}


def _pretty_tool_result(name: str, args_json: str, result: str) -> str | None:
    """Pretty summary for a known tool call + result; None keeps the old style."""
    builder = _PRETTY_BUILDERS.get(name)
    if builder is None:
        return None
    return builder(args_json, result)


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
    """
    show_full = get_verbose_tool_results() if verbose is None else verbose
    use_markdown = get_markdown_enabled() if markdown is None else markdown
    pretty = bool(use_markdown and markdown_render_available())

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
                if not pretty_tool:
                    preview = _preview(args, _TOOL_ARGS_PREVIEW)
                    click.secho(f"\n[tool] {name}({preview})", fg="yellow")
            else:
                pending_tools.append(("custom", call.id, False))
                click.secho(f"\n[tool] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            flush_assistant()
            content = event.message.content
            name, args, pretty_tool = pending_tools.pop(0) if pending_tools else ("", "", False)
            pretty_line = _pretty_tool_result(name, args, content) if pretty_tool else None
            if pretty_line is not None and not show_full:
                click.echo(f"\n{pretty_line}")
            elif show_full:
                click.secho(f"[tool ok]\n{content}", fg="magenta")
            else:
                preview = _preview_result(content, _TOOL_RESULT_PREVIEW)
                click.secho(f"[tool ok] {preview}", fg="magenta")
        elif isinstance(event, ErrorEvent):
            flush_assistant()
            suffix = ""
            if event.source:
                suffix += f" source={event.source}"
            if not event.retryable:
                suffix += " (fatal)"
            click.secho(f"\n[error]{suffix} {event.message}", fg="bright_red")
        elif isinstance(event, CancelledEvent):
            flush_assistant()
            if event.reason:
                click.secho(f"\n[cancelled] {event.reason}", fg="yellow")
            else:
                click.secho("\n[cancelled]", fg="yellow")
        elif isinstance(event, MaxRoundsEvent):
            flush_assistant()
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
    if assistant_buf or printed_assistant:
        flush_assistant()
    elif printed_reasoning:
        click.echo()
    click.echo()
