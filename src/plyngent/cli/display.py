from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
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
_ASK_SUBJECT_PREVIEW = 60
_MCP_TOOL_PREFIX = "mcp__"
_SKILL_MAIN_FILE = "SKILL.md"

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


def _json_int_arg(args_json: str, key: str) -> int | None:
    value = _json_arg(args_json, key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _status_fields(result: str) -> dict[str, str]:
    """Parse the leading ``key=value`` block of a structured tool result.

    The block ends at the first ``--- <payload> ---`` marker (``run_argv``,
    ``fetch``, ``read_pty``).
    """
    fields: dict[str, str] = {}
    for line in result.splitlines():
        if line.startswith("--- "):
            break
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key] = value
    return fields


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


def _range_detail(result: str) -> str | None:
    """``L{begin}-{end} (done)`` detail of a ranged result, when it carries one."""
    first = result.splitlines()[0] if result else ""
    if first.startswith("L") and "-" in first:
        return _pretty_segments((f"{first} ", "dim"), ("(done)", "green"))
    return None


def _read_file_detail(_args_json: str, result: str) -> str:
    """``read_file`` outcome: range/done, or which failure kind occurred."""
    if result.startswith("error: file not found"):
        return _pretty_segments(("(file not found)", "red"))
    if result.startswith("error: not a file"):
        return _pretty_segments(("(not a file)", "red"))
    if result.startswith("error:"):
        return _pretty_segments(("(error)", "red"))
    return _range_detail(result) or _pretty_segments(("(done)", "green"))


_TODO_LABELS: dict[str, str] = {
    "todo_list": "Todo List",
    "todo_push": "Todo Push",
    "todo_pop": "Todo Pop",
    "todo_update": "Todo Update",
    "todo_clear": "Todo Clear",
}


def _todo_line(name: str) -> PrettyLine:
    """Todo tools: header first, the rendered stack as the detail."""
    label = _TODO_LABELS[name]

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


def _regex_prefix(args_json: str) -> str:
    pattern = _json_str_arg(args_json, "pattern") or "?"
    path = _json_str_arg(args_json, "path") or "."
    return _pretty_prefix(f"* Search '{pattern}' in '{path}' ")


def _regex_detail(_args_json: str, result: str) -> str:
    """``regex_files`` outcome: matches and file count, or none/error."""
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
    fields = _status_fields(result)
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
    status = _status_fields(result).get("status", "")
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


def _vcs_kind_prefix(_args_json: str) -> str:
    return _pretty_prefix("* VCS Kind ")


def _vcs_status_prefix(_args_json: str) -> str:
    return _pretty_prefix("* VCS Status ")


def _vcs_status_detail(_args_json: str, result: str) -> str:
    """``vcs_status`` outcome: done, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _pretty_segments(("(done)", "green"))


def _vcs_log_prefix(_args_json: str) -> str:
    return _pretty_prefix("* VCS Log ")


def _vcs_diff_prefix(args_json: str) -> str:
    staged = _json_arg(args_json, "staged") is True
    return _pretty_prefix("* VCS Diff (staged) " if staged else "* VCS Diff ")


def _vcs_branch_prefix(_args_json: str) -> str:
    return _pretty_prefix("* VCS Branch ")


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
# PTY sessions share one prefix shape: ``* PTY <verb> <session> ``.
def _pty_session_label(args_json: str) -> str:
    session_id = _json_int_arg(args_json, "session_id")
    return "?" if session_id is None else str(session_id)


def _pty_session_line(verb: str, detail: Callable[[str, str], str]) -> PrettyLine:
    """``* PTY <verb> <session> `` plus the verb's own outcome."""
    return PrettyLine(
        prefix=lambda args_json: _pretty_prefix(f"* PTY {verb} {_pty_session_label(args_json)} "),
        detail=detail,
    )


def _open_pty_prefix(args_json: str) -> str:
    argv = _json_list_arg(args_json, "command")
    cmd = shlex.join(argv) if argv else "?"
    return _pretty_prefix(f"* PTY Open $ {cmd} ")


def _open_pty_detail(_args_json: str, result: str) -> str:
    """``open_pty`` outcome: the new session id, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    session_id = _status_fields(result).get("session_id", "")
    return _pretty_segments((f"(session {session_id or '?'})", "green"))


def _read_pty_detail(args_json: str, result: str) -> str:
    """``read_pty`` outcome: exit status, matched wait, payload size, or error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    fields = _status_fields(result)
    data = result.split("--- data ---", 1)[1].removeprefix("\n") if "--- data ---" in result else ""
    count = len(data.splitlines())
    if fields.get("alive") == "false":
        code = fields.get("exit_code", "")
        return _pretty_segments((f"(exited {code})" if code else "(exited)", "green"))
    unit = "line" if count == 1 else "lines"
    if _json_str_arg(args_json, "until") and fields.get("matched") == "true":
        return _pretty_segments((f"(matched, {count} {unit})", "green"))
    if count == 0:
        return _pretty_segments(("(alive, no output)", "dim"))
    return _pretty_segments((f"(alive, {count} {unit})", None))


def _pty_write_detail(_args_json: str, result: str) -> str:
    """``write_pty`` / ``write_pty_keys`` / ``ask_into_pty``: size, sent, error.

    The human's answer never appears here — only that it was written.
    """
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    fields = _status_fields(result)
    if fields.get("source") == "human":
        extra = ", secret" if fields.get("secret") == "true" else ""
        return _pretty_segments((f"(answer sent{extra})", "green"))
    wrote = fields.get("wrote", "")
    if wrote:
        unit = "char" if wrote == "1" else "chars"
        return _pretty_segments((f"({wrote} {unit})", None))
    return _pretty_segments(("(done)", "green"))


def _close_pty_detail(_args_json: str, result: str) -> str:
    """``close_pty`` outcome: closed, already closed, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if _status_fields(result).get("closed") == "true":
        return _pretty_segments(("(closed)", "green"))
    return _pretty_segments(("(already closed)", "dim"))


def _ask_line(verb: str, field: str, detail: Callable[[str, str], str]) -> PrettyLine:
    """``* Ask <verb> '<subject>' `` plus the verb's own outcome.

    *field* names the argument that carries the human-facing subject: the
    question for ``ask_user_line`` / ``ask_user_choice``, the title for a form.
    """
    label = f"Ask {verb}" if verb else "Ask"

    def prefix(args_json: str) -> str:
        subject = _json_str_arg(args_json, field) or "?"
        return _pretty_prefix(f"* {label} '{_preview(subject, _ASK_SUBJECT_PREVIEW)}' ")

    return PrettyLine(prefix=prefix, detail=detail)


def _ask_detail(_args_json: str, result: str) -> str:
    """``ask_user_line`` outcome: an answer came back, or the prompt failed."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _pretty_segments(("(answered)" if result else "(empty)", "green"))


def _ask_choice_detail(_args_json: str, result: str) -> str:
    """``ask_user_choice`` outcome: the chosen label, or the prompt failed."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if not result:
        return _pretty_segments(("(no selection)", "dim"))
    return _pretty_segments((f"({_preview(result, _ASK_SUBJECT_PREVIEW)})", "green"))


def _ask_form_detail(_args_json: str, result: str) -> str:
    """``ask_user_form`` outcome: answer count, or the prompt failed."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    try:
        loaded: object = json.loads(result)
    except json.JSONDecodeError:
        return _pretty_segments(("(answered)", "green"))
    if not isinstance(loaded, dict):
        return _pretty_segments(("(answered)", "green"))
    count = len(cast("dict[str, object]", loaded))
    unit = "answer" if count == 1 else "answers"
    return _pretty_segments((f"({count} {unit})", "green"))


def _access_prefix(args_json: str) -> str:
    path = _json_str_arg(args_json, "path") or "?"
    mode = _json_str_arg(args_json, "mode") or "read"
    return _pretty_prefix(f"* Access '{path}' ({mode}) ")


def _access_detail(_args_json: str, result: str) -> str:
    """``request_directory_access``: granted, already allowed, or denied."""
    if result.startswith("access granted:"):
        return _pretty_segments(("(granted)", "green"))
    if result.startswith("already accessible:"):
        return _pretty_segments(("(already accessible)", "dim"))
    if result.startswith("already granted:"):
        return _pretty_segments(("(already granted)", "dim"))
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _pretty_segments(("(done)", None))


def _temp_dir_prefix(_args_json: str) -> str:
    return _pretty_prefix("* Temp Dir ")


def _temp_dir_detail(_args_json: str, result: str) -> str:
    """``new_temporary_workspace`` outcome: the created path, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    path = _status_fields(result).get("temporary_workspace", "")
    return _pretty_segments((f"({path})" if path else "(done)", "dim"))


def _resume_prefix(_args_json: str) -> str:
    return _pretty_prefix("* Resume Truncated ")


def _resume_detail(_args_json: str, result: str) -> str:
    """``get_truncated`` outcome: the resumed range, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _range_detail(result) or _pretty_segments(("(done)", "green"))


def _vcs_kind_detail(_args_json: str, result: str) -> str:
    """``vcs_kind`` outcome: the detected kind, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _pretty_segments((f"({result})", None))


def _vcs_branch_detail(_args_json: str, result: str) -> str:
    """``vcs_branch`` outcome: the current branch or head, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _pretty_segments((f"({result})", None))


def _vcs_diff_detail(_args_json: str, result: str) -> str:
    """``vcs_diff`` outcome: changed files and line counts, none, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if result == "(no diff)":
        return _pretty_segments(("(no diff)", "dim"))
    files = 0
    added = 0
    removed = 0
    for line in result.splitlines():
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    file_unit = "file" if files == 1 else "files"
    return _pretty_segments((f"({files} {file_unit}, +{added}/-{removed})", None))


def _mcp_detail(_args_json: str, result: str) -> str:
    """MCP tool outcome: done, or the error the server returned."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _pretty_segments(("(done)", "green"))


def _skill_list_prefix(_args_json: str) -> str:
    return _pretty_prefix("* Skills ")


def _skill_list_detail(_args_json: str, result: str) -> str:
    """``skill_list`` outcome: how many skills were found, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    match = re.match(r"skills: (\d+)", result)
    if match is None:
        return _pretty_segments(("(none)", "dim"))
    count = int(match.group(1))
    unit = "skill" if count == 1 else "skills"
    return _pretty_segments((f"({count} {unit})", None if count else "dim"))


def _skill_read_prefix(args_json: str) -> str:
    name = _json_str_arg(args_json, "name") or "?"
    file = _json_str_arg(args_json, "file") or ""
    subject = f"'{name}'" if file in {"", _SKILL_MAIN_FILE} else f"'{name}' {file}"
    return _pretty_prefix(f"* Skill {subject} ")


def _skill_read_detail(_args_json: str, result: str) -> str:
    """``skill_read`` outcome: the range read, or the failure kind."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    return _range_detail(result) or _pretty_segments(("(done)", "green"))


def _skill_search_prefix(args_json: str) -> str:
    pattern = _json_str_arg(args_json, "pattern") or "?"
    scope = _json_str_arg(args_json, "skill") or ""
    where = f" in '{scope}'" if scope else ""
    return _pretty_prefix(f"* Skill Search '{_preview(pattern, _ASK_SUBJECT_PREVIEW)}'{where} ")


def _skill_search_detail(_args_json: str, result: str) -> str:
    """``skill_search`` outcome: matches and skills, none, or an error."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    if result == "(no matches)":
        return _pretty_segments(("(no matches)", "dim"))
    if result == "(no skills to search)":
        return _pretty_segments(("(no skills)", "dim"))
    matches = [line for line in result.splitlines() if line and not line.startswith("...[truncated")]
    skills = {line.split("/", 1)[0] for line in matches}
    match_unit = "match" if len(matches) == 1 else "matches"
    skill_unit = "skill" if len(skills) == 1 else "skills"
    return _pretty_segments((f"({len(matches)} {match_unit} in {len(skills)} {skill_unit})", None))


def _skill_create_prefix(args_json: str) -> str:
    name = _json_str_arg(args_json, "name") or "?"
    return _pretty_prefix(f"* Skill Create '{name}' ")


def _skill_create_detail(_args_json: str, result: str) -> str:
    """``skill_create`` outcome: created (with any notes), or the failure."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    detail = _pretty_segments(("(created)", "green"))
    notes = result.partition("\nnotes: ")[2]
    if notes:
        detail += _pretty_segments((f" — {notes}", "yellow"))
    return detail


def _skill_edit_prefix(args_json: str) -> str:
    name = _json_str_arg(args_json, "name") or "?"
    file = _json_str_arg(args_json, "file") or ""
    subject = f"'{name}'" if file in {"", _SKILL_MAIN_FILE} else f"'{name}' {file}"
    return _pretty_prefix(f"* Skill Edit {subject} ")


def _skill_edit_detail(_args_json: str, result: str) -> str:
    """``skill_edit`` outcome: the change made, or the failure."""
    if result.startswith("error:"):
        return _pretty_segments((f"({result})", "red"))
    first = result.splitlines()[0] if result else ""
    match = re.search(r"\((.+)\)$", first)
    text = match.group(1) if match is not None else "updated"
    detail = _pretty_segments((f"({text})", "green"))
    notes = result.partition("\nnotes: ")[2]
    if notes:
        detail += _pretty_segments((f" — {notes}", "yellow"))
    return detail


# Pretty tools: ``prefix`` prints as the call starts, ``detail`` when it lands.
_PRETTY_LINES: dict[str, PrettyLine] = {
    "read_file": PrettyLine(prefix=_read_file_prefix, detail=_read_file_detail),
    "tree": PrettyLine(prefix=_tree_prefix, detail=_tree_detail),
    "todo_list": _todo_line("todo_list"),
    "todo_push": _todo_line("todo_push"),
    "todo_pop": _todo_line("todo_pop"),
    "todo_update": _todo_line("todo_update"),
    "todo_clear": _todo_line("todo_clear"),
    "listdir": PrettyLine(prefix=_listdir_prefix, detail=_listdir_detail),
    "glob_paths": PrettyLine(prefix=_glob_prefix, detail=_glob_detail),
    "regex_files": PrettyLine(prefix=_regex_prefix, detail=_regex_detail),
    "get_truncated": PrettyLine(prefix=_resume_prefix, detail=_resume_detail),
    "run_argv": PrettyLine(prefix=_run_argv_prefix, detail=_run_argv_detail),
    "run_argv_batch": PrettyLine(prefix=_run_argv_batch_prefix, detail=_run_argv_batch_detail),
    "fetch": PrettyLine(prefix=_fetch_prefix, detail=_fetch_detail),
    "edit_replace": _mutator_line("edit_replace"),
    "edit_lineno": _mutator_line("edit_lineno"),
    "write_file": _mutator_line("write_file"),
    "copy_path": _mutator_line("copy_path"),
    "move_path": _mutator_line("move_path"),
    "delete_path": _mutator_line("delete_path"),
    "new_temporary_workspace": PrettyLine(prefix=_temp_dir_prefix, detail=_temp_dir_detail),
    "request_directory_access": PrettyLine(prefix=_access_prefix, detail=_access_detail),
    "skill_list": PrettyLine(prefix=_skill_list_prefix, detail=_skill_list_detail),
    "skill_read": PrettyLine(prefix=_skill_read_prefix, detail=_skill_read_detail),
    "skill_search": PrettyLine(prefix=_skill_search_prefix, detail=_skill_search_detail),
    "skill_create": PrettyLine(prefix=_skill_create_prefix, detail=_skill_create_detail),
    "skill_edit": PrettyLine(prefix=_skill_edit_prefix, detail=_skill_edit_detail),
    "open_pty": PrettyLine(prefix=_open_pty_prefix, detail=_open_pty_detail),
    "read_pty": _pty_session_line("Read", _read_pty_detail),
    "write_pty": _pty_session_line("Write", _pty_write_detail),
    "write_pty_keys": _pty_session_line("Keys", _pty_write_detail),
    "ask_into_pty": _pty_session_line("Ask", _pty_write_detail),
    "close_pty": _pty_session_line("Close", _close_pty_detail),
    "ask_user_line": _ask_line("", "question", _ask_detail),
    "ask_user_choice": _ask_line("Choose", "question", _ask_choice_detail),
    "ask_user_form": _ask_line("Form", "title", _ask_form_detail),
    "vcs_kind": PrettyLine(prefix=_vcs_kind_prefix, detail=_vcs_kind_detail),
    "vcs_status": PrettyLine(prefix=_vcs_status_prefix, detail=_vcs_status_detail),
    "vcs_diff": PrettyLine(prefix=_vcs_diff_prefix, detail=_vcs_diff_detail),
    "vcs_log": PrettyLine(prefix=_vcs_log_prefix, detail=_vcs_log_detail),
    "vcs_branch": PrettyLine(prefix=_vcs_branch_prefix, detail=_vcs_branch_detail),
    "wait": PrettyLine(prefix=_wait_prefix, detail=_wait_detail),
}


def _mcp_line(name: str) -> PrettyLine | None:
    """Generic renderer for namespaced MCP tools (``mcp__<server>__<tool>``)."""
    if not name.startswith(_MCP_TOOL_PREFIX):
        return None
    server, _, tool = name.removeprefix(_MCP_TOOL_PREFIX).partition("__")
    label = f"{server or '?'}:{tool or '?'}"
    return PrettyLine(
        prefix=lambda _args_json: _pretty_prefix(f"* MCP {label} "),
        detail=_mcp_detail,
    )


def _pretty_line_for(name: str) -> PrettyLine | None:
    """Resolve the renderer for *name*: a named entry, else the MCP naming rule."""
    return _PRETTY_LINES.get(name) or _mcp_line(name)


def _pretty_parts(line: PrettyLine, args_json: str, result: str) -> tuple[str, str]:
    """Prefix + detail for a resolved pretty tool call."""
    return line.prefix(args_json), line.detail(args_json, result)


def _echo_stream(text: str) -> None:
    """Write without newline and flush so assistant text appears token-by-token."""
    click.echo(text, nl=False)
    with contextlib.suppress(OSError):
        _ = sys.stdout.flush()


def _clear_streamed_lines(line_count: int, *, resume_above: bool = False) -> None:
    """Clear exactly *line_count* terminal lines ending at the cursor line.

    The cursor ends at column 0 of the topmost cleared line, so a re-render
    overwrites the erased block in place instead of appending below it. With
    ``resume_above`` the cursor steps one line further up: the next
    ``"\\n"``-prefixed write then reuses the last cleared line — needed by the
    parallel-batch fallback, which must not touch the line above the prefix.

    ``line_count`` counts *physical* rows; callers fold wrapped lines via
    :func:`_line_count_for_clear`.
    """
    if line_count <= 0:
        return
    # Clear the cursor line, then each line above it while moving up.
    for _ in range(line_count - 1):
        _ = sys.stdout.write("\r\033[2K\033[1A")
    _ = sys.stdout.write("\r\033[2K")
    if resume_above:
        _ = sys.stdout.write("\033[1A")
    with contextlib.suppress(OSError):
        _ = sys.stdout.flush()


def _terminal_columns() -> int:
    """Terminal width for wrap-aware row counts (fallback: 80)."""
    try:
        columns = shutil.get_terminal_size().columns
    except OSError, ValueError:
        return 80
    return columns if columns > 0 else 80


def _physical_line_count(text: str) -> int:
    """Rows streamed *text* occupies on the terminal, counting wrapping.

    ``_echo_stream`` writes without a trailing newline, so the cursor sits at
    the end of the last row and owns one row even when the text ends in ``"\\n"``.
    A row filled to the exact width is counted once: the wrap of the final
    column is pending, so erasing must not claim the row below it.
    """
    columns = _terminal_columns()
    return sum(max(1, -(-len(line) // columns)) for line in text.split("\n"))


def _line_count_for_clear(label: str, body: str) -> int:
    """Physical terminal rows used by ``label\\n + body`` for cursor erase."""
    if not body and not label:
        return 0
    # Label is on its own line; body may contain newlines and wrap.
    text = f"{label}\n{body}" if label else body
    return _physical_line_count(text)


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
            # Erase only the prefix line — the line above may be streamed
            # reasoning — and let the fallback's leading newline reuse it.
            _clear_streamed_lines(1, resume_above=True)
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
        # +1: the blank separator ``begin_assistant`` printed above the label.
        lines = _line_count_for_clear("assistant:", body) + 1
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
    pending_tools: list[tuple[str, str, PrettyLine | None]] = []

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
                line = _pretty_line_for(name)
                pending_tools.append((name, args, line))
                if line is not None:
                    tool_lines.start_call(line.prefix(args))
                else:
                    tool_lines.close_line()
                    preview = _preview(args, _TOOL_ARGS_PREVIEW)
                    click.secho(f"\n[tool] {name}({preview})", fg="yellow")
            else:
                tool_lines.close_line()
                pending_tools.append(("custom", call.id, None))
                click.secho(f"\n[tool] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            flush_assistant()
            content = event.message.content
            name, args, line = pending_tools.pop(0) if pending_tools else ("", "", None)
            parts = _pretty_parts(line, args, content) if line is not None else None
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
