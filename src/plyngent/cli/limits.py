from __future__ import annotations

import contextlib
import selectors
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from plyngent.agent import BatchConfirmDecision, ToolConfirmRequest
from plyngent.cli.display import close_open_pretty_line
from plyngent.cli.interrupt import off_loop_prompt, pause_task_cancel_for_prompt
from plyngent.prompting import (
    ChoiceOption,
    NonInteractiveError,
    PromptAnswerSource,
    ask,
    ask_async,
    choose,
    choose_async,
    configure_prompting,
    confirm,
    get_prompt_backend,
    prompt_wait,
    set_prompt_output_hook,
)
from plyngent.tools.access import AccessDecision
from plyngent.tools.process.pty_session import PtyManager
from plyngent.tools.skills import SkillWriteDecision
from plyngent.tools.workspace import AccessMode, parse_access_mode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type WorkspaceMismatchChoice = Literal["keep", "rebind", "abort"]

# Answers a notification offers (see `cli/notify.py`) mapped onto the keys the
# prompts below already accept, so one answer source fits every confirm.
_NOTIFY_ANSWERS = {"allow": "y", "deny": "n"}

_BOX_MIN_WIDTH = 40
_BOX_MAX_WIDTH = 100
_BOX_PAD = 2  # spaces inside left/right borders

# How long one poll slice waits before a pending notification answer is checked.
_ANSWER_POLL_SECONDS = 0.1


def _answer_source(wait: PromptAnswerSource | None) -> PromptAnswerSource | None:
    """The wait's answer source, but only when the prompt offered answers."""
    return wait if wait is not None and wait.supports_actions else None


def _read_answer(
    timeout_seconds: float | None,
    wait: PromptAnswerSource | None,
) -> str | None:
    """One answer line, from stdin or the notification (``None`` on timeout)."""
    raw = _read_yes_no_line_with_timeout(timeout_seconds, source=_answer_source(wait))
    if raw is None:
        return None
    return _NOTIFY_ANSWERS.get(raw.strip().lower(), raw)


# Limit prompts accept ``yyy`` to stop asking for the rest of the turn.
_CONTINUE_PROMPT = "Raise limit and continue? [Y/n/yyy] (yyy = stop asking this turn): "
_CONTINUE_YES_TOKENS = frozenset({"", "y", "yes"})
_CONTINUE_AUTO_TOKEN = "yyy"

# The answers each prompt accepts, as notification buttons: the key is the
# token the terminal read takes, the label is what a human reads there.
_LIMIT_CHOICES: tuple[tuple[str, str], ...] = (
    ("y", "Continue"),
    (_CONTINUE_AUTO_TOKEN, "Continue, stop asking"),
    ("n", "Stop here"),
)

# Batch confirm answers: ``y`` allows the round's calls, ``m`` reviews them one
# at a time through the per-call prompt.
_BATCH_YES_TOKENS = frozenset({"y", "yes"})
_BATCH_MANUAL_TOKENS = frozenset({"m", "manual", "each"})

_BATCH_CHOICES: tuple[tuple[str, str], ...] = (
    ("y", "Allow all"),
    ("m", "Review one by one"),
    ("n", "Deny"),
)

# Skill writes: ``o`` allows this change, ``s`` the whole directory for the rest
# of the chat.
_SKILL_CHOICES: tuple[tuple[str, str], ...] = (
    ("o", "Allow once"),
    ("s", "This chat"),
    ("n", "Deny"),
)


@dataclass
class _AutoContinue:
    """Auto-continue layers: process default (flag/config) + ``yyy`` turn skip."""

    default: bool = False
    turn: bool = False


_auto_continue = _AutoContinue()


def auto_continue_enabled() -> bool:
    """Whether limit prompts are auto-approved (process default or ``yyy``)."""
    return _auto_continue.default or _auto_continue.turn


def set_auto_continue_default(*, enabled: bool) -> None:
    """Set the process-wide default (``--auto-continue`` / ``[agent]`` config)."""
    _auto_continue.default = bool(enabled)


def reset_auto_continue_turn() -> None:
    """Clear the ``yyy`` skip at a turn boundary."""
    _auto_continue.turn = False


def reset_auto_continue() -> None:
    """Clear both layers (chat exit / tests)."""
    _auto_continue.default = False
    _auto_continue.turn = False


def _terminal_width() -> int:
    try:
        return max(_BOX_MIN_WIDTH, min(_BOX_MAX_WIDTH, shutil.get_terminal_size(fallback=(80, 24)).columns))
    except OSError:
        return 80


def _wrap_line(text: str, width: int) -> list[str]:
    """Hard-wrap a single logical line to *width* (no word-break library)."""
    width = max(width, 8)
    if not text:
        return [""]
    if len(text) <= width:
        return [text]
    out: list[str] = []
    rest = text
    while rest:
        if len(rest) <= width:
            out.append(rest)
            break
        # Prefer break at last space in the window.
        window = rest[:width]
        break_at = window.rfind(" ")
        if break_at >= width // 2:
            out.append(rest[:break_at])
            rest = rest[break_at + 1 :]
        else:
            out.append(rest[:width])
            rest = rest[width:]
    return out


def _box(header: str, lines: Sequence[str]) -> str:
    """Rounded box: a header row, a rule, then the wrapped body lines."""
    term = _terminal_width()
    inner = max(20, term - 4)  # room for ``│ `` + `` │``
    body_lines: list[str] = []
    for raw in lines:
        body_lines.extend(_wrap_line(raw.replace("\r\n", "\n").replace("\r", "\n"), inner - _BOX_PAD))
    content = [header, "─" * min(inner, max(len(header), 12)), *body_lines]
    width = min(inner, max(len(line) for line in content) + _BOX_PAD)
    width = max(width, min(inner, len(header) + _BOX_PAD))
    top = "┌" + "─" * (width + 2) + "┐"
    bottom = "└" + "─" * (width + 2) + "┘"
    rows = [top]
    for line in content:
        # Second line is a separator drawn with box dashes already in content.
        if line.startswith("─") and set(line) <= {"─"}:
            rows.append("├" + "─" * (width + 2) + "┤")
            continue
        padded = line[:width].ljust(width)
        rows.append(f"│ {padded} │")
    rows.append(bottom)
    return "\n".join(rows)


def format_tool_confirm_box(name: str, reason: str) -> str:
    """Multi-line boxed confirm body for one tool call.

    Printed with :meth:`PromptBackend.echo` before a short ``confirm()`` prompt
    so terminals keep newlines (unlike a single jammed readline line).
    """
    return _box(f"confirm · tool {name!r}", reason.split("\n"))


def _echo_tool_confirm(name: str, reason: str) -> None:
    backend = get_prompt_backend()
    backend.echo()
    backend.secho(format_tool_confirm_box(name, reason), fg="yellow")
    backend.echo()


def _echo_continue_limit(reason: str) -> None:
    """Echo the limit reason as a standalone line (not jammed into a confirm prompt)."""
    backend = get_prompt_backend()
    backend.secho(f"[limit] {reason}", fg="yellow")
    backend.echo()


def _echo_auto_continue(reason: str) -> None:
    """Echo the limit reason plus the ``yyy`` notice (no prompt)."""
    backend = get_prompt_backend()
    backend.secho(f"[limit] {reason}", fg="yellow")
    backend.secho("[limit] auto-continuing (yyy) — no more limit prompts this turn", fg="bright_black")
    backend.echo()


def _decide_continue_limit(reason: str) -> bool:
    """Shared limit decision for the sync/async prompts (may read a line)."""
    if auto_continue_enabled():
        _echo_auto_continue(reason)
        return True
    backend = get_prompt_backend()
    if not backend.is_interactive():
        return False
    with prompt_wait("limit", "limit reached — continue?", reason, choices=_LIMIT_CHOICES) as wait:
        _echo_continue_limit(reason)
        source = _answer_source(wait)
        if source is None:
            token = backend.read_line(_CONTINUE_PROMPT, default="y").strip().lower()
        else:
            # Same read as the timed confirms: stdin or the notification,
            # whichever answers first, with the terminal's default on silence.
            token = (_read_answer(None, wait) or "").strip().lower()
            if not token:
                token = "y"
    if token == _CONTINUE_AUTO_TOKEN:
        _auto_continue.turn = True
        return True
    return token in _CONTINUE_YES_TOKENS


def _prompt_continue_limit_sync(reason: str) -> bool:
    try:
        return _decide_continue_limit(reason)
    except NonInteractiveError:
        return False


def prompt_continue_limit(reason: str) -> bool:
    """Ask the user whether to raise a limit and continue (TTY, sync)."""
    with pause_task_cancel_for_prompt():
        return _prompt_continue_limit_sync(reason)


async def prompt_continue_limit_async(reason: str) -> bool:
    """Async variant: prompt off the event loop so the turn is not cancelled."""
    try:
        from plyngent.prompting import run_prompt_async

        return await run_prompt_async(lambda: _decide_continue_limit(reason))
    except NonInteractiveError:
        return False


def _confirm_with(wait: PromptAnswerSource | None, prompt: str, *, default: bool) -> bool:
    """Yes/no, also taking an answer that arrived through a notification."""
    return confirm(prompt, default=default, source=_answer_source(wait))


def _tool_facts(args: Mapping[str, object]) -> tuple[str, ...]:
    """What a call is about, short enough to travel in a notification.

    Only the *object* — the program or the file — never an argument, a URL or a
    piece of code: those stay in the reason, which a notification carries only
    when ``include_reason`` allows it.
    """
    facts: list[str] = []
    argv = args.get("argv")
    if isinstance(argv, list) and argv and isinstance(argv[0], str):
        facts.append(f"program: {Path(argv[0]).name}")
    for key in ("path", "dst", "src"):
        value = args.get(key)
        if isinstance(value, str) and value:
            facts.append(f"target: {Path(value).name}")
            break
    return tuple(facts)


def _prompt_confirm_tool_sync(name: str, args: Mapping[str, object], reason: str) -> bool | str:
    """True allow; False deny; non-empty str = deny with comment for the model."""
    try:
        with prompt_wait(
            "confirm",
            f"tool {name!r} needs confirmation",
            reason,
            facts=_tool_facts(args),
        ) as wait:
            _echo_tool_confirm(name, reason)
            allowed = _confirm_with(wait, "Allow this tool call?", default=False)
    except NonInteractiveError:
        return False
    if allowed:
        return True
    try:
        comment = ask(
            "Optional comment for the agent (why denied; empty to skip):",
            default="",
        ).strip()
    except NonInteractiveError:
        return False
    return comment or False


def prompt_confirm_tool(name: str, args: Mapping[str, object], reason: str) -> bool | str:
    """Ask whether to allow a dangerous tool call (TTY). Default is deny.

    Prints a multi-line boxed summary, then a short y/N prompt. On deny,
    optionally collect a free-text comment for the model.
    """
    with pause_task_cancel_for_prompt():
        return _prompt_confirm_tool_sync(name, args, reason)


def _try_readline() -> str | None:
    try:
        return sys.stdin.readline()
    except OSError, EOFError:
        return None


def _stdin_selector() -> selectors.BaseSelector | None:
    """A selector watching process stdin, or ``None`` when it cannot be polled."""
    try:
        fd = sys.stdin.fileno()
        selector = selectors.DefaultSelector()
        _ = selector.register(fd, selectors.EVENT_READ)
    except OSError, ValueError, AttributeError:
        return None
    return selector


def _read_yes_no_line_with_timeout(
    timeout_seconds: float | None,
    *,
    source: PromptAnswerSource | None = None,
) -> str | None:
    """Read one line from stdin with *timeout_seconds*; ``None`` on timeout/EOF.

    Polls the process stdin FD in short slices, so a *source* (a notification
    answer) can end the read too; stdin wins whenever it becomes readable. A
    ``None`` timeout waits for a line or an answer, whichever comes first; with
    neither stdin nor a *source* pollable the read gives up at once.
    On Windows without selector support, falls back to blocking readline (no
    true timeout, and no answers from a *source*).
    """
    try:
        if not sys.stdin.isatty():
            return None if source is None else source.take()
    except OSError, ValueError, AttributeError:
        return _try_readline()

    selector = _stdin_selector()
    if selector is None:
        return _try_readline()

    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    result: str | None = None
    try:
        while True:
            if deadline is None:
                remaining = _ANSWER_POLL_SECONDS
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
            if selector.select(min(remaining, _ANSWER_POLL_SECONDS)):
                try:
                    result = sys.stdin.readline()
                except OSError, EOFError:
                    result = None
                break
            answer = None if source is None else source.take()
            if answer is not None:
                result = answer
                break
    finally:
        selector.close()
    return result


def prompt_policy_command_confirm(
    basename: str,
    argv: Sequence[str],
    timeout_seconds: float,
) -> bool:
    """Ask whether to allow a denylisted command for this session (timed).

    Independent of YOLO: always prompts when interactive. Default is **deny**.
    Timeout, cancel, or non-interactive → False.
    """
    argv_preview = " ".join(str(part) for part in argv)
    reason = (
        f"policy denylist blocks basename {basename!r}\n"
        f"argv: {argv_preview}\n"
        f"Allow this basename for the rest of this process?\n"
        f"(timeout {timeout_seconds:g}s defaults to DENY; not skipped by YOLO)"
    )
    try:
        with (
            pause_task_cancel_for_prompt(),
            prompt_wait("policy", f"allow command {basename!r}?", reason, expires_in=timeout_seconds) as wait,
        ):
            backend = get_prompt_backend()
            if not backend.is_interactive():
                return False
            backend.echo()
            backend.secho(format_tool_confirm_box("policy", reason), fg="yellow")
            backend.echo()
            prompt = f"[policy] allow {basename!r}? [y/N] (timeout {timeout_seconds:g}s): "
            with contextlib.suppress(OSError):
                _ = sys.stderr.write(prompt)
                _ = sys.stderr.flush()
            raw = _read_answer(timeout_seconds, wait)
            if raw is None:
                with contextlib.suppress(OSError, NonInteractiveError):
                    backend.secho(
                        f"[policy] timed out after {timeout_seconds:g}s — denied",
                        fg="red",
                        err=True,
                    )
                return False
            return raw.strip().lower() in {"y", "yes"}
    except NonInteractiveError, KeyboardInterrupt, EOFError:
        return False


def prompt_policy_fetch_confirm(
    host: str,
    port: int,
    url: str,
    timeout_seconds: float,
) -> bool:
    """Ask whether to allow fetch_local to a private/loopback host (timed).

    Independent of YOLO: always prompts when interactive. Default is **deny**.
    Timeout, cancel, or non-interactive → False. On approve, the host:port is
    granted for the rest of this process (see net grants).
    """
    reason = (
        f"fetch_local targets private/loopback host {host!r} port {port}\n"
        f"url: {url}\n"
        f"Allow this host:port for the rest of this process?\n"
        f"(timeout {timeout_seconds:g}s defaults to DENY; not skipped by YOLO)"
    )
    try:
        with (
            pause_task_cancel_for_prompt(),
            prompt_wait("policy", f"allow fetch_local {host!r}?", reason, expires_in=timeout_seconds) as wait,
        ):
            backend = get_prompt_backend()
            if not backend.is_interactive():
                return False
            backend.echo()
            backend.secho(format_tool_confirm_box("policy", reason), fg="yellow")
            backend.echo()
            grant = f"{host}:{port}"
            prompt = f"[policy] allow fetch_local {grant!r}? [y/N] (timeout {timeout_seconds:g}s): "
            with contextlib.suppress(OSError):
                _ = sys.stderr.write(prompt)
                _ = sys.stderr.flush()
            raw = _read_answer(timeout_seconds, wait)
            if raw is None:
                with contextlib.suppress(OSError, NonInteractiveError):
                    backend.secho(
                        f"[policy] timed out after {timeout_seconds:g}s — denied",
                        fg="red",
                        err=True,
                    )
                return False
            return raw.strip().lower() in {"y", "yes"}
    except NonInteractiveError, KeyboardInterrupt, EOFError:
        return False


def prompt_skill_write_confirm(
    skill_dir: Path,
    action: str,
    detail: str,
    timeout_seconds: float,
) -> SkillWriteDecision:
    """Ask whether to write into *skill_dir* (timed; default deny).

    ``o``/``y`` allows this write; ``s`` allows this skill directory for the rest
    of the chat; anything else (or a timeout, cancel, or non-interactive stdin)
    denies. Independent of YOLO: a skill directory is not a path grant, so the
    human is asked even with ``--yes`` unless ``[skills].allow_write`` is set.
    """
    lines = [
        f"action: {action} skill {skill_dir.name!r}",
        f"path: {skill_dir}",
        f"change: {detail}",
        f"(timeout {timeout_seconds:g}s defaults to DENY; not skipped by YOLO)",
    ]
    try:
        with (
            pause_task_cancel_for_prompt(),
            prompt_wait(
                "skills",
                f"allow {action} in {skill_dir.name!r}?",
                detail,
                choices=_SKILL_CHOICES,
                facts=(f"path: {skill_dir}",),
                expires_in=timeout_seconds,
            ) as wait,
        ):
            backend = get_prompt_backend()
            if not backend.is_interactive():
                return SkillWriteDecision(allow=False)
            backend.echo()
            backend.secho(format_tool_confirm_box(f"skills {action}", "\n".join(lines)), fg="yellow")
            backend.echo()
            prompt = (
                f"[skills] allow {action} in {skill_dir.name!r}? [o]nce/[s]ession/[n]o (timeout {timeout_seconds:g}s): "
            )
            with contextlib.suppress(OSError):
                _ = sys.stderr.write(prompt)
                _ = sys.stderr.flush()
            raw = _read_answer(timeout_seconds, wait)
            if raw is None:
                with contextlib.suppress(OSError, NonInteractiveError):
                    backend.secho(f"[skills] timed out after {timeout_seconds:g}s — denied", fg="red", err=True)
                return SkillWriteDecision(allow=False)
            token = raw.strip().lower()
            if token in {"s", "session"}:
                return SkillWriteDecision(allow=True, session=True)
            if token in {"o", "once", "y", "yes"}:
                return SkillWriteDecision(allow=True)
            return SkillWriteDecision(allow=False)
    except NonInteractiveError, KeyboardInterrupt, EOFError:
        return SkillWriteDecision(allow=False)


def prompt_directory_access_confirm(
    path: Path,
    mode: AccessMode,
    reason: str,
    timeout_seconds: float,
) -> AccessDecision | str | None:
    """Ask whether to grant access to *path* at *mode* (timed; default deny).

    ``y`` approves the requested level; ``read`` / ``write`` / ``exec`` approve
    that level instead. The host's hook decides YOLO/``--yes`` before calling
    this. Timeout, cancel, or non-interactive → ``None`` (deny).
    """
    kind = "directory" if path.is_dir() else "file"
    lines = [
        f"request: {mode.name.lower()} access to {path}",
        f"kind: {kind}",
    ]
    if reason:
        lines.append(f"why: {reason}")
    lines.append(f"(timeout {timeout_seconds:g}s defaults to DENY)")
    try:
        with (
            pause_task_cancel_for_prompt(),
            prompt_wait(
                "access",
                f"allow {mode.name.lower()} access to {path}?",
                reason,
                facts=(f"kind: {kind}",),
                expires_in=timeout_seconds,
            ) as wait,
        ):
            backend = get_prompt_backend()
            if not backend.is_interactive():
                return None
            backend.echo()
            backend.secho(
                format_tool_confirm_box("request_directory_access", "\n".join(lines)),
                fg="yellow",
            )
            backend.echo()
            prompt = (
                f"[access] allow {path} ({mode.name.lower()})? [y/N|read|write|exec] (timeout {timeout_seconds:g}s): "
            )
            with contextlib.suppress(OSError):
                _ = sys.stderr.write(prompt)
                _ = sys.stderr.flush()
            raw = _read_answer(timeout_seconds, wait)
            if raw is None:
                with contextlib.suppress(OSError, NonInteractiveError):
                    backend.secho(
                        f"[access] timed out after {timeout_seconds:g}s — denied",
                        fg="red",
                        err=True,
                    )
                return None
            token = raw.strip().lower()
            if token in {"y", "yes"}:
                return AccessDecision(mode)
            level = parse_access_mode(token)
            if level is not None:
                return AccessDecision(level)
            return None
    except NonInteractiveError, KeyboardInterrupt, EOFError:
        return None


async def prompt_confirm_tool_async(name: str, args: Mapping[str, object], reason: str) -> bool | str:
    """Async confirm: True allow, False deny, str = deny with user comment."""
    facts = _tool_facts(args)
    try:
        # Box is printed inside the worker thread via confirm's backend.
        def _run() -> bool:
            with prompt_wait("confirm", f"tool {name!r} needs confirmation", reason, facts=facts) as wait:
                _echo_tool_confirm(name, reason)
                return _confirm_with(wait, "Allow this tool call?", default=False)

        from plyngent.prompting import run_prompt_async

        allowed = await run_prompt_async(_run)
    except NonInteractiveError:
        return False
    if allowed:
        return True
    try:
        comment = (
            await ask_async(
                "Optional comment for the agent (why denied; empty to skip):",
                default="",
            )
        ).strip()
    except NonInteractiveError:
        return False
    return comment or False


async def prompt_confirm_batch_async(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
    """One confirm for one round's soft-confirm requests (``y`` / ``N`` / ``m``).

    The answer covers exactly those calls — the next round asks again; ``m``
    asks them one at a time instead, and a deny carries one comment shared by
    every one of them.
    """
    try:
        from plyngent.prompting import run_prompt_async

        return await run_prompt_async(lambda: _prompt_confirm_batch_sync(requests))
    except NonInteractiveError:
        return BatchConfirmDecision(allowed=False)


def _prompt_confirm_batch_sync(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
    backend = get_prompt_backend()
    if not backend.is_interactive():
        # Non-interactive: deny, like the default of the single prompt.
        return BatchConfirmDecision(allowed=False)
    tools = tuple(dict.fromkeys(request.name for request in requests))
    with prompt_wait(
        "batch",
        f"{len(requests)} tool calls need confirmation",
        choices=_BATCH_CHOICES,
        facts=tools,
    ) as wait:
        backend.echo()
        backend.secho(format_batch_confirm_box(requests), fg="yellow")
        backend.secho("m = review them one call at a time", fg="bright_black")
        backend.echo()
        source = _answer_source(wait)
        try:
            if source is None:
                token = (
                    backend.read_line(
                        f"Allow these {len(requests)} tool calls? [y/N/m]:",
                        default="n",
                    )
                    .strip()
                    .lower()
                )
            else:
                with contextlib.suppress(OSError):
                    _ = sys.stderr.write(f"Allow these {len(requests)} tool calls? [y/N]: ")
                    _ = sys.stderr.flush()
                token = (_read_answer(None, wait) or "n").strip().lower()
        except NonInteractiveError:
            return BatchConfirmDecision(allowed=False)
    if token in _BATCH_MANUAL_TOKENS:
        return BatchConfirmDecision(allowed=False, each=True)
    if token in _BATCH_YES_TOKENS:
        return BatchConfirmDecision(allowed=True)
    try:
        comment = ask(
            "Optional comment for the agent (why denied; empty to skip):",
            default="",
        ).strip()
    except NonInteractiveError:
        comment = ""
    return BatchConfirmDecision(allowed=False, comment=comment or None)


def format_batch_confirm_box(requests: Sequence[ToolConfirmRequest]) -> str:
    """Boxed body listing every call a round wants to make, one numbered block each."""
    lines: list[str] = []
    for index, request in enumerate(requests, start=1):
        reason_lines = request.reason.split("\n")
        lines.append(f"{index}. [{request.name}] {reason_lines[0]}")
        lines.extend(f"   {line}" for line in reason_lines[1:])
    return _box(f"confirm · {len(requests)} tool calls", lines)


def _prompt_workspace_mismatch_sync(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    selected = choose(
        f"[workspace] session {session_id} is bound to a different directory:\n"
        f"  session: {session_workspace}\n"
        f"  current: {current_workspace}",
        [
            ChoiceOption(
                label="keep",
                description="keep session workspace (switch tools root to session path)",
                value="keep",
            ),
            ChoiceOption(
                label="update",
                description="update binding to current workspace",
                value="rebind",
            ),
            ChoiceOption(
                label="abort",
                description="abort resume",
                value="abort",
            ),
        ],
        default="keep",
        allow_custom=False,
    )
    if selected == "rebind":
        return "rebind"
    if selected == "abort":
        return "abort"
    return "keep"


def prompt_workspace_mismatch(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    """Ask how to handle resuming a session bound to a different directory."""
    try:
        with pause_task_cancel_for_prompt():
            return _prompt_workspace_mismatch_sync(session_id, session_workspace, current_workspace)
    except NonInteractiveError:
        return "abort"


async def prompt_workspace_mismatch_async(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    """Async variant of workspace mismatch prompt."""
    try:
        selected = await choose_async(
            f"[workspace] session {session_id} is bound to a different directory:\n"
            f"  session: {session_workspace}\n"
            f"  current: {current_workspace}",
            [
                ChoiceOption(
                    label="keep",
                    description="keep session workspace (switch tools root to session path)",
                    value="keep",
                ),
                ChoiceOption(
                    label="update",
                    description="update binding to current workspace",
                    value="rebind",
                ),
                ChoiceOption(
                    label="abort",
                    description="abort resume",
                    value="abort",
                ),
            ],
            default="keep",
            allow_custom=False,
        )
    except NonInteractiveError:
        return "abort"
    if selected == "rebind":
        return "rebind"
    if selected == "abort":
        return "abort"
    return "keep"


def install_cli_limit_hooks() -> None:
    """Register interactive continue hooks and prompt guards.

    Off-loop prompts (worker-thread reads) register their own SIGINT target so a
    Ctrl+C there never cancels the enclosing turn; synchronous prompts read on
    the main thread and take SIGINT as a KeyboardInterrupt instead. Command
    denylist policy confirm is set on
    :class:`~plyngent.tools.context.InstanceState` when the REPL starts (no
    process-global hook).
    """
    configure_prompting(pause_factory=off_loop_prompt)
    set_prompt_output_hook(close_open_pretty_line)
    PtyManager.set_limit_continue_hook(prompt_continue_limit)
