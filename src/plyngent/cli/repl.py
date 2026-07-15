from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import click
from msgspec import UNSET

from plyngent.cli.readline_setup import setup_readline
from plyngent.cli.retry import retry_pending_with_retries, run_user_text_with_retries
from plyngent.cli.selection import select_model, select_provider
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ToolChatMessage,
    UserChatMessage,
)
from plyngent.runtime import ProviderNotSupportedError

if TYPE_CHECKING:
    from plyngent.cli.state import ReplState
    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

HELP_TEXT = """\
Commands:
  /help              Show this help
  /quit, /exit       Leave the REPL
  /clear             Clear in-memory conversation (keeps session id)
  /history [n]       Show last n messages in this session (default 20)
  /sessions          List sessions for this workspace (newest first)
  /new [name]        Start a new session (bound to workspace)
  /resume [id]       Resume session id, or latest for this workspace if omitted
  /rename <name>     Rename the current session
  /delete [id]       Hard-delete a session (confirm; current → new empty)
  /export [md|json] [path]  Export session transcript from DB
  /compact [name]    Soft-compact + model-summarize into a new session
  /provider [name]   Show or switch provider
  /model [id]        Show or switch model
  /tools [on|off]    Show or toggle tools
  /stream [on|off]   Show or toggle streaming model output
  /verbose [on|off]  Show or toggle full tool-result dumps
  /rounds [n]        Show or set max tool-loop rounds
  /retry             Re-run incomplete last user turn (DB/orphan user; no retype)
  /status            Show session/provider/tools/rounds status

User messages are saved immediately. On API errors or Ctrl+C, partial
assistant/tool output is discarded but the user message stays (so /retry
works after resume, not only via readline history). Auto-retry: 10s/20s/30s.

Tab completes slash commands and some arguments (provider, model, tools,
stream, verbose). Use --session ID or /resume to continue a prior chat
after restart.
"""

type SlashHandler = Callable[[], None | Awaitable[None]]

_DEFAULT_HISTORY_LINES = 20
_CONTENT_PREVIEW = 200


def _cmd_status(state: ReplState) -> None:
    from plyngent.agent.budget import estimate_messages_chars

    pending = state.agent.pending_retry_text
    pending_disp = "yes" if pending else "no"
    ctx_chars = estimate_messages_chars(state.agent.messages)
    ctx_tokens = state.agent.context_tokens
    ctx_src = state.agent.context_tokens_source
    ctx_budget = state.agent.max_context_tokens
    session_u = state.agent.session_usage
    last_u = state.agent.last_turn_usage
    last_req = state.agent.last_request_usage
    last_rounds = state.agent.last_turn_rounds
    # API prompt_tokens from the last model call is real context size for that request.
    ctx_tag = "api" if ctx_src == "api" else "est"
    ctx_tilde = "" if ctx_src == "api" else "~"
    click.echo(
        f"provider={state.provider_name}  model={state.model}\n"
        f"session={state.session_id}  messages={len(state.agent.messages)}  "
        f"pending_retry={pending_disp}\n"
        f"tools={'on' if state.tools_enabled else 'off'}  "
        f"rounds={state.max_rounds}  "
        f"stream={'on' if state.agent.stream else 'off'}  "
        f"verbose={'on' if state.verbose else 'off'}\n"
        f"context_tokens={ctx_tilde}{ctx_tokens}/{ctx_budget} ({ctx_tag})  "
        f"context_chars={ctx_chars}  "
        f"tool_result_max={state.agent.max_tool_result_chars}\n"
        f"last_request={last_req.format_line()}\n"
        f"usage_last_turn={last_u.format_line(billed=True)}  "
        f"rounds={last_rounds}\n"
        f"usage_session={session_u.format_line(billed=True)}\n"
        f"workspace={state.workspace}"
    )


async def _cmd_sessions(state: ReplState) -> None:
    sessions = await state.memory.list_sessions(workspace=state.workspace)
    if not sessions:
        click.echo(f"(no sessions for workspace {state.workspace})")
        return
    for session in sessions:
        marker = "*" if session.sid == state.session_id else " "
        ws = session.workspace or "(unbound)"
        click.echo(f"{marker} {session.sid}\t{session.name}\tworkspace={ws}\tupdated={session.updated_at}")


async def _cmd_new(state: ReplState, arg: str) -> None:
    name = arg.strip() or "chat"
    await state.new_session(name=name)
    click.echo(f"new session id={state.session_id} name={name}")


async def _cmd_rename(state: ReplState, arg: str) -> None:
    name = arg.strip()
    if not name:
        click.echo("usage: /rename <name>")
        return
    try:
        row = await state.rename_current_session(name)
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    click.echo(f"renamed session {row.sid} -> {row.name}")


async def _cmd_delete(state: ReplState, arg: str) -> None:
    from plyngent.prompting import NonInteractiveError, confirm_async

    token = arg.strip()
    if token:
        try:
            sid = int(token)
        except ValueError:
            click.echo("usage: /delete [session id]")
            return
    else:
        if state.session_id is None:
            click.echo("error: no active session")
            return
        sid = state.session_id
    try:
        allowed = await confirm_async(
            f"Permanently delete session {sid} and all messages?",
            default=False,
        )
    except NonInteractiveError:
        click.echo("error: delete requires interactive confirm (or TTY)")
        return
    if not allowed:
        click.echo("delete cancelled")
        return
    try:
        was_current = await state.delete_session_and_maybe_replace(sid)
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    if was_current:
        click.echo(f"deleted session {sid}; new session {state.session_id}")
    else:
        click.echo(f"deleted session {sid}")


async def _cmd_export(state: ReplState, arg: str) -> None:
    from plyngent.cli.export import (
        encode_session_export_json,
        format_session_export_md,
        resolve_export_path,
        session_export_payload,
        write_export_file,
    )

    if state.session_id is None:
        click.echo("error: no active session")
        return
    parts = arg.split()
    fmt = "md"
    path_arg: str | None = None
    if parts:
        first = parts[0].lower()
        if first in {"md", "markdown", "json"}:
            fmt = "json" if first == "json" else "md"
            path_arg = parts[1] if len(parts) > 1 else None
        else:
            path_arg = parts[0]
            if len(parts) > 1:
                click.echo("usage: /export [md|json] [path]")
                return
    row = await state.memory.get_session(state.session_id)
    if row is None:
        click.echo(f"error: session not found: {state.session_id}")
        return
    messages = await state.memory.list_messages(state.session_id)
    out_path = resolve_export_path(state.session_id, fmt, path_arg)
    if fmt == "json":
        text = encode_session_export_json(
            session_export_payload(
                sid=row.sid,
                name=row.name,
                workspace=row.workspace,
                created_at=row.created_at,
                updated_at=row.updated_at,
                messages=messages,
            )
        )
    else:
        text = format_session_export_md(
            messages,
            sid=row.sid,
            name=row.name,
            workspace=row.workspace,
        )
    try:
        written = write_export_file(out_path, text)
    except OSError as exc:
        click.echo(f"error: write failed: {exc}")
        return
    click.echo(f"exported session {row.sid} ({fmt}) -> {written}")


async def _cmd_resume(state: ReplState, arg: str) -> None:
    if not arg.strip():
        mode = await state.resume_latest_or_new()
        if mode == "new":
            click.echo(f"no prior session; created new session {state.session_id}")
        else:
            click.echo(
                f"resumed latest session {state.session_id} "
                f"({len(state.agent.messages)} messages) workspace={state.workspace}"
            )
        return
    try:
        session_id = int(arg.strip())
    except ValueError:
        click.echo("session id must be an integer (or omit for latest)")
        return
    try:
        await state.resume_session(session_id)
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    click.echo(f"resumed session {session_id} ({len(state.agent.messages)} messages) workspace={state.workspace}")


async def _cmd_compact(state: ReplState, arg: str) -> None:
    name = arg.strip() or None
    click.secho("compacting (soft-compact + model summary)…", fg="yellow")
    try:
        old_id, new_id, summary = await state.compact_to_new_session(name=name)
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — surface model/API failures
        click.secho(f"error: compact failed: {exc}", fg="red")
        return
    preview = summary if len(summary) <= 400 else summary[:400] + "…"  # noqa: PLR2004
    click.echo(f"compacted session {old_id} -> new session {new_id}")
    click.secho(preview, fg="bright_black")


def _cmd_provider(state: ReplState, arg: str) -> None:
    if not arg.strip():
        click.echo(f"provider={state.provider_name}")
        return
    try:
        name, provider = select_provider(state.config.providers, preferred=arg.strip())
        state.provider_name = name
        state.provider = provider
        state.rebuild_client()
        click.echo(f"switched provider to {name}")
    except (click.ClickException, ProviderNotSupportedError) as exc:
        click.echo(f"error: {exc}")


def _cmd_model(state: ReplState, arg: str) -> None:
    if not arg.strip():
        click.echo(f"model={state.model}")
        return
    try:
        state.model = select_model(state.provider, preferred=arg.strip())
        state.rebuild_client()
        click.echo(f"switched model to {state.model}")
    except click.ClickException as exc:
        click.echo(f"error: {exc}")


def _cmd_tools(state: ReplState, arg: str) -> None:
    token = arg.strip().lower()
    if not token:
        click.echo(f"tools={'on' if state.tools_enabled else 'off'}")
        return
    if token in {"on", "1", "true", "yes"}:
        state.tools_enabled = True
    elif token in {"off", "0", "false", "no"}:
        state.tools_enabled = False
    else:
        click.echo("usage: /tools [on|off]")
        return
    state.rebuild_client()
    click.echo(f"tools={'on' if state.tools_enabled else 'off'}")


def _cmd_stream(state: ReplState, arg: str) -> None:
    token = arg.strip().lower()
    if not token:
        click.echo(f"stream={'on' if state.agent.stream else 'off'}")
        return
    if token in {"on", "1", "true", "yes"}:
        enabled = True
    elif token in {"off", "0", "false", "no"}:
        enabled = False
    else:
        click.echo("usage: /stream [on|off]")
        return
    state.stream_enabled = enabled
    state.agent.stream = enabled
    click.echo(f"stream={'on' if enabled else 'off'}")


def _cmd_verbose(state: ReplState, arg: str) -> None:
    token = arg.strip().lower()
    if not token:
        click.echo(f"verbose={'on' if state.verbose else 'off'}")
        return
    if token in {"on", "1", "true", "yes"}:
        enabled = True
    elif token in {"off", "0", "false", "no"}:
        enabled = False
    else:
        click.echo("usage: /verbose [on|off]")
        return
    state.verbose = enabled
    state.sync_display_flags()
    click.echo(f"verbose={'on' if enabled else 'off'}")


def _cmd_rounds(state: ReplState, arg: str) -> None:
    token = arg.strip()
    if not token:
        click.echo(f"max_rounds={state.max_rounds}")
        return
    try:
        value = int(token)
    except ValueError:
        click.echo("usage: /rounds <positive integer>")
        return
    if value < 1:
        click.echo("max_rounds must be >= 1")
        return
    state.max_rounds = value
    state.agent.max_rounds = value
    click.echo(f"max_rounds={state.max_rounds}")


def _cmd_clear(state: ReplState) -> None:
    state.agent.messages.clear()
    click.echo("conversation cleared (in-memory only; DB history kept)")


def _preview_content(text: str | None) -> str:
    if not text:
        return ""
    if len(text) <= _CONTENT_PREVIEW:
        return text
    return text[:_CONTENT_PREVIEW] + "…"


def _format_history_message(index: int, message: AnyChatMessage) -> str:
    if isinstance(message, UserChatMessage):
        return f"{index}. user: {_preview_content(message.content)}"
    if isinstance(message, AssistantChatMessage):
        parts: list[str] = []
        if isinstance(message.content, str) and message.content:
            parts.append(_preview_content(message.content))
        tool_calls = message.tool_calls
        if tool_calls is not UNSET and tool_calls:
            names: list[str] = []
            for call in tool_calls:
                if isinstance(call, AssistantFunctionToolCall):
                    names.append(call.function.name)
                else:
                    names.append("custom")
            parts.append(f"tool_calls=[{', '.join(names)}]")
        body = " ".join(parts) if parts else "(empty)"
        return f"{index}. assistant: {body}"
    if isinstance(message, ToolChatMessage):
        return f"{index}. tool({message.tool_call_id}): {_preview_content(message.content)}"
    role = getattr(message, "role", type(message).__name__)
    content = getattr(message, "content", "")
    return f"{index}. {role}: {_preview_content(str(content))}"


def _cmd_history(state: ReplState, arg: str) -> None:
    token = arg.strip()
    limit = _DEFAULT_HISTORY_LINES
    if token:
        try:
            limit = int(token)
        except ValueError:
            click.echo("usage: /history [n]")
            return
        if limit < 1:
            click.echo("n must be >= 1")
            return
    messages = state.agent.messages
    if not messages:
        click.echo("(no messages in this session)")
        return
    start = max(0, len(messages) - limit)
    click.echo(f"session={state.session_id}  messages={len(messages)}  showing={len(messages) - start}")
    for offset, message in enumerate(messages[start:]):
        click.echo(_format_history_message(start + offset, message))
    if state.agent.pending_retry_text is not None:
        click.secho(
            f"(pending retry) user: {_preview_content(state.agent.pending_retry_text)}",
            fg="yellow",
        )


async def _cmd_retry(state: ReplState) -> None:
    _ = await retry_pending_with_retries(state.agent)


async def _dispatch_slash(state: ReplState, command: str, arg: str) -> bool:
    if command in {"quit", "exit", "q"}:
        return False

    handlers: dict[str, SlashHandler] = {
        "help": lambda: click.echo(HELP_TEXT),
        "clear": lambda: _cmd_clear(state),
        "history": lambda: _cmd_history(state, arg),
        "sessions": lambda: _cmd_sessions(state),
        "new": lambda: _cmd_new(state, arg),
        "rename": lambda: _cmd_rename(state, arg),
        "delete": lambda: _cmd_delete(state, arg),
        "export": lambda: _cmd_export(state, arg),
        "resume": lambda: _cmd_resume(state, arg),
        "compact": lambda: _cmd_compact(state, arg),
        "provider": lambda: _cmd_provider(state, arg),
        "model": lambda: _cmd_model(state, arg),
        "tools": lambda: _cmd_tools(state, arg),
        "stream": lambda: _cmd_stream(state, arg),
        "verbose": lambda: _cmd_verbose(state, arg),
        "rounds": lambda: _cmd_rounds(state, arg),
        "retry": lambda: _cmd_retry(state),
        "status": lambda: _cmd_status(state),
    }
    handler = handlers.get(command)
    if handler is None:
        click.echo(f"unknown command /{command}; try /help")
        return True
    result = handler()
    if inspect.isawaitable(result):
        await result
    return True


async def handle_slash(state: ReplState, line: str) -> bool:
    """Handle a slash command. Returns False if the REPL should exit."""
    body = line[1:].strip()
    if not body:
        return True
    command, _, rest = body.partition(" ")
    return await _dispatch_slash(state, command.lower(), rest.strip())


def _read_line() -> str:
    """Blocking readline input (intentional for TTY REPL)."""
    return input("> ").strip()


async def run_repl(state: ReplState) -> None:
    """Interactive chat loop with readline editing, history, and Tab completion."""
    setup_readline(state)
    click.echo(
        f"plyngent chat  provider={state.provider_name}  model={state.model}  "
        f"session={state.session_id}  tools={'on' if state.tools_enabled else 'off'}  "
        f"rounds={state.max_rounds}  messages={len(state.agent.messages)}  "
        f"stream={'on' if state.agent.stream else 'off'}  "
        f"verbose={'on' if state.verbose else 'off'}"
    )
    click.echo("Type /help for commands. Empty line is ignored.")

    while True:
        try:
            line = _read_line()
        except EOFError:
            click.echo()
            break
        except KeyboardInterrupt:
            click.echo()
            continue

        if not line:
            continue
        if line.startswith("/"):
            cont = await handle_slash(state, line)
            if not cont:
                break
            continue

        click.secho("user: ", fg="green", nl=False)
        click.echo(line)
        _ = await run_user_text_with_retries(state.agent, line)
