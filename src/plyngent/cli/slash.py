from __future__ import annotations

import os
import shlex
from typing import TYPE_CHECKING, Any, cast, override

import awaitlet  # pyright: ignore[reportMissingTypeStubs]
import click
from click.shell_completion import CompletionItem
from msgspec import UNSET

from plyngent.cli.models_source import DEFAULT_MODELS_CACHE_TTL, model_choices_for_provider
from plyngent.cli.retry import retry_pending_with_retries
from plyngent.cli.selection import select_model, select_provider
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionToolCall,
    DeveloperChatMessage,
    SystemChatMessage,
    UserChatMessage,
)
from plyngent.runtime import ProviderNotSupportedError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from plyngent.cli.state import ReplState, YoloMode
    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

_DEFAULT_HISTORY_LINES = 20
_CONTENT_PREVIEW = 200
_COMPACT_PREVIEW = 400
_ON_OFF_CHOICES = ("on", "off")
_YOLO_MODE_CHOICES = ("on", "off", "once")
_EXPORT_FORMAT_CHOICES = ("md", "json")


class HistoryLimitType(click.ParamType[int]):
    """``N`` (int >= 1) or the shortcut ``last`` (equivalent to ``1``)."""

    name: str = "history_limit"

    @override
    def convert(self, value: object, param: click.Parameter | None, ctx: click.Context | None) -> int:
        if isinstance(value, int):
            if value < 1:
                self.fail("must be >= 1", param, ctx)
            return value
        text = str(value).strip().lower()
        if text == "last":
            return 1
        try:
            n = int(text, 10)
        except ValueError:
            self.fail("expected an integer N or 'last'", param, ctx)
        if n < 1:
            self.fail("must be >= 1", param, ctx)
        return n

    @override
    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        del ctx, param
        tokens = ("last", "1", "5", "10", "20")
        return [CompletionItem(token) for token in tokens if incomplete == "" or token.startswith(incomplete.lower())]


HELP_FOOTER = (
    "User messages are saved immediately. On API errors or Ctrl+C, partial\n"
    "assistant/tool output is discarded but the user message stays (so /retry\n"
    "works after resume, not only via readline history). Auto-retry: 10s/20s/30s.\n"
    "\n"
    "Tab completes slash commands and some arguments (provider, model, tools,\n"
    "stream, verbose, yolo, export). Use --session ID or /resume to continue a prior\n"
    "chat after restart.\n"
    "\n"
    'Multiline: start a message with """ then end a later line with """.\n'
    "Long prompts: /edit opens $EDITOR.\n"
)


class ReplExitError(Exception):
    """Signal that the REPL should leave (not a process exit)."""


def _filter_choices(incomplete: str, choices: Sequence[str]) -> list[CompletionItem]:
    return [CompletionItem(c) for c in choices if c.startswith(incomplete)]


def _repl_state(ctx: click.Context | None) -> ReplState | None:
    if ctx is None or ctx.obj is None:
        return None
    return cast("ReplState", ctx.obj)


class OnOffParam(click.ParamType[bool]):
    """Accept on/off (and common synonyms); convert to bool."""

    name: str = "on_off"

    @override
    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> bool:
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in {"on", "1", "true", "yes"}:
            return True
        if token in {"off", "0", "false", "no"}:
            return False
        msg = "expected on or off"
        raise click.BadParameter(msg, ctx=ctx, param=param)

    @override
    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        del ctx, param
        return _filter_choices(incomplete, _ON_OFF_CHOICES)


ON_OFF = OnOffParam()


class YoloModeParam(click.ParamType[str]):
    """Accept on|off|once for soft destructive-tool confirms."""

    name: str = "yolo_mode"

    @override
    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> str:
        if isinstance(value, str) and value in _YOLO_MODE_CHOICES:
            return value
        token = str(value).strip().lower()
        if token in _YOLO_MODE_CHOICES:
            return token
        msg = "expected on, off, or once"
        raise click.BadParameter(msg, ctx=ctx, param=param)

    @override
    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        del ctx, param
        return _filter_choices(incomplete, _YOLO_MODE_CHOICES)


YOLO_MODE = YoloModeParam()


class ExportFormatParam(click.ParamType[str]):
    """First token of /export: md|json (or a path if not a format)."""

    name: str = "export_format"

    @override
    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> str:
        del param, ctx
        return str(value)

    @override
    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        del ctx, param
        return _filter_choices(incomplete, _EXPORT_FORMAT_CHOICES)


EXPORT_FORMAT = ExportFormatParam()


class ProviderNameParam(click.ParamType[str]):
    name: str = "provider"

    @override
    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> str:
        del param, ctx
        return str(value)

    @override
    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        del param
        state = _repl_state(ctx)
        if state is None:
            return []
        return _filter_choices(incomplete, sorted(state.config.selectable_providers().keys()))


PROVIDER_NAME = ProviderNameParam()


class ModelIdParam(click.ParamType[str]):
    name: str = "model"

    @override
    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> str:
        del param, ctx
        return str(value)

    @override
    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        del param
        state = _repl_state(ctx)
        if state is None:
            return []
        return _filter_choices(incomplete, state.model_choice_ids())


MODEL_ID = ModelIdParam()


class SlashCommandNameParam(click.ParamType[str]):
    """Command name for ``/help <cmd>``."""

    name: str = "slash_command"

    @override
    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> str:
        del param, ctx
        return str(value)

    @override
    def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
        del param
        names = sorted(slash.list_commands(ctx))
        token = incomplete.lstrip("/")
        return _filter_choices(token, names)


SLASH_COMMAND_NAME = SlashCommandNameParam()


class SlashGroup(click.Group):
    """Click group for REPL slash commands (no process-level ownership).

    Commands never expose Click's ``--help``; use ``/help`` / ``/help <cmd>``.
    """

    @override
    def command(self, *args: Any, **kwargs: Any) -> Any:
        # No auto --help; drop [OPTIONS] metavar when the command has no options.
        kwargs.setdefault("add_help_option", False)
        kwargs.setdefault("options_metavar", "")
        return super().command(*args, **kwargs)

    @override
    def get_help(self, ctx: click.Context) -> str:
        lines = ["Commands:"]
        for name in sorted(self.list_commands(ctx)):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            brief = cmd.get_short_help_str(limit=60)
            lines.append(f"  /{name:<16} {brief}")
        lines.append("")
        lines.append(HELP_FOOTER.rstrip())
        return "\n".join(lines)


slash = SlashGroup(
    "slash",
    help="REPL slash commands",
    add_help_option=False,
    context_settings={"help_option_names": [], "max_content_width": 100},
)


def slash_command_names() -> list[str]:
    """Slash tokens for Tab completion (including leading /)."""
    ctx = click.Context(slash)
    return sorted(f"/{name}" for name in slash.list_commands(ctx))


def complete_slash_args(state: ReplState, command: str, incomplete: str) -> list[str]:
    """Tab-complete arguments for ``command`` (e.g. ``/stream``) from ParamTypes.

    Uses the first :class:`click.Argument` on the registered command whose type
    implements :meth:`~click.ParamType.shell_complete` with candidates.
    """
    name = command.lstrip("/").lower()
    ctx = click.Context(slash, obj=state)
    cmd = slash.get_command(ctx, name)
    if cmd is None:
        return []
    with click.Context(cmd, info_name=name, parent=ctx, obj=state) as sub:
        for param in cmd.params:
            if not isinstance(param, click.Argument):
                continue
            items = param.type.shell_complete(sub, param, incomplete)
            if items:
                return [item.value for item in items]
    return []


def _await[T](awaitable: Awaitable[T]) -> T:
    # Greenlet parks until the awaitable completes on the running loop.
    return awaitlet.awaitlet(awaitable)


# --- commands -----------------------------------------------------------------


@slash.command("help")
@click.argument("command", required=False, type=SLASH_COMMAND_NAME)
@click.pass_context
def help_cmd(ctx: click.Context, command: str | None) -> None:
    """Show this help, or help for one command."""
    if command:
        name = command.lstrip("/").lower()
        cmd = slash.get_command(ctx, name)
        if cmd is None:
            click.echo(f"unknown command /{name}; try /help")
            return
        # Standalone context (no parent=help) so usage is "/compact …" not "help … compact".
        with click.Context(cmd, info_name=f"/{name}") as sub:
            click.echo(cmd.get_help(sub))
        return
    click.echo(slash.get_help(ctx))


@slash.command("quit")
@click.pass_obj
def quit_cmd(_state: ReplState) -> None:
    """Leave the REPL."""
    raise ReplExitError


slash.add_command(quit_cmd, name="exit")
slash.add_command(quit_cmd, name="q")


@slash.command("clear")
@click.pass_obj
def clear_cmd(state: ReplState) -> None:
    """Clear in-memory conversation (keeps session id)."""
    state.agent.messages.clear()
    click.echo("conversation cleared (in-memory only; DB history kept)")


@slash.command("edit")
@click.pass_obj
def edit_cmd(state: ReplState) -> None:
    """Compose a user message in ``$EDITOR``, then send it.

    Opens a temporary buffer; save and quit the editor to submit.
    Empty buffer cancels. Requires ``EDITOR`` (e.g. ``codium --wait``).
    """
    from plyngent.cli.editor import edit_text_in_editor

    try:
        text = edit_text_in_editor("")
    except click.ClickException as exc:
        click.echo(f"error: {exc}")
        return
    if text is None:
        click.echo("edit cancelled (empty)")
        return
    state.pending_user_text = text
    click.echo(f"(edit) {len(text)} characters ready to send")


@slash.command("config")
@click.pass_obj
def config_cmd(state: ReplState) -> None:
    """Open plyngent.toml in ``$EDITOR``, then reload providers/agent settings.

    Same file as ``plyngent config edit``. After the editor exits, config is
    re-read; current provider/model are kept when still valid.
    """
    from plyngent import config as config_mod
    from plyngent.cli.editor import open_in_editor

    path = state.config.path
    try:
        open_in_editor(path)
    except click.ClickException as exc:
        click.echo(f"error: {exc}")
        return
    try:
        state.reload_config_from_disk()
    except (config_mod.ConfigFormatError, ValueError, OSError) as exc:
        click.secho(f"error: config reload failed: {exc}", fg="red")
        click.echo(f"config file: {path}")
        return
    if state.config.recoverable_providers:
        names = ", ".join(sorted(state.config.recoverable_providers.keys()))
        click.secho(
            f"recoverable providers (empty models): {names}",
            fg="yellow",
        )
    if state.config.bad_providers:
        names = ", ".join(sorted(state.config.bad_providers.keys()))
        click.secho(f"warning: ignored bad providers: {names}", fg="yellow")
    click.echo(f"config reloaded from {path}\nprovider={state.provider_name}  model={state.model}")


@slash.command("status")
@click.pass_obj
def status_cmd(state: ReplState) -> None:
    """Show session/provider/tools/rounds status."""
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
    ctx_tag = "api" if ctx_src == "api" else "est"
    ctx_tilde = "" if ctx_src == "api" else "~"
    click.echo(
        f"provider={state.provider_name}  model={state.model}\n"
        f"session={state.session_id}  messages={len(state.agent.messages)}  "
        f"pending_retry={pending_disp}\n"
        f"tools={'on' if state.tools_enabled else 'off'}  "
        f"rounds={state.max_rounds}  "
        f"stream={'on' if state.agent.stream else 'off'}  "
        f"verbose={'on' if state.verbose else 'off'}  "
        f"yolo={state.effective_yolo()}\n"
        f"context_tokens={ctx_tilde}{ctx_tokens}/{ctx_budget} ({ctx_tag})  "
        f"context_chars={ctx_chars}  "
        f"tool_result_max={state.agent.max_tool_result_chars}\n"
        f"last_request={last_req.format_line()}\n"
        f"usage_last_turn={last_u.format_line(billed=True)}  "
        f"rounds={last_rounds}\n"
        f"usage_session={session_u.format_line(billed=True)}\n"
        f"workspace={state.workspace}"
    )


@slash.command("sessions")
@click.pass_obj
def sessions_cmd(state: ReplState) -> None:
    """List sessions for this workspace (newest first)."""
    sessions = _await(state.memory.list_sessions(workspace=state.workspace))
    if not sessions:
        click.echo(f"(no sessions for workspace {state.workspace})")
        return
    for session in sessions:
        marker = "*" if session.sid == state.session_id else " "
        ws = session.workspace or "(unbound)"
        click.echo(f"{marker} {session.sid}\t{session.name}\tworkspace={ws}\tupdated={session.updated_at}")


@slash.command("new")
@click.argument("name", required=False, default="chat")
@click.pass_obj
def new_cmd(state: ReplState, name: str) -> None:
    """Start a new session (bound to workspace)."""
    label = name.strip() or "chat"
    _await(state.new_session(name=label))
    click.echo(f"new session id={state.session_id} name={label}")


@slash.command("rename")
@click.argument("name", nargs=-1, required=True)
@click.pass_obj
def rename_cmd(state: ReplState, name: tuple[str, ...]) -> None:
    """Rename the current session."""
    full = " ".join(name).strip()
    if not full:
        msg = "NAME is required"
        raise click.UsageError(msg)
    try:
        row = _await(state.rename_current_session(full))
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    click.echo(f"renamed session {row.sid} -> {row.name}")


@slash.command("delete")
@click.argument("session_id", type=int, required=False)
@click.pass_obj
def delete_cmd(state: ReplState, session_id: int | None) -> None:
    """Hard-delete a session (confirm; current → new empty)."""
    from plyngent.prompting import NonInteractiveError, confirm_async

    sid = session_id
    if sid is None:
        if state.session_id is None:
            click.echo("error: no active session")
            return
        sid = state.session_id
    try:
        allowed = _await(
            confirm_async(
                f"Permanently delete session {sid} and all messages?",
                default=False,
            )
        )
    except NonInteractiveError:
        click.echo("error: delete requires interactive confirm (or TTY)")
        return
    if not allowed:
        click.echo("delete cancelled")
        return
    try:
        was_current = _await(state.delete_session_and_maybe_replace(sid))
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    if was_current:
        click.echo(f"deleted session {sid}; new session {state.session_id}")
    else:
        click.echo(f"deleted session {sid}")


@slash.command("export")
@click.argument("parts", nargs=-1, type=EXPORT_FORMAT)
@click.pass_obj
def export_cmd(state: ReplState, parts: tuple[str, ...]) -> None:
    """Export session transcript from DB: /export [md|json] [path]."""
    from plyngent.cli.export import (
        encode_session_export_json,
        format_session_export_md,
        resolve_export_path,
        session_export_payload,
        write_export_file,
    )

    fmt = "md"
    path: str | None = None
    if parts:
        first = parts[0].lower()
        if first in {"md", "markdown", "json"}:
            fmt = "json" if first == "json" else "md"
            path = parts[1] if len(parts) > 1 else None
            if len(parts) > 2:  # noqa: PLR2004
                msg = "usage: /export [md|json] [path]"
                raise click.UsageError(msg)
        else:
            path = parts[0]
            if len(parts) > 1:
                msg = "usage: /export [md|json] [path]"
                raise click.UsageError(msg)

    if state.session_id is None:
        click.echo("error: no active session")
        return
    row = _await(state.memory.get_session(state.session_id))
    if row is None:
        click.echo(f"error: session not found: {state.session_id}")
        return
    messages = _await(state.memory.list_messages(state.session_id))
    out_path = resolve_export_path(state.session_id, fmt, path)
    if fmt == "json":
        text = encode_session_export_json(
            session_export_payload(
                sid=row.sid,
                name=row.name,
                workspace=row.workspace,
                created_at=row.created_at,
                updated_at=row.updated_at,
                messages=messages,
                provider_name=row.provider_name,
                model=row.model,
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


@slash.command("resume")
@click.argument("session_id", type=int, required=False)
@click.pass_obj
def resume_cmd(state: ReplState, session_id: int | None) -> None:
    """Resume session id, or latest for this workspace if omitted."""
    if session_id is None:
        mode = _await(state.resume_latest_or_new())
        if mode == "new":
            click.echo(f"no prior session; created new session {state.session_id}")
        else:
            click.echo(
                f"resumed latest session {state.session_id} "
                f"({len(state.agent.messages)} messages) workspace={state.workspace}"
            )
        return
    try:
        _await(state.resume_session(session_id))
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    click.echo(f"resumed session {session_id} ({len(state.agent.messages)} messages) workspace={state.workspace}")


@slash.command("compact")
@click.argument("name", required=False)
@click.pass_obj
def compact_cmd(state: ReplState, name: str | None) -> None:
    """Soft-compact + model-summarize into a new session."""
    click.secho("compacting (soft-compact + model summary)…", fg="yellow")
    try:
        old_id, new_id, summary = _await(state.compact_to_new_session(name=name))
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — surface model/API failures
        click.secho(f"error: compact failed: {exc}", fg="red")
        return
    preview = summary if len(summary) <= _COMPACT_PREVIEW else summary[:_COMPACT_PREVIEW] + "…"
    click.echo(f"compacted session {old_id} -> new session {new_id}")
    click.secho(
        f"active session is summary-only ({len(state.agent.messages)} messages); "
        f"full history remains on /resume {old_id}",
        fg="bright_black",
    )
    click.secho(preview, fg="bright_black")


@slash.command("provider")
@click.argument("name", required=False, type=PROVIDER_NAME)
@click.pass_obj
def provider_cmd(state: ReplState, name: str | None) -> None:
    """Show or switch provider."""
    if not name:
        click.echo(f"provider={state.provider_name}")
        return
    try:
        from plyngent.cli.provider_recovery import ensure_provider_ready

        pname, provider = select_provider(
            state.config.selectable_providers(),
            preferred=name.strip(),
        )
        provider = _await(
            ensure_provider_ready(
                state.config,
                pname,
                provider,
                preferred_model=state.model,
                interactive=True,
            )
        )
        prev_model = state.model
        state.provider_name = pname
        state.provider = provider
        state.rebuild_client()
        # Always request remote catalog for the new provider (bypass stale cache).
        choices = _await(state.merged_model_choices(refresh=True))
        if prev_model and (prev_model in choices or prev_model in provider.models):
            state.model = prev_model
        else:
            # Current model not on the new provider — pick one (prompt when interactive).
            try:
                state.model = select_model(
                    provider,
                    preferred=None,
                    interactive=True,
                    choices=choices,
                )
            except click.ClickException as exc:
                click.echo(f"error: switched provider but model selection failed: {exc}")
                return
            if prev_model:
                click.secho(
                    f"model {prev_model!r} is not available on {pname}; using {state.model!r}",
                    fg="yellow",
                )
            state.rebuild_client()
        _await(state.persist_llm_selection())
        click.echo(f"switched provider to {pname}  model={state.model}")
    except (click.ClickException, ProviderNotSupportedError) as exc:
        click.echo(f"error: {exc}")


@slash.command("models")
@click.option("--refresh", is_flag=True, help="Bypass cache and re-fetch GET /models.")
@click.option(
    "--persist",
    is_flag=True,
    help="Merge remote (and recovered) model ids into plyngent.toml for this provider.",
)
@click.pass_obj
def models_cmd(state: ReplState, *, refresh: bool, persist: bool) -> None:  # noqa: C901
    """List models (remote-first, plus config-only ids). Always tries GET /models."""
    del refresh  # always re-fetch; flag kept for CLI compatibility / docs
    remote: list[str] | None = None
    remote_err: str | None = None
    try:
        remote = _await(state.ensure_remote_models(refresh=True))
    except (RuntimeError, TypeError, OSError, ValueError) as exc:
        remote_err = str(exc)
        remote = state.cached_remote_models()

    # Promote empty-models recoverable provider after a successful remote list.
    if remote and state.provider_name in state.config.recoverable_providers:
        try:
            state.provider = state.config.promote_provider(state.provider_name, remote)
            state.rebuild_client()
            click.secho(
                f"recovered provider {state.provider_name!r} from remote catalog",
                fg="yellow",
                err=True,
            )
        except (KeyError, ValueError) as exc:
            click.secho(f"could not recover provider: {exc}", fg="yellow", err=True)

    if persist:
        try:
            path = state.persist_models_to_config(mode="catalog", catalog_ids=remote or ())
            click.echo(f"persisted {len(state.provider.models)} model(s) for {state.provider_name!r} to {path}")
        except (KeyError, ValueError, OSError) as exc:
            click.secho(f"error: could not persist models: {exc}", fg="red")
            return

    config_ids = set(state.config_model_ids())
    choices = model_choices_for_provider(state.provider, remote_ids=remote)

    if not choices:
        click.echo("(no models in config or remote catalog)")
    else:
        remote_set = set(remote or ())
        for mid in choices:
            tags: list[str] = []
            if mid in remote_set:
                tags.append("remote")
            if mid in config_ids:
                tags.append("config")
            suffix = f"  ({', '.join(tags)})" if tags else ""
            mark = " *" if mid == state.model else ""
            click.echo(f"{mid}{mark}{suffix}")

    if remote_err is not None:
        click.secho(f"remote list unavailable: {remote_err}", fg="yellow", err=True)
    elif remote is not None:
        click.echo(
            f"(remote-first: {len(remote)} remote, {len(config_ids)} config; "
            f"cache TTL {int(DEFAULT_MODELS_CACHE_TTL)}s)",
            err=True,
        )


@slash.command("model")
@click.argument("model_id", required=False, type=MODEL_ID)
@click.option(
    "--persist",
    is_flag=True,
    help="Write the current (or newly selected) model id into plyngent.toml.",
)
@click.pass_obj
def model_cmd(state: ReplState, model_id: str | None, *, persist: bool) -> None:
    """Show or switch model (Tab: remote-first plus config; live fetch on pick).

    ``/model --persist`` saves the active model id into the provider catalog in
    TOML (faster Tab/list on next launch). ``/model <id> --persist`` switches
    then saves.
    """
    if model_id:
        try:
            choices = _await(state.merged_model_choices(refresh=True))
            state.model = select_model(
                state.provider,
                preferred=model_id.strip(),
                choices=choices,
            )
            state.rebuild_client()
            _await(state.persist_llm_selection())
            click.echo(f"switched model to {state.model}")
        except click.ClickException as exc:
            click.echo(f"error: {exc}")
            return
    elif not persist:
        click.echo(f"model={state.model}")
        return

    if persist:
        try:
            path = state.persist_models_to_config(mode="current")
            click.echo(f"persisted model {state.model!r} for {state.provider_name!r} to {path}")
        except (KeyError, ValueError, OSError) as exc:
            click.secho(f"error: could not persist model: {exc}", fg="red")


@slash.command("tools")
@click.argument("enabled", required=False, type=ON_OFF, metavar="[on|off]")
@click.pass_obj
def tools_cmd(state: ReplState, enabled: bool | None) -> None:  # noqa: FBT001
    """Show or set whether agent tools are enabled.

    Omit the argument to print the current value; pass ``on`` or ``off`` to change it.
    """
    if enabled is None:
        click.echo(f"tools={'on' if state.tools_enabled else 'off'}")
        return
    state.tools_enabled = enabled
    state.rebuild_client()
    click.echo(f"tools={'on' if enabled else 'off'}")


@slash.command("yolo")
@click.argument("mode", required=False, type=YOLO_MODE, metavar="[on|off|once]")
@click.pass_obj
def yolo_cmd(state: ReplState, mode: str | None) -> None:
    """Show or set YOLO mode for soft destructive-tool confirms.

    ``off`` (default when config ``confirm_destructive`` is true): prompt on
    delete/move/overwrite (deny in non-TTY). ``on``: skip confirms for the
    process. ``once``: skip for the next user turn only, then return to ``off``.
    Path/command denylists still apply. Omit the argument to print the value.
    """
    if mode is None:
        click.echo(f"yolo={state.effective_yolo()}")
        return
    state.set_yolo(cast("YoloMode", mode))
    click.echo(f"yolo={state.effective_yolo()}")


@slash.command("stream")
@click.argument("enabled", required=False, type=ON_OFF, metavar="[on|off]")
@click.pass_obj
def stream_cmd(state: ReplState, enabled: bool | None) -> None:  # noqa: FBT001
    """Show or set streaming model output.

    ``on`` (default): print assistant text and reasoning as tokens arrive.
    ``off``: wait for each full model response before printing.
    Omit the argument to print the current value.
    """
    if enabled is None:
        click.echo(f"stream={'on' if state.agent.stream else 'off'}")
        return
    state.stream_enabled = enabled
    state.agent.stream = enabled
    click.echo(f"stream={'on' if enabled else 'off'}")


@slash.command("verbose")
@click.argument("enabled", required=False, type=ON_OFF, metavar="[on|off]")
@click.pass_obj
def verbose_cmd(state: ReplState, enabled: bool | None) -> None:  # noqa: FBT001
    """Show or set full tool-result dumps in the terminal.

    ``off`` (default): short one-line tool result preview.
    ``on``: print the full tool result text.
    Omit the argument to print the current value.
    """
    if enabled is None:
        click.echo(f"verbose={'on' if state.verbose else 'off'}")
        return
    state.verbose = enabled
    state.sync_display_flags()
    click.echo(f"verbose={'on' if enabled else 'off'}")


@slash.command("markdown")
@click.argument("enabled", required=False, type=ON_OFF, metavar="[on|off]")
@click.pass_obj
def markdown_cmd(state: ReplState, enabled: bool | None) -> None:  # noqa: FBT001
    """Show or set end-of-turn Rich markdown for assistant text.

    ``on`` (default on TTY): stream plain tokens, then re-render as markdown.
    ``off``: leave streamed plain text as-is.
    Non-TTY / ``PLYNGENT_PLAIN=1`` never pretty-prints.
    """
    if enabled is None:
        click.echo(f"markdown={'on' if state.markdown_enabled else 'off'}")
        return
    state.markdown_enabled = enabled
    state.sync_display_flags()
    click.echo(f"markdown={'on' if enabled else 'off'}")


@slash.command("rounds")
@click.argument("n", required=False, type=int)
@click.pass_obj
def rounds_cmd(state: ReplState, n: int | None) -> None:
    """Show or set max tool-loop rounds."""
    if n is None:
        click.echo(f"max_rounds={state.max_rounds}")
        return
    if n < 1:
        msg = "max_rounds must be >= 1"
        raise click.UsageError(msg)
    state.max_rounds = n
    state.agent.max_rounds = n
    click.echo(f"max_rounds={state.max_rounds}")


@slash.command("history")
@click.argument("n", required=False, type=HistoryLimitType())
@click.option(
    "--full",
    "mode_full",
    is_flag=True,
    default=False,
    help="Print full message bodies (markdown for assistant when TTY).",
)
@click.option(
    "--preview",
    "mode_preview",
    is_flag=True,
    default=False,
    help="Force short previews even for a single message.",
)
@click.pass_obj
def history_cmd(
    state: ReplState,
    n: int | None,
    *,
    mode_full: bool,
    mode_preview: bool,
) -> None:
    """Show recent messages (default: last 20 as short previews).

    ``/history last`` or ``/history 1`` shows the last message in full with Rich
    markdown when available. Use ``--full`` for multiple messages in full;
    ``--preview`` forces the short form even for a single message.
    """
    if mode_full and mode_preview:
        msg = "use only one of --full / --preview"
        raise click.UsageError(msg)
    limit = _DEFAULT_HISTORY_LINES if n is None else n
    messages = state.agent.messages
    if not messages:
        click.echo("(no messages in this session)")
        return
    start = max(0, len(messages) - limit)
    slice_msgs = messages[start:]
    # Full: --full, or single-message window (last / 1) unless --preview.
    full = mode_full or (len(slice_msgs) == 1 and not mode_preview)
    mode_tag = "full" if full else "preview"
    click.echo(f"session={state.session_id}  messages={len(messages)}  showing={len(slice_msgs)}  mode={mode_tag}")
    for offset, message in enumerate(slice_msgs):
        idx = start + offset
        if full:
            _print_history_message_full(idx, message)
        else:
            click.echo(_format_history_message(idx, message))
    if state.agent.pending_retry_text is not None:
        pending = state.agent.pending_retry_text
        if full:
            click.secho("(pending retry) user:", fg="yellow")
            click.echo(pending)
        else:
            click.secho(
                f"(pending retry) user: {_preview_content(pending)}",
                fg="yellow",
            )


@slash.command("todos")
@click.argument("action", required=False, type=str)
@click.argument("rest", required=False, nargs=-1, type=str)
@click.pass_obj
def todos_cmd(  # noqa: C901, PLR0911, PLR0912, PLR0915
    state: ReplState, action: str | None, rest: tuple[str, ...]
) -> None:
    """Show or edit the LIFO stack of **task groups**.

    ``/todos`` — list (top group first)
    ``/todos push T1; T2`` — one new group of siblings
    ``/todos pop`` — pop entire top group
    ``/todos done <id>`` / ``/todos cancel <id>`` — set status
    ``/todos clear`` — wipe stack
    """
    from plyngent.agent.todo_stack import parse_push_titles

    stack = state.todo_stack
    act = (action or "list").strip().lower()
    if act in {"list", "show", "ls"}:
        click.echo(stack.render())
        return
    if act == "push":
        raw = " ".join(rest).strip()
        if not raw:
            click.echo("error: usage: /todos push <title> | /todos push T1; T2")
            return
        titles = parse_push_titles(raw)
        if not titles:
            click.echo("error: no titles to push")
            return
        try:
            group = stack.push_group(titles)
        except ValueError as exc:
            click.echo(f"error: {exc}")
            return
        _await(state.persist_todo_stack())
        ids = ", ".join(i.id for i in group.items)
        click.echo(f"pushed group depth={stack.depth} items=[{ids}]")
        click.echo(stack.render())
        return
    if act == "pop":
        group = stack.pop()
        if group is None:
            click.echo("todo stack empty")
            return
        _await(state.persist_todo_stack())
        titles = ", ".join(f"{i.id}:{i.title}" for i in group.items) or "(empty)"
        click.echo(f"popped TOP group ({titles})")
        click.echo(stack.render())
        return
    if act in {"done", "cancel", "pending", "in_progress"}:
        if not rest:
            click.echo(f"error: usage: /todos {act} <id>")
            return
        if act == "done":
            new_status = "done"
        elif act == "cancel":
            new_status = "cancelled"
        elif act == "pending":
            new_status = "pending"
        else:
            new_status = "in_progress"
        try:
            item = stack.update(rest[0], status=new_status)
        except (KeyError, ValueError) as exc:
            click.echo(f"error: {exc}")
            return
        _await(state.persist_todo_stack())
        click.echo(f"updated {item.id} → {item.status}")
        click.echo(stack.render())
        return
    if act == "clear":
        n = stack.clear()
        _await(state.persist_todo_stack())
        click.echo(f"cleared {n} item(s)")
        return
    click.echo("error: usage: /todos [list|push|pop|done|cancel|clear] …")


@slash.command("retry")
@click.pass_obj
def retry_cmd(state: ReplState) -> None:
    """Continue an incomplete turn (user-only or after committed tools).

    Does not retype the user message. After tools already ran, continues the
    model loop without re-executing those tool calls.
    """
    _ = _await(retry_pending_with_retries(state.agent))


def _preview_content(text: str | None) -> str:
    if not text:
        return ""
    if len(text) <= _CONTENT_PREVIEW:
        return text
    return text[:_CONTENT_PREVIEW] + "…"


def _print_history_assistant_full(index: int, message: AssistantChatMessage) -> None:
    from plyngent.cli.display import markdown_render_available, print_markdown

    click.secho(f"{index}. assistant:", fg="cyan")
    content = message.content
    has_text = isinstance(content, str) and content.strip()
    if has_text and isinstance(content, str):
        if markdown_render_available():
            print_markdown(content, label="")
        else:
            click.echo(content)
    reasoning = message.reasoning_content
    has_reason = isinstance(reasoning, str) and reasoning.strip()
    if has_reason and isinstance(reasoning, str):
        click.secho("reasoning:", fg="bright_black")
        click.echo(reasoning)
    tool_calls = message.tool_calls
    has_tools = tool_calls is not UNSET and bool(tool_calls)
    if has_tools and tool_calls is not UNSET:
        for call in tool_calls:
            if isinstance(call, AssistantFunctionToolCall):
                click.secho(
                    f"  tool_call {call.id}: {call.function.name}({call.function.arguments})",
                    fg="yellow",
                )
            else:
                click.secho(f"  tool_call custom id={call.id}", fg="yellow")
    if not (has_text or has_reason or has_tools):
        click.echo("(empty)")
    click.echo()


def _print_history_message_full(index: int, message: AnyChatMessage) -> None:
    """Print one history message with full body; Rich markdown for assistant text."""
    if isinstance(message, UserChatMessage):
        click.secho(f"{index}. user:", fg="green")
        click.echo(message.content or "")
        click.echo()
        return
    if isinstance(message, DeveloperChatMessage):
        click.secho(f"{index}. developer:", fg="blue")
        click.echo(message.content or "")
        click.echo()
        return
    if isinstance(message, SystemChatMessage):
        click.secho(f"{index}. system:", fg="bright_black")
        click.echo(message.content or "")
        click.echo()
        return
    if isinstance(message, AssistantChatMessage):
        _print_history_assistant_full(index, message)
        return
    # ToolChatMessage (remaining AnyChatMessage arm)
    click.secho(f"{index}. tool({message.tool_call_id}):", fg="magenta")
    click.echo(message.content or "")
    click.echo()


def _format_history_message(index: int, message: AnyChatMessage) -> str:
    if isinstance(message, UserChatMessage):
        return f"{index}. user: {_preview_content(message.content)}"
    if isinstance(message, DeveloperChatMessage):
        return f"{index}. developer: {_preview_content(message.content)}"
    if isinstance(message, SystemChatMessage):
        return f"{index}. system: {_preview_content(message.content)}"
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
    # ToolChatMessage
    return f"{index}. tool({message.tool_call_id}): {_preview_content(message.content)}"


def _run_slash_argv(args: Sequence[str], state: ReplState) -> None:
    """Sync Click entrypoint; may call awaitlet() for async work."""
    # standalone_mode=False → UsageError/ClickException instead of SystemExit.
    slash.main(
        args=list(args),
        prog_name="",
        obj=state,
        standalone_mode=False,
    )


async def handle_slash(state: ReplState, line: str) -> bool:
    """Handle a slash command. Returns False if the REPL should exit."""
    body = line[1:].strip()
    if not body:
        return True
    try:
        # Windows paths use backslashes; POSIX shlex would treat them as escapes.
        args = shlex.split(body, posix=os.name != "nt")
    except ValueError as exc:
        click.echo(f"error: {exc}")
        return True
    if not args:
        return True
    args[0] = args[0].lower()
    try:
        await awaitlet.async_def(_run_slash_argv, args, state)
    except ReplExitError:
        return False
    except click.ClickException as exc:
        exc.show()
    except click.exceptions.Exit as exc:
        # Click may still raise Exit(0) for some paths; ignore non-error codes.
        if exc.exit_code not in {0, None}:
            click.echo(f"error: exit {exc.exit_code}")
    except click.Abort:
        click.echo("aborted")
    return True
