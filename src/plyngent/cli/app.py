from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click
import msgspec
from platformdirs import user_data_path

from plyngent import config as config_mod
from plyngent.agent.loop import DEFAULT_MAX_ROUNDS
from plyngent.cli.editor import (
    load_config_with_optional_edit,
    open_in_editor,
    resolve_config_path,
)
from plyngent.cli.exit_codes import EXIT_CANCELLED, EXIT_OK, EXIT_TURN_FAILED
from plyngent.cli.limits import install_cli_limit_hooks
from plyngent.cli.repl import run_repl
from plyngent.cli.retry import run_user_text_with_retries
from plyngent.cli.selection import select_model, select_provider
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig
from plyngent.memory import MemoryStore
from plyngent.prompting import NonInteractiveBackend, configure_prompting
from plyngent.runtime import ProviderNotSupportedError, create_client

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.config.store import ConfigStore

_DEFAULT_DB_FILENAME = "chat.db"


def _load_config(config_path: Path | None, *, require_providers: bool = True) -> ConfigStore:
    """Load config; optionally skip the interactive “no providers” recovery path.

    Plugin management only needs a valid TOML file (and will create one if
    missing via :func:`load_config_with_optional_edit` when providers are required
    for chat). For plugins, use ``require_providers=False``.
    """
    try:
        if require_providers:
            return load_config_with_optional_edit(config_path)
        path = resolve_config_path(config_path)
        return config_mod.load(path)
    except config_mod.ConfigFormatError as exc:
        path = resolve_config_path(config_path)
        msg = f"invalid config TOML ({path}): {exc}"
        raise click.ClickException(msg) from exc


def _warn_bad_providers(bad: Mapping[str, object]) -> None:
    """Surface ignored provider entries (parse errors, unknown fields, …)."""
    if not bad:
        return
    names = ", ".join(sorted(bad.keys()))
    click.secho(
        f"warning: ignored bad providers ({len(bad)}): {names}",
        fg="yellow",
        err=True,
    )
    for name in sorted(bad.keys()):
        entry = bad[name]
        reason = "invalid or incomplete entry"
        if isinstance(entry, dict):
            entry_map = cast("dict[str, object]", entry)
            raw_reason = entry_map.get("_reason")
            if isinstance(raw_reason, str) and raw_reason:
                reason = raw_reason
            elif "preset" not in entry_map and not any(k in entry_map for k in ("access_key_or_token", "url")):
                reason = "not a provider table"
        click.secho(f"  - {name}: {reason}", fg="yellow", err=True)


def _warn_recoverable_providers(recoverable: Mapping[str, object]) -> None:
    """Surface providers with empty models (recoverable via GET /models)."""
    if not recoverable:
        return
    names = ", ".join(sorted(recoverable.keys()))
    click.secho(
        f"warning: providers with empty models ({len(recoverable)}): {names} (will try GET /models or --model on use)",
        fg="yellow",
        err=True,
    )


def _database_config(store: ConfigStore, *, quiet: bool = False) -> DatabaseConfig:
    raw = dict(store.database)
    url = raw.get("url")
    impl = raw.get("implementation", "sqlite")
    # Unset/empty → durable user-data chat.db so interactive sessions persist.
    # Explicit ``:memory:`` is kept as-is (ephemeral; warn so it is not accidental).
    if impl == "sqlite" and url in {None, ""}:
        db_path = user_data_path("plyngent", ensure_exists=True) / _DEFAULT_DB_FILENAME
        raw = {**raw, "implementation": "sqlite", "url": str(db_path)}
        if not quiet:
            click.secho(f"using database: {db_path}", fg="bright_black", err=True)
    elif not quiet and impl == "sqlite" and url == ":memory:":
        click.secho(
            "warning: database url is :memory: — sessions are not persisted to disk",
            fg="yellow",
            err=True,
        )
    return msgspec.convert(raw, DatabaseConfig)


def _read_prompt_text(prompt: str | None, *, stdin_isatty: bool) -> str | None:
    """Resolve one-shot prompt from ``-p`` and/or non-TTY stdin."""
    chunks: list[str] = []
    if prompt is not None and prompt.strip():
        chunks.append(prompt)
    if not stdin_isatty:
        data = sys.stdin.read()
        if data.strip():
            chunks.append(data.rstrip("\n"))
    if not chunks:
        return None
    text = "\n".join(chunks).strip()
    return text or None


def _setup_hooks(*, interactive: bool) -> None:
    """Install interactive limit hooks (workspace policy is set on ReplState)."""
    if interactive:
        install_cli_limit_hooks()
    else:
        from plyngent.tools.process.pty_session import PtyManager

        PtyManager.set_limit_continue_hook(None)


async def _bind_session(
    state: ReplState,
    *,
    session_id: int | None,
    new_session: bool,
    oneshot: bool,
    quiet: bool,
) -> None:
    if session_id is not None:
        try:
            await state.resume_session(session_id)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if not quiet and not oneshot:
            click.echo(
                f"resumed session {session_id} ({len(state.agent.messages)} messages) workspace={state.workspace}",
                err=True,
            )
        return
    if new_session or oneshot:
        await state.new_session()
        if not quiet and not oneshot:
            click.echo(
                f"new session {state.session_id} (workspace={state.workspace})",
                err=True,
            )
        return
    mode = await state.resume_latest_or_new()
    if quiet:
        return
    if mode == "resume":
        click.echo(
            f"resumed latest session {state.session_id} for this workspace "
            f"({len(state.agent.messages)} messages); use --new for a fresh chat",
            err=True,
        )
    else:
        click.echo(
            f"new session {state.session_id} (workspace={state.workspace})",
            err=True,
        )


async def _run_oneshot(state: ReplState, prompt_text: str) -> int:
    try:
        ok = await run_user_text_with_retries(state.agent, prompt_text, delays=())
    except asyncio.CancelledError:
        state.expire_yolo_once(quiet=True)
        return EXIT_CANCELLED
    except KeyboardInterrupt:
        state.expire_yolo_once(quiet=True)
        return EXIT_CANCELLED
    state.expire_yolo_once(quiet=True)
    return EXIT_OK if ok else EXIT_TURN_FAILED


async def _run_chat(  # noqa: C901, PLR0912, PLR0915 — chat orchestration
    *,
    config_path: Path | None,
    provider_name: str | None,
    model: str | None,
    tools: bool,
    workspace: Path,
    session_id: int | None,
    max_rounds: int,
    new_session: bool,
    prompt_text: str | None,
    stream: bool,
    yes: bool,
    quiet: bool,
) -> int:
    oneshot = prompt_text is not None
    interactive = not oneshot and sys.stdin.isatty() and sys.stdout.isatty()

    if oneshot:
        configure_prompting(backend=NonInteractiveBackend())

    store = _load_config(config_path)
    if not quiet:
        if store.bad_providers:
            _warn_bad_providers(store.bad_providers)
        if store.recoverable_providers:
            _warn_recoverable_providers(store.recoverable_providers)

    _setup_hooks(interactive=interactive)
    # --yes forces sticky YOLO; else derive from config.confirm_destructive.
    from plyngent.cli.state import YoloMode

    yolo: YoloMode | None = "on" if yes else None

    memory = await MemoryStore.open(_database_config(store, quiet=quiet or oneshot))
    try:
        from plyngent.cli.provider_recovery import ensure_provider_ready

        # Prefer session-remembered LLM when resuming (unless flags override).
        preferred_provider = provider_name
        preferred_model = model
        if not oneshot and not new_session and preferred_provider is None:
            if session_id is not None:
                row = await memory.get_session(session_id)
            else:
                row = await memory.get_latest_session(workspace=workspace)
            if row is not None:
                if preferred_provider is None and row.provider_name:
                    preferred_provider = row.provider_name
                if preferred_model is None and row.model:
                    preferred_model = row.model

        try:
            pname, provider = select_provider(
                store.selectable_providers(),
                preferred=preferred_provider,
                interactive=interactive,
            )
            provider = await ensure_provider_ready(
                store,
                pname,
                provider,
                preferred_model=preferred_model,
                interactive=interactive,
            )
            # Avoid blocking ready on GET /models unless interactive pick needs it.
            from plyngent.cli.models_source import (
                client_supports_models,
                fetch_remote_model_ids,
                model_choices_for_provider,
                needs_remote_models_for_selection,
            )

            remote_ids: list[str] | None = None
            if needs_remote_models_for_selection(
                provider,
                preferred_model=preferred_model,
                interactive=interactive,
            ):
                client = create_client(provider)
                try:
                    if client_supports_models(client):
                        remote_ids = await fetch_remote_model_ids(client)
                except RuntimeError, TypeError, OSError, ValueError, TimeoutError:
                    remote_ids = None
            choices = model_choices_for_provider(provider, remote_ids=remote_ids)
            model_id = select_model(
                provider,
                preferred=preferred_model,
                interactive=interactive,
                choices=choices,
            )
        except ProviderNotSupportedError as exc:
            raise click.ClickException(str(exc)) from exc

        state = ReplState(
            config=store,
            memory=memory,
            workspace=workspace,
            provider_name=pname,
            provider=provider,
            model=model_id,
            tools_enabled=tools,
            max_rounds=max_rounds,
            stream_enabled=stream,
            interactive_limits=interactive,
            yolo=yolo,
        )
        # Path denylist and policy confirm live on instance.workspace (no process bag).
        state.instance_state.workspace.path_denylist = tuple(store.agent_config.path_denylist or ())
        if interactive:
            from plyngent.cli.limits import prompt_policy_command_confirm

            state.instance_state.workspace.policy_confirm_hook = prompt_policy_command_confirm
        # Seed cache if we already fetched; else warm in background for Tab.
        if remote_ids is not None:
            state.seed_remote_models(remote_ids)
        elif not oneshot and interactive:
            state.schedule_remote_models_warm()
        if not quiet and not oneshot:
            click.secho(f"workspace: {state.workspace}", fg="bright_black", err=True)

        await _bind_session(
            state,
            session_id=session_id,
            new_session=new_session,
            oneshot=oneshot,
            quiet=quiet,
        )
        # Ensure new sessions / flag overrides are stored for next resume.
        await state.persist_llm_selection()

        if oneshot:
            assert prompt_text is not None
            return await _run_oneshot(state, prompt_text)

        await run_repl(state)
        return EXIT_OK
    finally:
        await memory.close()
        # PTY + temp workspace cleanup via instance shutdown when state exists.
        state_obj = locals().get("state")
        if isinstance(state_obj, ReplState):
            state_obj.instance_state.workspace.policy_confirm_hook = None
            state_obj.instance_state.workspace.policy_allowed_commands.clear()
            await state_obj.instance_state.shutdown()
        else:
            # No ReplState: only PTY class cleanup (temps require instance allowlist).
            from plyngent.tools.process.pty_session import PtyManager

            PtyManager.close_all()


def _configure_logging(level: str) -> None:
    import logging

    name = level.upper()
    numeric = getattr(logging, name, None)
    if not isinstance(numeric, int):
        msg = f"invalid --log-level {level!r}"
        raise click.ClickException(msg)
    logging.basicConfig(
        level=numeric,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Avoid accidental secret leakage via HTTP libraries at DEBUG.
    if numeric <= logging.DEBUG:
        logging.getLogger("niquests").setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.INFO)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="plyngent")
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    help="Logging level for stderr (DEBUG, INFO, WARNING, ERROR).",
)
@click.pass_context
def main(ctx: click.Context, log_level: str) -> None:
    """Plyngent — LLM chat and agent toolkit."""
    _ = ctx
    _configure_logging(log_level)


@main.command("chat")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml (default: platform config dir).",
)
@click.option("--provider", "provider_name", default=None, help="Provider name from config.")
@click.option("--model", default=None, help="Model id.")
@click.option("--tools/--no-tools", default=True, show_default=True, help="Enable catalog tools (local surface).")
@click.option(
    "--workspace",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="Workspace root for tools (default: cwd).",
)
@click.option("--session", "session_id", type=int, default=None, help="Resume session id.")
@click.option(
    "--new",
    "new_session",
    is_flag=True,
    default=False,
    help="Start a new session instead of resuming the latest.",
)
@click.option(
    "--max-rounds",
    type=int,
    default=DEFAULT_MAX_ROUNDS,
    show_default=True,
    help="Max tool-loop rounds per user turn.",
)
@click.option(
    "-p",
    "--prompt",
    "prompt",
    default=None,
    help="One-shot user message (non-interactive). Also reads stdin when not a TTY.",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    show_default=True,
    help="Stream model output (one-shot and REPL default).",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Enable YOLO: skip destructive-tool confirms (sticky for this process).",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Less status noise on stderr.",
)
def chat_cmd(
    config_path: Path | None,
    provider_name: str | None,
    model: str | None,
    tools: bool,  # noqa: FBT001
    workspace: Path | None,
    session_id: int | None,
    new_session: bool,  # noqa: FBT001
    max_rounds: int,
    prompt: str | None,
    stream: bool,  # noqa: FBT001
    yes: bool,  # noqa: FBT001
    quiet: bool,  # noqa: FBT001
) -> None:
    """Interactive chat REPL, or one-shot with ``-p`` / stdin.

    Exit codes (one-shot): 0 ok, 1 config/usage, 2 cancelled, 3 turn failed.
    """
    if max_rounds < 1:
        msg = "--max-rounds must be >= 1"
        raise click.ClickException(msg)
    if session_id is not None and new_session:
        msg = "use either --session or --new, not both"
        raise click.ClickException(msg)

    prompt_text = _read_prompt_text(prompt, stdin_isatty=sys.stdin.isatty())
    if prompt is None and prompt_text is None and not sys.stdin.isatty():
        # Non-TTY with empty stdin and no -p: still require an explicit prompt.
        msg = "no prompt: pass -p/--prompt or pipe text on stdin"
        raise click.ClickException(msg)

    root = workspace if workspace is not None else Path.cwd()
    code = asyncio.run(
        _run_chat(
            config_path=config_path,
            provider_name=provider_name,
            model=model,
            tools=tools,
            workspace=root,
            session_id=session_id,
            max_rounds=max_rounds,
            new_session=new_session,
            prompt_text=prompt_text,
            stream=stream,
            yes=yes,
            quiet=quiet,
        )
    )
    if code != EXIT_OK:
        raise SystemExit(code)


@main.command("providers")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml.",
)
def providers_cmd(config_path: Path | None) -> None:
    """List configured providers."""
    store = _load_config(config_path)
    if not store.providers and not store.recoverable_providers:
        click.echo("(no providers)")
    for name, provider in sorted(store.providers.items()):
        tag = type(provider).__struct_config__.tag
        models = ", ".join(sorted(provider.models.keys())) or "(none listed)"
        click.echo(f"{name}\tpreset={tag}\tmodels={models}")
    for name, provider in sorted(store.recoverable_providers.items()):
        tag = type(provider).__struct_config__.tag
        click.echo(f"{name}\tpreset={tag}\tmodels=(empty; recoverable)")
    if store.bad_providers:
        _warn_bad_providers(store.bad_providers)


def print_plugins_table(store: ConfigStore) -> None:
    """Print discovered plugins with allowlist status (CLI + slash)."""
    from plyngent.tools.plugins import list_plugin_statuses

    cfg = store.plugins_config
    click.echo(f"config={store.path}")
    enable_s = ", ".join(cfg.enable) if cfg.enable else "(none)"
    disable_s = ", ".join(cfg.disable) if cfg.disable else "(none)"
    click.echo(f"enable=[{enable_s}]  disable=[{disable_s}]")
    rows = list_plugin_statuses(enable=cfg.enable, disable=cfg.disable)
    if not rows:
        click.echo("(no plyngent.tools entry points installed)")
        return
    click.echo("id\tstatus\tpackage\tvalue")
    for row in rows:
        if row.disabled:
            status = "disabled"
        elif row.enabled:
            status = "enabled"
        else:
            status = "off"
        pkg = row.plugin.package or "-"
        if row.plugin.version:
            pkg = f"{pkg}@{row.plugin.version}"
        click.echo(f"{row.plugin.id}\t{status}\t{pkg}\t{row.plugin.value}")


@main.group("plugins")
def plugins_group() -> None:
    """Manage third-party plugins (``[plugins]`` allowlist in config)."""


@plugins_group.command("list")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml.",
)
def plugins_list_cmd(config_path: Path | None) -> None:
    """List installed ``plyngent.tools`` entry points and enable/disable status."""
    store = _load_config(config_path, require_providers=False)
    print_plugins_table(store)


@plugins_group.command("enable")
@click.argument("name")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml.",
)
def plugins_enable_cmd(name: str, config_path: Path | None) -> None:
    """Allowlist a plugin entry-point name (or ``*`` for all) and write config.

    Removes the name from ``[plugins].disable`` if present. Takes effect for
    new chat sessions (registry is built at chat start).
    """
    store = _load_config(config_path, require_providers=False)
    try:
        _ = store.enable_plugin(name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    store.write()
    click.echo(f"enabled {name.strip()!r} in {store.path}")
    enable_s = ", ".join(store.plugins_config.enable) or "(none)"
    disable_s = ", ".join(store.plugins_config.disable) or "(none)"
    click.echo(f"enable=[{enable_s}]  disable=[{disable_s}]")


@plugins_group.command("disable")
@click.argument("name")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml.",
)
def plugins_disable_cmd(name: str, config_path: Path | None) -> None:
    """Block a plugin via ``[plugins].disable`` and write config.

    Disable always wins over enable / ``*``. Use ``plugins undeny`` to drop the
    block without adding the plugin to enable.
    """
    store = _load_config(config_path, require_providers=False)
    try:
        _ = store.disable_plugin(name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    store.write()
    click.echo(f"disabled {name.strip()!r} in {store.path}")
    enable_s = ", ".join(store.plugins_config.enable) or "(none)"
    disable_s = ", ".join(store.plugins_config.disable) or "(none)"
    click.echo(f"enable=[{enable_s}]  disable=[{disable_s}]")


@plugins_group.command("undeny")
@click.argument("name")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml.",
)
def plugins_undeny_cmd(name: str, config_path: Path | None) -> None:
    """Remove *name* from ``[plugins].disable`` only (does not enable it)."""
    store = _load_config(config_path, require_providers=False)
    try:
        _ = store.undeny_plugin(name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    store.write()
    click.echo(f"undenied {name.strip()!r} in {store.path}")
    disable_s = ", ".join(store.plugins_config.disable) or "(none)"
    click.echo(f"disable=[{disable_s}]")


@plugins_group.command("clear")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml.",
)
@click.option(
    "--yes",
    "confirm_yes",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def plugins_clear_cmd(config_path: Path | None, *, confirm_yes: bool) -> None:
    """Clear ``[plugins].enable`` and ``disable`` (load no plugins)."""
    store = _load_config(config_path, require_providers=False)
    needs_confirm = not confirm_yes and (store.plugins_config.enable or store.plugins_config.disable)
    if needs_confirm and not click.confirm("Clear all plugin enable/disable entries?", default=False):
        raise click.Abort
    _ = store.clear_plugins()
    store.write()
    click.echo(f"cleared plugins lists in {store.path}")


@main.group("config")
def config_group() -> None:
    """Manage plyngent configuration."""


@config_group.command("path")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override config path (prints the path that would be used).",
)
def config_path_cmd(config_path: Path | None) -> None:
    """Print the resolved config file path."""
    click.echo(str(resolve_config_path(config_path)))


@config_group.command("edit")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to plyngent.toml (default: platform config dir).",
)
def config_edit_cmd(config_path: Path | None) -> None:
    """Open the config file in $VISUAL/$EDITOR, or system default if unset.

    Blocking editors (e.g. ``codium --wait``) wait for exit. Without VISUAL/EDITOR,
    falls back to xdg-open / open / startfile (non-blocking).
    """
    path = resolve_config_path(config_path)
    outcome = open_in_editor(path, allow_system_open=True)
    if outcome == "system":
        click.secho(
            f"opened {path} with system default (not waiting for the app to exit)",
            fg="yellow",
            err=True,
        )
    else:
        click.echo(f"edited {path}")


if __name__ == "__main__":
    main()
