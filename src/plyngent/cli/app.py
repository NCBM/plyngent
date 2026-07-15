from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import click
import msgspec
from platformdirs import user_data_path

from plyngent.agent.loop import DEFAULT_MAX_ROUNDS
from plyngent.cli.editor import (
    load_config_with_optional_edit,
    open_in_editor,
    resolve_config_path,
)
from plyngent.cli.limits import install_cli_limit_hooks
from plyngent.cli.repl import run_repl
from plyngent.cli.selection import select_model, select_provider
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig
from plyngent.memory import MemoryStore
from plyngent.runtime import ProviderNotSupportedError, create_client
from plyngent.tools import set_workspace_root

if TYPE_CHECKING:
    from plyngent.config.store import ConfigStore

_DEFAULT_DB_FILENAME = "chat.db"


def _load_config(config_path: Path | None) -> ConfigStore:
    return load_config_with_optional_edit(config_path)


def _database_config(store: ConfigStore) -> DatabaseConfig:
    raw = dict(store.database)
    # Prefer a durable file DB so sessions survive CLI restarts.
    if raw.get("url") in {None, "", ":memory:"} and raw.get("implementation", "sqlite") == "sqlite":
        db_path = user_data_path("plyngent", ensure_exists=True) / _DEFAULT_DB_FILENAME
        raw = {**raw, "implementation": "sqlite", "url": str(db_path)}
        click.secho(f"using database: {db_path}", fg="bright_black")
    return msgspec.convert(raw, DatabaseConfig)


async def _run_chat(
    *,
    config_path: Path | None,
    provider_name: str | None,
    model: str | None,
    tools: bool,
    workspace: Path,
    session_id: int | None,
    max_rounds: int,
    new_session: bool,
) -> None:
    store = _load_config(config_path)
    if store.bad_providers:
        names = ", ".join(sorted(store.bad_providers.keys()))
        click.secho(f"warning: ignored bad providers: {names}", fg="yellow")

    try:
        pname, provider = select_provider(store.providers, preferred=provider_name)
        model_id = select_model(provider, preferred=model)
        _ = create_client(provider)  # fail early if unsupported
    except ProviderNotSupportedError as exc:
        raise click.ClickException(str(exc)) from exc

    _ = set_workspace_root(workspace)
    from plyngent.tools import set_path_denylist

    set_path_denylist(store.agent_config.path_denylist or None)
    install_cli_limit_hooks()
    memory = await MemoryStore.open(_database_config(store))
    try:
        state = ReplState(
            config=store,
            memory=memory,
            workspace=workspace,
            provider_name=pname,
            provider=provider,
            model=model_id,
            tools_enabled=tools,
            max_rounds=max_rounds,
        )
        click.secho(f"workspace: {state.workspace}", fg="bright_black")
        if session_id is not None:
            try:
                await state.resume_session(session_id)
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
            click.echo(
                f"resumed session {session_id} ({len(state.agent.messages)} messages) workspace={state.workspace}"
            )
        elif new_session:
            await state.new_session()
            click.echo(f"new session {state.session_id} (workspace={state.workspace})")
        else:
            mode = await state.resume_latest_or_new()
            if mode == "resume":
                click.echo(
                    f"resumed latest session {state.session_id} for this workspace "
                    f"({len(state.agent.messages)} messages); use --new for a fresh chat"
                )
            else:
                click.echo(f"new session {state.session_id} (workspace={state.workspace})")
        await run_repl(state)
    finally:
        await memory.close()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="plyngent")
def main() -> None:
    """Plyngent — LLM chat and agent toolkit."""


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
@click.option("--tools/--no-tools", default=True, show_default=True, help="Enable DEFAULT_TOOLS.")
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
def chat_cmd(
    config_path: Path | None,
    provider_name: str | None,
    model: str | None,
    tools: bool,  # noqa: FBT001
    workspace: Path | None,
    session_id: int | None,
    new_session: bool,  # noqa: FBT001
    max_rounds: int,
) -> None:
    """Interactive chat REPL with optional tools and session memory."""
    if max_rounds < 1:
        msg = "--max-rounds must be >= 1"
        raise click.ClickException(msg)
    if session_id is not None and new_session:
        msg = "use either --session or --new, not both"
        raise click.ClickException(msg)
    root = workspace if workspace is not None else Path.cwd()
    asyncio.run(
        _run_chat(
            config_path=config_path,
            provider_name=provider_name,
            model=model,
            tools=tools,
            workspace=root,
            session_id=session_id,
            max_rounds=max_rounds,
            new_session=new_session,
        )
    )


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
    if not store.providers:
        click.echo("(no providers)")
    for name, provider in sorted(store.providers.items()):
        tag = type(provider).__struct_config__.tag
        models = ", ".join(sorted(provider.models.keys())) or "(none listed)"
        click.echo(f"{name}\tpreset={tag}\tmodels={models}")
    if store.bad_providers:
        click.secho(f"bad: {', '.join(sorted(store.bad_providers.keys()))}", fg="yellow")


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
    """Open the config file in $EDITOR (supports e.g. ``codium --wait``)."""
    path = resolve_config_path(config_path)
    open_in_editor(path)
    click.echo(f"edited {path}")


if __name__ == "__main__":
    main()
