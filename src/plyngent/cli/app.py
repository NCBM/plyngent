from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import click
import msgspec

from plyngent import config as config_mod
from plyngent.cli.repl import run_repl
from plyngent.cli.selection import select_model, select_provider
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig
from plyngent.memory import MemoryStore
from plyngent.runtime import ProviderNotSupportedError, create_client
from plyngent.tools import set_workspace_root

if TYPE_CHECKING:
    from plyngent.config.store import ConfigStore


def _load_config(config_path: Path | None) -> ConfigStore:
    return config_mod.load(config_path)


def _database_config(store: ConfigStore) -> DatabaseConfig:
    return msgspec.convert(dict(store.database), DatabaseConfig)


async def _run_chat(  # noqa: PLR0913
    *,
    config_path: Path | None,
    provider_name: str | None,
    model: str | None,
    tools: bool,
    workspace: Path,
    session_id: int | None,
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
        )
        if session_id is not None:
            await state.resume_session(session_id)
        else:
            await state.new_session()
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
def chat_cmd(  # noqa: PLR0913
    config_path: Path | None,
    provider_name: str | None,
    model: str | None,
    tools: bool,  # noqa: FBT001
    workspace: Path | None,
    session_id: int | None,
) -> None:
    """Interactive chat REPL with optional tools and session memory."""
    root = workspace if workspace is not None else Path.cwd()
    asyncio.run(
        _run_chat(
            config_path=config_path,
            provider_name=provider_name,
            model=model,
            tools=tools,
            workspace=root,
            session_id=session_id,
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


if __name__ == "__main__":
    main()
