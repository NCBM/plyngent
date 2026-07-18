from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.cli.models_source import fetch_remote_model_ids
from plyngent.runtime import ProviderNotSupportedError, create_client

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.config.models import Provider
    from plyngent.config.store import ConfigStore


async def discover_model_ids(provider: Provider) -> list[str]:
    """``GET /models`` for *provider*; empty list if unsupported or empty catalog."""
    try:
        client = create_client(provider)
    except ProviderNotSupportedError:
        return []
    try:
        return await fetch_remote_model_ids(client)
    except RuntimeError, TypeError, OSError, ValueError, TimeoutError:
        return []


async def try_promote_provider(
    store: ConfigStore,
    name: str,
    *,
    seed_model_ids: Sequence[str] | None = None,
) -> Provider | None:
    """Promote recoverable *name* into ready providers.

    Prefers *seed_model_ids* when non-empty; otherwise remote ``models()``.
    Returns the promoted provider, or ``None`` if recovery failed.
    """
    if name in store.providers:
        return store.providers[name]
    if name not in store.recoverable_providers:
        return None

    provider = store.recoverable_providers[name]
    ids: list[str] = []
    if seed_model_ids:
        ids = [mid.strip() for mid in seed_model_ids if mid and str(mid).strip()]
    if not ids:
        ids = await discover_model_ids(provider)
    if not ids:
        return None
    return store.promote_provider(name, ids)


async def ensure_provider_ready(
    store: ConfigStore,
    name: str,
    provider: Provider,
    *,
    preferred_model: str | None = None,
    interactive: bool = True,
) -> Provider:
    """Return a ready provider; recover empty-models entries when possible.

    Raises:
        click.ClickException: When recovery is required but fails.
    """
    if provider.models:
        return provider
    if name not in store.recoverable_providers and name not in store.providers:
        msg = f"provider {name!r} has no models and is not recoverable"
        raise click.ClickException(msg)

    seeds: list[str] = []
    if preferred_model and preferred_model.strip():
        seeds = [preferred_model.strip()]

    promoted = await try_promote_provider(store, name, seed_model_ids=seeds or None)
    if promoted is not None:
        click.secho(
            f"recovered provider {name!r} with {len(promoted.models)} model(s) (was empty models in config)",
            fg="yellow",
            err=True,
        )
        return promoted

    # Remote failed and no preferred seed: interactive free-form model id.
    if interactive and not seeds:
        from plyngent.prompting import ask

        mid = ask(f"Model id for provider {name!r} (empty models; remote list failed)")
        if mid.strip():
            promoted = store.promote_provider(name, [mid.strip()])
            click.secho(
                f"recovered provider {name!r} with model {mid.strip()!r}",
                fg="yellow",
                err=True,
            )
            return promoted

    if seeds:
        # Prefer explicit model even when remote list failed.
        promoted = store.promote_provider(name, seeds)
        click.secho(
            f"recovered provider {name!r} with model {seeds[0]!r}",
            fg="yellow",
            err=True,
        )
        return promoted

    msg = f"provider {name!r} has empty models and could not be recovered (pass --model or fix GET /models)"
    raise click.ClickException(msg)
