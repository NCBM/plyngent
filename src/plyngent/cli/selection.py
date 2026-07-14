from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.config.models import Provider


def select_provider(
    providers: Mapping[str, Provider],
    *,
    preferred: str | None = None,
) -> tuple[str, Provider]:
    """Pick a provider by name or interactive prompt."""
    if not providers:
        msg = "no providers configured; edit your plyngent.toml"
        raise click.ClickException(msg)

    names = sorted(providers.keys())
    if preferred is not None:
        if preferred not in providers:
            msg = f"unknown provider {preferred!r}; available: {', '.join(names)}"
            raise click.ClickException(msg)
        return preferred, providers[preferred]

    if len(names) == 1:
        name = names[0]
        click.echo(f"Using provider: {name}")
        return name, providers[name]

    click.echo("Available providers:")
    for index, name in enumerate(names, start=1):
        preset = type(providers[name]).__struct_config__.tag
        click.echo(f"  {index}. {name} ({preset})")
    choice = click.prompt("Select provider", type=click.Choice(names), show_choices=True)
    return choice, providers[choice]


def select_model(
    provider: Provider,
    *,
    preferred: str | None = None,
) -> str:
    """Pick a model id from provider.models or free-form prompt."""
    model_names = sorted(provider.models.keys())
    if preferred is not None:
        if model_names and preferred not in provider.models:
            msg = f"unknown model {preferred!r}; available: {', '.join(model_names)}"
            raise click.ClickException(msg)
        return preferred

    if len(model_names) == 1:
        model = model_names[0]
        click.echo(f"Using model: {model}")
        return model

    if model_names:
        click.echo("Available models:")
        for index, name in enumerate(model_names, start=1):
            click.echo(f"  {index}. {name}")
        return click.prompt("Select model", type=click.Choice(model_names), show_choices=True)

    return click.prompt("Model id (not listed in config)", type=str)
