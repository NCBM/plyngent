from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.prompting import ChoiceOption, choose

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.config.models import Provider


def select_provider(
    providers: Mapping[str, Provider],
    *,
    preferred: str | None = None,
    interactive: bool = True,
) -> tuple[str, Provider]:
    """Pick a provider by name or interactive prompt (readline + Tab)."""
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
        click.echo(f"Using provider: {name}", err=True)
        return name, providers[name]

    if not interactive:
        msg = f"multiple providers; pass --provider ({', '.join(names)})"
        raise click.ClickException(msg)

    options = [
        ChoiceOption(
            label=name,
            description=str(type(providers[name]).__struct_config__.tag),
            value=name,
        )
        for name in names
    ]
    choice = choose("Select provider", options, allow_custom=False)
    return choice, providers[choice]


def select_model(
    provider: Provider,
    *,
    preferred: str | None = None,
    interactive: bool = True,
) -> str:
    """Pick a model id from provider.models or free-form prompt (readline + Tab)."""
    model_names = sorted(provider.models.keys())
    if preferred is not None:
        if model_names and preferred not in provider.models:
            msg = f"unknown model {preferred!r}; available: {', '.join(model_names)}"
            raise click.ClickException(msg)
        return preferred

    if len(model_names) == 1:
        model = model_names[0]
        click.echo(f"Using model: {model}", err=True)
        return model

    if not interactive:
        if model_names:
            msg = f"multiple models; pass --model ({', '.join(model_names)})"
        else:
            msg = "no models listed; pass --model"
        raise click.ClickException(msg)

    if model_names:
        return choose(
            "Select model",
            model_names,
            allow_custom=False,
        )

    from plyngent.prompting import ask

    return ask("Model id (not listed in config)")
