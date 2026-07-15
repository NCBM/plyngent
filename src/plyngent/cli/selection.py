from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.cli.models_source import config_model_ids, model_choices_for_provider
from plyngent.prompting import ChoiceOption, choose

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    choices: Sequence[str] | None = None,
) -> str:
    """Pick a model id from config/remote choices or free-form prompt.

    *choices* overrides the default list (config keys). Explicit *preferred*
    is accepted even when not in the list (API validates at chat time).
    """
    model_names = list(choices) if choices is not None else config_model_ids(provider)
    if preferred is not None:
        token = preferred.strip()
        if not token:
            msg = "model id must not be empty"
            raise click.ClickException(msg)
        return token

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
            allow_custom=True,
        )

    from plyngent.prompting import ask

    return ask("Model id (not listed in config)")


def default_model_choices(provider: Provider, remote_ids: Sequence[str] | None = None) -> list[str]:
    """Config plus optional remote ids (for Tab / interactive pick)."""
    return model_choices_for_provider(provider, remote_ids=remote_ids)
