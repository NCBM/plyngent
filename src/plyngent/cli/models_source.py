from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from plyngent.config.models import Provider

# Cache remote catalog this long (seconds) unless /models --refresh.
DEFAULT_MODELS_CACHE_TTL = 300.0


@runtime_checkable
class SupportsModels(Protocol):
    async def models(self) -> list[str]: ...


def config_model_ids(provider: Provider) -> list[str]:
    """Sorted model ids declared in provider config."""
    return sorted(provider.models.keys())


def merge_model_choices(
    config_ids: Iterable[str],
    remote_ids: Iterable[str] | None = None,
) -> list[str]:
    """Union config and remote ids (sorted, unique)."""
    merged: set[str] = {i for i in config_ids if i}
    if remote_ids is not None:
        merged.update(i for i in remote_ids if i)
    return sorted(merged)


def client_supports_models(client: object) -> bool:
    """True when *client* exposes OpenAI-compatible ``models()``."""
    return isinstance(client, SupportsModels) or callable(getattr(client, "models", None))


async def fetch_remote_model_ids(client: object) -> list[str]:
    """Call ``client.models()``; raise if missing or the call fails."""
    method = getattr(client, "models", None)
    if not callable(method):
        msg = "client does not support listing models"
        raise TypeError(msg)
    result = method()
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, list):
        msg = f"models() returned unexpected type {type(result)!r}"
        raise TypeError(msg)
    return [str(item) for item in cast("list[object]", result) if item]


def model_choices_for_provider(
    provider: Provider,
    *,
    remote_ids: Sequence[str] | None = None,
) -> list[str]:
    """Config plus remote catalog for selection / Tab complete."""
    return merge_model_choices(config_model_ids(provider), remote_ids)
