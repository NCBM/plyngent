from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from plyngent.config.models import Provider

# Cache remote catalog this long (seconds) unless /models --refresh.
DEFAULT_MODELS_CACHE_TTL = 300.0

type ModelListPrefer = Literal["remote", "union", "config"]


@runtime_checkable
class SupportsModels(Protocol):
    async def models(self) -> list[str]: ...


def config_model_ids(provider: Provider) -> list[str]:
    """Sorted model ids declared in provider config."""
    return sorted(provider.models.keys())


def merge_model_choices(
    config_ids: Iterable[str],
    remote_ids: Iterable[str] | None = None,
    *,
    prefer: ModelListPrefer = "remote",
) -> list[str]:
    """Merge config and remote model ids.

    *prefer*:
    - ``remote`` (default): remote catalog first (sorted), then config-only ids
    - ``union``: sorted unique union
    - ``config``: config first, then remote-only ids
    """
    config_list = [i for i in config_ids if i]
    remote_list = [i for i in (remote_ids or ()) if i]
    if not remote_list:
        return sorted(set(config_list))
    if prefer == "union":
        return sorted(set(config_list) | set(remote_list))
    remote_sorted = sorted(set(remote_list))
    config_only = sorted(set(config_list) - set(remote_sorted))
    if prefer == "remote":
        return [*remote_sorted, *config_only]
    # config first
    config_sorted = sorted(set(config_list))
    remote_only = sorted(set(remote_list) - set(config_sorted))
    return [*config_sorted, *remote_only]


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
    prefer: ModelListPrefer = "remote",
) -> list[str]:
    """Config plus remote catalog for selection / Tab complete (remote-first)."""
    return merge_model_choices(config_model_ids(provider), remote_ids, prefer=prefer)
