from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import msgspec
import tomlkit

from .models import AgentConfig, DatabaseConfig, ModelConfig, Provider

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence
    from pathlib import Path


class ConfigFormatError(ValueError):
    """Raised when the config file contains invalid TOML."""


def _parse_database(raw: dict[str, object]) -> DatabaseConfig:
    """Parse the ``[database]`` section, falling back to defaults."""
    try:
        return msgspec.convert(raw, DatabaseConfig)
    except msgspec.ValidationError:
        return DatabaseConfig()


def _parse_agent(raw: dict[str, object]) -> AgentConfig:
    """Parse the ``[agent]`` section, falling back to defaults."""
    try:
        return msgspec.convert(raw, AgentConfig)
    except msgspec.ValidationError:
        return AgentConfig()


def _parse_providers(
    document: tomlkit.TOMLDocument,
) -> tuple[dict[str, Provider], dict[str, object], dict[str, Provider]]:
    """Parse provider entries from the document.

    Returns:
        (providers, bad_providers, recoverable_providers)

        *providers* — ready to use (non-empty ``models``).
        *bad_providers* — unparseable / unknown fields.
        *recoverable_providers* — parsed OK but ``models`` empty; may be
        promoted after a successful remote ``GET /models`` (or explicit model id).
    """
    providers: dict[str, Provider] = {}
    bad_providers: dict[str, object] = {}
    recoverable: dict[str, Provider] = {}

    raw: dict[str, object] = document.unwrap()
    providers_raw: object = raw.get("providers", {})
    if not isinstance(providers_raw, dict):
        return providers, bad_providers, recoverable

    for name, raw_entry in cast("dict[str, object]", providers_raw).items():
        if not isinstance(raw_entry, dict):
            bad_providers[name] = raw_entry
            continue

        entry = cast("dict[str, object]", raw_entry)
        # Missing preset → OpenAI platform (Responses-capable defaults).
        if "preset" not in entry:
            entry = {**entry, "preset": "openai"}

        # Try tagged-union dispatch via msgspec
        try:
            provider: Provider = msgspec.convert(entry, Provider)
        except msgspec.ValidationError:
            bad_providers[name] = raw_entry
            continue

        # msgspec silently ignores unknown fields — reject them manually.
        # The tag_field ("preset") is not a struct field, so we add it back.
        cls = type(provider)
        known = set(cls.__struct_fields__)
        known.add("preset")  # tag_field, excluded from __struct_fields__
        # Validate against the entry we actually converted (may have injected preset).
        if set(entry.keys()) - known:
            bad_providers[name] = raw_entry
            continue

        # Empty models: recoverable. OpenAI/DeepSeek seed defaults when
        # ``models`` is omitted; only explicit models={} lands here.
        if not provider.models:
            recoverable[name] = provider
            continue

        providers[name] = provider

    return providers, bad_providers, recoverable


class ConfigStore:
    """Stateful wrapper around a TOML config file."""

    _path: Path
    _document: tomlkit.TOMLDocument
    _database: DatabaseConfig
    _agent: AgentConfig
    _providers: dict[str, Provider]
    _bad_providers: dict[str, object]
    _recoverable_providers: dict[str, Provider]

    def __init__(self, path: Path, document: tomlkit.TOMLDocument) -> None:
        self._path = path
        self._document = document
        raw: dict[str, object] = document.unwrap()
        self._database = _parse_database(cast("dict[str, object]", raw.get("database", {})))
        self._agent = _parse_agent(cast("dict[str, object]", raw.get("agent", {})))
        self._providers, self._bad_providers, self._recoverable_providers = _parse_providers(document)

    @property
    def path(self) -> Path:
        """Filesystem path of the TOML config file."""
        return self._path

    # -- database (read-only) --

    @property
    def database(self) -> MappingProxyType[str, object]:
        """Read-only mapping view of database configuration."""
        return MappingProxyType(msgspec.structs.asdict(self._database))

    # -- agent (read-only) --

    @property
    def agent(self) -> MappingProxyType[str, object]:
        """Read-only mapping view of agent profile configuration."""
        return MappingProxyType(msgspec.structs.asdict(self._agent))

    @property
    def agent_config(self) -> AgentConfig:
        """Typed agent profile (system prompt, tool budgets, etc.)."""
        return self._agent

    # -- providers (read/write) --

    @property
    def providers(self) -> MappingProxyType[str, Provider]:
        """Read-only mapping view of recognised providers.

        Supports ``|`` for augmented-assignment merge.
        """
        return MappingProxyType(self._providers)

    @providers.setter
    def providers(self, value: Mapping[str, Provider]) -> None:  # pyright: ignore[reportPropertyTypeMismatch]
        """Replace all ready providers (clears recoverable/bad)."""
        self._providers = dict(value)
        self._bad_providers = {}
        self._recoverable_providers = {}

    # -- bad_providers (read-only) --

    @property
    def bad_providers(self) -> MappingProxyType[str, object]:
        """Read-only view of unrecognised / malformed provider entries."""
        return MappingProxyType(self._bad_providers)

    @property
    def recoverable_providers(self) -> MappingProxyType[str, Provider]:
        """Parsed providers with empty ``models`` (recoverable via remote list)."""
        return MappingProxyType(self._recoverable_providers)

    def selectable_providers(self) -> dict[str, Provider]:
        """Ready providers plus recoverable ones (ready wins on name clash)."""
        return {**self._recoverable_providers, **self._providers}

    def get_provider(self, name: str) -> Provider | None:
        """Look up a ready or recoverable provider by config name."""
        if name in self._providers:
            return self._providers[name]
        return self._recoverable_providers.get(name)

    def promote_provider(self, name: str, model_ids: Sequence[str]) -> Provider:
        """Seed ``models`` from *model_ids* and move recoverable → ready.

        Also re-seeds an already-ready provider that somehow has empty models.
        Does not write the TOML file (in-memory session only unless ``write()``).
        """
        ids = [mid.strip() for mid in model_ids if mid and mid.strip()]
        if not ids:
            msg = f"cannot promote provider {name!r}: no model ids"
            raise ValueError(msg)

        if name in self._providers:
            provider = self._providers[name]
        elif name in self._recoverable_providers:
            provider = self._recoverable_providers.pop(name)
        else:
            msg = f"unknown provider {name!r}"
            raise KeyError(msg)

        models = {mid: ModelConfig() for mid in ids}
        promoted = msgspec.structs.replace(provider, models=models)
        self._providers[name] = promoted
        _ = self._bad_providers.pop(name, None)
        return promoted

    # -- persistence --

    def write(self) -> None:
        """Serialize current state to the TOML file."""
        self._sync_to_document()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            tomlkit.dump(self._document, f)

    def reload(self) -> None:
        """Re-read the TOML file and re-parse all sections."""
        with self._path.open() as f:
            self._document = tomlkit.parse(f.read())
        raw: dict[str, object] = self._document.unwrap()
        self._database = _parse_database(cast("dict[str, object]", raw.get("database", {})))
        self._agent = _parse_agent(cast("dict[str, object]", raw.get("agent", {})))
        self._providers, self._bad_providers, self._recoverable_providers = _parse_providers(self._document)

    # -- internal sync helpers --

    def _toml_table(self, key: str) -> MutableMapping[str, object]:
        """Return a mutable table for ``key``, creating it if missing."""
        if key not in self._document:
            self._document[key] = tomlkit.table()
        return cast("MutableMapping[str, object]", self._document[key])

    def _sync_section(self, key: str, data: object) -> None:
        raw: dict[str, object] = msgspec.to_builtins(data)
        if not raw:
            if key in self._document:
                del self._document[key]
            return
        section = self._toml_table(key)
        for k in list(section.keys()):
            if k not in raw:
                del section[k]
        for k, v in raw.items():
            section[k] = v

    def _sync_database_section(self) -> None:
        """Sync ``[database]`` to the document."""
        self._sync_section("database", self._database)

    def _sync_agent_section(self) -> None:
        """Sync ``[agent]`` to the document."""
        self._sync_section("agent", self._agent)

    def _sync_providers_section(self) -> None:
        """Sync ``[providers]`` to the document (ready + recoverable)."""
        section = self._toml_table("providers")
        keep = set(self._providers) | set(self._recoverable_providers)

        for name in list(section.keys()):
            if name not in keep:
                del section[name]

        for name, provider in {**self._recoverable_providers, **self._providers}.items():
            raw: dict[str, object] = msgspec.to_builtins(provider)
            if name in section:
                entry = cast("MutableMapping[str, object]", section[name])
                entry.clear()
                for k, v in raw.items():
                    entry[k] = v
            else:
                section[name] = raw

    def _sync_to_document(self) -> None:
        """Incrementally sync all sections into the document."""
        self._sync_database_section()
        self._sync_agent_section()
        self._sync_providers_section()
