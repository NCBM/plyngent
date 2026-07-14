from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import msgspec
import tomlkit

from .models import DatabaseConfig, Provider

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class ConfigFormatError(ValueError):
    """Raised when the config file contains invalid TOML."""


def _parse_database(raw: dict[str, object]) -> DatabaseConfig:
    """Parse the ``[database]`` section, falling back to defaults."""
    try:
        return msgspec.convert(raw, DatabaseConfig)
    except msgspec.ValidationError:
        return DatabaseConfig()


def _parse_providers(
    document: tomlkit.TOMLDocument,
) -> tuple[dict[str, Provider], dict[str, object]]:
    """Parse provider entries from the document.

    Returns:
        (providers, bad_providers) — valid and invalid entries respectively.
    """
    providers: dict[str, Provider] = {}
    bad_providers: dict[str, object] = {}

    raw: dict[str, object] = document.unwrap()
    providers_raw: object = raw.get("providers", {})
    if not isinstance(providers_raw, dict):
        return providers, bad_providers

    for name, raw_entry in cast("dict[str, object]", providers_raw).items():
        if not isinstance(raw_entry, dict):
            bad_providers[name] = raw_entry
            continue

        # Try tagged-union dispatch via msgspec
        try:
            provider: Provider = msgspec.convert(raw_entry, Provider)
        except msgspec.ValidationError:
            bad_providers[name] = raw_entry
            continue

        # msgspec silently ignores unknown fields — reject them manually.
        # The tag_field ("preset") is not a struct field, so we add it back.
        cls = type(provider)
        known = set(cls.__struct_fields__)
        known.add("preset")  # tag_field, excluded from __struct_fields__
        if set(cast("dict[str, object]", raw_entry).keys()) - known:
            bad_providers[name] = raw_entry
            continue

        providers[name] = provider

    return providers, bad_providers


class ConfigStore:
    """Stateful wrapper around a TOML config file."""

    _path: Path
    _document: tomlkit.TOMLDocument
    _database: DatabaseConfig
    _providers: dict[str, Provider]
    _bad_providers: dict[str, object]

    def __init__(self, path: Path, document: tomlkit.TOMLDocument) -> None:
        self._path = path
        self._document = document
        raw: dict[str, object] = document.unwrap()
        self._database = _parse_database(
            cast("dict[str, object]", raw.get("database", {}))
        )
        self._providers, self._bad_providers = _parse_providers(document)

    # -- database (read-only) --

    @property
    def database(self) -> MappingProxyType[str, object]:
        """Read-only mapping view of database configuration."""
        return MappingProxyType(msgspec.structs.asdict(self._database))

    # -- providers (read/write) --

    @property
    def providers(self) -> MappingProxyType[str, Provider]:
        """Read-only mapping view of recognised providers.

        Supports ``|`` for augmented-assignment merge.
        """
        return MappingProxyType(self._providers)

    @providers.setter
    def providers(self, value: Mapping[str, Provider]) -> None:  # pyright: ignore[reportPropertyTypeMismatch]
        """Replace all providers."""
        self._providers = dict(value)
        self._bad_providers = {}

    # -- bad_providers (read-only) --

    @property
    def bad_providers(self) -> MappingProxyType[str, object]:
        """Read-only view of unrecognised / malformed provider entries."""
        return MappingProxyType(self._bad_providers)

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
        self._database = _parse_database(
            cast("dict[str, object]", raw.get("database", {}))
        )
        self._providers, self._bad_providers = _parse_providers(self._document)

    # -- internal sync helpers --

    def _sync_database_section(self) -> None:
        """Sync ``[database]`` to the document."""
        raw_db: dict[str, object] = msgspec.to_builtins(self._database)
        if not raw_db:
            if "database" in self._document:
                del self._document["database"]
            return

        if "database" not in self._document:
            self._document["database"] = tomlkit.table()
        section = self._document["database"]

        for k in list(section.keys()):  # type: ignore[union-attr]
            if k not in raw_db:
                del section[k]  # type: ignore[union-attr]
        for k, v in raw_db.items():
            section[k] = v  # type: ignore[union-attr]

    def _sync_providers_section(self) -> None:
        """Sync ``[providers]`` to the document."""
        if "providers" not in self._document:
            self._document["providers"] = tomlkit.table()
        section = self._document["providers"]

        for name in list(section.keys()):  # type: ignore[union-attr]
            if name not in self._providers:
                del section[name]  # type: ignore[union-attr]

        for name, provider in self._providers.items():
            raw: dict[str, object] = msgspec.to_builtins(provider)
            if name in section:
                entry = section[name]  # type: ignore[union-attr]
                entry.clear()  # type: ignore[union-attr]
                for k, v in raw.items():
                    entry[k] = v  # type: ignore[union-attr]
            else:
                section[name] = raw  # type: ignore[union-attr]

    def _sync_to_document(self) -> None:
        """Incrementally sync all sections into the document."""
        self._sync_database_section()
        self._sync_providers_section()
