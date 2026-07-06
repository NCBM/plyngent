from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import msgspec
import tomlkit

from .models import Provider

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class ConfigFormatError(ValueError):
    """Raised when the config file contains invalid TOML."""


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
    _providers: dict[str, Provider]
    _bad_providers: dict[str, object]

    def __init__(self, path: Path, document: tomlkit.TOMLDocument) -> None:
        self._path = path
        self._document = document
        self._providers, self._bad_providers = _parse_providers(document)

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
        """Serialize current providers to the TOML file."""
        self._sync_to_document()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            tomlkit.dump(self._document, f)

    def reload(self) -> None:
        """Re-read the TOML file and re-parse providers."""
        with self._path.open() as f:
            self._document = tomlkit.parse(f.read())
        self._providers, self._bad_providers = _parse_providers(self._document)

    # -- internal --

    def _sync_to_document(self) -> None:
        """Incrementally sync ``self._providers`` into the ``[providers]`` section.

        Only adds, updates, or removes individual provider entries — the rest of
        the document (comments, formatting, other top-level sections) is untouched.
        """
        if "providers" not in self._document:
            self._document["providers"] = tomlkit.table()
        section = self._document["providers"]

        # Remove deleted providers
        for name in list(section.keys()):  # type: ignore[union-attr]
            if name not in self._providers:
                del section[name]  # type: ignore[union-attr]

        # Add / update providers
        for name, provider in self._providers.items():
            raw: dict[str, object] = msgspec.to_builtins(provider)
            if name in section:
                entry = section[name]  # type: ignore[union-attr]
                # Clear and rebuild the existing entry to match current state
                entry.clear()  # type: ignore[union-attr]
                for k, v in raw.items():
                    entry[k] = v  # type: ignore[union-attr]
            else:
                section[name] = raw  # type: ignore[union-attr]
