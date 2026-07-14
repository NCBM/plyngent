from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

if TYPE_CHECKING:
    from plyngent.config.models import DatabaseConfig


class UnsupportedDatabaseError(NotImplementedError):
    """Raised when the configured database implementation is not supported."""


def build_async_url(config: DatabaseConfig) -> str:
    """Build a SQLAlchemy async database URL from :class:`DatabaseConfig`."""
    implementation = config.implementation.lower()
    if implementation != "sqlite":
        msg = f"database implementation {config.implementation!r} is not supported"
        raise UnsupportedDatabaseError(msg)

    url = config.url
    if url in {":memory:", ""}:
        return "sqlite+aiosqlite:///:memory:"
    if url.startswith("sqlite+aiosqlite://"):
        return url
    if url.startswith("sqlite://"):
        return "sqlite+aiosqlite://" + url.removeprefix("sqlite://")
    # File path (relative or absolute)
    return f"sqlite+aiosqlite:///{url}"


def create_engine(config: DatabaseConfig, *, echo: bool = False) -> AsyncEngine:
    """Create an async SQLAlchemy engine for the given database config."""
    return create_async_engine(build_async_url(config), echo=echo)
