from __future__ import annotations

from typing import TYPE_CHECKING

from platformdirs import user_config_path

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG_FILE_NAME: str = "plyngent.toml"


def get_default_path() -> Path:
    """Return the default config file path using the platform-standard config directory.

    On Linux, this is ``~/.config/plyngent/plyngent.toml``.
    ``ensure_exists=True`` creates the directory if it does not exist.
    """
    return user_config_path("plyngent", ensure_exists=True) / _CONFIG_FILE_NAME
