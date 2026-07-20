from __future__ import annotations

from typing import TYPE_CHECKING

import tomlkit
from tomlkit.exceptions import TOMLKitError

from .models import DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from .models import AgentConfig as AgentConfig
from .models import AnthropicProvider as AnthropicProvider
from .models import DatabaseConfig as DatabaseConfig
from .models import DeepseekProvider as DeepseekProvider
from .models import HttpTimeoutConfig as HttpTimeoutConfig
from .models import ModelConfig as ModelConfig
from .models import OpenAICompatibleProvider as OpenAICompatibleProvider
from .models import OpenAIProvider as OpenAIProvider
from .models import ProviderConfig as ProviderConfig
from .path import get_default_path as _get_default_path
from .store import ConfigFormatError as ConfigFormatError
from .store import ConfigStore as ConfigStore

if TYPE_CHECKING:
    from pathlib import Path

default_config_source: Path = _get_default_path()


def load(path: Path | None = None) -> ConfigStore:
    """Load configuration from a TOML file.

    Args:
        path: Path to the config file. If ``None``, uses :data:`default_config_source`.

    Returns:
        A :class:`ConfigStore` with parsed providers.

    Raises:
        ConfigFormatError: If the file contains invalid TOML.
    """
    if path is None:
        path = default_config_source
    try:
        with path.open() as f:
            doc = tomlkit.parse(f.read())
    except FileNotFoundError:
        doc = tomlkit.document()
    except TOMLKitError as exc:
        raise ConfigFormatError(str(exc)) from exc
    return ConfigStore(path=path, document=doc)
