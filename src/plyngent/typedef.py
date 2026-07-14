from typing import Any

from msgspec import UnsetType

type JSONSchema = dict[str, Any]
# Assignment (not PEP 695 `type`) so msgspec recognizes UnsetType in unions.
Unset = UnsetType
