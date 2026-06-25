from typing import Any, Literal

from msgspec import UnsetType

type JSONSchema = dict[str, Any]
type Unset = Literal[UnsetType.UNSET]
