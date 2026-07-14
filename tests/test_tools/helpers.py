from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from plyngent.agent import ToolDefinition


def call_sync(definition: ToolDefinition, *args: object, **kwargs: object) -> str:
    result: Any = definition.handler(*args, **kwargs)
    assert not inspect.isawaitable(result)
    return cast("str", result)


async def call_async(definition: ToolDefinition, *args: object, **kwargs: object) -> str:
    result: Any = definition.handler(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return cast("str", result)
