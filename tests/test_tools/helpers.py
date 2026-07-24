from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from plyngent.agent import ToolDefinition


def call_sync(definition: ToolDefinition, *args: object, **kwargs: object) -> str:
    """Invoke a tool handler from a **sync** test.

    Async handlers are run with ``asyncio.run``. Inside an already-running
    event loop (async tests), use :func:`call_async` instead.
    """
    result: Any = definition.handler(*args, **kwargs)
    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(result)
        else:
            msg = "call_sync cannot await inside a running event loop; use call_async"
            raise RuntimeError(msg)
    return cast("str", result)


async def call_async(definition: ToolDefinition, *args: object, **kwargs: object) -> str:
    result: Any = definition.handler(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return cast("str", result)
