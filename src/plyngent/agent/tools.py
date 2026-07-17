from __future__ import annotations

import asyncio
import inspect
import types
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast, get_args, get_origin, get_type_hints, overload

import msgspec

from plyngent.lmproto.openai_compatible.model import ToolFunction, ToolFunctionItem
from plyngent.typedef import JSONSchema  # noqa: TC001

type ToolHandler = Callable[..., Any | Awaitable[Any]]
type DangerClassifier = Callable[[str, Mapping[str, object]], str | None]
type ToolConfirmHook = Callable[
    [str, Mapping[str, object], str],
    bool | Awaitable[bool],
]

_PRIMITIVE_SCHEMA: dict[type, JSONSchema] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


class ToolDefinition:
    """A registered tool: schema for the model plus a callable handler."""

    name: str
    description: str
    parameters: JSONSchema
    handler: ToolHandler

    def __init__(
        self,
        name: str,
        description: str,
        parameters: JSONSchema,
        handler: ToolHandler,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_tool_item(self) -> ToolFunctionItem:
        return ToolFunctionItem(
            function=ToolFunction(
                name=self.name,
                description=self.description or msgspec.UNSET,
                parameters=self.parameters or msgspec.UNSET,
            )
        )


def _annotation_to_schema(annotation: object) -> JSONSchema:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    if origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _annotation_to_schema(args[0]) if len(args) == 1 else {}
    if origin is list:
        args = get_args(annotation)
        items = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": items or True}
    if origin is dict:
        return {"type": "object"}
    if isinstance(annotation, type) and annotation in _PRIMITIVE_SCHEMA:
        return dict(_PRIMITIVE_SCHEMA[annotation])
    return {}


def schema_from_callable(func: ToolHandler) -> JSONSchema:
    """Build a JSON Schema object for ``func`` parameters from type hints."""
    hints = get_type_hints(func)
    signature = inspect.signature(func)
    properties: dict[str, JSONSchema] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = hints.get(name, param.annotation)
        properties[name] = _annotation_to_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: JSONSchema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _build_definition(
    func: ToolHandler,
    *,
    name: str | None,
    description: str | None,
) -> ToolDefinition:
    tool_name = name or func.__name__
    tool_description = description if description is not None else (inspect.getdoc(func) or "")
    return ToolDefinition(
        name=tool_name,
        description=tool_description,
        parameters=schema_from_callable(func),
        handler=func,
    )


@overload
def tool[**PS, R](func: Callable[PS, R], /) -> ToolDefinition: ...


@overload
def tool[**PS, R](
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[PS, R]], ToolDefinition]: ...


def tool[**PS, R](
    func: Callable[PS, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolDefinition | Callable[[Callable[PS, R]], ToolDefinition]:
    """Register a function as an agent tool (decorator).

    Schema is inferred from type hints; description defaults to the docstring.
    """

    def decorator(fn: Callable[PS, R]) -> ToolDefinition:
        return _build_definition(fn, name=name, description=description)

    if func is not None:
        return decorator(func)
    return decorator


class ToolRegistry:
    """Name → tool definition map with execution helpers."""

    _tools: dict[str, ToolDefinition]
    _danger: DangerClassifier | None
    _on_confirm: ToolConfirmHook | None
    _confirm_lock: asyncio.Lock

    def __init__(
        self,
        tools: Mapping[str, ToolDefinition] | list[ToolDefinition] | None = None,
        *,
        danger: DangerClassifier | None = None,
        on_confirm: ToolConfirmHook | None = None,
    ) -> None:
        self._tools = {}
        self._danger = danger
        self._on_confirm = on_confirm
        self._confirm_lock = asyncio.Lock()
        if tools is None:
            return
        if isinstance(tools, list):
            for item in tools:
                self.register(item)
        else:
            for item in tools.values():
                self.register(item)

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def tool_items(self) -> list[ToolFunctionItem]:
        return [t.to_tool_item() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def soft_confirm(self) -> bool:
        """True when dangerous tools are gated by ``on_confirm``."""
        return self._danger is not None and self._on_confirm is not None

    async def _invoke(self, definition: ToolDefinition, args: dict[str, object]) -> str:
        try:
            result = definition.handler(**args)
            if inspect.isawaitable(result):
                result = await result
        except asyncio.CancelledError:
            raise
        except TypeError as exc:
            return f"error: invalid tool arguments: {exc}"
        except Exception as exc:  # noqa: BLE001 — surface tool failures to the model
            return f"error: tool {definition.name!r} failed: {exc}"
        if isinstance(result, str):
            return result
        return msgspec.json.encode(result).decode()

    async def _maybe_confirm(self, name: str, args: dict[str, object]) -> str | None:
        """Return an error string if the user denies a dangerous tool, else None."""
        if self._danger is None or self._on_confirm is None:
            return None
        reason = self._danger(name, args)
        if reason is None:
            return None
        async with self._confirm_lock:
            # Re-check under the lock so parallel tools do not race the prompt.
            reason = self._danger(name, args)
            if reason is None:
                return None
            allowed = self._on_confirm(name, args, reason)
            if inspect.isawaitable(allowed):
                allowed = await allowed
            if allowed:
                return None
        return f"error: user denied tool {name!r} ({reason})"

    async def execute(self, name: str, arguments_json: str) -> str:
        """Run a tool by name; returns a string result (errors become error text)."""
        definition = self._tools.get(name)
        if definition is None:
            return f"error: unknown tool {name!r}"
        try:
            raw_args: object = msgspec.json.decode(arguments_json.encode())
        except (msgspec.DecodeError, UnicodeEncodeError) as exc:
            return f"error: invalid tool arguments JSON: {exc}"
        if not isinstance(raw_args, dict):
            return "error: tool arguments must be a JSON object"
        args = {str(key): value for key, value in cast("dict[object, object]", raw_args).items()}
        denied = await self._maybe_confirm(name, args)
        if denied is not None:
            return denied
        return await self._invoke(definition, args)
