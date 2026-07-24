from __future__ import annotations

import asyncio
import inspect
import types
from collections.abc import Awaitable, Callable, Mapping
from enum import Flag, auto
from typing import Any, cast, get_args, get_origin, get_type_hints, overload

import msgspec

from plyngent.lmproto.openai_compatible.model import ToolFunction, ToolFunctionItem
from plyngent.typedef import JSONSchema  # noqa: TC001

type ToolHandler = Callable[..., Any | Awaitable[Any]]
type DangerClassifier = Callable[[str, Mapping[str, object]], str | None]
# Returns True to allow, False to deny, or a non-empty str as denial reason for the model.
type ToolConfirmResult = bool | str
type ToolConfirmHook = Callable[
    [str, Mapping[str, object], str],
    ToolConfirmResult | Awaitable[ToolConfirmResult],
]

_PRIMITIVE_SCHEMA: dict[type, JSONSchema] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


class ToolTag(Flag):
    """Host policy / affinity bits for a tool (not I/O taxonomy).

    Default when omitted on ``@tool`` is :attr:`LOCAL`. A tool should set at
    least one of :attr:`LOCAL` or :attr:`PUBLIC` (register rejects neither).
    """

    LOCAL = auto()  # local agent frontend (CLI)
    PUBLIC = auto()  # may be exposed on shared / multi-tenant frontends
    TRUSTABLE = auto()  # soft-confirm: grant once, then reuse
    YOLO = auto()  # soft-confirm: eligible for YOLO auto-approve
    INSTANCE_STATE = auto()  # needs instance-scoped state
    SESSION_STATE = auto()  # needs session-scoped state


class ToolDefinition:
    """A registered tool: schema for the model plus a callable handler."""

    name: str
    description: str
    parameters: JSONSchema
    handler: ToolHandler
    tags: ToolTag

    def __init__(
        self,
        name: str,
        description: str,
        parameters: JSONSchema,
        handler: ToolHandler,
        *,
        tags: ToolTag = ToolTag.LOCAL,
    ) -> None:
        if not (tags & (ToolTag.LOCAL | ToolTag.PUBLIC)):
            msg = f"tool {name!r} tags must include LOCAL and/or PUBLIC"
            raise ValueError(msg)
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.tags = tags

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
    tags: ToolTag,
) -> ToolDefinition:
    tool_name = name or func.__name__
    tool_description = description if description is not None else (inspect.getdoc(func) or "")
    return ToolDefinition(
        name=tool_name,
        description=tool_description,
        parameters=schema_from_callable(func),
        handler=func,
        tags=tags,
    )


def _register_definition(definition: ToolDefinition) -> None:
    """Push *definition* into the process tool catalog (lazy import)."""
    # Imported lazily so agent.tools does not hard-depend on plyngent.tools.
    from plyngent.tools.catalog import register_tool

    register_tool(definition)


@overload
def tool[**PS, R](func: Callable[PS, R], /) -> ToolDefinition: ...


@overload
def tool[**PS, R](
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: ToolTag = ToolTag.LOCAL,
    register: bool = True,
) -> Callable[[Callable[PS, R]], ToolDefinition]: ...


def tool[**PS, R](
    func: Callable[PS, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: ToolTag = ToolTag.LOCAL,
    register: bool = True,
) -> ToolDefinition | Callable[[Callable[PS, R]], ToolDefinition]:
    """Define (and by default catalog-register) a function as an agent tool.

    Schema is inferred from type hints; description defaults to the docstring.
    Default ``tags`` is :attr:`ToolTag.LOCAL`. When ``register`` is true, the
    definition is added to the process :class:`~plyngent.tools.catalog.ToolCatalog`
    with the current registration source (builtin unless a plugin context is set).
    Catalog registration does **not** alone make the tool model-visible — hosts
    still **select** tools into a :class:`ToolRegistry`.
    """

    def decorator(fn: Callable[PS, R]) -> ToolDefinition:
        definition = _build_definition(
            fn,
            name=name,
            description=description,
            tags=tags,
        )
        if register:
            _register_definition(definition)
        return definition

    if func is not None:
        return decorator(func)
    return decorator


class ToolRegistry:
    """Name → tool definition map with execution helpers."""

    _tools: dict[str, ToolDefinition]
    _danger: DangerClassifier | None
    _on_confirm: ToolConfirmHook | None
    _confirm_lock: asyncio.Lock
    _yolo: bool
    _auto_bind_state: bool
    _instance: object | None
    _session: object | None

    def __init__(
        self,
        tools: Mapping[str, ToolDefinition] | list[ToolDefinition] | None = None,
        *,
        danger: DangerClassifier | None = None,
        on_confirm: ToolConfirmHook | None = None,
        yolo: bool = False,
        auto_bind_state: bool = False,
        instance_state: object | None = None,
        session_state: object | None = None,
    ) -> None:
        self._tools = {}
        self._danger = danger
        self._on_confirm = on_confirm
        self._confirm_lock = asyncio.Lock()
        self._yolo = yolo
        self._auto_bind_state = auto_bind_state
        self._instance = instance_state
        self._session = session_state
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

    def set_yolo(self, *, enabled: bool) -> None:
        self._yolo = enabled

    def set_session_state(self, session_state: object | None) -> None:
        self._session = session_state

    def set_instance_state(self, instance_state: object | None) -> None:
        self._instance = instance_state

    @property
    def yolo(self) -> bool:
        """Whether YOLO mode may auto-approve YOLO-tagged soft confirms."""
        return self._yolo

    @property
    def soft_confirm(self) -> bool:
        """True when a soft-confirm path is configured (danger + on_confirm).

        YOLO mode does not clear this: non-YOLO-tagged tools still prompt.
        """
        return self._danger is not None and self._on_confirm is not None

    def _check_state_tags(self, definition: ToolDefinition) -> str | None:
        """Return an error string if required state context is missing.

        Only enforced when the host opted into ``auto_bind_state`` (CLI). Hand
        registries in tests may still rely on process globals during migration.
        """
        if not self._auto_bind_state:
            return None
        tags = definition.tags
        if tags & ToolTag.INSTANCE_STATE:
            from plyngent.tools.context import get_instance

            if get_instance() is None and self._instance is None:
                return f"error: tool {definition.name!r} requires instance state (INSTANCE_STATE) but none is bound"
        if tags & ToolTag.SESSION_STATE:
            from plyngent.tools.context import get_session

            if get_session() is None and self._session is None:
                return f"error: tool {definition.name!r} requires session state (SESSION_STATE) but none is bound"
        return None

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

    def _session_for_grants(self) -> Any | None:
        from plyngent.tools.context import get_session

        session = get_session()
        if session is None and self._session is not None:
            return cast("Any", self._session)
        return session

    def _grant_allows(self, name: str, *, tags: ToolTag) -> bool:
        if not (tags & ToolTag.TRUSTABLE):
            return False
        from plyngent.tools.grants import has_grant

        session = self._session_for_grants()
        return session is not None and has_grant(session, name)

    async def _record_grant(self, name: str, *, tags: ToolTag) -> None:
        if not (tags & ToolTag.TRUSTABLE):
            return
        from plyngent.tools.grants import add_grant

        session = self._session_for_grants()
        if session is not None:
            await add_grant(session, name)

    async def _prompt_soft_confirm(
        self,
        name: str,
        args: dict[str, object],
        reason: str,
        *,
        tags: ToolTag,
    ) -> str | None:
        if self._on_confirm is None:
            return f"error: tool {name!r} denied by policy ({reason}; no confirm hook)"
        decision = self._on_confirm(name, args, reason)
        if inspect.isawaitable(decision):
            decision = await decision
        if decision is True:
            await self._record_grant(name, tags=tags)
            return None
        if isinstance(decision, str) and decision.strip():
            return f"error: tool {name!r} denied by user confirm ({reason}); user comment: {decision.strip()}"
        return f"error: tool {name!r} denied by user confirm ({reason})"

    async def _maybe_confirm(
        self,
        name: str,
        args: dict[str, object],
        *,
        tags: ToolTag,
    ) -> str | None:
        """Soft-confirm gate driven by danger reason + tags + grants + YOLO.

        Pipeline (soft gate only; hard denylists stay in tool handlers)::

            no soft reason → run
            YOLO mode and (tags & YOLO) → allow
            (tags & TRUSTABLE) and grant exists → allow
            else on_confirm; on approve + TRUSTABLE → store grant
        """
        if self._danger is None:
            return None
        reason = self._danger(name, args)
        if reason is None:
            return None

        async with self._confirm_lock:
            reason = self._danger(name, args)
            if reason is None:
                return None
            if self._yolo and (tags & ToolTag.YOLO):
                return None
            if self._grant_allows(name, tags=tags):
                return None
            return await self._prompt_soft_confirm(name, args, reason, tags=tags)

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

        async def _run() -> str:
            missing = self._check_state_tags(definition)
            if missing is not None:
                return missing
            denied = await self._maybe_confirm(name, args, tags=definition.tags)
            if denied is not None:
                return denied
            return await self._invoke(definition, args)

        # Bind host-provided state for tools that read contextvars (todo, workspace).
        # Tag enforcement remains gated by ``auto_bind_state``; binding is always useful
        # when the registry holds instance/session handles.
        if self._instance is not None or self._session is not None:
            from plyngent.tools.context import bind_tool_context

            with bind_tool_context(
                instance=cast("Any", self._instance),
                session=cast("Any", self._session),
            ):
                return await _run()
        return await _run()
