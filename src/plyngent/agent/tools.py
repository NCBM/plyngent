from __future__ import annotations

import asyncio
import inspect
import types
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import Flag, auto
from pathlib import Path
from typing import Any, cast, get_args, get_origin, get_type_hints, overload

import msgspec
from msgspec import Struct

from plyngent.lmproto.openai_compatible.model import ChatImagePart, ToolFunction, ToolFunctionItem
from plyngent.typedef import JSONSchema  # noqa: TC001


@dataclass(frozen=True, slots=True)
class DangerReason:
    """Why a call needs a soft confirm, and whether YOLO mode may drop that prompt.

    ``yolo_skip`` is false for a call that reaches rights the session does not
    have (``sudo`` / ``doas`` / ``pkexec`` and friends): the human answers it
    however YOLO mode is set.
    """

    text: str
    yolo_skip: bool = True


type ToolHandler = Callable[..., Any | Awaitable[Any]]
type DangerClassifier = Callable[[str, Mapping[str, object]], str | DangerReason | None]
# Returns True to allow, False to deny, or a non-empty str as denial reason for the model.
type ToolConfirmResult = bool | str
type ToolConfirmHook = Callable[
    [str, Mapping[str, object], str],
    ToolConfirmResult | Awaitable[ToolConfirmResult],
]


class ToolOutput(Struct):
    """A tool result: the model-facing text, plus images the model should see.

    Most tools return their text directly; one that produced an image (a
    screenshot, a chart, a downloaded picture) returns this so the image rides
    along with the text that explains it. ``execute`` hands the pair to the
    loop, which keeps the text as the tool message's content and the images
    beside it.
    """

    text: str
    images: list[ChatImagePart] = msgspec.field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)


def _without_refused_images(output: ToolOutput) -> ToolOutput:
    """Drop a picture the selected model cannot take, and say why.

    ``view_image`` and ``fetch_local`` refuse on their own, before any bytes are
    read, so the model gets a clean error. This is the safety net for the rest —
    an MCP server's image content, a plugin's ``ToolOutput`` — where the picture
    would otherwise reach a model that answers ``[Unsupported Image]``, or a
    gateway that hands it raw base64 (see ``doc/images.md``).
    """
    if not output.images:
        return output
    from plyngent.tools.workspace import image_input_refusal

    refusal = image_input_refusal()
    if refusal is None:
        return output
    note = refusal.removeprefix("error: ")
    text = f"{output.text}\n(error: image dropped: {note})" if output.text else f"error: image dropped: {note}"
    return ToolOutput(text=text)


@dataclass(frozen=True, slots=True)
class ToolConfirmRequest:
    """One call of a batch that needs a soft confirm before it may run."""

    name: str
    args: Mapping[str, object]
    reason: str


@dataclass(frozen=True, slots=True)
class BatchConfirmDecision:
    """Host answer for one round's soft-confirm requests.

    It covers the calls of that round only; the next round asks again. With
    ``each`` the host hands the calls back to the one-call prompt instead of
    answering for the batch (``allowed``/``comment`` then carry no meaning).
    """

    allowed: bool
    comment: str | None = None
    each: bool = False


# One prompt for a round's requests; a single request keeps ``ToolConfirmHook``.
type ToolBatchConfirmHook = Callable[
    [Sequence[ToolConfirmRequest]],
    BatchConfirmDecision | Awaitable[BatchConfirmDecision],
]

MIN_BATCH_CONFIRM_REQUESTS = 2
"""Below this many requests a round asks one call at a time (see ``prepare_batch_confirm``)."""

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
    READ_ONLY = auto()  # never mutates host state; safe for read-only contexts


class ToolDefinition:
    """A registered tool: schema for the model plus a callable handler.

    ``writes`` names the arguments whose values name files the handler writes.
    Calls of one batch that name the same file apply one at a time, in call
    order, so the later one sees what the earlier one wrote.
    """

    name: str
    description: str
    parameters: JSONSchema
    handler: ToolHandler
    tags: ToolTag
    writes: tuple[str, ...]

    def __init__(
        self,
        name: str,
        description: str,
        parameters: JSONSchema,
        handler: ToolHandler,
        *,
        tags: ToolTag = ToolTag.LOCAL,
        writes: tuple[str, ...] = (),
    ) -> None:
        if not (tags & (ToolTag.LOCAL | ToolTag.PUBLIC)):
            msg = f"tool {name!r} tags must include LOCAL and/or PUBLIC"
            raise ValueError(msg)
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.tags = tags
        self.writes = writes

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
    writes: tuple[str, ...] = (),
) -> ToolDefinition:
    tool_name = name or func.__name__
    tool_description = description if description is not None else (inspect.getdoc(func) or "")
    return ToolDefinition(
        name=tool_name,
        description=tool_description,
        parameters=schema_from_callable(func),
        handler=func,
        tags=tags,
        writes=writes,
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
    writes: tuple[str, ...] = (),
    register: bool = True,
) -> Callable[[Callable[PS, R]], ToolDefinition]: ...


def tool[**PS, R](
    func: Callable[PS, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: ToolTag = ToolTag.LOCAL,
    writes: tuple[str, ...] = (),
    register: bool = True,
) -> ToolDefinition | Callable[[Callable[PS, R]], ToolDefinition]:
    """Define (and by default catalog-register) a function as an agent tool.

    Schema is inferred from type hints; description defaults to the docstring.
    Default ``tags`` is :attr:`ToolTag.LOCAL`. ``writes`` names the arguments
    that carry the files the handler writes (:attr:`ToolDefinition.writes`).
    When ``register`` is true, the definition is added to the process
    :class:`~plyngent.tools.catalog.ToolCatalog` with the current registration
    source (builtin unless a plugin context is set). Catalog registration does
    **not** alone make the tool model-visible — hosts still **select** tools
    into a :class:`ToolRegistry`.
    """

    def decorator(fn: Callable[PS, R]) -> ToolDefinition:
        definition = _build_definition(
            fn,
            name=name,
            description=description,
            tags=tags,
            writes=writes,
        )
        if register:
            _register_definition(definition)
        return definition

    if func is not None:
        return decorator(func)
    return decorator


def _missing_required_error(definition: ToolDefinition, args: Mapping[str, object]) -> str | None:
    """Model-facing error when a call omits a required argument.

    The JSON schema already declares what is required, so the model gets the
    exact argument names back instead of the handler's bare ``TypeError``
    (``missing 1 required positional argument``). Unknown *extra* arguments are
    left to the handler: a declared schema is not proof that no other key is
    accepted (MCP servers may document only part of their input).
    """
    schema = definition.parameters
    required = schema.get("required")
    if not isinstance(required, list):
        return None
    missing = [name for name in cast("list[object]", required) if isinstance(name, str) and name not in args]
    if not missing:
        return None
    properties = schema.get("properties")
    accepted: list[str] = []
    if isinstance(properties, dict):
        accepted = sorted(str(key) for key in cast("dict[object, object]", properties))
    hint = f"; it accepts: {', '.join(accepted)}" if accepted else ""
    return f"error: `{definition.name}` is missing required argument(s): {', '.join(missing)}{hint}"


def _decode_tool_args(arguments_json: str) -> dict[str, object] | str:
    """Decode one call's arguments; a returned string is the model-facing error."""
    try:
        raw_args: object = msgspec.json.decode(arguments_json.encode())
    except (msgspec.DecodeError, UnicodeEncodeError) as exc:
        return f"error: invalid tool arguments JSON: {exc}"
    if not isinstance(raw_args, dict):
        return "error: tool arguments must be a JSON object"
    return {str(key): value for key, value in cast("dict[object, object]", raw_args).items()}


def _denied_error(name: str, reason: str, comment: str | None) -> str:
    """Model-facing text for a denied soft confirm (carries a user comment if one was given)."""
    if comment is not None and comment.strip():
        return f"error: tool {name!r} denied by user confirm ({reason}); user comment: {comment.strip()}"
    return f"error: tool {name!r} denied by user confirm ({reason})"


def _as_danger_reason(raw: str | DangerReason | None) -> DangerReason | None:
    """Normalize a classifier answer; a bare string is a reason YOLO mode may drop."""
    if raw is None:
        return None
    return raw if isinstance(raw, DangerReason) else DangerReason(text=raw)


def _write_path_key(value: str) -> str:
    """Comparable key for one path argument of a writing tool.

    Relative arguments all mean the same workspace, so the text itself separates
    two spellings of one file (``a.txt`` / ``./a.txt``) from two files.
    """
    return str(Path(value).expanduser())


class ToolRegistry:
    """Name → tool definition map with execution helpers."""

    _tools: dict[str, ToolDefinition]
    _danger: DangerClassifier | None
    _on_confirm: ToolConfirmHook | None
    _on_confirm_batch: ToolBatchConfirmHook | None
    _confirm_lock: asyncio.Lock
    # One lock per written file, kept for the registry's life (a session's edits
    # bound the map), so a batch's writes to one file apply one at a time.
    _write_locks: dict[str, asyncio.Lock]
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
        on_confirm_batch: ToolBatchConfirmHook | None = None,
        yolo: bool = False,
        auto_bind_state: bool = False,
        instance_state: object | None = None,
        session_state: object | None = None,
    ) -> None:
        self._tools = {}
        self._danger = danger
        self._on_confirm = on_confirm
        self._on_confirm_batch = on_confirm_batch
        self._confirm_lock = asyncio.Lock()
        self._write_locks = {}
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

    def definitions(self) -> list[ToolDefinition]:
        """Return tool definitions in registration order (for cloning registries)."""
        return list(self._tools.values())

    def clone(
        self,
        *,
        instance_state: object | None = None,
        session_state: object | None = None,
        yolo: bool | None = None,
        auto_bind_state: bool | None = None,
        read_only_only: bool = False,
    ) -> ToolRegistry:
        """New registry with the same tools/hooks and rebound host state.

        *instance_state* / *session_state* are applied as given (use the parent
        registry's handles only when the caller passes them through).

        With *read_only_only*, only :attr:`ToolTag.READ_ONLY` tools are kept
        (e.g. for safe side turns / planning); hooks and policy flags are
        preserved.
        """
        definitions = self.definitions()
        if read_only_only:
            definitions = [d for d in definitions if d.tags & ToolTag.READ_ONLY]
        return ToolRegistry(
            definitions,
            danger=self._danger,
            on_confirm=self._on_confirm,
            on_confirm_batch=self._on_confirm_batch,
            yolo=self._yolo if yolo is None else yolo,
            auto_bind_state=self._auto_bind_state if auto_bind_state is None else auto_bind_state,
            instance_state=instance_state,
            session_state=session_state,
        )

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

    async def _invoke(self, definition: ToolDefinition, args: dict[str, object]) -> str | ToolOutput:
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
        if isinstance(result, str | ToolOutput):
            return result
        return msgspec.json.encode(result).decode()

    def _session_for_grants(self) -> Any | None:
        from plyngent.tools.context import get_session

        session = get_session()
        if session is None and self._session is not None:
            return cast("Any", self._session)
        return session

    def _write_locks_for(self, definition: ToolDefinition, args: Mapping[str, object]) -> list[asyncio.Lock]:
        """Locks for the files this call writes, in a stable order (no deadlock).

        Empty for a tool that declares no :attr:`ToolDefinition.writes`.
        """
        keys: set[str] = set()
        for name in definition.writes:
            value = args.get(name)
            if isinstance(value, str) and value.strip():
                keys.add(_write_path_key(value))
        locks: list[asyncio.Lock] = []
        for key in sorted(keys):
            lock = self._write_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._write_locks[key] = lock
            locks.append(lock)
        return locks

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
        comment = decision if isinstance(decision, str) else None
        return _denied_error(name, reason, comment)

    async def _apply_batch_decision(
        self,
        name: str,
        reason: str,
        decision: BatchConfirmDecision,
        *,
        tags: ToolTag,
    ) -> str | None:
        if decision.allowed:
            await self._record_grant(name, tags=tags)
            return None
        return _denied_error(name, reason, decision.comment)

    def _prepare_confirm(self, name: str, arguments_json: str) -> ToolConfirmRequest | None:
        """Soft-confirm request for one call, or ``None`` when it needs none.

        Mirrors the front half of :meth:`execute`: a call whose arguments fail to
        decode or miss a required argument is answered with an error there and
        must not be put in front of the user now, and a call whose reason YOLO
        mode covers runs unprompted, so it must not take a slot in the round's
        aggregated prompt either.
        """
        definition = self._tools.get(name)
        if self._danger is None or definition is None:
            return None
        args = _decode_tool_args(arguments_json)
        if isinstance(args, str) or _missing_required_error(definition, args) is not None:
            return None
        reason = _as_danger_reason(self._danger(name, args))
        if reason is None or (self._yolo and (definition.tags & ToolTag.YOLO) and reason.yolo_skip):
            return None
        return ToolConfirmRequest(name=name, args=args, reason=reason.text)

    def _batch_requests(
        self,
        calls: Sequence[tuple[str, str]],
    ) -> tuple[list[ToolConfirmRequest], list[int]]:
        """Requests for a round, plus the index each one came from."""
        requests: list[ToolConfirmRequest] = []
        positions: list[int] = []
        for index, (name, arguments_json) in enumerate(calls):
            request = self._prepare_confirm(name, arguments_json)
            if request is not None:
                requests.append(request)
                positions.append(index)
        return requests, positions

    async def _record_batch_allow(
        self,
        requests: Sequence[ToolConfirmRequest],
        decision: BatchConfirmDecision,
    ) -> None:
        """Record the trust a batch allow carries (the trustable grants)."""
        if decision.allowed:
            for request in requests:
                definition = self._tools.get(request.name)
                if definition is not None:
                    await self._record_grant(request.name, tags=definition.tags)

    async def prepare_batch_confirm(
        self,
        calls: Sequence[tuple[str, str]],
    ) -> list[BatchConfirmDecision | None]:
        """Ask once for every soft confirm a round's calls would raise.

        Returns one entry per call: a decision to apply instead of prompting, or
        ``None`` to fall through to the per-call path. Fewer than
        :data:`MIN_BATCH_CONFIRM_REQUESTS` requests keep the plain one-call
        prompt, a host without a batch hook keeps the per-call prompts, and a
        :attr:`BatchConfirmDecision.each` answer hands this round back to them.
        """
        decisions: list[BatchConfirmDecision | None] = [None] * len(calls)
        if self._danger is None or self._on_confirm_batch is None:
            return decisions
        requests, positions = self._batch_requests(calls)
        if len(requests) < MIN_BATCH_CONFIRM_REQUESTS:
            return decisions
        async with self._confirm_lock:
            decision = self._on_confirm_batch(requests)
            if inspect.isawaitable(decision):
                decision = await decision
            if decision.each:
                return decisions
            await self._record_batch_allow(requests, decision)
        for index in positions:
            decisions[index] = decision
        return decisions

    async def _maybe_confirm(
        self,
        name: str,
        args: dict[str, object],
        *,
        tags: ToolTag,
        decision: BatchConfirmDecision | None = None,
    ) -> str | None:
        """Soft-confirm gate driven by danger reason + tags + grants + YOLO.

        Pipeline (soft gate only; hard denylists stay in tool handlers)::

            no soft reason → run
            YOLO mode, (tags & YOLO) and a reason it may drop → allow
            batch decision for this call → apply it
            (tags & TRUSTABLE) and grant exists → allow
            else → on_confirm (a tool without the YOLO tag, or a reason YOLO
                   may not drop — a privilege escalation — still prompts
                   under YOLO mode)
        """
        if self._danger is None:
            return None
        reason = _as_danger_reason(self._danger(name, args))
        if reason is None:
            return None
        if self._yolo and (tags & ToolTag.YOLO) and reason.yolo_skip:
            return None

        async with self._confirm_lock:
            if decision is not None:
                return await self._apply_batch_decision(name, reason.text, decision, tags=tags)
            if self._grant_allows(name, tags=tags):
                return None
            return await self._prompt_soft_confirm(name, args, reason.text, tags=tags)

    async def execute(
        self,
        name: str,
        arguments_json: str,
        *,
        decision: BatchConfirmDecision | None = None,
    ) -> str:
        """Run a tool by name; returns its text (errors become error text).

        An image the tool returned is dropped here — this is the text-only entry
        point. :meth:`execute_output` returns it; the agent loop uses that one.
        """
        return (await self.execute_output(name, arguments_json, decision=decision)).text

    async def execute_output(
        self,
        name: str,
        arguments_json: str,
        *,
        decision: BatchConfirmDecision | None = None,
    ) -> ToolOutput:
        """Run a tool by name; returns its text plus any images it produced.

        Errors become error text. *decision* is the host's answer for this call
        from :meth:`prepare_batch_confirm`; ``None`` means the call was not part
        of an aggregated prompt and confirms on its own.
        """
        definition = self._tools.get(name)
        if definition is None:
            return ToolOutput(text=f"error: unknown tool {name!r}")
        args = _decode_tool_args(arguments_json)
        if isinstance(args, str):
            return ToolOutput(text=args)
        # Argument shape/coverage problems are answered before any confirm prompt:
        # the human should never be asked about a call that cannot run.
        missing_error = _missing_required_error(definition, args)
        if missing_error is not None:
            return ToolOutput(text=missing_error)

        async def _run() -> ToolOutput:
            missing = self._check_state_tags(definition)
            if missing is not None:
                return ToolOutput(text=missing)
            # Held before the confirm prompt, so a sibling call writing the same
            # file waits for this one instead of overtaking its prompt.
            async with AsyncExitStack() as held:
                for lock in self._write_locks_for(definition, args):
                    await held.enter_async_context(lock)
                denied = await self._maybe_confirm(name, args, tags=definition.tags, decision=decision)
                if denied is not None:
                    return ToolOutput(text=denied)
                result = await self._invoke(definition, args)
                return _without_refused_images(result if isinstance(result, ToolOutput) else ToolOutput(text=result))

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
