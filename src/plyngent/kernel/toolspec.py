"""What a tool declares about itself, and the things generated from that declaration.

A tool is one value: the resources it reaches for, the kind of change it may cause, which of its
arguments are paths (and whether they are read or written), how a call gets approved, and **one
structured argument package**. The JSON Schema the model sees is generated from that package, and so
is the validator the call goes through — so the schema is never a second, hand-maintained
description of the same thing.

Three rules the shape enforces:

* **The schema is a function of the tool, not of the run.** Nothing here reads config, limits or the
  workspace, because anything that changed with them would silently rewrite the prompt prefix that a
  provider caches (see §10.5).
* **Argument packages live next to their handler.** There is no central table of tools: a module
  defines ``Args`` and the handler that takes it, and declares the spec beside them.
* **A tool states what it is, never how this one call should be judged.** Approval is derived from
  the declarations (:meth:`ToolSpec.approval`), and the only way to ask for more is to declare it —
  there is no hook that looks at a call and answers "less than the baseline", because that is the
  one shape that turns a dangerous call into an automatic one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, StrEnum
from functools import cache
from typing import TYPE_CHECKING, Annotated, cast

import msgspec
from msgspec import Struct

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from typing import Any

    from plyngent.kernel.env import Env
    from plyngent.kernel.toolresult import ToolResult

#: How much room a call gets to say why it was made: a sentence, not a place to think out loud.
MAX_INTENTION_CHARS = 280


class Res(Flag):
    """The resources a tool may reach for. Nothing outside this set is reachable from its handler."""

    PATHS = 1 << 0
    EXEC = 1 << 1
    PTY = 1 << 2
    NET = 1 << 3
    HUMAN = 1 << 4
    HISTORY = 1 << 5
    DOCUMENT = 1 << 6
    CLIENT = 1 << 7
    #: The tool set itself: what exists, what this session can reach, what it has not been shown. The
    #: one resource whose subject is the tools rather than the world (§10.4).
    TOOLS = 1 << 8


class Mutation(Flag):
    """What a call may change, ordered so a batch's posture is the union of its members'."""

    NONE = 0
    WORKSPACE = 1 << 0
    SESSION = 1 << 1
    HOST = 1 << 2
    OUTSIDE = 1 << 3  # decided by somebody else: a child process, a remote system, a client
    #: The tool's ordinary use takes away something that was already there — a delete, a move. Not
    #: "an optional parameter could": ``write_file(overwrite=true)`` belongs to the approval layer.
    DESTRUCTIVE = 1 << 4


class Role(StrEnum):
    """What an argument means when it is a path, so the executor can resolve and lock it.

    ``READ`` and ``WRITE`` are both paths; the difference is what a batch's locks say about them and
    which access mode the resolver is asked for. ``CWD`` is a directory a command runs in, ``ARGV``
    is a program and its arguments (judged by the command policy), ``COMMAND`` is a whole command
    line for the same purpose. There is no "opaque": a path is either ours to resolve or not a path.
    """

    READ = "read"
    WRITE = "write"
    CWD = "cwd"
    ARGV = "argv"
    COMMAND = "command"


#: The roles that name a path: a tool with one of these resolves something through the workspace, so it
#: has to declare :attr:`Res.PATHS` or its handler could not be honoured in the first place.
_PATH_ROLES: frozenset[Role] = frozenset({Role.READ, Role.WRITE, Role.CWD})


class Group(StrEnum):
    """The disclosure unit: a group arrives together, with its own playbook section."""

    DOCUMENT = "document"
    FS = "fs"
    #: The tool set talking about itself — "what else is there". Always available, so a hint that
    #: points at a group nobody has been shown always has somewhere to point.
    META = "meta"
    PROCESS = "process"
    NET = "net"
    VCS = "vcs"
    SESSION = "session"
    SKILLS = "skills"


class Posture(StrEnum):
    """How a call is approved, in the three ways that differ to the person being asked.

    ``AUTO`` runs without asking. ``ONCE`` asks the first time and then remembers, by whatever
    :class:`Remember` says. ``ALWAYS`` asks every time. The level is derived from what a tool
    declares (:meth:`ToolSpec.approval`), and policies may only raise it.
    """

    AUTO = "auto"
    ONCE = "once"
    ALWAYS = "always"


class Remember(StrEnum):
    """What an approval is remembered by — the four things a person can mean by "don't ask again".

    A path is not a program and neither is a session: an approval to run one command says nothing
    about another, and an approval to write into a live terminal cannot be scoped to a file.
    """

    PATH = "path"
    PROGRAM = "program"
    SESSION = "session"
    HOST = "host"


class ArgsError(Exception):
    """The model's arguments could not be read as this tool's package — nothing ran.

    ``message`` is the validator's whole text, which names the field itself. ``field`` is the place in
    the package when one is reported structurally, and it is always empty today: ``msgspec`` 0.22's
    ``ValidationError`` carries the location only inside the message (`` - at $.steps[0].stdin``), and
    slicing a sentence to get at it would be a parser for somebody else's prose. The field stays so
    that a validator which does report a path puts it here without a second error type.
    """

    field: str
    message: str

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}" if field else message)
        self.field = field
        self.message = message


class Intention(StrEnum):
    """Whether a call has to say why it is being made.

    The sentence is worth asking for where nothing else tells the human what is about to happen: a
    call that destroys, moves, executes or asks leaves a workspace that no longer says what was
    there. For everything else the arguments already say what the call is — asking anyway teaches a
    model to write "read the file", which is worse than silence.
    """

    OPTIONAL = "optional"
    REQUIRED = "required"


class ToolArgs(Struct, forbid_unknown_fields=True, kw_only=True):
    """The base of every tool's argument package: what any call may say about itself.

    ``intention`` is the model's own account of why it is calling, and it is the one thing no other
    part of the call carries. It is **advisory text**: no policy, refusal, hint or lock may read it
    — a claim cannot grant anything — and where a human is shown it, it is shown as the model's
    words.

    Optional here, required per tool (:attr:`ToolSpec.intention`), because that is where the
    difference lives. ``kw_only`` has to be repeated on every concrete package — ``msgspec`` does not
    inherit struct config — and it is not a style choice: a base field with a default and a required
    field beside it is otherwise refused outright, and keyword-only arguments make a package
    impossible to get wrong by position. :meth:`ToolSpec.__post_init__` refuses a package that forgot.
    """

    intention: Annotated[str, msgspec.Meta(max_length=MAX_INTENTION_CHARS)] = ""


@dataclass(frozen=True, slots=True)
class ExternalArgs:
    """Someone else's JSON Schema, taken as it is: MCP servers and the clients that offer tools.

    The kernel does not add, rename or tighten a field here. A provider's schema is the provider's
    promise about its tool, and rewriting it would make the prefix disagree with what that provider
    validates against — so the only thing asked of an external package is that it be a JSON object
    with the declared properties. It is a different kind of thing from :class:`ToolArgs` rather than
    a subclass, because there is no base to inherit: the schema is the whole declaration.
    """

    schema: Mapping[str, Any]
    #: The tool's description, from the provider; the schema may carry its own.
    description: str = ""

    @property
    def properties(self) -> frozenset[str]:
        """The names the schema declares, which is what a ``roles`` entry may refer to."""
        declared = self.schema.get("properties")
        if not isinstance(declared, dict):
            return frozenset()
        return frozenset(cast("dict[str, Any]", declared))


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what it may touch, what its arguments are, and what runs them.

    ``group`` is the disclosure unit; a remote or client-provided tool uses a prefixed group name
    (``mcp:<server>``, ``client:<id>``) and its own lifetime, which is why the field is a plain
    string rather than :class:`Group`.

    The three approval fields are declarations, not judgements: ``posture`` overrides the derived
    baseline for a tool whose reach is not expressed by ``uses``/``mutates`` (a temporary workspace
    widens what is reachable), ``remember`` overrides what an ``ONCE`` is keyed on, and ``escalates``
    says which argument values make the ordinary use of the tool irreversible.
    """

    name: str
    group: Group | str
    args: type[ToolArgs] | ExternalArgs
    run: Callable[[Env, Any], Awaitable[ToolResult]]
    uses: Res
    mutates: Mutation = Mutation.NONE
    roles: Mapping[str, Role] | None = None
    intention: Intention | None = None
    posture: Posture | None = None
    remember: Remember | None = None
    escalates: Mapping[str, object] | None = None
    #: What this tool asks of a package beyond its types. A rule two arguments make *together* — one
    #: locator and not the other — is not something a JSON Schema can say about either field alone, and
    #: checking it in the handler would mean the rule is skipped by every caller that does not reach the
    #: handler (a replay, a refusal, a program). So it is declared here and run by :meth:`check`.
    guard: Callable[[Any], None] | None = None

    def __post_init__(self) -> None:
        declared = self._declared_fields()
        roles = self.roles or {}
        escalates = self.escalates or {}
        for name in (*roles, *escalates):
            if name not in declared:
                msg = f"{self.name}: names an argument {name!r} that {self._args_name()} does not have"
                raise ValueError(msg)
        if self.intention is None:
            object.__setattr__(self, "intention", _asked_for(self.uses, self.mutates))
        named = {name for name, role in roles.items() if role in _PATH_ROLES}
        if named and not self.uses & Res.PATHS:
            msg = f"{self.name}: {', '.join(sorted(named))} is a path, so the tool has to declare Res.PATHS"
            raise ValueError(msg)
        if self.intention is Intention.REQUIRED and "intention" not in declared:
            msg = f"{self.name}: needs every call to say why, but its own schema declares no 'intention'"
            raise ValueError(msg)
        if self.posture is Posture.ONCE:
            resolved = self.remember if self.remember is not None else _baseline(self.uses, self.mutates)[1]
            if resolved is None:
                msg = f"{self.name}: approved once per what? declare a Remember for a once-only posture"
                raise ValueError(msg)

    def schema(self) -> dict[str, Any]:
        """The JSON Schema the model is given; identical bytes for identical declarations."""
        if isinstance(self.args, ExternalArgs):
            declared: dict[str, Any] = {}
            declared.update(self.args.schema)
            return declared
        return schema_of(self.args)

    def decode(self, arguments: str | bytes) -> Any:
        """Read an argument package, or raise :class:`ArgsError` saying which field and why."""
        parsed = self._read(arguments)
        self.check(parsed)
        return parsed

    def check(self, args: Any) -> None:
        """What the declaration asks of a package beyond its types.

        It lives beside :meth:`decode` rather than in a handler so that every entry into a tool —
        a call from the model, a call from a program, a replayed journal entry — is held to the same
        rule, and so that nothing has to remember the rule at the point where it runs. A rule that is
        about one argument is decided here; a rule two arguments make together is the tool's own
        :attr:`guard`, for the same reason.
        """
        if self.intention is Intention.REQUIRED and not _said(args):
            msg = "field required: this tool must be told why it is called"
            raise ArgsError("intention", msg)
        if self.guard is not None:
            self.guard(args)

    def approval(self, args: Any) -> tuple[Posture, Remember | None]:
        """The posture this call is approved at, and what the approval is remembered by.

        ``Remember`` is only meaningful for ``ONCE`` and is ``None`` otherwise. What a *policy*
        answers about this call (:class:`~plyngent.kernel.command.CommandPolicy`, the network policy,
        a grant for a path outside the root) is not here: it arrives at the executor and may only
        raise this value.
        """
        posture, remembers = _baseline(self.uses, self.mutates)
        if self.posture is not None:
            posture = self.posture
        for name, value in (self.escalates or {}).items():
            if argument(args, name) == value:
                return Posture.ALWAYS, None
        remember = self.remember if self.remember is not None else remembers
        return posture, remember if posture is Posture.ONCE else None

    def _read(self, arguments: str | bytes) -> Any:
        """The arguments as a package, in whichever of the two kinds this tool's are."""
        if isinstance(self.args, ExternalArgs):
            try:
                parsed = msgspec.json.decode(arguments)
            except msgspec.DecodeError as exc:
                raise ArgsError("", str(exc)) from exc
            if not isinstance(parsed, dict):
                msg = f"{self.name} takes a JSON object, not {type(parsed).__name__}"
                raise ArgsError("", msg)
            return cast("dict[str, Any]", parsed)
        try:
            return _decoder(self.args).decode(arguments)
        except msgspec.ValidationError as exc:
            # No path: 0.22 keeps the location in the message (" - at $.conns[2].port") and nowhere
            # else, and a regex over somebody else's wording is not a place to get a field name from.
            raise ArgsError("", str(exc)) from exc
        except ValueError as exc:
            # A package that refuses its own values (`Filter.ignore_lists` and its kind) is still
            # saying the arguments were wrong, which is an argument error and not a bug.
            raise ArgsError("", str(exc)) from exc

    def _declared_fields(self) -> frozenset[str]:
        """The argument names this tool declares, from the package it was given."""
        if isinstance(self.args, ExternalArgs):
            return self.args.properties
        if ToolArgs not in self.args.__mro__:
            msg = f"{self.name}: {self.args.__name__} must inherit ToolArgs, so that every call can be asked why"
            raise TypeError(msg)
        return frozenset(self.args.__struct_fields__)

    def _args_name(self) -> str:
        return type(self.args).__name__ if isinstance(self.args, ExternalArgs) else self.args.__name__


def _asked_for(uses: Res, mutates: Mutation) -> Intention:
    """Whether a tool with these declarations has to be told why it is called.

    Derived, not written per tool: forty-odd packages each spelling out the same rule is how the
    rule ends up with forty-one spellings. A tool may still say it explicitly.
    """
    if uses & (Res.EXEC | Res.HUMAN) or mutates & (Mutation.OUTSIDE | Mutation.DESTRUCTIVE):
        return Intention.REQUIRED
    return Intention.OPTIONAL


def _baseline(uses: Res, mutates: Mutation) -> tuple[Posture, Remember | None]:
    """The approval a tool's declarations ask for, before any policy looks at a call.

    First match wins, and the order is the point: a tool that *is* a question to the human is not
    approved like a tool that runs a program, and a write into a live process is not approved like a
    write into a file.

    ``EXEC`` comes before ``OUTSIDE`` because the tool that runs programs also declares that what they
    change cannot be verified here (``OUTSIDE | WORKSPACE``), and the other order would make *every*
    call of it ``ALWAYS`` — the per-command judgement of the command policy would never be reached.
    ``OUTSIDE`` is left to tools whose effects are in somebody else's hands with no policy able to
    judge a single call: writing a user directory, calling a client.
    """
    if uses & Res.HUMAN:
        outcome = (Posture.AUTO, None)  # the tool *is* the question; asking it again is a bug
    elif uses & Res.CLIENT:
        outcome = (Posture.ALWAYS, None)  # the effect happens in somebody else's process
    elif uses & Res.EXEC:
        outcome = (Posture.AUTO, None)  # whether a program is dangerous is the command policy's answer
    elif mutates & Mutation.OUTSIDE:
        outcome = (Posture.ALWAYS, None)  # and it cannot be verified here either
    elif uses & Res.PTY and mutates & Mutation.HOST:
        outcome = (Posture.ONCE, Remember.SESSION)  # a live process cannot be un-written, but one batch is enough
    elif mutates & Mutation.WORKSPACE:
        outcome = (Posture.ONCE, Remember.PATH)
    else:
        outcome = (Posture.AUTO, None)  # reading, and the bookkeeping a session does for itself
    return outcome


def argument(args: Any, name: str) -> Any:
    """One argument of a package, whichever of the two kinds it is.

    Public because more than one reader needs the *same* answer: approval escalations, the "did it say
    why" check, and the announcement of a call all look a named field up in a package that is either a
    tree-internal ``Struct`` or an external JSON object.
    """
    if isinstance(args, dict):
        return cast("dict[str, Any]", args).get(name)
    return getattr(args, name, None)


def _said(args: Any) -> bool:
    """Whether the package carries a reason that is not whitespace."""
    return bool(str(argument(args, "intention") or "").strip())


@cache
def _decoder(args: type[Struct]) -> msgspec.json.Decoder[Any]:
    return msgspec.json.Decoder(args)


def schema_of(args: type[Struct]) -> dict[str, Any]:
    """The schema for one argument package, with its nested types under ``$defs``.

    ``msgspec`` describes a struct as ``{"$ref": "#/$defs/Name"}`` plus the definitions, which is
    right for nesting and wrong for a tool's ``parameters``: a provider expects the object schema
    itself, and some reject a bare reference outright. So the top-level reference is resolved into
    the definition it names and only the *nested* types stay under ``$defs``.
    """
    schema = dict(msgspec.json.schema(args))
    definitions = dict(schema.pop("$defs", {}))
    reference = schema.pop("$ref", None)
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        named = definitions.pop(reference.removeprefix("#/$defs/"), {})
        schema = {**named, **schema}
    if definitions:
        schema["$defs"] = definitions
    return schema
