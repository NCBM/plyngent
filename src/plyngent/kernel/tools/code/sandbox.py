"""The process a program runs in, and the one method it may ask this end for.

A program is Python, and the point of it is that a loop, a filter and a dozen calls cost one answer
instead of a dozen rounds. So it runs where it cannot reach anything by accident: a process of its own,
started with ``-I -S``, speaking one line of JSON at a time over the pipes this end already holds. The
conversation is
:class:`~plyngent.kernel.rpc.peer.Peer` with a :class:`~plyngent.kernel.rpc.serving.Serving` — the same
framing the child servers speak — and it is **not** the wire protocol: no frontend and no model ever
sees one of these lines (E190).

Three decisions are in :data:`BOOTSTRAP` and nowhere else, because they are the whole of what keeps a
program inside its plan:

* **The namespace is the fence.** ``__builtins__`` is a curated dict with no ``getattr``, no ``open`` and
  no ``__import__``, so "reach it by name" does not parse. The seven modules a program gets are injected
  as names, and the plan's tools are the only functions it can call.
* **``sys.stdout`` is a buffer and the protocol has the real handle.** A program's ``print`` is the one
  way back into the conversation, so it must not share a line with a protocol message: the real handle is
  kept before the swap, and everything printed is answered at the end.
* **The audit hook is the teeth.** It is installed after this end has done its own importing and
  compiling, and it refuses the nine events a program has no business raising. A static scan is evidence
  and says so; this is what holds.

What the child answers is not a result: it answers the printed text and, when the program stopped, the
kind of failure and the sentence to read. Turning that into a
:class:`~plyngent.kernel.toolresult.ToolResult` is this module's, because ``bodies.kept`` needs the store
and the store is here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from itertools import count
from typing import TYPE_CHECKING, Any, cast

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.render import render_body
from plyngent.kernel.rpc.child import CannotStart, Child
from plyngent.kernel.rpc.peer import PeerGone, ProtocolBroken, RpcError
from plyngent.kernel.rpc.serving import Serving
from plyngent.kernel.toolresult import DenialKind, Meta, ToolResult
from plyngent.kernel.tools.bodies import kept
from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.kernel.env import Env
    from plyngent.kernel.toolresult import Part
    from plyngent.kernel.toolspec import ToolSpec

#: The one method a program may ask this end for: run one step of its plan, under the call that runs it.
STEP = "call"

#: How long one program may run, counted from the moment its process is asked to. The turn's own cancel is
#: a different thing: it needs somebody to press something, and a loop nobody is watching has no such
#: moment (E190).
WALL_SECONDS = 120.0

#: How many steps one program may take. Two hundred is far past any plan somebody wrote by hand and far
#: short of a loop over a directory nothing told it the size of.
MAX_CALLS = 200

#: How much text one program may be, and how much it may print. One number for both ends, because it is
#: one cost: what this end hands a child on its command line, and what the child holds in a buffer to hand
#: back. The printed half is counted in the child, since a loop that prints is what the limit is for.
MAX_CHARS = 1 << 20

#: The program that runs the program: a JSON-RPC server on its own pipes, and the fence around the code.
#:
#: It is a string because it is not importable: the child is started with ``-I -S``, which is what keeps a
#: module the model's program could name out of its path, so everything it needs is written here once.
BOOTSTRAP = r'''
import builtins, io, json, keyword, sys

_real_out = sys.stdout
_real_in = sys.stdin
_number = 0


class Exhausted(Exception):
    """The program ran past what a program here is given: a step, or the text going in and out."""


class Broken(Exception):
    """The conversation with the kernel ended, so nothing more can be called."""


class _Buffer(io.StringIO):
    """Where a program's prints go, and the one place the size of them is counted.

    The limit is on the buffer and not on the answer, because a loop that prints a line forever is what it
    is for: counting afterwards is counting after the buffer was filled.
    """

    def write(self, text):
        if _MAX_CHARS and self.tell() + len(text) > _MAX_CHARS:
            _over.append("the program printed more than %d characters" % _MAX_CHARS)
            raise Exhausted(_over[-1])
        return io.StringIO.write(self, text)


_over = []
_printed = _Buffer()


def _say(message):
    _real_out.write(json.dumps(message) + "\n")
    _real_out.flush()


def _hear():
    line = _real_in.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def _ask(method, params):
    global _number
    _number += 1
    _say({"jsonrpc": "2.0", "id": _number, "method": method, "params": params})
    while True:
        message = _hear()
        if "method" in message:
            continue
        error = message.get("error")
        if error is not None:
            raise Broken(str(error.get("message") or "the kernel could not answer"))
        return message.get("result")


class Result:
    """What one call came back with: the text, whether it worked, and the parts that are not text."""

    def __init__(self, answer):
        self.status = answer.get("status") or "failed"
        self.failure = answer.get("failure") or ""
        self.denial = answer.get("denial") or ""
        self.summary = answer.get("summary") or ""
        self.parts = answer.get("parts") or []
        self.documents = answer.get("documents") or []
        self.hints = answer.get("hints") or []
        self._text = answer.get("text") or ""

    @property
    def ok(self):
        """Whether the call did what it was asked."""
        return self.status == "ok"

    def __str__(self):
        return self._text

    def __repr__(self):
        return "Result(%r, %r)" % (self.status, self.summary)


class ToolFailed(Exception):
    """A call this program made did not do what it was asked; the result it carried is in ``result``."""

    def __init__(self, result, name=""):
        self.result = result
        told = "%s (call it as: %s)" % (result.summary, _spelled(name)) if name else result.summary
        super().__init__(told)


class Objection(Exception):
    """What the audit hook raises: this program tried to do something it was not given."""


def _fields(schema):
    return schema.get("properties") or {}


def _required(schema):
    return set(schema.get("required") or ())


def _ordered(schema):
    """The declared fields with the required ones first: Python refuses a default before a plain name."""
    properties = _fields(schema)
    required = _required(schema)
    return [name for name in properties if name in required] + [
        name for name in properties if name not in required
    ]


def _spelled(name):
    """One function's signature, written out: how to call it, and which fields it takes."""
    schema = _tools.get(name) or {}
    properties = _fields(schema)
    required = _required(schema)
    said = []
    for field in _ordered(schema):
        said.append(field if field in required else "%s=%r" % (field, properties[field].get("default")))
    return "%s(%s)" % (name, ", ".join(said))


def _source(name):
    """The Python of one generated function: its real signature, its docstring, and its one call."""
    schema = _tools[name]
    properties = _fields(schema)
    required = _required(schema)
    said = []
    for field in _ordered(schema):
        said.append(field if field in required else "%s=%r" % (field, properties[field].get("default")))
    given = ", ".join("%s=%s" % (field, field) for field in _ordered(schema))
    docs = schema.get("description") or ""
    notes = [
        "%s: %s" % (field, properties[field]["description"])
        for field in properties
        if properties[field].get("description")
    ]
    if notes:
        docs = docs + "\n\n" + "\n".join(notes)
    return "def %s(%s):\n    %r\n    return call(%r, %s)\n" % (name, ", ".join(said), docs, name, given)


def _callable(name):
    """Whether this plan's tool can also be a function here: a real name, and not one already taken."""
    return name.isidentifier() and not keyword.iskeyword(name) and name not in _TAKEN


def call(name, **kwargs):
    """Call one tool of this program's plan, and answer what it came back with."""
    _calls[0] += 1
    if _MAX_CALLS and _calls[0] > _MAX_CALLS:
        _over.append("this program made more than %d calls" % _MAX_CALLS)
        raise Exhausted(_over[-1])
    result = Result(_ask("call", {"name": name, "kwargs": kwargs}))
    if not result.ok:
        raise ToolFailed(result, name)
    return result


def help(*what):
    """Print the signature and the fields of the tools this program may call."""
    names = []
    if not what:
        names = list(_tools)
    for item in what:
        for one in item if isinstance(item, (tuple, list, set)) else (item,):
            name = one if isinstance(one, str) else getattr(one, "__name__", "")
            if name not in names:
                names.append(name)
    unknown = [name for name in names if name not in _tools]
    for name in names:
        if name in _tools:
            print(_spelled(name))
            docs = _tools[name].get("description") or ""
            if docs:
                print(docs)
    if unknown:
        print("not here: %s. this program may call: %s" % (", ".join(unknown), ", ".join(_tools)))


def _safe():
    """The builtins a program is given: the ordinary ones, minus every way to reach past a name."""
    dropped = {
        "getattr", "setattr", "delattr", "vars", "globals", "locals", "eval", "exec", "compile", "type",
        "open", "input", "breakpoint", "__import__", "help", "dir", "exit", "quit", "memoryview",
        "__loader__", "__spec__", "__package__", "__doc__",
    }
    safe = {
        name: value
        for name, value in vars(builtins).items()
        if name not in dropped and (not name.startswith("_") or name == "__build_class__")
    }
    safe["__import__"] = _import
    return safe


_real_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    """The one way a program could load code: refuse every module this program was not given.

    This is the import fence, and it is a function instead of an audit event because the event cannot
    tell a program's ``import`` from a standard-library module loading something on its first use — the C
    side of ``datetime`` looks ``__import__`` up in the *calling* frame's builtins, which is this dict.
    So the list is what this bootstrap loaded for the program (:data:`_IMPORTABLE`), and a name outside it
    is a program reaching for a module nobody gave it.
    """
    if level == 0 and name.partition(".")[0] in _IMPORTABLE:
        return _real_import(name, globals, locals, fromlist, level)
    raise Objection("import %s" % name)


_BANNED = (
    "open", "socket.connect", "subprocess.Popen", "os.system", "os.remove", "os.rename", "compile", "exec",
)
_ours = [0]
_stopped = []


def _watch(event, args):
    """Refuse one of the eight events a program has no business raising, and say which after the fact."""
    if _ours[0]:
        return
    if event in _BANNED:
        _stopped.append(event)
        raise Objection(event)


_run = _hear()
_params = _run.get("params") or {}
_tools = _params.get("tools") or {}
_limits = _params.get("limits") or {}
_MAX_CALLS = int(_limits.get("calls") or 0)
_MAX_CHARS = int(_limits.get("chars") or 0)
_calls = [0]
_TAKEN = {
    "call", "help", "print", "Result", "ToolFailed", "math", "json", "re", "itertools", "collections",
    "datetime", "textwrap",
}
_modules = {}
for _name in ("math", "json", "re", "itertools", "collections", "datetime", "textwrap"):
    _modules[_name] = __import__(_name)
# One call into each module that loads something on first use, so that what a program reaches is loaded
# before the fence is up: `strptime` asks for `_strptime`, and a program refused for that would be refused
# for the standard library's own laziness.
_ = _modules["datetime"].datetime.strptime("2000-01-01", "%Y-%m-%d")
#: The modules a program may name in an import: the seven it is given, plus the one the C side of
#: ``datetime`` asks its caller's builtins for. Everything else is a module it was not given.
_IMPORTABLE = frozenset(_modules) | {"time"}

_namespace = dict(_modules)
_namespace.update(
    {
        "__builtins__": _safe(),
        "__name__": "toolscript",
        "call": call,
        "help": help,
        "Result": Result,
        "ToolFailed": ToolFailed,
    }
)

_generated = "\n".join(_source(name) for name in _tools if _callable(name))
_generated = "\n".join(_source(name) for name in _tools if _callable(name))
if _generated:
    exec(compile(_generated, "<tools>", "exec"), _namespace)

_failure = ""
_denial = ""
_summary = ""
_program = None
try:
    _program = compile(_params.get("code") or "", "<program>", "exec")
except SyntaxError as _exc:
    _failure, _summary = "args", "this program is not Python: %s" % _exc.msg
sys.addaudithook(_watch)
if _program is not None:
    sys.stdout = _printed
    try:
        _ours[0] += 1
        try:
            exec(_program, _namespace)
        finally:
            _ours[0] -= 1
    except ToolFailed as _exc:
        _failure = _exc.result.failure or "tool"
        _denial = _exc.result.denial or ""
        _summary = str(_exc)
    except Objection as _exc:
        _failure, _denial = "denied", "rule"
        _summary = "this program tried to %s, which it is not given" % _exc
    except Exhausted as _exc:
        _failure, _summary = "budget", str(_exc)
    except Broken as _exc:
        _failure, _summary = "protocol", str(_exc)
    except BaseException as _exc:
        _failure = "args"
        _summary = "the program stopped: %s: %s" % (type(_exc).__name__, _exc)
    finally:
        sys.stdout = _real_out
if _stopped and not _failure:
    _failure, _denial = "denied", "rule"
    _summary = "this program tried to %s, which it is not given" % _stopped[0]
if _over:
    _failure, _summary = "budget", _over[0]
_say(
    {
        "jsonrpc": "2.0",
        "id": _run.get("id"),
        "result": {"said": _printed.getvalue(), "failure": _failure, "denial": _denial, "summary": _summary},
    }
)
'''


async def played(env: Env, *, code: str, plan: Mapping[str, ToolSpec]) -> ToolResult:
    """Run one program in a process of its own, and answer what it printed (E190).

    The question that allowed the program was asked by the caller, so nothing here asks again: a program
    whose every step were put to a person is a wall of prompts in place of the one the plan is.

    Three budgets are enforced and none of them is a number a caller passes: the wall clock is a deadline on
    this call, and the other two are the child's own, because the child is what knows how many steps it has
    taken and how much it has printed. Going past any of them is :attr:`~plyngent.kernel.failure.Failure.
    BUDGET`, which is never retried — the same program reaches the same wall — and the ``stop`` on the way
    out takes the whole process group with it.

    A child that never started is a result, a child that stopped speaking is a result, and a child that
    was stopped with the turn is the caller's cancellation — the three are all things the caller can read,
    which is why the ``stop`` happens on the way out either way.
    """
    started = time.monotonic()
    child = Child(
        [sys.executable, "-I", "-S", "-c", BOOTSTRAP],
        cwd=env.workspace.root,
        serving=_serving(env, plan),
    )
    try:
        peer = await child.start()
    except CannotStart as exc:
        return ToolResult.of(Failure.of(exc.caused), str(exc))
    try:
        answer = await peer.call(
            "run",
            {
                "code": code,
                "tools": {name: spec.schema() for name, spec in plan.items()},
                "limits": {"calls": MAX_CALLS, "chars": MAX_CHARS},
            },
            timeout_seconds=WALL_SECONDS,
        )
    except TimeoutError:
        return ToolResult.of(
            Failure.BUDGET,
            f"the program ran for {WALL_SECONDS:g} seconds and was stopped; its process group is gone",
            meta=Meta(elapsed_ms=_since(started)),
        )
    except RpcError as exc:
        return ToolResult.of(Failure.PROTOCOL, str(exc), meta=Meta(elapsed_ms=_since(started)))
    except (PeerGone, ProtocolBroken) as exc:
        return ToolResult.of(
            Failure.PROTOCOL,
            f"the program stopped before it said what it printed: {exc}{_complaint(child)}",
            meta=Meta(elapsed_ms=_since(started)),
        )
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await child.stop()
    return await _answered(env, answer, elapsed=_since(started))


def _serving(env: Env, plan: Mapping[str, ToolSpec]) -> Serving:
    """The one method a program may ask this end for, and the steps it is counted in.

    A name outside the plan is refused here and not by the executor, because it never was a call: the
    plan is what the person agreed to, so a name the plan does not hold is this layer's own answer
    (§10.13).
    """
    steps = count(1)

    async def step(params: Mapping[str, Any]) -> dict[str, Any]:
        name, arguments = params.get("name"), params.get("kwargs")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            msg = "a step is a tool name and an object of keyword arguments"
            raise TypeError(msg)
        if name not in plan:
            return _as_read(
                ToolResult.denied(
                    DenialKind.RULE,
                    f"{name!r} is not one of this program's tools, and the plan is what was agreed to; "
                    f"it may call: {', '.join(plan) or 'nothing'}",
                )
            )
        return _as_read(await env.dispatch(name, json.dumps(arguments), step=next(steps)))

    return Serving({STEP: step})


def _as_read(result: ToolResult) -> dict[str, Any]:
    """One step's result as the program reads it: the text, the status, and the parts that are not text.

    ``text`` is the projection the model itself reads, so a program that prints a result prints what a
    round would have been told — the same sentence, from the same function, and no second rendering to
    drift from it.
    """
    return {
        "status": result.status.value,
        "failure": "" if result.failure is None else result.failure.value,
        "denial": "" if result.denial is None else result.denial.value,
        "summary": result.summary,
        "text": render_body(result.body),
        "parts": msgspec.to_builtins(result.body),
        "documents": list(result.handles),
        "hints": [
            {"tool": hint.tool, "given": [list(pair) for pair in hint.given], "why": hint.why} for hint in result.hints
        ],
    }


async def _answered(env: Env, answer: Any, *, elapsed: int) -> ToolResult:
    """What the program printed, as the one result the round reads.

    Nothing printed is nothing handed back, and it is said as what it is: a program that prints nothing
    has left whatever its steps left, which is in their own rows, and has told the conversation nothing.
    """
    heard = _heard(answer)
    said = str(heard.get("said") or "")
    body: Part | None = None
    if said:
        kept_part = await kept(env, said, summary="what the program printed")
        if isinstance(kept_part, ToolResult):
            return kept_part
        body = kept_part
    meta = Meta(elapsed_ms=elapsed)
    failure = _kind(heard)
    if failure is Failure.DENIED:
        return ToolResult.denied(_denial(heard), str(heard.get("summary") or ""), body=body, meta=meta)
    if failure is not None:
        return ToolResult.of(failure, str(heard.get("summary") or ""), body=body, meta=meta)
    if not said:
        return ToolResult.ok("the program printed nothing, so nothing came back", meta=meta)
    return ToolResult.ok(f"the program printed {counted(len(said), 'character')}", body=body, meta=meta)


def _heard(answer: Any) -> Mapping[str, Any]:
    """The child's answer as the object it is, or nothing when it said something that is not one."""
    return cast("Mapping[str, Any]", answer) if isinstance(answer, dict) else {}


def _kind(heard: Mapping[str, Any]) -> Failure | None:
    """The failure the child named, or ``None`` when the program finished."""
    named = str(heard.get("failure") or "")
    if not named:
        return None
    try:
        return Failure(named)
    except ValueError:
        return Failure.PROTOCOL


def _denial(heard: Mapping[str, Any]) -> DenialKind:
    """Why a program that was refused was refused: the step's own kind, or this build's rules."""
    named = str(heard.get("denial") or "")
    try:
        return DenialKind(named)
    except ValueError:
        return DenialKind.RULE


def _complaint(child: Child) -> str:
    """What the child said on its error stream, as the tail of a sentence, when it said anything."""
    told = child.complaint
    return f" — it said: {told}" if told else ""


def _since(started: float) -> int:
    """How long this took, in the milliseconds a result reports."""
    return int((time.monotonic() - started) * 1000)
