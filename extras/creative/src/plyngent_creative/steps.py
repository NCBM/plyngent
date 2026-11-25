"""What a script says, as steps a run can carry out, and what one step has to see happen.

A script is a linear sequence of static JSON steps, because a shape that is data can be written down,
diffed, and read by a person who is not the one who wrote it::

    [
      {"model": {"text": "I will need the file tools"}},
      {"send": "read my notes"},
      {"expect": {"tag": "tool-called", "fields": {"tool": "read_file"}}}
    ]

One step may hold all three keys, and then the order is fixed: the model answers, the line is sent, and
what arrives is looked for. An order written down once is what makes one step have one reading.

**``send``** is a string, which is a message somebody types, or an object, which is a command. A command is
the object the contract already defines — ``{"cmd": "compact", "text": "…"}`` — because the contract is
where its shape lives, and a second spelling of one command line would be a second grammar to keep in step.

**``expect``** looks forward in the events that arrived after the previous step and takes the first that
matches: ``tag`` is exact, ``fields`` is a recursive subset, and a scalar or a list inside it is equal or it
is not. What an author asserts on is the wire the kernel speaks, so an expectation names a ``tool-called``
and not a line somebody rendered (E184).

**Nothing here drives anything.** A step is a value, matching is a function of two values, and the run that
carries them out is :mod:`plyngent_creative.session`'s: a matcher that also waited would be untestable
without a kernel, and this module is where the rules are read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.model import Call, Reply
from plyngent.wire.command import AnyCommand
from plyngent_creative.model import Cut, Failing

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from plyngent.wire.event import AnyEvent

#: How many events a step that matched nothing shows. A script that failed is read by whoever wrote it, and
#: the whole conversation is one scroll away in the run; what fits on a line is what they look at first.
ARRIVED_SHOWN = 8


class ScriptError(ValueError):
    """A script this distribution will not run, with the sentence saying which step and what about it."""


def tag_of(event: AnyEvent) -> str:
    """The word this event has on the wire, which is the name an expectation uses for it."""
    return str(event.__struct_config__.tag)


def values(event: AnyEvent) -> dict[str, object]:
    """One event as the values the wire carries: an enum is its word and a tuple is a list.

    Read that way because that is what an expectation is written in: a script says ``"kind": "chat"`` and
    ``"targets": ["a"]``, so the two sides have to be the same kind of value before they are compared. The
    round trip through JSON is what makes a tuple a list, which is the one difference the encoder keeps.
    """
    return cast("dict[str, object]", msgspec.json.decode(msgspec.json.encode(event)))


@dataclass(frozen=True, slots=True)
class Modelled:
    """What the scripted model answers this time, in the three shapes trouble can take.

    ``failure`` is the one kind of trouble this step is about: the model could not be asked at all, or — with
    ``cut`` — it answered and the answer broke after that many pieces. ``text`` is what it says, and
    ``calls`` are the tool calls it asks for beside it, which is the pair a model answers with.
    """

    text: str = ""
    calls: tuple[Call, ...] = ()
    failure: Failure | None = None
    cut: int | None = None
    says: str = ""

    def scripted(self) -> Reply | Cut | Failing:
        """This answer as the model's own value, which is what a scripted model is handed."""
        if self.failure is None:
            return Reply(text=self.text, calls=self.calls)
        if self.cut is None:
            return Failing(failure=self.failure, says=self.says)
        return Cut(pieces=_said(self.text, self.cut), failure=self.failure, says=self.says)


@dataclass(frozen=True, slots=True)
class Expected:
    """One event a step waits for, named by its wire tag and the fields it has to carry."""

    tag: str
    fields: Mapping[str, object]

    def wants(self, event: AnyEvent) -> bool:
        """Whether this is the event the step was waiting for."""
        return tag_of(event) == self.tag and _contains(self.fields, values(event))

    def says(self) -> str:
        """What was expected, so that a step which matched nothing says what it was looking for."""
        return json.dumps({"tag": self.tag, "fields": self.fields}, sort_keys=True)


@dataclass(frozen=True, slots=True)
class Step:
    """One step of a script: what the model answers, what is sent, and what has to arrive."""

    model: Modelled | None = None
    send: str | AnyCommand | None = None
    expect: Expected | None = None


def parse(data: object) -> tuple[Step, ...]:
    """Read a script, or say which step is not one."""
    if not isinstance(data, list):
        msg = f"a script is a list of steps; this is a {type(data).__name__}"
        raise ScriptError(msg)
    steps = cast("list[object]", data)
    return tuple(_step(one, at) for at, one in enumerate(steps))


def arrived(events: Iterable[AnyEvent], *, at_most: int = ARRIVED_SHOWN) -> str:
    """What arrived, one line per event, newest last — the other half of a step that matched nothing.

    Only the values a line can hold are shown: a field that is a structure or a list says nothing in one
    line, and the event's own tag is the part of it an expectation names anyway.
    """
    said = [_line(one) for one in events]
    return "\n".join(said[-at_most:]) or "nothing"


def _line(event: AnyEvent) -> str:
    """One event as ``tag`` and the plain fields it carries."""
    held = [(name, value) for name, value in values(event).items() if _plain(value)]
    return " ".join([tag_of(event), *(f"{name}={value!r}" for name, value in held)])


def _plain(value: object) -> bool:
    """Whether a field's value fits on one line: a scalar and nothing else."""
    return value is None or isinstance(value, (str, int, float, bool))


def _step(one: object, at: int) -> Step:
    """One step, or the sentence naming what is wrong with it."""
    where = f"step {at + 1}"
    held = _object(one, where)
    _only(held, ("model", "send", "expect"), where)
    if not held:
        msg = f"{where} is empty: a step says what the model answers, what is sent, or what arrives"
        raise ScriptError(msg)
    return Step(
        model=None if "model" not in held else _modelled(held["model"], where),
        send=None if "send" not in held else _sent(held["send"], where),
        expect=None if "expect" not in held else _expected(held["expect"], where),
    )


def _modelled(data: object, where: str) -> Modelled:
    """What the model answers, in the three shapes trouble can take."""
    held = _object(data, where)
    _only(held, ("text", "calls", "fail", "cut_after", "failure", "says"), where)
    text = _text(held.get("text", ""), where)
    calls = tuple(_call(one, where) for one in _list(held.get("calls", []), where))
    fail = None if "fail" not in held else _failure(held["fail"], where)
    cut = None if "cut_after" not in held else _count(held["cut_after"], where)
    broke = None if "failure" not in held else _failure(held["failure"], where)
    says = _text(held.get("says", ""), where)
    if fail is not None and cut is not None:
        msg = f"{where} says fail and cut_after: the model could not be asked, or it answered and broke"
        raise ScriptError(msg)
    if cut is not None and broke is None:
        msg = f"{where} says cut_after and no failure: what broke after {cut} pieces is what has to be named"
        raise ScriptError(msg)
    if broke is not None and cut is None:
        msg = f"{where} says failure and no cut_after: a failure that ends an answer says how much arrived"
        raise ScriptError(msg)
    if says and fail is None and cut is None:
        msg = f"{where} says says and no trouble: what the model said about itself is said by fail or cut_after"
        raise ScriptError(msg)
    if cut is not None and cut < 1:
        msg = f"{where} says cut_after {cut}: an answer says at least one piece before it breaks"
        raise ScriptError(msg)
    return Modelled(text=text, calls=calls, failure=fail if fail is not None else broke, cut=cut, says=says)


def _call(data: object, where: str) -> Call:
    """One tool call the model asks for, as the contract spells it."""
    held = _object(data, where)
    _only(held, ("tool", "arguments", "id"), where)
    tool = _text(held.get("tool", ""), where)
    if not tool:
        msg = f"{where} has a call with no tool: a call says which tool it is for"
        raise ScriptError(msg)
    arguments = held.get("arguments", {})
    spelled = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return Call(id=_text(held.get("id", ""), where) or tool, tool=tool, arguments=spelled)


def _sent(data: object, where: str) -> str | AnyCommand:
    """A line somebody types, or a command in the shape the contract defines for it."""
    if isinstance(data, str):
        return data
    held = _object(data, where)
    try:
        return cast("AnyCommand", msgspec.json.decode(json.dumps(held), type=AnyCommand))
    except msgspec.ValidationError as exc:
        msg = f"{where} sends a command the contract does not have: {exc}"
        raise ScriptError(msg) from exc


def _expected(data: object, where: str) -> Expected:
    """One event to wait for: its tag, and the fields it has to carry."""
    held = _object(data, where)
    _only(held, ("tag", "fields"), where)
    tag = _text(held.get("tag", ""), where)
    if not tag:
        msg = f"{where} expects an event with no tag: a tag is how an event is named"
        raise ScriptError(msg)
    fields = _object(held.get("fields", {}), where)
    return Expected(tag=tag, fields=fields)


def _object(data: object, where: str) -> dict[str, object]:
    """One JSON object, or the sentence saying what was there instead."""
    if not isinstance(data, dict):
        msg = f"{where} is a {type(data).__name__} where an object belongs"
        raise ScriptError(msg)
    return cast("dict[str, object]", data)


def _list(data: object, where: str) -> list[object]:
    """One JSON list, or the sentence saying what was there instead."""
    if not isinstance(data, list):
        msg = f"{where} is a {type(data).__name__} where a list belongs"
        raise ScriptError(msg)
    return cast("list[object]", data)


def _text(data: object, where: str) -> str:
    """One string, or the sentence saying what was there instead."""
    if not isinstance(data, str):
        msg = f"{where} has a {type(data).__name__} where a string belongs"
        raise ScriptError(msg)
    return data


def _count(data: object, where: str) -> int:
    """One whole number, or the sentence saying what was there instead."""
    if not isinstance(data, int) or isinstance(data, bool):
        msg = f"{where} has a {type(data).__name__} where a number belongs"
        raise ScriptError(msg)
    return data


def _only(held: Mapping[str, object], allowed: tuple[str, ...], where: str) -> None:
    """Refuse a key this step does not know, since a key nothing reads is a step that says less than it looks."""
    unknown = sorted(set(held) - set(allowed))
    if unknown:
        msg = f"{where} says {', '.join(unknown)}, and the words here are {', '.join(allowed)}"
        raise ScriptError(msg)


def _failure(word: object, where: str) -> Failure:
    """One member of the kernel's closed set of trouble, by the word the contract spells it with."""
    known = _text(word, where)
    try:
        return _FAILURES[known]
    except KeyError:
        msg = f"{where} says {known!r}, and the kinds of trouble are {', '.join(sorted(_FAILURES))}"
        raise ScriptError(msg) from None


def _said(text: str, pieces: int) -> tuple[str, ...]:
    """The first ``pieces`` words of ``text``, each with the space that follows it, which is what arrived.

    A piece is one word and the space after it: an answer that broke after two pieces handed over two
    words and nothing after them, so a listener that joins them reads a sentence that stops where the
    stream stopped.
    """
    return tuple(f"{word} " for word in text.split(" ")[:pieces])


def _contains(wanted: object, found: object) -> bool:
    """Whether ``found`` carries ``wanted``: an object is a subset of another, and everything else is equal."""
    if not isinstance(wanted, dict):
        return bool(found == wanted)
    held = cast("dict[str, object]", found) if isinstance(found, dict) else {}
    return all(key in held and _contains(value, held[key]) for key, value in cast("dict[str, object]", wanted).items())


_FAILURES: dict[str, Failure] = {one.value: one for one in Failure}
