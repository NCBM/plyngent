"""What this terminal offers the kernel, and what it does when the kernel asks for one of them.

Asking a person is a tool the *client* provides: the kernel does not read stdin, and it has no way to
put a question in front of anybody. So the terminal offers the one way it can ask — ``ask_user`` — and
answers it on the invocation lane (§10.8). What a question *looks* like is the kernel's answer, not this
file's: an offer of that name carries no schema, and the field names read below are the ones the kernel's
own package declares.

Two rules this file keeps:

* **Nothing but asking.** This client offers no other tool: its display and its Ctrl+C are a renderer
  and a command, and putting them on this lane would be a privilege no other client has (§10.8 rule 5).
* **Every invocation is answered.** A call the kernel is waiting on ends in a deadline if nobody
  answers it, so a name this client does not have is answered as one it could not run, and nothing is
  dropped on the floor.

The call is an array of questions, and this lane cannot put a block of them up in one go: there is one
cursor here and one line at a time, so they are asked in sequence and the answer is one line per question
(§0.13, E95). A frontend that draws a real form answers in the same words, which is why the kernel's
shape — an array of questions with a label each — is not this file's invention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from msgspec.structs import replace as placed

from plyngent.wire.ask import (
    Agreed,
    By,
    Confirmation,
    FreeText,
    Invocation,
    Picked,
    Returned,
    Selection,
    Severity,
    Written,
)
from plyngent.wire.command import OfferedTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from plyngent.cli.line import LineEditor
    from plyngent.wire.ask import AnyAnswer

#: The names this terminal answers, and how to run one: the answer is one line of prose, either way,
#: and ``None`` from the editor means "nobody answered", where an empty line is an answer (§10.8, E48).
type Handler = Callable[[LineEditor, Invocation], Awaitable[Returned]]

#: Ask the person at the terminal: one question, or several as one array.
ASK = "ask_user"


@dataclass(frozen=True, slots=True)
class ClientTools:
    """One client's offer: the tools, and the one place each of them is run.

    One value, with no pair of module-level tables, because it is what a program is handed: a frontend
    that can implement more than the terminal does offers them the same way, and a story that needs a
    client-provided tool passes a bigger value, leaving this module untouched.
    """

    offer: tuple[OfferedTool, ...]
    handlers: Mapping[str, Handler]


async def dispatch(tools: ClientTools, editor: LineEditor, invocation: Invocation) -> Returned:
    """Run one invocation and answer it — including with "this client cannot".

    The answer names the invocation it belongs to, so the kernel never has to guess what came back. An
    exception from a handler is answered the same way as a missing one: a tool failing on this side is
    news for the model, not a reason for the loop to stop reading the lane.
    """
    handler = tools.handlers.get(invocation.tool)
    if handler is None:
        return Returned(id=invocation.id, text=f"this client has no tool named {invocation.tool!r}")
    try:
        return await handler(editor, invocation)
    except Exception:  # noqa: BLE001 — a tool that failed here is a result, and the lane must go on
        return Returned(id=invocation.id, text=f"the client could not run {invocation.tool}")


async def _ask(editor: LineEditor, invocation: Invocation) -> Returned:
    """Put the call's questions to the person one at a time, and answer one line per question.

    The block is one answer in the kernel's sense and a sequence of them here: this lane holds one cursor
    and draws one line, and a block of labelled questions is a screen's business (R8). Each line of the
    answer starts with the question's label when it has one, so what the model reads back is what it
    asked for — a frontend that did draw a real form would say the same words, and the kernel would not
    have to know which one answered.
    """
    fields = _fields(invocation.arguments)
    title = _text(fields, "title")
    filled: list[str] = []
    for raw in _items(fields.get("questions")):
        question = _question(_object(raw), title=title)
        value = _answer(await editor.ask(_where(question, invocation)))
        if value is None:
            return Returned(id=invocation.id, text="")
        filled.append(f"{question.label}: {value}" if question.label else value)
    return Returned(id=invocation.id, ok=bool(filled), text="\n".join(filled))


def _where[Q: Confirmation | FreeText | Selection](question: Q, invocation: Invocation) -> Q:
    """The question, at the place the call came from: a frontend with several sessions shows it there."""
    return placed(question, session=invocation.session, turn=invocation.turn)


def _question(raw: Mapping[str, Any], *, title: str) -> Confirmation | FreeText | Selection:
    """One question of a call, rebuilt as the value this lane puts.

    The kind is the tag the contract gives it, and the fields are the ones the kernel's own package
    declares — the call was validated against them before it arrived, so nothing here is invented or
    re-checked. Two things are deliberately not carried over: the addressing a question has on the lane
    (``session``, ``turn``, ``scope``, ``id``), which is this invocation's, and a deadline, which belongs
    to whoever is holding the cursor.
    """
    prompt = _titled(title, _text(raw, "prompt"))
    label = _text(raw, "label")
    default = _text(raw, "default") or None
    options = tuple(str(one) for one in _items(raw.get("options")))
    match raw.get("ask"):
        case "confirmation":
            offered = raw.get("default")
            return Confirmation(
                prompt=prompt,
                default=offered if isinstance(offered, bool) else None,
                severity=Severity(str(raw.get("severity") or Severity.CONFIRM)),
                label=label,
                executable_code=_text(raw, "executable_code"),
            )
        case "selection" if options:
            return Selection(
                prompt=prompt,
                options=options,
                default=default,
                multiple=bool(raw.get("multiple")),
                label=label,
            )
        case _:
            return FreeText(prompt=prompt, default=default, label=label)


def _titled(title: str, prompt: str) -> str:
    """A question as a person reads it in a sequence: the block's heading, then the question."""
    return f"{title}: {prompt}" if title else prompt


def _answer(answer: AnyAnswer) -> str | None:
    """What one question's answer said, or ``None`` when nobody answered it.

    A yes-or-no is written as the word a person reads, because the model reading a block back wants the
    answers side by side and ``true`` next to ``plain`` reads like two kinds of value. Several picks are
    one line with commas, which is how a line holds a list; a single pick is itself.

    "Nobody answered" is read from the answer's author and not from a value: a confirmation has a bool
    either way, so a declined one and a real "no" would otherwise be the same answer to a block that is
    one answer or none.
    """
    if answer.by is By.TIMEOUT or answer.by is By.NOBODY:
        return None
    match answer:
        case Agreed(accepted=accepted):
            return "yes" if accepted else "no"
        case Picked(values=values) if values:
            return ", ".join(values)
        case Picked(value=value):
            return value
        case Written(text=text):
            return text


def _fields(arguments: str) -> Mapping[str, Any]:
    """A call's arguments, as the object the kernel already validated them into."""
    parsed: object = json.loads(arguments) if arguments else {}
    return cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}


def _text(fields: Mapping[str, Any], name: str) -> str:
    """One string field of a call, or nothing: what is not a string is not a prompt."""
    value = fields.get(name)
    return value if isinstance(value, str) else ""


def _object(value: object) -> Mapping[str, Any]:
    """One object of a call, or an empty one: what is not an object names no field."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _items(value: object) -> list[object]:
    """One array of a call, or an empty one."""
    return cast("list[object]", value) if isinstance(value, list) else []


#: What the terminal offers: the one question it can put in front of a person, always there — a model
#: that may ask has to be able to ask before it has been shown anything (§10.8, E44).
TERMINAL = ClientTools(
    offer=(OfferedTool(name=ASK, resident=True),),
    handlers={ASK: _ask},
)
