"""What this terminal offers the kernel, and what it does when the kernel asks for one of them.

Asking a person is a tool the *client* provides: the kernel does not read stdin, and it has no way to
put a question in front of anybody. So the terminal offers the two ways it can ask, and answers them on
the invocation lane (§10.8). What a question *looks* like is the kernel's answer, not this file's: an
offer of one of these names carries no schema, and the field names read below are the ones the kernel's
own package declares.

Two rules this file keeps:

* **Nothing but asking.** This client offers no other tool: its display and its Ctrl+C are a renderer
  and a command, and putting them on this lane would be a privilege no other client has (§10.8 rule 5).
* **Every invocation is answered.** A call the kernel is waiting on ends in a deadline if nobody
  answers it, so a name this client does not have is answered as one it could not run rather than
  dropped on the floor.

A form is the third way to ask and the only one this lane cannot put up in one go: there is one cursor
here and one line at a time, so its fields are asked in sequence and the answer is one line per field
(§0.13, E95). A frontend that draws a real form answers in the same words, which is why the kernel's
shape — an array of named fields — is not this file's invention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from msgspec.structs import replace as placed

from plyngent.wire.ask import Agreed, By, FreeText, Invocation, Picked, Returned, Selection, Written
from plyngent.wire.command import OfferedTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from plyngent.cli.line import LineEditor
    from plyngent.wire.ask import AnyAnswer

#: The names this terminal answers, and how to run one: the answer is one line of prose, either way,
#: and ``None`` from the editor is "nobody answered" rather than an empty answer (§10.8, E48).
type Handler = Callable[[LineEditor, Invocation], Awaitable[Returned]]

#: Ask for a line of free text.
LINE = "ask_user_line"

#: Ask for one of a few options.
CHOICE = "ask_user_choice"

#: Ask for a set of fields, one line each.
FORM = "ask_user_form"


@dataclass(frozen=True, slots=True)
class ClientTools:
    """One client's offer: the tools, and the one place each of them is run.

    A value rather than two module-level tables, because it is what a program is handed: a frontend
    that can implement more than the terminal does offers them the same way, and a story that needs a
    client-provided tool passes a bigger value instead of patching this module.
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


async def _ask_line(editor: LineEditor, invocation: Invocation) -> Returned:
    """Put one line of free text to the person at the terminal, and answer what they wrote."""
    fields = _fields(invocation.arguments)
    question = FreeText(prompt=_text(fields, "question"), default=_text(fields, "default") or None)
    answer = await editor.ask(_where(question, invocation))
    written = answer if isinstance(answer, Written) else Written(by=By.NOBODY)
    return Returned(id=invocation.id, ok=written.text is not None, text=written.text or "")


async def _ask_choice(editor: LineEditor, invocation: Invocation) -> Returned:
    """Offer the options the model named, and answer the one the person picked."""
    fields = _fields(invocation.arguments)
    offered = cast("list[object]", fields.get("options"))
    question = Selection(
        prompt=_text(fields, "question"),
        options=tuple(str(option) for option in offered),
        default=_text(fields, "default") or None,
    )
    answer = await editor.ask(_where(question, invocation))
    picked = answer if isinstance(answer, Picked) else Picked(by=By.NOBODY)
    return Returned(id=invocation.id, ok=picked.value is not None, text=picked.value or "")


async def _ask_form(editor: LineEditor, invocation: Invocation) -> Returned:
    """Put a form's fields to the person one at a time, and answer one line per field.

    A form is one question in the kernel's sense and a sequence of them here: this lane holds one cursor
    and draws one line, and a block of labelled fields is a screen's business (R8). The answer names each
    field, so what the model reads back is what it asked for — a frontend that did draw a real form
    would say the same words, and the kernel would not have to know which one answered.
    """
    fields = _fields(invocation.arguments)
    title = _text(fields, "title")
    filled: list[str] = []
    for raw in _items(fields.get("fields")):
        field = _object(raw)
        answer = await editor.ask(_where(_field_question(title, field), invocation))
        value = _answer(answer)
        if value is None:
            return Returned(id=invocation.id, text="")
        filled.append(f"{_text(field, 'name')}: {value}")
    return Returned(id=invocation.id, ok=bool(filled), text="\n".join(filled))


def _where[Q: FreeText | Selection](question: Q, invocation: Invocation) -> Q:
    """The question, at the place the call came from: a frontend with several sessions shows it there."""
    return placed(question, session=invocation.session, turn=invocation.turn)


def _field_question(title: str, field: Mapping[str, Any]) -> FreeText | Selection:
    """One field as the question to put: a menu when it is a menu, a line otherwise.

    The title leads every line, because a person reading a sequence of questions has to know which form
    the line belongs to, and this lane has no block to put it above. A menu that also allows anything
    else cannot be a :class:`~plyngent.wire.ask.Selection` — that question refuses an answer outside
    ``options`` — so it is asked as a line with the menu spelled into it.
    """
    prompt = f"{title}: {_text(field, 'prompt')}"
    default = _text(field, "default") or None
    options = tuple(str(one) for one in _items(field.get("options")))
    if not options:
        return FreeText(prompt=prompt, default=default)
    if not field.get("allow_custom"):
        return Selection(prompt=prompt, options=options, default=default)
    return FreeText(prompt=f"{prompt} ({'/'.join(options)})", default=default)


def _answer(answer: AnyAnswer) -> str | None:
    """What a field's answer said, or ``None`` when nobody answered it.

    A form asks lines and menus and never a yes/no, so an agreement here would be an answer to a
    question this lane did not ask, and reading it as one would invent a value.
    """
    match answer:
        case Written(text=text):
            return text
        case Picked(value=value):
            return value
        case Agreed():
            return None


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


#: What the terminal offers: the three questions it can put in front of a person, one of them always
#: there — a model that may ask has to be able to ask before it has been shown anything (§10.8, E44).
TERMINAL = ClientTools(
    offer=(
        OfferedTool(name=LINE, resident=True),
        OfferedTool(name=CHOICE),
        OfferedTool(name=FORM),
    ),
    handlers={LINE: _ask_line, CHOICE: _ask_choice, FORM: _ask_form},
)
