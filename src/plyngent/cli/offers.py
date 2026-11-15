"""What this terminal offers the kernel, and what it does when the kernel asks for one of them.

Asking a person is a tool the *client* provides: the kernel does not read stdin, and it has no way to
put a question in front of anybody. So the terminal offers the two ways it can ask, and answers them on
the invocation lane (§10.8). What a question *looks* like is the kernel's answer, not this file's: an
offer of one of these names carries no schema, and the field names read below are the ones the kernel's
own package declares.

Two rules this file keeps:

* **Nothing but asking.** This client offers no other tool: its display and its Ctrl+C are a renderer
  and a command, and putting them on this lane would be a privilege no other client has (§10.8 rule 5).
  ``ask_user_form`` is not here either — a form needs a multi-field editor, which arrives with R8.
* **Every invocation is answered.** A call the kernel is waiting on ends in a deadline if nobody
  answers it, so a name this client does not have is answered as one it could not run rather than
  dropped on the floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from msgspec.structs import replace as placed

from plyngent.wire.ask import FreeText, Invocation, Picked, Returned, Selection, Written
from plyngent.wire.command import OfferedTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from plyngent.cli.line import LineEditor

#: The two names this terminal answers, and how to run one: the answer is one line of prose, either
#: way, and ``None`` from the editor is "nobody answered" rather than an empty answer (§10.8, E48).
type Handler = Callable[[LineEditor, Invocation], Awaitable[Returned]]

#: Ask for a line of free text.
LINE = "ask_user_line"

#: Ask for one of a few options.
CHOICE = "ask_user_choice"


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
    written = answer if isinstance(answer, Written) else Written()
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
    picked = answer if isinstance(answer, Picked) else Picked()
    return Returned(id=invocation.id, ok=picked.value is not None, text=picked.value or "")


def _where[Q: FreeText | Selection](question: Q, invocation: Invocation) -> Q:
    """The question, at the place the call came from: a frontend with several sessions shows it there."""
    return placed(question, session=invocation.session, turn=invocation.turn)


def _fields(arguments: str) -> Mapping[str, Any]:
    """A call's arguments, as the object the kernel already validated them into."""
    parsed: object = json.loads(arguments) if arguments else {}
    return cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}


def _text(fields: Mapping[str, Any], name: str) -> str:
    """One string field of a call, or nothing: what is not a string is not a prompt."""
    value = fields.get(name)
    return value if isinstance(value, str) else ""


#: What the terminal offers: the two questions it can put in front of a person, one of them always
#: there — a model that may ask has to be able to ask before it has been shown anything (§10.8, E44).
TERMINAL = ClientTools(
    offer=(
        OfferedTool(name=LINE, resident=True),
        OfferedTool(name=CHOICE),
    ),
    handlers={LINE: _ask_line, CHOICE: _ask_choice},
)
