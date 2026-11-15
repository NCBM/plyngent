"""What a client offers becomes tools: their shape, their posture, and what a call comes back as."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from plyngent.kernel.client import (
    ASKING_NAMES,
    DUPLICATE,
    NAMELESS,
    NO_SHAPE,
    NOT_AN_OBJECT,
    NOT_YOURS,
    SHAPES,
    answered,
    declaration,
    specs_from,
)
from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.toolresult import Hint, Status, Text
from plyngent.kernel.toolspec import ExternalArgs, Intention, Mutation, Posture, Res
from plyngent.wire.ask import Returned
from plyngent.wire.command import OfferedTool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.kernel.toolspec import ToolSpec

#: Somebody else's tool, as a client sends one: a schema of its own, written the way an MCP server
#: writes one.
SEARCH = json.dumps(
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "what to look for"}},
        "required": ["query"],
    }
)


def _one(tool: OfferedTool, *, taken: tuple[str, ...] = ()) -> tuple[ToolSpec | None, tuple[tuple[str, str], ...]]:
    """One tool through the fold, as ``(the spec it became or None, the refusals)``."""
    specs, refused = specs_from("c1", (tool,), taken=taken)
    return (specs[0] if specs else None), refused


def test_an_asking_name_carries_the_kernel_shape_and_not_the_clients() -> None:
    """A terminal says "I can ask" and no more: what a question looks like is the kernel's (§10.8)."""
    spec, refused = _one(OfferedTool(name="ask_user_line", resident=True))
    assert spec is not None

    assert refused == ()
    assert spec.args is SHAPES["ask_user_line"], "the kernel's package, byte for byte"
    assert set(spec.schema()["properties"]) == {"question", "default", "intention"}


def test_a_question_is_not_approved_like_a_call_to_a_client() -> None:
    """The tool that *is* the question runs without asking; everything else asks every time (§10.10)."""
    asked, _ = _one(OfferedTool(name="ask_user_line"))
    called, _ = _one(OfferedTool(name="search", schema=SEARCH))

    assert asked is not None
    assert called is not None
    assert asked.uses == Res.CLIENT | Res.HUMAN, "the question is the kernel's, the answer somebody else's"
    assert asked.approval({"question": "why?", "default": "", "intention": "because"}) == (Posture.AUTO, None)
    assert called.uses == Res.CLIENT
    assert called.approval({"query": "x"}) == (Posture.ALWAYS, None), "an effect nobody here can verify"


def test_a_clients_schema_cannot_be_asked_to_say_why() -> None:
    """``OUTSIDE`` would derive a required field the provider never declared, so it is spelled out."""
    called, _ = _one(OfferedTool(name="search", schema=SEARCH))

    assert called is not None
    assert called.mutates is Mutation.OUTSIDE, "the change happens in somebody else's process"
    assert called.intention is Intention.OPTIONAL
    assert called.decode('{"query": "x"}') == {"query": "x"}, "the package is the client's, as it was sent"
    assert isinstance(called.args, ExternalArgs)
    assert called.schema() == json.loads(SEARCH), "and the schema the model sees is that one, unchanged"


def test_an_offered_tool_travels_in_a_group_named_after_the_connection() -> None:
    """One group per connection, so a client's tools arrive and leave together (§10.8, E44)."""
    specs, refused = specs_from(
        "c7", (OfferedTool(name="search", schema=SEARCH), OfferedTool(name="ask_user_line")), taken=()
    )

    assert refused == ()
    assert {spec.group for spec in specs} == {"client:c7"}, "one group, named after the connection that offered it"
    assert specs[0].run is not specs[1].run, "each tool runs itself: the handler is closed over its own name"


@pytest.mark.parametrize(
    ("tool", "reason"),
    [
        (OfferedTool(name=""), NAMELESS),
        (OfferedTool(name="read", schema=SEARCH), DUPLICATE),
        (OfferedTool(name="ask_user_line", schema=SEARCH), NOT_YOURS),
        (OfferedTool(name="ask_user_form"), NO_SHAPE),
        (OfferedTool(name="search", schema="["), NOT_AN_OBJECT),
        (OfferedTool(name="search", schema="[]"), NOT_AN_OBJECT),
        (OfferedTool(name="search"), NOT_AN_OBJECT),
    ],
)
def test_an_offer_is_refused_one_tool_at_a_time(tool: OfferedTool, reason: str) -> None:
    """Every refusal names the rule it broke, so a client is told which one and can fix it."""
    spec, refused = _one(tool, taken=("read",))

    assert spec is None
    assert refused == ((tool.name, reason),)


def test_a_second_offer_of_one_name_inside_the_same_command_is_refused_too() -> None:
    """Two tools answering to one name is one name too many, whichever command carried them."""
    specs, refused = specs_from(
        "c1",
        (OfferedTool(name="search", schema=SEARCH), OfferedTool(name="search", schema=SEARCH)),
        taken=(),
    )

    assert len(specs) == 1, "the first one is the tool"
    assert refused == (("search", DUPLICATE),), "and the second is a name that already answered"


def test_every_asking_name_is_refused_or_shaped_but_never_both() -> None:
    """The three names are the kernel's; two of them have a shape and one waits for an editor (E48)."""
    assert set(SHAPES) <= ASKING_NAMES
    assert ASKING_NAMES - set(SHAPES) == {"ask_user_form"}


def test_a_client_answers_in_prose_and_the_kernel_makes_the_result() -> None:
    """The status is the kernel's, the text is the client's, and a client cannot forge a next step."""
    ran = answered("c1", "search", '{"query": "x"}', Returned(ok=True, text="two hits"))

    assert (ran.status, ran.failure, ran.retry) == (Status.OK, None, Retry.NEVER)
    assert ran.summary == "the client ran search"
    assert ran.body == (Text(value="two hits"),), "the body is what the client said, as text"
    assert ran.effects.external == (declaration("c1", "search", '{"query": "x"}'),)
    assert not any(isinstance(part, Hint) for part in ran.body), "a client hints at nothing"


def test_a_call_nobody_answered_is_a_failure_and_not_a_verdict() -> None:
    """Gone and out of time read the same here, and neither is retried (§10.8, E47)."""
    lost = answered("c1", "search", "{}", Returned(ok=False))

    assert (lost.status, lost.failure, lost.retry) == (Status.FAILED, Failure.CLIENT, Retry.NEVER)
    assert "never answered" in lost.summary
    assert lost.body == ()
    assert lost.effects.external == (declaration("c1", "search", "{}"),), "it was sent, so it is declared"


def test_a_client_that_says_it_failed_is_believed_about_that_and_nothing_else() -> None:
    """Its words are prose: they go in the body, and the status stays the table's."""
    failed = answered("c1", "search", "{}", Returned(ok=False, text="the index is rebuilding"))

    assert (failed.status, failed.failure) == (Status.FAILED, Failure.CLIENT)
    assert failed.summary == "search failed on the client"
    assert failed.body == (Text(value="the index is rebuilding"),)


def test_a_declaration_is_one_line_however_long_the_arguments_were() -> None:
    """It is a journal line, not a copy of the call: a body-sized argument cannot become a body-sized row."""
    said = declaration("c1", "search", json.dumps({"query": "x" * 500}))

    assert said.startswith('client c1: run search({"query": "xxx')
    assert "…" in said, "the rest is elided"
    assert len(said) < 100


def test_the_schema_of_an_offered_tool_is_theirs_and_survives_whole() -> None:
    """The schema arrives as text because that is what a provider writes; it is kept as the object it was."""
    spec, _ = _one(OfferedTool(name="search", schema=SEARCH))

    assert spec is not None
    schema: Mapping[str, Any] = spec.schema()

    assert schema["properties"]["query"]["description"] == "what to look for", "their words survive, unchanged"
