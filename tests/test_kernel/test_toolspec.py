"""A tool declares its arguments once: the schema, the validator and the approval all come from that."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, cast

import msgspec
import pytest
from msgspec import Struct

from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import (
    MAX_INTENTION_CHARS,
    ArgsError,
    ExternalArgs,
    Group,
    Intention,
    Mutation,
    Posture,
    Remember,
    Res,
    Role,
    ToolArgs,
    ToolSpec,
    schema_of,
)
from plyngent.wire.ask import Severity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from plyngent.kernel.session import TurnCtx


async def _run(ctx: TurnCtx, args: ToolArgs) -> ToolResult:
    """A handler that does nothing: these tests are about the declaration, not the work."""
    return ToolResult.ok("nothing to do")


class Probe(ToolArgs, kw_only=True):
    """A package with nothing in it, for asking the declaration questions one at a time."""


class Sweep(ToolArgs, kw_only=True):
    """A package with a switch that makes its ordinary use irreversible."""

    path: str
    permanent: bool = False


class Search(ToolArgs, kw_only=True):
    """A package whose constraints all have to reach the schema, not a handler."""

    query: Annotated[str, msgspec.Meta(min_length=1)]
    format: Literal["markdown", "json"] = "markdown"
    limit: Annotated[int, msgspec.Meta(ge=1)] = 20
    tags: list[str] | None = None


class Step(Struct, forbid_unknown_fields=True):
    """A recursive package, declared at module level: a forward reference cannot see a local class."""

    argv: list[str]
    then: Step | list[Step] | None = None
    else_: Step | None = msgspec.field(default=None, name="else")


class Batch(Struct, forbid_unknown_fields=True):
    steps: list[Step]


class Batched(ToolArgs, kw_only=True):
    """A tool package with a tree in it, for asking where in the tree a bad value sits."""

    steps: list[Step]


def _spec(
    *,
    args: type[ToolArgs] = Probe,
    uses: Res = Res.PATHS,
    mutates: Mutation = Mutation.NONE,
    posture: Posture | None = None,
    remember: Remember | None = None,
    severity: Severity | None = None,
    intention: Intention | None = None,
    escalates: Mapping[str, Severity] | None = None,
) -> ToolSpec:
    """A tool with one declaration changed, for asking the derivation what it makes of it."""
    return ToolSpec(
        name="probe",
        group=Group.FS,
        args=args,
        run=_run,
        uses=uses,
        mutates=mutates,
        posture=posture,
        remember=remember,
        severity=severity,
        intention=intention,
        escalates=escalates,
    )


def _asking(spec: ToolSpec, args: ToolArgs | None = None) -> tuple[Posture, Remember | None, Severity]:
    """What one call asks: whether to, remembered by what, and how heavily."""
    asking = spec.asking(Probe() if args is None else args)
    return asking.posture, asking.remember, asking.severity


def test_the_schema_is_generated_from_the_argument_package() -> None:
    schema = _spec(args=Search).schema()
    properties = schema["properties"]

    assert schema["type"] == "object"
    assert schema["required"] == ["query"], "a field with a default is not required"
    assert properties["query"]["minLength"] == 1
    assert properties["limit"]["minimum"] == 1
    assert properties["format"]["enum"] == ["json", "markdown"], "Literal is an enum, and it is sorted"
    assert any(branch.get("type") == "array" for branch in properties["tags"]["anyOf"])
    assert "$ref" not in schema, "a tool's parameters are the object itself, not a reference to it"


def test_a_recursive_package_becomes_definitions_rather_than_a_hand_written_validator() -> None:
    schema = schema_of(Batch)

    assert "Step" in schema["$defs"], "the nested type is a definition, not inlined"
    assert schema["properties"]["steps"]["items"] == {"$ref": "#/$defs/Step"}
    assert "else" in schema["$defs"]["Step"]["properties"], "a Python keyword is renamed on the wire"


def test_every_package_says_what_every_call_may_say_about_itself() -> None:
    schema = _spec().schema()
    intention = schema["properties"]["intention"]
    positional = cast("Callable[..., object]", Probe)

    with pytest.raises(TypeError):
        _ = positional("by position")
    assert intention["type"] == "string"
    assert intention["default"] == "", "saying why is optional in the package"
    assert intention["maxLength"] == MAX_INTENTION_CHARS
    assert "intention" not in schema["required"], "the declaration asks for it, the type does not"


def test_the_same_declaration_always_generates_the_same_bytes() -> None:
    first = msgspec.json.encode(_spec(args=Search).schema())
    second = msgspec.json.encode(_spec(args=Search).schema())

    assert first == second, "the prompt prefix is cached on these bytes; they may not wobble"


def test_a_package_that_forgets_the_base_is_refused_when_the_tool_is_declared() -> None:
    class Bare(Struct):
        path: str

    with pytest.raises(TypeError, match="must inherit ToolArgs"):
        _ = ToolSpec(name="bare", group=Group.FS, args=Bare, run=_run, uses=Res.PATHS)


def test_arguments_are_read_into_the_package_and_a_bad_one_says_where() -> None:
    spec = _spec(args=Search)
    parsed = spec.decode(b'{"query": "q"}')

    assert parsed == Search(query="q")
    assert parsed.limit == 20

    with pytest.raises(ArgsError) as caught:
        _ = spec.decode(b'{"query": ""}')
    assert "query" in caught.value.message, "the validator names the field even without a path"
    assert caught.value.field == "", "0.22 reports the location inside the message and nowhere else"

    with pytest.raises(ArgsError, match="unknown field"):
        _ = spec.decode(b'{"query": "q", "extra": 1}')

    with pytest.raises(ArgsError, match="enum"):
        _ = spec.decode(b'{"query": "q", "format": "yaml"}')


def test_where_in_a_nested_package_a_bad_value_sits_is_read_from_the_message() -> None:
    with pytest.raises(ArgsError) as caught:
        _ = _spec(args=Batched).decode(b'{"steps": [{"argv": "not a list"}]}')

    assert caught.value.field == "", "no structural path exists to put here"
    assert "$.steps[0].argv" in caught.value.message, "the place is in the sentence the validator wrote"


def test_a_tool_that_has_to_be_told_why_refuses_a_call_that_does_not_say() -> None:
    required = _spec(intention=Intention.REQUIRED)

    with pytest.raises(ArgsError) as caught:
        _ = required.decode(b"{}")
    assert caught.value.field == "intention"

    with pytest.raises(ArgsError, match="intention"):
        _ = required.decode(b'{"intention": "  "}')

    assert required.decode(b'{"intention": "because"}').intention == "because"


def test_the_reason_is_asked_for_by_what_the_tool_declares() -> None:
    assert _spec().intention is Intention.OPTIONAL, "reading a file says what it is by being read"
    assert _spec(intention=Intention.REQUIRED).intention is Intention.REQUIRED, "and an override is explicit"
    assert _spec(uses=Res.EXEC).intention is Intention.REQUIRED, "a program changed something we cannot see"
    assert _spec(uses=Res.HUMAN).intention is Intention.REQUIRED, "the human is asked, so say what for"
    assert _spec(mutates=Mutation.OUTSIDE).intention is Intention.REQUIRED
    assert _spec(mutates=Mutation.DESTRUCTIVE).intention is Intention.REQUIRED, "it took something away"


def test_an_external_schema_is_taken_as_it_is() -> None:
    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    external = ExternalArgs(schema=schema, description="search a documentation server")
    spec = ToolSpec(name="search", group="mcp:docs", args=external, run=_run, uses=Res.NET)

    assert spec.schema() == schema, "the kernel does not add, rename or tighten a field"
    assert spec.decode(b'{"query": "x"}') == {"query": "x"}

    with pytest.raises(ArgsError, match="JSON object"):
        _ = spec.decode(b'["not an object"]')
    with pytest.raises(ArgsError):
        _ = spec.decode(b"{not json")


def test_an_external_tool_can_only_ask_for_a_reason_it_declares() -> None:
    plain = ExternalArgs(schema={"type": "object", "properties": {"query": {"type": "string"}}})

    with pytest.raises(ValueError, match="declares no 'intention'"):
        _ = ToolSpec(
            name="search",
            group="mcp:docs",
            args=plain,
            run=_run,
            uses=Res.NET,
            intention=Intention.REQUIRED,
        )

    asking = ExternalArgs(schema={"type": "object", "properties": {"intention": {"type": "string"}}})
    spec = ToolSpec(
        name="search",
        group="mcp:docs",
        args=asking,
        run=_run,
        uses=Res.NET,
        intention=Intention.REQUIRED,
    )
    with pytest.raises(ArgsError):
        _ = spec.decode(b"{}")
    assert spec.decode(b'{"intention": "find the migration guide"}')["intention"] == "find the migration guide"


def test_a_role_that_names_no_field_is_refused_when_the_tool_is_declared() -> None:
    with pytest.raises(ValueError, match="does not have"):
        _ = ToolSpec(
            name="typo",
            group=Group.FS,
            args=Probe,
            run=_run,
            uses=Res.PATHS,
            roles={"paths": Role.READ},
        )


def test_a_tool_that_says_where_its_paths_are_has_to_say_it_has_any() -> None:
    """``holds`` is the declaration for a tree, and a tree of paths is still paths: same flag or no tool."""
    with pytest.raises(ValueError, match=r"Res\.PATHS"):
        _ = ToolSpec(
            name="tree",
            group=Group.PROCESS,
            args=Probe,
            run=_run,
            uses=Res.EXEC,
            holds=lambda _args: (),
        )


def test_a_subject_that_names_no_field_is_refused_when_the_tool_is_declared() -> None:
    """What a question is about is an argument, so a declaration that names nothing is a typo."""
    with pytest.raises(ValueError, match="does not have"):
        _ = ToolSpec(
            name="typo",
            group=Group.SKILLS,
            args=Probe,
            run=_run,
            uses=Res.SKILLS,
            mutates=Mutation.OUTSIDE,
            subject="Title",
        )


def test_the_approval_is_derived_from_what_the_tool_declares() -> None:
    assert _asking(_spec()) == (Posture.AUTO, None, Severity.ROUTINE), "a read asks nobody"
    assert _asking(_spec(uses=Res.HUMAN)) == (Posture.AUTO, None, Severity.DANGER), (
        "asking the human is not a reason to ask, but it is the heaviest thing there is"
    )
    assert _asking(_spec(uses=Res.CLIENT)) == (Posture.ALWAYS, None, Severity.DANGER), (
        "the effect is in another process"
    )
    assert _asking(_spec(mutates=Mutation.OUTSIDE)) == (Posture.ALWAYS, None, Severity.DANGER)
    assert _asking(_spec(uses=Res.EXEC)) == (Posture.AUTO, None, Severity.DANGER), (
        "the command policy judges the program"
    )
    assert _asking(_spec(uses=Res.PTY, mutates=Mutation.HOST)) == (Posture.ONCE, Remember.SESSION, Severity.CONFIRM)
    assert _asking(_spec(mutates=Mutation.WORKSPACE)) == (Posture.ONCE, Remember.PATH, Severity.CONFIRM)
    assert _asking(_spec(mutates=Mutation.HOST)) == (Posture.AUTO, None, Severity.ROUTINE), (
        "bookkeeping is not a change"
    )


def test_what_cannot_be_taken_back_is_heavy_even_where_the_posture_does_not_move() -> None:
    """``DESTRUCTIVE`` has no row of its own in the posture table and one of its own in the level table."""

    assert _asking(_spec(mutates=Mutation.WORKSPACE | Mutation.DESTRUCTIVE)) == (
        Posture.ONCE,
        Remember.PATH,
        Severity.DANGER,
    ), "a move is asked once per path, and it is still the heaviest thing a file call can be"


def test_a_wide_reach_is_declared_and_a_remembered_key_can_be_chosen() -> None:
    wide = _spec(mutates=Mutation.HOST, posture=Posture.ONCE, remember=Remember.HOST)

    assert _asking(wide) == (Posture.ONCE, Remember.HOST, Severity.ROUTINE), (
        "a temporary workspace widens what is reachable"
    )


def test_a_light_call_inside_a_heavy_declaration_says_so_itself() -> None:
    """``write_file`` is the one place this is needed: creating is not replacing, and both write."""

    makes = _spec(mutates=Mutation.WORKSPACE, severity=Severity.ROUTINE)

    assert _asking(makes) == (Posture.ONCE, Remember.PATH, Severity.ROUTINE), "the posture does not follow the level"


@pytest.mark.parametrize(
    ("uses", "mutates", "expected"),
    [
        (Res.HUMAN | Res.PATHS, Mutation.NONE, (Posture.AUTO, None, Severity.DANGER)),
        (Res.CLIENT | Res.NET, Mutation.OUTSIDE, (Posture.ALWAYS, None, Severity.DANGER)),
        (Res.EXEC | Res.PATHS, Mutation.OUTSIDE | Mutation.WORKSPACE, (Posture.AUTO, None, Severity.DANGER)),
        (Res.EXEC | Res.PTY, Mutation.HOST, (Posture.AUTO, None, Severity.DANGER)),
        (Res.PATHS, Mutation.OUTSIDE, (Posture.ALWAYS, None, Severity.DANGER)),
        (Res.PTY, Mutation.HOST, (Posture.ONCE, Remember.SESSION, Severity.CONFIRM)),
        (Res.PTY, Mutation.NONE, (Posture.AUTO, None, Severity.ROUTINE)),
        (Res.PATHS, Mutation.WORKSPACE, (Posture.ONCE, Remember.PATH, Severity.CONFIRM)),
        (Res.PATHS, Mutation.HOST, (Posture.AUTO, None, Severity.ROUTINE)),
        (Res.PATHS | Res.DOCUMENT, Mutation.NONE, (Posture.AUTO, None, Severity.ROUTINE)),
    ],
)
def test_the_baseline_is_one_row_of_the_table_per_declaration(
    uses: Res, mutates: Mutation, expected: tuple[Posture, Remember | None, Severity]
) -> None:
    assert _asking(_spec(uses=uses, mutates=mutates)) == expected


def test_a_tool_that_runs_programs_is_judged_per_command_and_not_always_asked() -> None:
    """The order of the table's rows is a decision, and this pair is why."""

    runs = _spec(uses=Res.EXEC | Res.PATHS, mutates=Mutation.OUTSIDE | Mutation.WORKSPACE)
    elsewhere = _spec(uses=Res.PATHS, mutates=Mutation.OUTSIDE)

    assert _asking(runs)[:2] == (Posture.AUTO, None), "the command policy answers per argv"
    assert _asking(elsewhere)[:2] == (Posture.ALWAYS, None), "nothing can judge this one call"


def test_a_once_only_posture_has_to_say_what_it_remembers() -> None:
    with pytest.raises(ValueError, match="declare a Remember"):
        _ = _spec(posture=Posture.ONCE)


def test_the_ordinary_use_of_a_tool_can_be_declared_irreversible() -> None:
    spec = _spec(
        args=Sweep,
        mutates=Mutation.WORKSPACE | Mutation.DESTRUCTIVE,
        escalates={"permanent": Severity.DANGER},
    )

    assert _asking(spec, Sweep(path="x")) == (Posture.ONCE, Remember.PATH, Severity.DANGER), "the ordinary use"
    assert _asking(spec, Sweep(path="x", permanent=True)) == (Posture.ALWAYS, None, Severity.DANGER), (
        "this much cannot be undone, so it is not remembered"
    )


def test_an_argument_may_raise_the_level_and_never_lower_it() -> None:
    with pytest.raises(ValueError, match="never lower it"):
        _ = _spec(args=Sweep, mutates=Mutation.DESTRUCTIVE, escalates={"permanent": Severity.CONFIRM})


def test_an_escalation_says_how_heavy_and_not_merely_that_it_is_heavier() -> None:
    with pytest.raises(TypeError, match="name a Severity"):
        _ = _spec(args=Sweep, escalates={"permanent": True})  # type: ignore[dict-item]


def test_an_escalation_that_names_no_field_is_refused_when_the_tool_is_declared() -> None:
    with pytest.raises(ValueError, match="does not have"):
        _ = _spec(escalates={"permanant": Severity.DANGER})


def test_the_tool_set_talks_about_itself_in_its_own_group() -> None:
    assert Group.META == "meta", "tool_require is about the tool set, not about a path"
