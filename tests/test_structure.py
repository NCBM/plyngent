"""The shape of the program, asserted mechanically.

None of these measures what a function does: they hold the *skeleton*, which no story can express —
who may import whom, that every event a client can receive has something to say on a terminal, and
that the stories run over every transport rather than over the one that happened to exist. This is
the only part of the suite that is not a story, and it is here because "the boundary must not rot" is
otherwise a matter of review (§11).
"""

from __future__ import annotations

import ast
import pathlib
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast, get_args

import pytest

from plyngent import boot, cli, kernel, wire
from plyngent.cli.grammar import ALIASES, build
from plyngent.cli.render import render
from plyngent.kernel.access import AccessMode
from plyngent.kernel.catalog import BUILTIN, EVERY, RESIDENT, ToolCatalog
from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.listing import Filter
from plyngent.kernel.net.web import Web
from plyngent.kernel.skills import NAME_PATTERN, Skills
from plyngent.kernel.toolresult import OUTCOMES, Effects, Hint, Status, Text, ToolResult, refusal
from plyngent.kernel.tools import fs, net, skills
from plyngent.kernel.tools.net import net_specs
from plyngent.kernel.tools.skills import skill_specs
from plyngent.kernel.toolspec import (
    MAX_INTENTION_CHARS,
    ArgsError,
    Group,
    Intention,
    Mutation,
    Posture,
    Res,
    ToolArgs,
    ToolSpec,
)
from plyngent.kernel.workspace import Denied, OutsideWorkspace
from plyngent.wire.event import Batch, Event, TurnStarted
from tests.stories import driver

if TYPE_CHECKING:
    from collections.abc import Callable

#: What a terminal library looks like from the outside: the kernel may not import one, because
#: width, colour, keystrokes and the cursor are the adapter's whole reason to exist.
TERMINAL_LIBRARIES = frozenset(
    {"blessed", "click", "curses", "msvcrt", "prompt_toolkit", "readline", "rich", "termios"}
)

#: The one module that may touch `termios`, because a pty *device* is not the user's terminal: there
#: it configures the slave a tool talks to, which is the kernel's own job (§9). Anywhere else, a
#: terminal library would mean the display had leaked out of the CLI.
_DEVICE = pathlib.Path("kernel") / "pty" / "backend.py"


def _ours(name: str) -> bool:
    """Whether this import stays inside the contract's own package."""
    return name == "plyngent.wire" or name.startswith("plyngent.wire.")


def _modules_of(package: ModuleType) -> list[pathlib.Path]:
    """Every module in a package, as files, so the rules are read off the source rather than imports."""
    return sorted(pathlib.Path(package.__file__).parent.rglob("*.py"))


def _imported(path: pathlib.Path) -> set[str]:
    """Every module one file imports, by dotted name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _offenders(paths: list[pathlib.Path], unwanted: Callable[[str], bool]) -> list[str]:
    """Every import in these files that the rule forbids, named so the failure says where to look."""
    return [f"{path.name}: {name}" for path in paths for name in _imported(path) if unwanted(name)]


def test_a_client_knows_only_the_contract() -> None:
    offenders = _offenders(
        _modules_of(cli),
        lambda name: name.startswith("plyngent.") and not name.startswith(("plyngent.wire", "plyngent.cli")),
    )

    assert offenders == [], "a client imports the contract and its own modules, nothing inward"


def test_the_kernel_does_not_know_terminals() -> None:
    everywhere = [*_modules_of(kernel), pathlib.Path(boot.__file__)]
    offenders = _offenders(
        [path for path in everywhere if path.parts[-len(_DEVICE.parts) :] != _DEVICE.parts],
        lambda name: name.split(".")[0] in TERMINAL_LIBRARIES,
    )

    assert offenders == [], "a terminal library in the kernel means the boundary is already gone"


def test_the_contract_depends_on_nothing_of_ours() -> None:
    offenders = _offenders(_modules_of(wire), lambda name: name.startswith("plyngent.") and not _ours(name))

    assert offenders == [], "the contract is importable by anyone, including a client with no kernel"


def test_the_composition_root_does_not_depend_on_a_client() -> None:
    offenders = _offenders([pathlib.Path(boot.__file__)], lambda name: name.startswith("plyngent.cli"))

    assert offenders == [], "boot hands a link over; what draws on it is the client's business"


def test_every_event_can_be_read() -> None:
    samples = [_sample(cls) for cls in get_args(wire.AnyEvent.__value__)]
    quiet = [type(sample).__name__ for sample in samples if not render(sample)]

    assert quiet == [], "an event with nothing to say is an event nobody sees"


def _sample(cls: type[Event]) -> Event:
    """One instance of an event kind, with defaults — enough to render, and nothing else."""
    return Batch(items=(TurnStarted(),)) if cls is Batch else cls()


def test_every_command_says_what_it_is_for() -> None:
    assert all(spec.summary for spec in wire.command_specs()), "a command with no summary is one /help cannot offer"


def test_every_command_is_either_spellable_or_withheld() -> None:
    catalog = _catalog()
    tags = {spec.name for spec in wire.command_specs()}

    assert set(catalog.offers) | set(catalog.withheld) == tags, "a command is offered or refused, never missing"
    assert not set(catalog.offers) & set(catalog.withheld), "and it cannot be both"


def test_every_alias_names_a_command_and_every_command_has_a_word() -> None:
    catalog = _catalog()

    assert {word: catalog.word(word) for word in ALIASES} == ALIASES, "an alias that names nothing is a lie"
    assert all(catalog.word(tag) == tag for tag in catalog.offers), "every command is spellable as itself"


def test_the_stories_run_over_every_transport() -> None:
    assert driver.TRANSPORTS is boot.KERNEL_TRANSPORTS, "the fixture's list is the kernel's own, not a copy"
    assert set(driver.TRANSPORTS) == set(boot.KERNEL_TRANSPORTS), "and every one of them is run"


# -- what a tool declares, over every tool there is ----------------------------------
#
# A declaration is what the model is shown and what the executor acts on, and most of it has no outside
# a story could read: a schema with an argument that says nothing about what it accepts, a closed set
# whose order came from a `set`, a package that can also be called by position. None of those have a
# screen. They are asserted here — over `EVERY`, not over the group whose turn it was to be written, and
# over the tools a kernel only serves with a backend too: a schema is a promise to a model whether or not
# this particular kernel has the backend behind it.

#: What makes a property a description: the keywords one of which says what a value may be.
_TYPED = frozenset({"type", "$ref", "enum", "const", "anyOf", "oneOf", "allOf"})


@pytest.mark.parametrize("spec", EVERY, ids=lambda spec: spec.name)
def test_every_tool_answers_for_itself(spec: ToolSpec) -> None:
    schema = spec.schema()

    assert spec.name and spec.name == spec.name.strip(), "a name with space in it is a name two spellings have"
    assert isinstance(spec.group, Group), "every builtin tool travels in a disclosure group"
    assert spec.uses, "a tool that reaches for nothing has no handler to write"
    assert isinstance(spec.args, type) and issubclass(spec.args, ToolArgs), "the package is one ToolArgs describes"
    assert schema["type"] == "object" and schema["properties"], "a tool is called with named arguments"
    assert schema["properties"]["intention"]["default"] == "", "every call may say why it is being made"
    assert schema["properties"]["intention"]["maxLength"] == MAX_INTENTION_CHARS
    assert "intention" not in schema.get("required", []), "the declaration asks for it, the type must not"
    assert spec.schema() == schema, "the same declaration twice is the same bytes"


@pytest.mark.parametrize("spec", EVERY, ids=lambda spec: spec.name)
def test_every_package_refuses_to_be_called_by_position(spec: ToolSpec) -> None:
    """Keyword only, so that no ordering of two calls can disagree about which value is which."""
    with pytest.raises(TypeError):
        _ = cast("Any", spec.args)(None)


@pytest.mark.parametrize("spec", EVERY, ids=lambda spec: spec.name)
def test_every_argument_the_model_can_give_says_what_it_accepts(spec: ToolSpec) -> None:
    assert _untyped(spec.schema()) == [], "an argument with no type is an argument the model has to guess at"


@pytest.mark.parametrize("spec", EVERY, ids=lambda spec: spec.name)
def test_every_closed_set_is_an_enum_in_an_order_of_its_own(spec: ToolSpec) -> None:
    for where, values in _enums(spec.schema()):
        assert values == sorted(values), f"{spec.name}: {where} is a closed set in some other order"
        assert len(set(values)) == len(values), f"{spec.name}: {where} lists a value twice"


def _untyped(node: object, where: str = "") -> list[str]:
    """Every subschema in *node* that says nothing about what it accepts, by its dotted path.

    A schema is a tree of three shapes: an object with ``properties``, an array with ``items``, and
    ``$defs`` beside them. Walking all three is what makes "no argument is left undescribed" a claim
    about the whole tree rather than about the top level.
    """
    if not isinstance(node, dict):
        return []
    schema = cast("dict[str, Any]", node)
    typed = bool(set(schema) & _TYPED)
    found: list[str] = [] if typed else [where or "."]
    for name, child in (schema.get("properties") or {}).items():
        found.extend(_untyped(child, f"{where}{name}"))
    if "items" in schema:
        found.extend(_untyped(schema["items"], f"{where}[]"))
    for name, definition in (schema.get("$defs") or {}).items():
        found.extend(_untyped(definition, f"{name}."))
    return found


def _enums(node: object, where: str = "") -> list[tuple[str, list[Any]]]:
    """Every closed set in *node*, by dotted path — the values a call may be given, and nothing else."""
    if not isinstance(node, dict):
        return []
    schema = cast("dict[str, Any]", node)
    found: list[tuple[str, list[Any]]] = []
    if isinstance(values := schema.get("enum"), list):
        found.append((where or ".", cast("list[Any]", values)))
    for name, child in (schema.get("properties") or {}).items():
        found.extend(_enums(child, f"{where}{name}"))
    if "items" in schema:
        found.extend(_enums(schema["items"], f"{where}[]"))
    for name, definition in (schema.get("$defs") or {}).items():
        found.extend(_enums(definition, f"{name}."))
    return found


def test_the_two_walkers_find_what_they_judge() -> None:
    """A checker that finds nothing because it looked nowhere looks exactly like a clean one.

    So the two walkers are shown a schema of the shape a tool produces, with one undescribed argument at
    the top and one inside a definition, and one closed set out of order.
    """
    planted = {
        "type": "object",
        "properties": {
            "good": {"type": "string"},
            "bad": {},
            "closed": {"enum": ["two", "one"]},
            "list": {"type": "array", "items": {"type": "integer"}},
        },
        "$defs": {"Nested": {"type": "object", "properties": {"also_bad": {}}}},
    }

    assert _untyped(planted) == ["bad", "Nested.also_bad"], "the top level and inside a definition, both"
    assert _enums(planted) == [("closed", ["two", "one"])]
    assert any(_enums(spec.schema()) for spec in BUILTIN), "and the tools do hand it closed sets to judge"


def test_the_tools_that_must_say_why_are_the_ones_that_leave_no_trace() -> None:
    """An intention is asked for by what a tool reaches and changes, not by the tool writing it down.

    The alternative is forty packages each deciding again whether "why" is worth asking, which is how the
    rule ends up with forty-one answers — and a *read* that asks is a model taught to write "read it".

    Over ``EVERY`` and not over ``BUILTIN``, for the same reason the schema walks are: ``client_fetch`` and
    the two skill writers are declarations whether or not this particular kernel was started with what they
    need behind them, and a tool left out of the walk is a tool nobody checks.
    """
    leaves_no_trace = {
        spec.name
        for spec in EVERY
        if spec.uses & (Res.EXEC | Res.HUMAN) or spec.mutates & (Mutation.OUTSIDE | Mutation.DESTRUCTIVE)
    }
    asks = {spec.name for spec in EVERY if spec.intention is Intention.REQUIRED}

    assert asks == leaves_no_trace, "the classification is derived from the declarations, so it cannot drift"


def test_the_group_table_is_a_fold_over_the_tools() -> None:
    """A group is what a tool says it travels in, so no second list may disagree with one.

    Written as a walk over the tools rather than over a table: a tool that forgot its group, or a group
    listed twice with different members, would otherwise be a thing only a reviewer could notice.
    """
    catalog = ToolCatalog()
    grouped = [name for _group, members in catalog.groups() for name in members]

    assert sorted(grouped) == sorted(catalog.names()), "every tool travels in exactly one group"
    assert [group for group, _ in catalog.groups()] == sorted(group for group, _ in catalog.groups()), (
        "a group table is written in name order, like the tools inside it"
    )
    assert all(list(members) == sorted(members) for _group, members in catalog.groups())


def test_a_session_starts_with_the_residents_and_is_told_what_is_missing() -> None:
    """What is always there, what is missing, and the one line that says so — all from the same fold."""
    catalog = ToolCatalog()
    nothing_shown = catalog.view(())
    everything_shown = catalog.view({group for group, _ in catalog.groups()})

    assert nothing_shown.names() == tuple(sorted(RESIDENT)), "a resident name that names nothing would show here"
    assert everything_shown.index() == "", "a session that has been shown everything is told nothing"
    assert nothing_shown.index() == (
        "5 more groups (14 tools): document(1), fs(5), net(1), process(6), session(1) — "
        "use tool_require(group=…) to add one"
    ), "the index counts what *this* session cannot reach, and it is a function of the set"
    assert catalog.view({"document", "process", "session"}).index() == (
        "2 more groups (6 tools): fs(5), net(1) — use tool_require(group=…) to add one"
    ), "one group left reads as one group"


def test_a_shown_group_arrives_whole_and_leaves_the_index() -> None:
    catalog = ToolCatalog()
    shown = catalog.view({"fs"})

    assert {"listdir", "tree", "copy_path", "move_path", "delete_path"} <= set(shown.names())
    assert "fs" not in shown.index(), "a group that has arrived is not still described as missing"
    assert shown.get("execute") == catalog.get("execute"), "and the residents did not move: both views hold it"


def test_a_tool_that_needs_a_backend_is_served_only_when_one_is_installed() -> None:
    """Existence is decided where the set is assembled, so the table is a function of what was installed.

    Nothing installed is a kernel with one net tool, and the index line says so; a backend is two more
    tools and a longer line. What never changes with the answer is a name or a schema — that is what keeps
    the block a function of the *set* rather than of how the program was started (§0.9, E26).
    """
    assert net_specs(None) == (), "a kernel with no backend serves nothing because of one"
    with_backend = ToolCatalog((*BUILTIN, *net_specs(Web(pages=_Any(), sources=(_Any(),)))))

    assert with_backend.view(()).index() == (
        "5 more groups (16 tools): document(1), fs(5), net(3), process(6), session(1) — "
        "use tool_require(group=…) to add one"
    ), "the line counts what this kernel has, and it is a function of the set"
    assert {"client_fetch", "builtin_websearch", "web_fetch"} <= set(with_backend.names())


def test_a_net_body_is_kept_rather_than_paged_through() -> None:
    """A body too long to send is a handle, so no net tool takes an offset to page through one (§0.9, E28).

    ``read`` has an offset, and it counts *lines of a file* — that is a window on something already here.
    A response body is not read that way: it is kept, and the handle is the one way to the rest of it.
    """
    for spec in (*net.SPECS, *net.CONDITIONAL):
        assert "offset" not in spec.schema()["properties"], f"{spec.name} pages a body instead of keeping it"


class _Any:
    """A backend that is only ever asked whether it is there."""


def test_the_tool_that_adds_a_group_is_one_a_session_cannot_lose() -> None:
    """A fallback nobody can reach is not a fallback: the asking tool is resident, and it is about the set.

    It is also the one group that is never offered: ``meta`` holds nothing a session is missing, so the
    index line never names it — which is what keeps "there are: document, fs, process" honest.
    """
    catalog = ToolCatalog()
    spec = catalog.get("tool_require")

    assert spec is not None, "a hint pointing at this tool would point at nothing"
    assert spec.group is Group.META, "its subject is the tool set, not a path or a process"
    assert spec.name in RESIDENT
    assert "meta" not in catalog.view(()).index()


def test_a_listing_tool_declares_its_bounds_rather_than_checking_them() -> None:
    """A bound in the schema is a call the model can see is wrong; a bound in the handler is a surprise."""
    schema = fs.GLOB.schema()
    properties = schema["properties"]

    assert schema["required"] == ["pattern"], "the one thing a call cannot leave out"
    assert properties["pattern"]["minLength"] == 1
    assert properties["max_matches"]["minimum"] == 1, "a budget of nothing is not a budget"
    assert properties["path"]["default"] == "."
    assert properties["filter"]["$ref"] == "#/$defs/Filter", "the listing knobs are one definition, not fields"
    with pytest.raises(ValueError, match="is a path"):
        _ = Filter(ignore_lists=["docs/.gitignore"])


def test_a_listing_filter_is_a_declaration_and_not_a_setting() -> None:
    """The knobs a listing takes are in the schema, with the values the schema declares.

    A default read from config would rewrite this tool's schema whenever the workspace changed, and the
    schema sits in the prompt prefix a provider caches (§10.5) — so it is the one thing that must not be
    a function of the run.
    """
    knobs = fs.GLOB.schema()["$defs"]["Filter"]["properties"]

    assert sorted(knobs) == ["ignore_lists", "show_hidden"], "build noise is a claim the ignore sources make"
    assert knobs["show_hidden"]["default"] is False
    assert knobs["ignore_lists"]["default"] is None, "the config's list of names does not go into the schema"


def test_the_tool_block_is_a_function_of_the_tool_set() -> None:
    """The block a model is given has to be the same bytes for the same tools, whatever the imports did.

    A provider caches the prompt prefix, and the tool block sits in front of everything else in it: an
    order that came from discovery order would rewrite the prefix whenever a tool was added, renamed or
    moved, and the cost is paid on every request that follows (§10.5).
    """
    catalog = ToolCatalog()

    assert len(catalog) == len(BUILTIN), "every declared tool is in the catalog, and only once"
    assert catalog.names() == tuple(sorted(catalog.names())), "name order, not registration order"
    assert [decl.name for decl in catalog.prompts()] == list(catalog.names())
    assert catalog.prompts() == ToolCatalog(reversed(BUILTIN)).prompts(), "the same set, the same bytes"


def _catalog():
    """The client's vocabulary, derived the one way it is ever derived."""
    return build(wire.KernelManifest(commands=wire.command_specs()), session="main")


# -- what a result is, which no reader can ask it -------------------------------------
#
# A result is four words and a line, and every reader of it — the model, the screen, the journal — reads
# one of those. The rules that decide them have no outside, so they are held here: the closed set that is
# total over `Failure`, the one table that makes a status and a retry together, and the two ways a result
# may not contradict itself.


def test_the_four_outcomes_are_the_words_the_contract_carries() -> None:
    """The wire spells these as plain strings, so the spelling is the contract, not our business."""
    assert [status.value for status in Status] == ["ok", "refused", "failed", "partial"]


def test_every_kind_of_trouble_is_one_row_of_one_table() -> None:
    """A failure with no row is a failure with no status and no answer to "try again?"."""
    assert set(OUTCOMES) == set(Failure)

    for failure, (status, retry) in OUTCOMES.items():
        result = ToolResult.of(failure, "something went wrong")

        assert (result.status, result.retry) == (status, retry), "the table decides, never the raise site"


def test_the_world_can_fix_two_of_them_by_itself_and_the_rest_ask_for_something() -> None:
    assert {kind for kind, (_, retry) in OUTCOMES.items() if retry is Retry.SAME_CALL} == {
        Failure.BUSY,
        Failure.TIMEOUT,
    }, "nothing else is worth trying again as it was"


def test_a_result_cannot_disagree_with_itself() -> None:
    with pytest.raises(ValueError, match="has to say what kind of failure"):
        _ = ToolResult(Status.FAILED, "it broke")

    with pytest.raises(ValueError, match="cannot also carry"):
        _ = ToolResult(Status.OK, "all good", failure=Failure.BUG)


def test_a_body_is_measured_through_the_projection_and_not_as_it_was_produced() -> None:
    """``chars`` is what the model is sent, so a byte the renderer has to spell out counts as spelled."""
    result = ToolResult.ok("read it", body=Text(value="a\x1bb"))

    assert result.body[0] == Text(value="a\x1bb"), "the part keeps what the tool produced"
    assert result.meta.chars == len("a\\x1bb"), "and the count is what the model reads"


def test_a_partial_result_is_never_retried_as_a_whole() -> None:
    result = ToolResult.partial("wrote 1 of 2 files", effects=Effects(changed=(pathlib.Path("/ws/a.txt"),)))

    assert result.status is Status.PARTIAL
    assert result.retry is Retry.NEVER, "running it again would do the parts that worked a second time"
    assert result.effects.changed == (pathlib.Path("/ws/a.txt"),), "what landed is still recorded"


def test_a_refusal_reads_the_way_the_workspace_put_it() -> None:
    """One template per kind of refusal: a second wording at the tool layer is a second answer to fix."""
    escaped = OutsideWorkspace(
        pathlib.Path("../x"),
        resolved=pathlib.Path("/elsewhere/x"),
        root=pathlib.Path("/ws"),
        need=AccessMode.EXEC,
    )
    lifted = refusal(escaped.with_tools("request_directory_access"))
    denied = refusal(Denied(pathlib.Path("/ws/key.pem"), "/key"))

    assert (lifted.status, lifted.failure, lifted.retry) == (Status.REFUSED, Failure.BLOCKED, Retry.FIX_ARGS)
    assert lifted.summary == str(escaped), "the workspace's sentence, not a rewrite of it"
    assert lifted.hints == (Hint(tool="request_directory_access", given=(("mode", "exec"),)),), (
        "the tool is offered as data, with the mode the call needed"
    )
    assert "request_directory_access" not in lifted.summary, "and the kernel names no tool in prose"
    assert (denied.failure, denied.hints) == (Failure.DENIED, ()), "nothing lifts a denial, so nothing is offered"


# -- the skill tools: a group that exists only when there is somewhere to look -------------
#
# Five tools whose whole group is conditional, which is the second kind of conditional tool this kernel
# has (the net group was the first). What exists is decided where the set is assembled, so what each
# declaration says about itself is asserted here — over the declared specs, not over a catalog built for
# one of the answers.


def _given(spec: ToolSpec) -> object:
    """A package of this tool with its required arguments filled in, as a call would carry them."""
    required: list[str] = spec.schema().get("required", [])
    assert isinstance(spec.args, type), "a builtin tool's arguments are a package, not somebody else's schema"
    return spec.args(**dict.fromkeys(required, "x"))


def test_a_kernel_with_no_skills_directory_has_no_skill_tools() -> None:
    """Existence is a function of what the host holds, and nothing else about the index line moves."""
    assert skill_specs(None) == (), "a kernel with no store serves nothing because of one"
    assert skill_specs(Skills(())) == skills.SPECS, "and one that has a store serves all five"

    without = ToolCatalog((*BUILTIN, *net_specs(None)))
    plain = ToolCatalog((*BUILTIN, *net_specs(None), *skill_specs(None)))
    with_store = ToolCatalog((*BUILTIN, *net_specs(None), *skill_specs(Skills(()))))

    assert plain.view(()).index() == without.view(()).index(), "nothing installed is the line it always was"
    assert f"skills({len(skills.SPECS)})" in with_store.view(()).index(), "a store is one more group in it, counted"


def test_the_skill_tools_are_one_group_and_none_of_them_is_resident() -> None:
    """A group is what arrives together; nothing in this one is part of what a conversation starts with."""
    catalog = ToolCatalog((*BUILTIN, *skills.SPECS))

    assert {spec.group for spec in skills.SPECS} == {Group.SKILLS}
    assert dict(catalog.groups())["skills"] == tuple(sorted(spec.name for spec in skills.SPECS))
    assert not {spec.name for spec in skills.SPECS} & set(RESIDENT), "the residents are the same twelve"


def test_the_skill_reading_tools_are_reads() -> None:
    """Reading a directory nobody in this workspace wrote changes nothing, so nothing is asked."""
    for spec in skills.SPECS:
        if spec.mutates is not Mutation.NONE:
            continue
        assert spec.approval(_given(spec)) == (Posture.AUTO, None), f"{spec.name} only reads"
        assert spec.intention is Intention.OPTIONAL, "a read that asks why teaches a model to write 'read it'"


def test_a_skill_name_is_one_rule_for_the_schema_and_for_the_handler() -> None:
    """A name that passes one check and fails the other makes the refusal depend on which ran first."""
    for spec in skills.SPECS:
        declared = spec.schema()["properties"].get("name")
        if declared is not None:
            assert declared["pattern"] == NAME_PATTERN, f"{spec.name} spells the name rule its own way"

    with pytest.raises(ArgsError):
        _ = skills.SKILL_READ.decode('{"name": "Not A Name"}')
    assert skills.SKILL_READ.decode('{"name": "fine-name"}').name == "fine-name", "and a good one decodes"


def test_the_skill_writing_tools_ask_every_time() -> None:
    """A directory that is not the workspace's is not something a path policy can judge one call about.

    Loosening this to a session grant is a change to §10.10, not a habit: ``ALWAYS`` does not consult a
    table, and the only way to have one would be a second posture that is lower than the declaration.
    """
    for spec in (skills.SKILL_CREATE, skills.SKILL_EDIT):
        assert spec.mutates is Mutation.OUTSIDE, "the effect is in somebody else's directory"
        assert spec.approval(_given(spec)) == (Posture.ALWAYS, None), f"{spec.name} asks every single time"
        assert spec.intention is Intention.REQUIRED, "and every call says why it is being made"


def test_the_skill_group_is_three_readers_and_two_writers() -> None:
    """What the group is made of, so that a tool added to it has to say which of the two it is."""
    writing = {spec.name for spec in skills.SPECS if spec.mutates is not Mutation.NONE}

    assert writing == {"skill_create", "skill_edit"}, "only these two write"
    assert {spec.name for spec in skills.SPECS} == {
        "skill_list",
        "skill_read",
        "skill_search",
        "skill_create",
        "skill_edit",
    }
