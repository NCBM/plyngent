"""The shape of the program, asserted mechanically.

None of these measures what a function does: they hold the *skeleton*, which no story can express —
who may import whom, that every event a client can receive has something to say on a terminal, and
that the stories run over every transport, not over the one that happened to exist. This is
the only part of the suite that is not a story, and it is here because "the boundary must not rot" is
otherwise a matter of review (§11).
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast, get_args

import pytest

import plyngent
from plyngent import boot, cli, kernel, wire
from plyngent.cli import offers
from plyngent.cli.grammar import ALIASES, build
from plyngent.cli.render import Role, render
from plyngent.kernel.access import AccessMode
from plyngent.kernel.catalog import BUILTIN, EVERY, NOBODY, RESIDENT, ToolCatalog, ToolView
from plyngent.kernel.client import ASKING_NAMES, CLIENT
from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.journal import _decode, _encode
from plyngent.kernel.listing import Filter
from plyngent.kernel.lm import DIALECTS, Caps, codec_named
from plyngent.kernel.lm.route import Route, RouteWeb
from plyngent.kernel.model import AnyMessage, Note, Request, User
from plyngent.kernel.net.web import Engine, Engines, Hit, Page
from plyngent.kernel.prompt import GROUP_PLAYBOOK, PromptPlan
from plyngent.kernel.skills import NAME_PATTERN, Skills
from plyngent.kernel.toolresult import (
    DENIAL_SAYS,
    OUTCOMES,
    DenialKind,
    Effects,
    Hint,
    Status,
    Text,
    ToolResult,
    refusal,
)
from plyngent.kernel.tools import fs, meta, net, session, skills
from plyngent.kernel.tools import lifted as lifted_path
from plyngent.kernel.tools.net import net_specs
from plyngent.kernel.tools.skills import skill_specs
from plyngent.kernel.toolspec import (
    MAX_INTENTION_CHARS,
    ArgsError,
    Group,
    Intention,
    Mutation,
    Posture,
    Remember,
    Res,
    ToolArgs,
    ToolSpec,
)
from plyngent.kernel.trail import (
    NOTICE_MARKER,
    SUMMARY_HEADING,
    AnyRecord,
    Attempt,
    Consent,
    Effect,
    Notice,
    Promoted,
    StateChange,
    Summary,
    TurnEnded,
    Usage,
    Withheld,
    as_message,
    kind_of,
)
from plyngent.kernel.workspace import Denied, OutsideWorkspace
from plyngent.wire.ask import SEVERITIES, Severity
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

#: The words that name a provider or one of its endpoints. They belong to ``kernel/lm`` and nowhere else,
#: because a loop that compares one of them is a loop that has to be edited to add a provider.
DIALECT_WORDS = frozenset({"openai", "anthropic", "deepseek", "chat-completions", "chat/completions", "/v1/messages"})


def _ours(name: str) -> bool:
    """Whether this import stays inside the contract's own package."""
    return name == "plyngent.wire" or name.startswith("plyngent.wire.")


def _modules_of(package: ModuleType) -> list[pathlib.Path]:
    """Every module in a package, as files, so the rules are read off the source, not off imports."""
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


def test_the_kernel_names_no_dialect() -> None:
    """One dialect is one file and one row: a provider's name outside ``kernel/lm`` is a row that failed.

    This is the acceptance for the whole provider layer written as a machine check (§6, R11). It is a
    *word* scan, because the failure it guards against is a comparison and not a dependency: the day a
    loop writes ``if provider == "…"``, nothing new is imported and the layer is
    gone. A route names a dialect by a string it was handed; the kernel is not allowed to know any.

    ``kernel/net`` is out of scope, and not as a loophole: a vendor's *search* endpoint is named after
    the company that runs it by design (§0.9), which is a different claim from "the model loop speaks
    one dialect". What this asserts is that nothing that *asks a model* knows who answered.
    """
    package = pathlib.Path(kernel.__file__).parent.parent
    everywhere = [
        *_modules_of(kernel),
        pathlib.Path(boot.__file__),
        pathlib.Path(boot.__file__).with_name("__main__.py"),
    ]
    outside = [
        path for path in everywhere if path.relative_to(package).parts[:2] not in {("kernel", "lm"), ("kernel", "net")}
    ]
    offenders = [
        f"{path.name}: {word}"
        for path in outside
        for word in DIALECT_WORDS
        if word in path.read_text(encoding="utf-8").lower()
    ]

    assert offenders == [], "a dialect's name outside kernel/lm means the loop knows a provider"


def test_every_dialect_is_one_row_in_one_module() -> None:
    """The table is the list: every dialect is filed under its own name, and each lives in its own module.

    A codec that exists and is not filed is a dialect nobody can route to; one filed from somewhere other
    than this package is a row that a second place could also have written.
    """
    filed = [codec.name for codec in DIALECTS]

    assert len(filed) == len(set(filed)), "two dialects are filed under one name"
    assert [codec_named(name) for name in filed] == list(DIALECTS), "a row the lookup cannot find, in order"
    for codec in DIALECTS:
        assert type(codec).__module__.startswith("plyngent.kernel.lm."), f"{codec.name} is filed from elsewhere"


def test_a_dialect_that_reads_a_stream_is_the_one_that_asks_for_one() -> None:
    """``Caps.stream`` and ``Codec.sse`` are one fact in two places, and the body is where they must agree.

    A route is streamed only where both hold — the endpoint will send frames and this build can read
    them — so a dialect that writes ``stream: true`` while saying it cannot read a stream would be a
    request nobody can answer, and one that claims to read a stream while asking for a whole body would
    never see one. This is the same two-gate rule a picture has: what the endpoint can do, and whether
    the wire has a channel for it.
    """
    document = Request(messages=(User(text="hi"),))

    for codec in DIALECTS:
        body = codec.encode(document, model="m", caps=Caps(max_output=1, stream=True))

        assert body["stream"] is codec.sse, f"{codec.name} and the body it writes disagree about streaming"


def test_the_composition_root_does_not_depend_on_a_client() -> None:
    offenders = _offenders([pathlib.Path(boot.__file__)], lambda name: name.startswith("plyngent.cli"))

    assert offenders == [], "boot hands a link over; what draws on it is the client's business"


def test_the_kernel_does_not_know_how_it_was_configured() -> None:
    """A settings value is the composition root's, and it becomes values before a kernel sees anything.

    The kernel is handed a route table, a skill store, a prompt plan and three budgets; none of them is a
    document, and a kernel that could read one would be a second place deciding what this program is
    (§0.11, E69). The rule is asserted as an import because that is the only way it can be broken by
    accident: a value passed down is visible in a signature, a file opened three layers in is not.
    """
    everywhere = [*_modules_of(kernel), pathlib.Path(boot.__file__)]
    offenders = _offenders(everywhere, lambda name: name == "plyngent.config" or name.startswith("plyngent.config."))

    assert offenders == [], "a kernel that imports the configuration is a kernel that reads a file"


def test_only_the_loader_reads_the_environment() -> None:
    """E76: the environment holds a secret and never a setting, so one function reads it — the loader's.

    The readers that are not configuration are named, because a name in
    this list is a claim somebody has to write down: a child process is handed the environment its parent
    had — twice, once for a tool's command and once for the git this build keeps versions with — then
    ``$EDITOR`` is how a person says which program they write in, and the last two are facts about
    the stream in front of this run — there is no terminal here, and ``NO_COLOR`` says not to paint. They
    are the exception to E76 and they are narrow on purpose: they say what the *input and output* are,
    not what the program does, and setting them where a terminal was reading is a worse picture,
    not different behaviour. Any other reader — a value that quietly changed because of how the program
    was launched — arrives here as a failure, not as a habit (§0.11, E70/E76).
    """
    readers = {found for path in _modules_of(plyngent) for found in _environment_readers(path)}

    assert readers == {
        "config.py:load",
        "process.py:_environment",
        "history.py:_environment",
        "line.py:_composed",
        "__main__.py:_plainly_asked",
        "__main__.py:_colour",
    }, (
        "the loader, the two places a child is handed an environment, the person's editor, "
        "and the two facts about a stream"
    )


def _environment_readers(path: pathlib.Path) -> list[str]:
    """Every function in one file that reads the process's environment, as ``file.py:function``."""
    return [
        f"{path.name}:{node.name}"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _reads_environment(node)
    ]


def _reads_environment(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether this function touches ``os.environ`` or one of the functions that change it."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Attribute) and inner.attr == "environ":
            return True
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr in {"getenv", "putenv", "unsetenv"}
        ):
            return True
    return False


def test_every_event_can_be_read() -> None:
    samples = [_sample(cls) for cls in get_args(wire.AnyEvent.__value__)]
    quiet = [type(sample).__name__ for sample in samples if not render(sample)]

    assert quiet == [], "an event with nothing to say is an event nobody sees"


def test_every_line_an_event_makes_says_what_kind_of_line_it_is() -> None:
    """E90: the roles are a closed set, so a new kind of line is a decision, not a colour."""
    roles = set(get_args(Role.__value__))
    wrong = [
        (type(sample).__name__, shown.role)
        for cls in get_args(wire.AnyEvent.__value__)
        for sample in (_sample(cls),)
        for shown in render(sample)
        if shown.role not in roles
    ]

    assert wrong == [], "a line whose role is not one of the eight has no reader the frontend knows"
    assert roles == {"assistant", "tool", "notice", "refusal", "usage", "thought", "added", "removed"}, (
        "and the eight are the whole set"
    )


def _sample(cls: type[Event]) -> Event:
    """One instance of an event kind, with defaults — enough to render, and nothing else."""
    return Batch(items=(TurnStarted(),)) if cls is Batch else cls()


def test_every_command_says_what_it_is_for() -> None:
    assert all(spec.summary for spec in wire.command_specs()), "a command with no summary is one /help cannot offer"


def test_every_read_is_a_row_a_frontend_can_offer() -> None:
    """The greeting carries the reads beside the commands, because a window with buttons has no line.

    What a frontend cannot work out for itself is which questions this kernel answers and what each one
    takes; what it must not be told is a number to put on one, because that number is the connection's
    and a client that chose it could answer somebody else's question (E155).
    """
    specs = wire.query_specs()
    assert {spec.name for spec in specs} == {"grants", "trajectory", "skills", "turns"}, "the four reads"
    assert all(spec.summary for spec in specs), "a read with no summary is one a frontend cannot offer"
    assert all('"id"' not in spec.args for spec in specs), "the reply's number is the link's, not a client's"


#: The contract as a stranger reads it: in the package, because `docs/` is not in git.
SPEC = pathlib.Path(wire.__file__).parent / "SPEC.md"


def test_the_spec_names_exactly_the_words_this_build_answers() -> None:
    """The document a frontend is written from is kept true by a test, not by care.

    A table one word away from the union is worse than no table at all: the client is written, and the
    word it sends is refused forever. So the two word lists are the derived ones, exactly (E155).
    """
    text = SPEC.read_text(encoding="utf-8")

    assert set(_spec_words(text, "Commands")) == {spec.name for spec in wire.command_specs()}, "the commands"
    assert set(_spec_words(text, "Reads")) == {spec.name for spec in wire.query_specs()}, "and the reads"


def _spec_words(text: str, heading: str) -> tuple[str, ...]:
    """The first cell of every table row under one ``## heading``, with the backticks off."""
    lines = text.splitlines()
    start = lines.index(f"## {heading}")
    words: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        first = line.split("|")[1].strip().strip("`")
        if first and first != "word" and not set(first) <= {"-"}:
            words.append(first)
    return tuple(words)


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
    about the whole tree, not about the top level.
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

    Written as a walk over the tools, not over a table: a tool that forgot its group, or a group
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
    """What is always there, what is missing, and the line that says so — all from the same fold."""
    catalog = ToolCatalog()
    nothing_shown = catalog.view(())
    everything_shown = catalog.view({group for group, _ in catalog.groups()})

    assert nothing_shown.names() == tuple(sorted(RESIDENT)), "a resident name that names nothing would show here"
    assert everything_shown.index() == NOBODY, "a session with every group is told one thing: nobody to ask"
    assert not everything_shown.asks, "and the sentence is the group table's answer, not a second flag"
    assert nothing_shown.index() == (
        "6 more groups (21 tools): document(1), fs(7), net(1), process(6), session(1), vcs(5) — "
        "use tool_require(group=…) to add one\n" + NOBODY
    ), "the index counts what *this* session cannot reach, and it is a function of the set"
    assert catalog.view({"document", "process", "session"}).index() == (
        "3 more groups (13 tools): fs(7), net(1), vcs(5) — use tool_require(group=…) to add one\n" + NOBODY
    ), "one group left reads as one group"


def test_a_client_group_is_what_says_a_client_is_attached() -> None:
    """The two things a catalog takes as input: residency, and whether one is here.

    Both are the same fact seen twice — a client's tools travel in a ``client:<id>`` group that exists
    only while it is attached — so neither is a field anybody has to remember to set (§10.8, E44).
    """
    offered = replace(meta.TOOL_REQUIRE, name="client_tool", group=f"{CLIENT}:c1")
    resident = ToolCatalog((*BUILTIN, offered), resident=(*RESIDENT, "client_tool"))
    hidden = ToolCatalog((*BUILTIN, offered))

    assert resident.view(()).asks and hidden.view(()).asks, "attached is the group, resident or not"
    assert "client_tool" in resident.view(()).names(), "a resident offer is there without being asked for"
    assert f"{CLIENT}:c1" not in resident.view(()).index(), "and a group with nothing left in it is not missing"
    assert dict(hidden.view(()).missing)[f"{CLIENT}:c1"] == 1, "unresident, it is one more group to ask for"
    assert not ToolCatalog().view(()).asks, "a kernel with no client has nobody to ask"


def test_the_asking_tools_are_an_offer_and_never_a_builtin() -> None:
    """A kernel with nobody attached cannot ask, so these names are not in its table at all (§10.8).

    Written over ``EVERY`` because that is the walk of what this build declares: a name that reached it
    would be a tool that exists without a client, which is the state the offer exists to end.
    """
    assert not {spec.name for spec in EVERY} & ASKING_NAMES, "asking is a client's tool, not this kernel's"
    assert "ask_user" not in ToolCatalog().names()


def test_the_terminal_offers_one_way_to_ask_and_nothing_else() -> None:
    """§10.8: a client offers questions and no other privilege, and it does not shape them itself.

    One name and not three, because the questions are one array: a line, a menu, and a block of five are
    the same tool with a different list, and the array is the kernel's shape, not the client's.
    """
    names = {tool.name for tool in offers.TERMINAL.offer}

    assert names == {"ask_user"} <= ASKING_NAMES
    assert set(offers.TERMINAL.handlers) == names, "every tool it offers is one it can run"
    assert all(not tool.schema for tool in offers.TERMINAL.offer), "the shape of a question is the kernel's"


def test_a_shown_group_arrives_whole_and_leaves_the_index() -> None:
    catalog = ToolCatalog()
    shown = catalog.view({"fs"})

    assert {"listdir", "tree", "copy_path", "move_path", "delete_path"} <= set(shown.names())
    assert "fs" not in shown.index(), "a group that has arrived is not still described as missing"
    assert shown.get("execute") == catalog.get("execute"), "and the residents did not move: both views hold it"


def test_every_group_has_a_playbook_section() -> None:
    """The table is the exchange rate between a group and what a model is told about its tools.

    A group is the disclosure unit, so its instructions are one too: filled for every :class:`Group` and
    for nothing else, and the section arrives exactly when a tool of that group is reachable (§10.4, E64).
    A group with instructions and no group would be a paragraph nobody ever reads; a group without one
    would arrive as schemas with no how-to, which is the state R9 exists to end.
    """
    catalog = ToolCatalog()
    by_group = dict(catalog.groups())
    plan = PromptPlan()

    assert set(GROUP_PLAYBOOK) == set(Group), "every group has a section, and nothing else does"
    assert all(text.strip() for text in GROUP_PLAYBOOK.values()), "an empty section is a group with no how-to"
    assert len(set(GROUP_PLAYBOOK.values())) == len(Group), "two groups sharing a section is one section lost"
    assert not [text for text in GROUP_PLAYBOOK.values() if text in plan.compose(catalog.view(()))], (
        "a session that reached nothing is told no group's how-to"
    )

    for group in Group:
        members = by_group.get(group.value, ())
        if not members:
            continue
        alone = ToolView(
            specs={name: spec for name in members if (spec := catalog.get(name)) is not None},
            groups=catalog.groups(),
            disclosed=frozenset({group.value}),
        )
        arrived = [text for text in GROUP_PLAYBOOK.values() if text in plan.compose(alone)]

        assert arrived == [GROUP_PLAYBOOK[group]], f"one group's tools brought {len(arrived)} sections"


def test_a_tool_that_needs_a_backend_is_served_only_when_one_is_installed() -> None:
    """Existence is decided where the set is assembled, so the table is a function of what was installed.

    Nothing installed is a kernel with one net tool, and the index line says so; a backend is two more
    tools and a longer line. What never changes with the answer is a name or a schema — that is what keeps
    the block a function of the *set*, not of how the program was started (§0.9, E26).
    """
    assert net_specs(None) == (), "a kernel with no backend serves nothing because of one"
    any_engine = _Any()
    every_verb = Engine(name="either", search=any_engine.search, page=any_engine.page)
    with_backend = ToolCatalog((*BUILTIN, *net_specs(Engines([every_verb]))))

    assert with_backend.view(()).index() == (
        "6 more groups (23 tools): document(1), fs(7), net(3), process(6), session(1), vcs(5) — "
        "use tool_require(group=…) to add one\n" + NOBODY
    ), "the line counts what this kernel has, and it is a function of the set"
    assert {"client_fetch", "web_search", "web_fetch"} <= set(with_backend.names())


def test_a_route_installs_the_web_engine_it_names() -> None:
    """Engines come from the routes, and the tools come from the engines — in that order and no other.

    A route that names no provider is a model endpoint and nothing else, so a program configured with only
    those has one net tool. One that names an implementation gets that engine, and with it every verb the
    engine has. ``share`` is not consulted here: whether a tool exists and whose key pays for a search are
    two questions, and this is the first one.
    """
    plain = Route(name="local", dialect="chat-completions", base="http://127.0.0.1:1", caps=Caps(max_output=64))
    named = Route(
        name="ds-a",
        dialect="chat-completions",
        base="https://vendor.example",
        caps=Caps(max_output=64),
        provider="deepseek",
        web=RouteWeb(share=True),
    )

    assert boot.engines_for([plain]).names() == (), "a route that names no provider is not an engine"
    assert boot.engines_for([plain, named]).names() == ("deepseek",), "and one that names it installs exactly it"
    assert [spec.name for spec in net_specs(boot.engines_for([named]))] == ["web_fetch", "web_search"], (
        "an engine with both verbs is both tools, and the search is the second one it offers"
    )


def test_an_engine_gives_the_tools_its_verbs_can_answer() -> None:
    """A verb that is not there is a tool that is not there: nothing promises a page read to Bing."""
    any_engine = _Any()

    assert [spec.name for spec in net_specs(Engines([Engine(name="only-search", search=any_engine.search)]))] == [
        "web_search"
    ], "an engine that only searches is one tool"
    assert [spec.name for spec in net_specs(Engines([Engine(name="only-pages", page=any_engine.page)]))] == [
        "web_fetch"
    ], "and one that only reads pages is the other"


def test_the_network_section_names_the_tools_this_kernel_has() -> None:
    """A prompt must never name a tool that is not in the table, and a search's argument is its value domain.

    The names travel here and nowhere else: a model that cannot read them cannot pass one, and a section
    that promised ``web_fetch`` to a program with one engine that only searches would be a round lost.
    """
    assert "web_search" not in GROUP_PLAYBOOK[Group.NET], "the floor is the tool every kernel has"
    assert "web_fetch" not in GROUP_PLAYBOOK[Group.NET]

    both = Engine(name="either", search=_Any().search, page=_Any().page)
    catalog = ToolCatalog((*BUILTIN, *net_specs(Engines([both]))))
    net_members = dict(catalog.groups())[Group.NET.value]
    view = ToolView(
        specs={name: spec for name in net_members if (spec := catalog.get(name)) is not None},
        groups=catalog.groups(),
        disclosed=frozenset({Group.NET.value}),
    )

    said = PromptPlan(engines=("bing", "deepseek")).compose(view)

    assert "— bing, deepseek —" in said, "the value domain of the argument, in the order a search asks"
    assert "`web_search` finds pages" in said and "`web_fetch` reads one" in said


def test_a_net_body_is_kept_rather_than_paged_through() -> None:
    """A body too long to send is a handle, so no net tool takes an offset to page through one (§0.9, E28).

    ``read`` has an offset, and it counts *lines of a file* — that is a window on something already here.
    A response body is not read that way: it is kept, and the handle is the one way to the rest of it.
    """
    for spec in (*net.SPECS, *net.CONDITIONAL):
        assert "offset" not in spec.schema()["properties"], f"{spec.name} pages a body, and does not keep it"


class _Any:
    """An engine that is only ever asked whether it is there: both verbs, and neither reaches anything."""

    async def search(self, query: str, *, gate: object, on: str = "") -> tuple[Hit, ...]:
        del query, gate, on
        return ()

    async def page(self, url: str, *, on: str = "") -> Page:
        del url, on
        return Page()


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


# -- what may enter a history as words, which is what a prompt section must never do -------
#
# R9's other half: a section of the prompt is not a message. The record vocabulary is the whole check,
# because a writer can only put something in a history by carrying a record — so if every record either is
# a message, is dropped from what a model reads, or is announced as what it is (the summary standing in
# for what was compressed, the harness saying a reminder in its own marked line), no writer anywhere can
# put words into a person's mouth.


def test_every_record_arrives_as_who_actually_spoke() -> None:
    """Two records are read as messages and neither of them is a person's own words.

    A summary is made into the person's turn it stands in for, and a notice into the harness speaking —
    with a marker in its text, so the model does not take it for somebody else in the conversation. Every
    other record is dropped, and that split is the whole check: a new record has to be classified here,
    and not arrive in a history because nobody asked who said it.
    """
    assert get_args(AnyRecord.__value__) == (
        AnyMessage,
        StateChange,
        Summary,
        Notice,
        Withheld,
        Attempt,
        Effect,
        Consent,
        Usage,
        TurnEnded,
        Promoted,
    ), "a record nobody classified would be a writer nobody asked about"
    kinds = {type(as_message(cls())).__name__ for cls in get_args(AnyMessage.__value__)}
    assert kinds == {"User", "Assistant", "Result", "Note"}, "a message is a message, whichever of the four it is"
    assert as_message(Notice(text=NOTICE_MARKER)) == Note(text=NOTICE_MARKER), (
        "a notice is the harness speaking, and the marker is in the text it is read from"
    )
    assert as_message(Notice()) is None, "a notice that stopped being true said nothing"
    assert [
        as_message(record)
        for record in (StateChange(), Withheld(), Attempt(), Effect(), Consent(), Usage(), TurnEnded())
    ] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ], (
        "what a session knows, what it stopped showing, what became of a request, what a call left "
        "different, what a person was asked, what a request cost and that a turn ended are not things a "
        "model reads"
    )
    says = as_message(Summary(text="it was settled"))

    assert says is not None and says.text.startswith(SUMMARY_HEADING), "a summary is announced as one"


def test_every_record_written_is_a_record_read_back() -> None:
    """The tag is a column and the reader is one branch per kind, so a new kind needs a new branch.

    Nothing else would notice: a row of a kind the reader does not know decodes as a message and fails
    there, which is a journal that cannot be listed — so the shape is checked here, once, over the whole
    union, not by each writer remembering.
    """
    for kind in get_args(AnyRecord.__value__):
        if kind is AnyMessage:
            continue
        record = kind()
        assert _decode(kind_of(record), _encode(record)) == record, f"{kind.__name__} is written and not read"


def test_the_four_outcomes_are_the_words_the_contract_carries() -> None:
    """The wire spells these as plain strings, so the spelling is the contract, not our business."""
    assert [status.value for status in Status] == ["ok", "refused", "failed", "partial"]


def test_every_kind_of_trouble_is_one_row_of_one_table() -> None:
    """A failure with no row is a failure with no status and no answer to "try again?"."""
    assert set(OUTCOMES) == set(Failure)

    for failure, (status, retry) in OUTCOMES.items():
        result = (
            ToolResult.denied(DenialKind.NEVER_ALLOWED, "something went wrong")
            if failure is Failure.DENIED
            else ToolResult.of(failure, "something went wrong")
        )

        assert (result.status, result.retry) == (status, retry), "the table decides, never the raise site"


def test_a_denial_is_a_kind_and_never_a_bare_word() -> None:
    """``DENIED`` with a sentence beside it is the shape that made a refusal unreadable (§0.15, E118)."""
    with pytest.raises(ValueError, match="a denial is a kind"):
        _ = ToolResult.of(Failure.DENIED, "not allowed")

    with pytest.raises(ValueError, match="a denial is a kind"):
        _ = ToolResult(Status.OK, "all good", denial=DenialKind.HUMAN)

    refused = ToolResult.denied(DenialKind.TIMEOUT, "nobody answered in time")
    assert (refused.status, refused.failure, refused.denial) == (Status.REFUSED, Failure.DENIED, DenialKind.TIMEOUT)


def test_every_denial_kind_has_one_sentence() -> None:
    """The word is for a client and the sentence is for a model: a kind with neither is unreadable."""
    assert set(DENIAL_SAYS) == set(DenialKind), "a kind of refusal with no sentence says nothing at all"

    for kind, says in DENIAL_SAYS.items():
        assert "{about}" in says, f"{kind} does not say what was refused"


def test_the_kernel_does_not_know_the_names_of_the_policies() -> None:
    """A policy is a frontend's business — a kernel that knew either word would be a kernel with a mode.

    ``--yolo`` and ``--away`` are what a *client* answers with (§0.15, E114). The kernel asks and is
    answered, so the words cannot appear below the boundary: a tool, a policy or the wire that knew
    them could act on "what mode is this run in", which is the coupling this rewrite exists to end.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "plyngent"
    named = [path for package in ("kernel", "wire") for path in (root / package).rglob("*.py")]

    assert named, "the walk found no modules, so this asserts nothing"
    for path in named:
        written = path.read_text(encoding="utf-8").lower()
        assert "yolo" not in written, f"{path} names a client's policy"
        assert "afk" not in written, f"{path} names a client's policy"


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
    """One template per kind of refusal: a second wording at the tool layer is a second answer to fix.

    What a hint may carry is the refusal's to say and the tool's to accept: the mode is offered to the
    tool that takes a ``mode`` and to that one only, because a hint naming an argument a tool does not
    have is a call the model cannot make.
    """
    escaped = OutsideWorkspace(
        pathlib.Path("../x"),
        resolved=pathlib.Path("/elsewhere/x"),
        root=pathlib.Path("/ws"),
        need=AccessMode.EXEC,
    )
    lifted = lifted_path(escaped)
    denied = refusal(Denied(pathlib.Path("/ws/key.pem"), "/key"))

    assert (lifted.status, lifted.failure, lifted.retry) == (Status.REFUSED, Failure.BLOCKED, Retry.FIX_ARGS)
    assert lifted.summary == str(escaped), "the workspace's sentence, not a rewrite of it"
    assert lifted.hints == (
        Hint(tool="request_directory_access", given=(("mode", "exec"),)),
        Hint(tool="new_temporary_workspace"),
    ), "both ways to lift it are offered, and only the one that takes a mode is told it"
    assert "request_directory_access" not in lifted.summary, "and the kernel names no tool in prose"
    assert (denied.failure, denied.hints) == (Failure.DENIED, ()), "nothing lifts a denial, so nothing is offered"


def test_the_two_ways_out_of_the_workspace_declare_what_they_are() -> None:
    """Acceptance ④: the grant tool is the question itself, and a scratch directory is agreed to once.

    Neither judgement is written in a handler: the posture comes off ``uses``/``mutates``, and the two
    that cannot be derived — nothing to ask twice, and one yes per host — say why beside the field.
    """
    asking = ToolCatalog().get("request_directory_access")
    scratch = ToolCatalog().get("new_temporary_workspace")

    assert asking is not None and scratch is not None, "a refusal that hints at a name has to be a name"
    assert asking.uses == Res.PATHS | Res.HUMAN, "it *is* the question, so asking it again would be a bug"
    assert _posed(asking, _given(asking)) == (Posture.AUTO, None)
    assert asking.level() is Severity.DANGER, "the effect is a person's, and there is no undo for it"
    assert asking.intention is Intention.REQUIRED, "a call that puts a question to a person says why"
    assert sorted(next(iter(dict(_enums(asking.schema())).values()))) == ["exec", "read", "write"], (
        "the closed set is the words a person reads, not the integers an IntEnum would give"
    )
    assert scratch.uses == Res.PATHS and scratch.mutates is Mutation.HOST, "a directory is the host's"
    assert _posed(scratch, _given(scratch)) == (Posture.ONCE, Remember.HOST), "one yes for this host"
    assert scratch.level() is Severity.ROUTINE, "handing out a scratch directory changes nothing of yours"
    assert not {asking.name, scratch.name} & set(RESIDENT), "neither is resident: a path is allowed per session"


def test_waiting_is_the_third_way_a_turn_spends_a_person_s_time() -> None:
    """Acceptance ④ for the last name: a wait *is* a question, and it is always available.

    ``HUMAN`` derives both answers §10.10 asks of it — nothing to ask twice, and every call says why —
    and ``TOOLS`` rides along because a wait has to know whether anybody is there to wait for.
    """
    waiting = ToolCatalog().get("wait")
    assert waiting is not None, "a resident that names nothing would show in a view's names"
    assert waiting.name in RESIDENT, "a model that had to be shown `wait` before it could wait cannot wait"
    assert waiting.group is Group.SESSION, "it spends a turn, and the turn's own subject is this group"
    assert waiting.uses == Res.HUMAN | Res.TOOLS
    assert _posed(waiting, session.WaitArgs(seconds=1.0, intention="let them read it")) == (Posture.AUTO, None), (
        "the wait is the question, so asking it again would be a bug"
    )
    assert waiting.intention is Intention.REQUIRED and waiting.mutates is Mutation.NONE


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


def _posed(spec: ToolSpec, args: object) -> tuple[Posture, Remember | None]:
    """Whether one call is put to a person, and by what an answer is remembered."""
    asking = spec.asking(args)
    return asking.posture, asking.remember


def test_every_tool_states_how_heavy_its_ordinary_use_is() -> None:
    """One level per tool, read off ``uses``/``mutates`` — with one declared exception and its reason."""
    declared = {spec.name: spec.severity for spec in EVERY if spec.severity is not None}

    assert declared == {"write_file": Severity.ROUTINE}, (
        "creating a path is lighter than replacing one, and both are writes into the workspace"
    )
    for spec in EVERY:
        reaches = spec.uses & (Res.HUMAN | Res.CLIENT | Res.EXEC) or spec.mutates & (
            Mutation.OUTSIDE | Mutation.DESTRUCTIVE
        )
        if reaches:
            assert spec.level() is Severity.DANGER, f"{spec.name} can do what nothing here can undo"
        elif spec.mutates is Mutation.NONE:
            assert spec.level() is Severity.ROUTINE, f"{spec.name} changes nothing"


def test_an_argument_that_makes_a_call_heavier_never_makes_it_lighter() -> None:
    """``escalates`` raises the level and takes the memory away; the other direction is a bug."""
    for spec in EVERY:
        for name, level in (spec.escalates or {}).items():
            assert SEVERITIES.index(level) >= SEVERITIES.index(spec.level()), (
                f"{spec.name}({name}=true) would be lighter than the tool's own use"
            )


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
    assert not {spec.name for spec in skills.SPECS} & set(RESIDENT), "the residents are the same thirteen"


def test_the_skill_reading_tools_are_reads() -> None:
    """Reading a directory nobody in this workspace wrote changes nothing, so nothing is asked."""
    for spec in skills.SPECS:
        if spec.mutates is not Mutation.NONE:
            continue
        assert _posed(spec, _given(spec)) == (Posture.AUTO, None), f"{spec.name} only reads"
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
        assert _posed(spec, _given(spec)) == (Posture.ALWAYS, None), f"{spec.name} asks every single time"
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
