"""What a typed line means: plain text is a message, and a slash word is a command.

The command table is not written here. The kernel's manifest already carries each command's tag, its
one-line help and the JSON Schema of its own arguments, so this module *derives* the parser, the help
text and the completion candidates from that one document — three things that cannot drift apart
because there is only one source (R2.4 C5). A command whose arguments cannot be spelled on a line is
withheld with a reason, and never offered half-way.

Two halves, both terminal business:

* a slash word is the command's own tag, plus a short alias table (:data:`ALIASES`) for the words a
  person reaches for first. The alias table is the adapter's: a kernel command may be named anything,
  and ``/new`` is a terminal convenience, with no second command behind it (R2.4 C4);
* text without a slash is a message — ``Submit`` — because "say something" is the one thing a person
  does all day and should never have to spell.

A line is split the way a shell splits one (``shlex``), because an argument is a value and a value may
contain a space: ``/export --path "my notes.md"`` is one path, not two. An argument the person wrote as
``""`` is the empty string and stays one — dropping it would turn "say nothing here" into "say nothing
at all", which click reads as a missing option.

``/submit`` is the exception that proves the rule: it exists so a message can be composed in the
human's editor, which is the one thing a single line cannot do. It takes no arguments; the text comes
from the editor and the session is the one in focus.

``/travel`` is the second word this adapter owns, and the only one that sends nothing at all: where a
frontend is looking is its own state, so standing somewhere else is a change here, and no request
reaches the kernel. Everything else on a line is either a command from the manifest or a read.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, get_args, override

import click
import msgspec
from prompt_toolkit.completion import CompleteEvent, Completer, Completion

from plyngent.wire.command import AnyCommand, Command, Submit
from plyngent.wire.event import TurnKind
from plyngent.wire.query import AnyQuery, Grants, Query, Skills, Trajectory, Turns

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from prompt_toolkit.document import Document

    from plyngent.wire.manifest import CommandSpec, KernelManifest

#: The words a person reaches for first, and the tags they mean. Every alias must name a tag that
#: exists, and every tag must be reachable — by itself, through an alias, or by being withheld with a
#: reason; ``tests/test_structure.py`` holds those rules together.
ALIASES: dict[str, str] = {"new": "open", "sessions": "list-sessions", "model": "set-route"}

#: The words that *read*, and ask for nothing to be done: each names one query, and none is in the manifest.
#: A manifest lists what a client may ask the kernel to **do**; these ask it to say something and travel
#: on their own lane (§5, E134), so the table is this adapter's — the word is the query's tag the same way
#: a command's word is its tag, and ``tests/test_structure.py`` holds the two tables to one shape.
#:
#: Every read so far is about one session, which is why the table says so: a read that was about nothing
#: in particular would be a second shape here, and naming it today would be a dict of ``Query`` that no
#: read could be built from.
READS: dict[str, type[AnyQuery]] = {
    "grants": Grants,
    "history": Trajectory,
    "running": Turns,
    "skills": Skills,
}

#: The one word whose text is written in an editor, and never on the line.
COMPOSED = "submit"

#: What that word does, said in its own terms: the wire's summary describes a ``submit`` whose text is
#: an argument, and this client withholds that argument in favour of an editor.
COMPOSE_NOTE = "Write the message in your editor, then send it."

#: The word that moves where this client stands in the forest. It is the adapter's and has no command
#: behind it at all: which branch a frontend is reading is the frontend's own state, so there is nothing
#: to ask the kernel — the entry becomes the next message's ``begin_after``, and the kernel validates it
#: when that message arrives (R2.1 D22, §5 P5).
TRAVELLED = "travel"

#: What standing somewhere means, in the words a person reads: nothing about this program, and nothing
#: about what it avoids.
TRAVEL_NOTE = "Continue this conversation from an earlier entry."

#: The only argument that word takes: which entry to stand at.
TRAVEL_USAGE = "--entry ENTRY"

#: The word that asks what the words are. It is the adapter's, like the two above: the manifest lists the
#: commands, and the sentence about them is this client's.
HELP = "help"

#: How a slash line is spelled.
SLASH = "/"

#: The word that asks something beside the conversation. Its text goes as a ``Submit`` of the aside
#: kind: the kernel runs that turn with the tools that write nothing, and leaves its rows out of the
#: model's history until somebody keeps them (E163).
ASIDE = "btw"

#: What that word is for, in the words a person reads.
ASIDE_NOTE = "Ask something beside the conversation; the answer joins it only if you say to keep it."

#: The only argument that word takes: the question itself, spelled as it was typed.
ASIDE_USAGE = "<question>"

#: The only argument a read word takes: which conversation to ask about, and nothing else.
READ_USAGE = "[--session TEXT]"


def _scoped(kind: type[Query]) -> bool:
    """Whether this read word asks about one conversation, and not about the program itself.

    One word asks the other question: the skills are found when the program starts and every conversation
    sees the same table, so ``/skills`` has no name to take and no session to resolve.
    """
    return kind is not Skills


def _asks_about(kind: type[AnyQuery], session: str) -> AnyQuery:
    """The question one read word puts: the conversation it names, or none at all.

    A builder per word, since one keyword argument cannot carry both shapes: a
    ``Grants`` about a conversation takes its name and a ``Skills`` has no field to put one in.
    """
    return _ASKS[kind](session)


#: How each read word becomes its query, keyed by the query it names.
_ASKS: dict[type[AnyQuery], Callable[[str], AnyQuery]] = {
    Grants: lambda session: Grants(session=session),
    Trajectory: lambda session: Trajectory(session=session),
    Turns: lambda session: Turns(session=session),
    Skills: lambda _session: Skills(),
}


#: How many entries an option that stands for a tuple takes, when the schema fixes its length.
PAIR_LENGTH = 2

#: The plain JSON Schema types this client can put on a line.
_BY_TYPE: dict[str, type] = {"string": str, "integer": int}

#: One object out of a JSON Schema: read by shape, never trusted to be there.
type _Object = dict[str, object]


@dataclass(frozen=True)
class Entry:
    """One slash word: what to send, how to argue with it, and what to say about it."""

    tag: str
    command: click.Command
    #: Options that must be given zero or twice — a tuple the wire carries as a fixed-length pair.
    pairs: list[str] = field(default_factory=list)
    #: Whether this word opens an editor, taking no arguments from the line.
    composes: bool = False

    @property
    def summary(self) -> str:
        """The one line a help listing shows."""
        return self.command.help or ""


@dataclass(frozen=True)
class Catalog:
    """The words this client can spell, and the ones it honestly cannot.

    ``withheld`` is not a list of bugs: a command may carry an argument shape a line cannot express,
    and saying "not from here" beats a parser that silently drops a field.
    """

    offers: dict[str, Entry] = field(default_factory=dict)
    withheld: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=lambda: dict(ALIASES))

    def word(self, said: str) -> str | None:
        """The tag a word or alias names, or ``None`` when this client cannot spell it."""
        tag = self.aliases.get(said, said)
        return tag if tag in self.offers else None


@dataclass(frozen=True)
class Runs:
    """A command ready to send, spelled out in full."""

    command: AnyCommand


@dataclass(frozen=True)
class Composes:
    """A message to write in an editor and then send as a ``Submit`` to this session."""

    session: str


@dataclass(frozen=True)
class Help:
    """What to show when nothing is done."""

    text: str


@dataclass(frozen=True)
class Wrong:
    """A line this client cannot make sense of, and what to say about it."""

    text: str


@dataclass(frozen=True)
class Reads:
    """A question to put to the kernel, whose answer this client prints and does not forward.

    It is not a ``Runs``: a command changes the world and its outcome arrives as an event, while a read
    asks for a value and gets it back on the lane that promised one (§5, E134).
    """

    query: AnyQuery


@dataclass(frozen=True)
class Travels:
    """A place in the forest to stand at from now on: what the next message will continue.

    It is not a ``Runs`` either, and for the opposite reason to a read: nothing is sent at all. The
    kernel keeps no "current branch", so where this frontend is looking is this frontend's business.
    """

    entry: str


#: What one line came to. Six outcomes, six different things for the loop to do.
type Intent = Runs | Composes | Reads | Travels | Help | Wrong


def build(manifest: KernelManifest, *, session: str) -> Catalog:
    """Derive the slash vocabulary from the manifest, once per connection.

    ``session`` is the session in focus, so ``--session`` can default to it, and no command has to say
    which conversation it means.
    """
    classes = {cls.__struct_config__.tag: cls for cls in get_args(AnyCommand.__value__)}
    offers: dict[str, Entry] = {}
    withheld: dict[str, str] = {}
    for spec in manifest.commands:
        tag = spec.name
        if tag not in classes:
            withheld[tag] = "the manifest names a command this client has no value for"
            continue
        if tag == COMPOSED:
            offers[tag] = Entry(tag=tag, command=_bare(spec), composes=True)
            continue
        options, pairs, missing = _options(spec, session=session)
        if missing:
            withheld[tag] = missing
            continue
        offers[tag] = Entry(tag=tag, command=_command(spec, options), pairs=pairs)
    return Catalog(offers=offers, withheld=withheld)


def decode(line: str, catalog: Catalog, *, session: str) -> Intent:
    """What this line came to: a command to send, an editor to open, or something to say."""
    if not line.startswith(SLASH):
        return Runs(command=Submit(session=session, text=line))
    return _slash(line[len(SLASH) :], catalog, session=session)


def word_help(catalog: Catalog, word: str) -> str:
    """One word's own help, as the word itself answers it (``/open --help``).

    The one way a word describes itself, so the two are one function: a command's text is click's own,
    rendered from the options the parser was built from, while a read word and the one word this client
    answers itself have no command behind them and say their two lines here.

    A word nobody has is a sentence, and never an exception: a person typed it on the other end.
    """
    tag = catalog.word(word)
    if tag is not None:
        return _help_of(catalog.offers[tag])
    if word in READS:
        kind = READS[word]
        return f"{SLASH}{word} {_read_usage(kind)}\n\n  {_read_summary(kind)}"
    if word == TRAVELLED:
        return f"{SLASH}{TRAVELLED} {TRAVEL_USAGE}\n\n  {TRAVEL_NOTE}"
    if word == ASIDE:
        return f"{SLASH}{ASIDE} {ASIDE_USAGE}\n\n  {ASIDE_NOTE}"
    return f"no command {SLASH}{word} here"


def help_text(catalog: Catalog) -> str:
    """The listing: every word this client can spell, and the sentence that says what it is for."""
    spelled = [
        (f"{SLASH}{tag}" + (" <editor>" if entry.composes else _usage(entry)), entry.summary)
        for tag, entry in catalog.offers.items()
    ]
    spelled.extend((f"{SLASH}{word} {_read_usage(kind)}".rstrip(), _read_summary(kind)) for word, kind in READS.items())
    spelled.append((f"{SLASH}{TRAVELLED} {TRAVEL_USAGE}", TRAVEL_NOTE))
    spelled.append((f"{SLASH}{ASIDE} {ASIDE_USAGE}", ASIDE_NOTE))
    width = max((len(usage) for usage, _ in spelled), default=0) + 2
    lines = ["commands you can type:", *(f"  {usage:<{width}}{summary}" for usage, summary in spelled)]
    lines.append(
        "the same commands, shorter: "
        + ", ".join(f"{SLASH}{word} = {SLASH}{tag}" for word, tag in catalog.aliases.items())
    )
    if catalog.withheld:
        refused = ", ".join(f"{SLASH}{tag} ({why})" for tag, why in catalog.withheld.items())
        lines.append(f"not from here: {refused}")
    lines.append("anything else you type is a message")
    return "\n".join(lines)


def _slash(text: str, catalog: Catalog, *, session: str) -> Intent:
    """A line that begins with the slash: a word, then the arguments it was given."""
    said, _, rest = text.partition(" ")
    if not said or said == HELP:
        return _asking_help(said, rest, catalog)
    reading = READS.get(said)
    if reading is not None:
        return _reading(said, reading, rest, catalog=catalog, session=session)
    if said == TRAVELLED:
        return _travelling(rest, catalog=catalog)
    if said == ASIDE:
        return _betwixt(rest, catalog=catalog, session=session)
    tag = catalog.word(said)
    if tag is None:
        return Wrong(text=f"no command {SLASH}{said} here; {SLASH}help lists the ones there are")
    return _known(tag, catalog.offers[tag], rest, catalog=catalog, session=session)


def _asking_help(said: str, rest: str, catalog: Catalog) -> Intent:
    """The listing — or a sentence saying that a word describes itself, and how to ask that word.

    A bare slash asks what the word for asking does, so both are the same answer. A word after it used to
    be answered here too, and that was two answers to one question: the word's own text is the one that
    cannot go stale, because it is rendered from the options the parser was built from (E147).
    """
    asked = rest.strip().split(" ")[0].lstrip(SLASH) if rest.strip() else ""
    if said and asked:
        # The example is the word they asked about when this client has it, so the sentence points at the
        # line that would actually answer them.
        known = catalog.word(asked) is not None or asked in READS or asked in (TRAVELLED, ASIDE)
        example = asked if known else "open"
        return Wrong(text=f"{SLASH}{HELP} takes no argument; a word describes itself, as {SLASH}{example} --help")
    return Help(text=help_text(catalog))


def _reading(word: str, kind: type[AnyQuery], rest: str, *, catalog: Catalog, session: str) -> Intent:
    """A read word: the query it names, and the session it is about when it is about one.

    Click is not used here and that is not a shortcut: there is no command behind this word to derive a
    parser from, and a read that grew a second argument would be a read that should have been a command.
    ``--help`` is answered from the query's own docstring, the way a command's help is answered from its.
    """
    try:
        parts = shlex.split(rest)
    except ValueError as exc:
        return Wrong(text=f"{SLASH}{word}: {exc}")
    if parts in (["--help"], ["-h"]):
        return Help(text=word_help(catalog, word))
    if not _scoped(kind):
        return Reads(query=_asks_about(kind, session)) if not parts else Wrong(text=f"{SLASH}{word} takes no argument")
    match parts:
        case []:
            return Reads(query=_asks_about(kind, session))
        case ["--session", asked]:
            return Reads(query=_asks_about(kind, asked))
        case _:
            return Wrong(text=f"{SLASH}{word} takes {READ_USAGE} and nothing else")


def _read_usage(kind: type[Query]) -> str:
    """How one read word is spelled: the conversation it asks about, or nothing when it asks about the program."""
    return READ_USAGE if _scoped(kind) else ""


def _read_summary(kind: type[Query]) -> str:
    """What one read is for: the first line of its own docstring, as a command's summary is its own."""
    return (kind.__doc__ or "").strip().splitlines()[0]


def _betwixt(rest: str, *, catalog: Catalog, session: str) -> Intent:
    """A question beside the conversation: the words after the slash, sent as the aside kind of turn.

    The text is taken as it was typed and not through ``shlex``, because a question is a sentence: the
    quotes in it are the person's and there is no second argument here to take them apart for. ``--help``
    is the one exception, and it is every word's.
    """
    text = rest.strip()
    if text in ("--help", "-h"):
        return Help(text=word_help(catalog, ASIDE))
    if not text:
        return Wrong(text=f"{SLASH}{ASIDE} takes {ASIDE_USAGE}")
    return Runs(command=Submit(session=session, text=text, kind=TurnKind.ASIDE))


def _travelling(rest: str, *, catalog: Catalog) -> Intent:
    """Where to stand: one entry id, spelled as the option a person can see in the listing.

    One argument and no second: an entry is a name the kernel minted, so there is nothing here to
    interpret — and an empty one is a line that forgot its value, and no place to stand.
    """
    try:
        parts = shlex.split(rest)
    except ValueError as exc:
        return Wrong(text=f"{SLASH}{TRAVELLED}: {exc}")
    if parts in (["--help"], ["-h"]):
        return Help(text=word_help(catalog, TRAVELLED))
    match parts:
        case ["--entry", entry] if entry:
            return Travels(entry=entry)
        case _:
            return Wrong(text=f"{SLASH}{TRAVELLED} takes {TRAVEL_USAGE} and nothing else")


def _known(tag: str, entry: Entry, rest: str, *, catalog: Catalog, session: str) -> Intent:
    """A word this client can spell: either it wants an editor, or click does the parsing.

    The arguments are split like a shell's, so a quoted value stays one argument and an unquoted one is
    several; a line with an unclosed quote is a sentence, and never an exception, because the person is
    still typing on the other end of it.
    """
    try:
        args = shlex.split(rest)
    except ValueError as exc:
        return Wrong(text=f"{SLASH}{tag}: {exc}")
    if entry.composes:
        if args:
            return Wrong(text=f"{SLASH}{tag} takes no argument: it opens an editor for the text")
        return Composes(session=session)
    if "--help" in args or "-h" in args:
        return Help(text=word_help(catalog, tag))
    return _parsed(tag, entry, args)


def _parsed(tag: str, entry: Entry, args: list[str]) -> Intent:
    """Let click parse the arguments, then make the value the wire expects."""
    try:
        parsed = entry.command.make_context(tag, list(args), parent=None)
    except click.UsageError as exc:
        return Wrong(text=exc.format_message())
    except click.exceptions.Exit:
        return Wrong(text=f"{SLASH}{tag} did not understand that line; try {SLASH}help {tag}")
    values = _pairs(parsed.params, entry)
    if isinstance(values, str):
        return Wrong(text=values)
    return Runs(command=_command_value(tag, values))


# -- deriving the parser -----------------------------------------------------------


def _command(spec: CommandSpec, options: list[click.Parameter]) -> click.Command:
    """One click command: the tag is its name, the wire's summary is its help, the options its fields.

    Its callback is never run — this client builds a value and sends it — but click still does the
    parsing, the validation and the usage message, which is the point of deriving it: a second tokenizer
    would be one more thing to keep in step.
    """
    return click.Command(name=spec.name, params=options, help=spec.summary, callback=_ignored, add_help_option=False)


def _bare(spec: CommandSpec) -> click.Command:
    """A command that takes nothing: its one input is an editor."""
    return click.Command(name=spec.name, params=[], help=COMPOSE_NOTE, callback=_ignored, add_help_option=False)


def _ignored(**_values: object) -> None:
    """Click insists on a callback; sending happens in the loop, not here."""


def _options(spec: CommandSpec, *, session: str) -> tuple[list[click.Parameter], list[str], str]:
    """The options of one command, the ones that are pairs, and why it cannot be spelled (if so).

    Every command carries ``session`` — the contract gives each one a target — so it is an option here
    too, defaulting to the session in focus, where the wire allows the empty string.

    ``workspace`` is on the wire's base class as well and is *not* an option: which tree a command means
    is the connection's own declaration, made once when it connects, and a per-command override is a thing
    a frontend with more than one tree does deliberately, and typing a flag fifteen times is not that
    (E154).

    ``sole`` is whether this command's whole argument is the one field besides ``session``: a word with a
    single value takes it bare (``/goal on``), because an option for it would be a name a person has to
    type to say a word they already know (E146).
    """
    document = _object(json.loads(spec.args))
    name = _text(document.get("$ref")).rpartition("/")[2]
    defs = _object(document.get("$defs"))
    properties = _object(_object(defs.get(name)).get("properties"))
    fields = {field: schema for field, schema in properties.items() if field not in {"cmd", "session", "workspace"}}
    options: list[click.Parameter] = []
    pairs: list[str] = []
    for field_name, schema in fields.items():
        option, pair, why = _option(field_name, _object(schema), defs, sole=len(fields) == 1)
        if option is None:
            return [], [], why
        if pair:
            pairs.append(option.name or field_name)
        options.append(option)
    options.append(
        click.Option(["--session"], type=str, default=session, show_default=True, help="the conversation to act on")
    )
    return options, pairs, ""


def _option(
    name: str, schema: _Object, defs: _Object, *, sole: bool = False
) -> tuple[click.Parameter | None, bool, str]:
    """One field as a click option: its type, whether it is a pair, and why it cannot be spoken.

    ``None`` and a reason travel together: a field this client cannot spell has no option at all, and
    building a placeholder one would give a click command that cannot be constructed, where a command
    that is withheld can be.
    """
    # A default belongs to the property, and not to the type it refers to, so it is read before the
    # reference is followed — otherwise an enum with a default would silently lose it.
    offered = schema.get("default")
    branches = [_object(branch) for branch in _objects(schema.get("anyOf"))]
    nullable = any(branch.get("type") == "null" for branch in branches)
    return _shaped(name, _unwrap(schema, defs), offered=offered, nullable=nullable, sole=sole)


def _unwrap(schema: _Object, defs: _Object) -> _Object:
    """The real shape behind a nullable or referenced schema; empty when there is more than one."""
    branches = [_object(branch) for branch in _objects(schema.get("anyOf")) if _object(branch).get("type") != "null"]
    if "anyOf" in schema:
        if len(branches) != 1:
            return {}
        schema = branches[0]
    if "$ref" in schema:
        return _object(defs.get(_text(schema["$ref"]).rpartition("/")[2]))
    return schema


def _shaped(
    name: str, schema: _Object, *, offered: object, nullable: bool, sole: bool = False
) -> tuple[click.Parameter | None, bool, str]:
    """The option for one field, by the shape its schema has."""
    choices = [str(choice) for choice in _objects(schema.get("enum"))]
    if choices:
        default = None if nullable else offered if offered is not None else ""
        if sole:
            # The whole argument of the word is one of a few words, so it is typed bare: an option here
            # would make a person name a value they are looking at in the help.
            return click.Argument([name], type=click.Choice(choices), default=default, required=False), False, ""
        # The choices are the option's metavar, so they are already on the line: help here would say
        # the same values a second time, in a narrower column.
        return click.Option([_flag(name)], type=click.Choice(choices), default=default), False, ""
    kind = schema.get("type")
    if isinstance(kind, str) and kind in _BY_TYPE:
        default = None if nullable else offered if offered is not None else ""
        return click.Option([_flag(name)], type=_BY_TYPE[kind], default=default), False, ""
    if kind == "array":
        return _listed(name, schema)
    if kind == "boolean":
        # A boolean is a switch and not a value: nothing to type after it, and no default to send when
        # it is absent — which is why this is read before the table of types that take an argument.
        return click.Option([_flag(name)], is_flag=True, default=bool(offered)), False, ""
    return None, False, "has a shape this client cannot put on a line"


def _listed(name: str, schema: _Object) -> tuple[click.Option | None, bool, str]:
    """An array field: either a list of strings, or a pair whose length the schema fixes."""
    fixed = schema.get("minItems") == PAIR_LENGTH and schema.get("maxItems") == PAIR_LENGTH
    # A position's shape is ``prefixItems`` (one object per position) and an element's is ``items`` (one
    # object covering every element), and the draft-07 spelling of a fixed tuple puts a list in
    # ``items`` instead — so both readings are taken. What matters either way: an element shaped as
    # anything but a string is a value a line cannot carry, and offering it would build a command the
    # kernel cannot read.
    shapes = schema.get("prefixItems", schema.get("items"))
    parts = [_object(part) for part in _objects(shapes)] or [_object(shapes)]
    if any(part.get("type") != "string" for part in parts):
        return None, False, "is a list this client cannot put on a line"
    help_note = "give this option twice" if fixed else "this option may be given more than once"
    return click.Option([_flag(name)], type=str, multiple=True, default=(), help=help_note), fixed, ""


def _pairs(params: dict[str, object], entry: Entry) -> dict[str, object] | str:
    """Settle the pairs: nothing given is ``None``, two given is a tuple, anything else is a problem."""
    values = dict(params)
    for name in entry.pairs:
        given = values.get(name)
        if not isinstance(given, tuple):
            continue
        entered = cast("tuple[object, ...]", given)
        if not entered:
            values[name] = None
        elif len(entered) == PAIR_LENGTH:
            values[name] = entered
        else:
            return f"{_flag(name)} takes exactly {PAIR_LENGTH} entries, and got {len(entered)}"
    return values


def _command_value(tag: str, values: dict[str, object]) -> AnyCommand:
    """The value to send: the struct the manifest's schema was taken from, built from parsed fields.

    Built through ``msgspec`` and not by calling the struct, because a struct does not validate on
    construction: click answers the *word* a choice was picked from (``"on"``), and a value that skipped
    the wire's own decoding would arrive at the kernel as a string where the contract says a closed set —
    a difference only the in-process transport would ever see.
    """
    return cast("AnyCommand", msgspec.convert(values, _class_of(tag)))


def _class_of(tag: str) -> type[Command]:
    """The wire class a tag names."""
    for cls in get_args(AnyCommand.__value__):
        if cls.__struct_config__.tag == tag:
            return cls
    msg = f"no command carries the tag {tag!r}"
    raise KeyError(msg)


def _flag(name: str) -> str:
    """A field name as the option a person types."""
    return f"--{name.replace('_', '-')}"


def _object(value: object) -> _Object:
    """One JSON object, or an empty one: a schema is read by shape, and a missing part refuses."""
    return cast("_Object", value) if isinstance(value, dict) else {}


def _objects(value: object) -> list[object]:
    """One JSON array, or an empty one."""
    return cast("list[object]", value) if isinstance(value, list) else []


def _text(value: object) -> str:
    """One JSON string, or the empty one."""
    return value if isinstance(value, str) else ""


def _taken(entry: Entry, tail: str) -> Iterator[Completion]:
    """What one command's half-typed argument may still become.

    Two shapes, read off the same click command the parser uses: a value its bare argument accepts
    (``/goal o``), or the name of one of its options (``/goal --mod``).
    """
    if not tail.startswith("--"):
        for value in _chosen(entry):
            if value.startswith(tail):
                yield Completion(value, start_position=-len(tail))
        return
    for param in entry.command.get_params(click.Context(entry.command)):
        if isinstance(param, click.Option) and param.opts[0].startswith(tail):
            yield Completion(param.opts[0], start_position=-len(tail), display_meta=param.help or "")


def _chosen(entry: Entry) -> tuple[str, ...]:
    """The values a command's bare argument accepts, and nothing when it has no such argument.

    Read from the same click command the parser uses, so a word that grew or lost its closed set cannot
    disagree with what completion offers (E147).
    """
    for param in entry.command.get_params(click.Context(entry.command)):
        if isinstance(param, click.Argument) and isinstance(param.type, click.Choice):
            return tuple(str(choice) for choice in param.type.choices)
    return ()


def _usage(entry: Entry) -> str:
    """The options of one command, as the listing shows them: names only, since ``/help <word>`` has the rest."""
    names = [
        param.opts[0]
        for param in entry.command.get_params(click.Context(entry.command))
        if isinstance(param, click.Option)
    ]
    return "".join(f" [{name}]" for name in names)


def _help_of(entry: Entry) -> str:
    """One command's own help, generated by click from the same options the parser uses."""
    return entry.command.get_help(click.Context(entry.command, info_name=entry.tag))


class SlashCompleter(Completer):
    """Completion of slash words and their options, from the same table that parses them.

    Derived, never maintained by hand: a command that gains an option gains a completion, because both
    read the catalog. A word this client cannot spell is not offered, which is the answer the parser
    gives too.

    Every candidate carries the line that explains it — the command's summary for a word, the option's
    own help for an option — because a list of bare names makes the person guess, and the guess is what
    completion exists to save.
    """

    _catalog: Catalog

    def __init__(self, catalog: Catalog) -> None:
        super().__init__()
        self._catalog = catalog

    @override
    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterator[Completion]:
        """Complete the word being typed, or the argument of the command it names."""
        del complete_event
        text = document.text_before_cursor
        if not text.startswith(SLASH):
            return
        head, _, tail = text.partition(" ")
        if not tail:
            yield from self._words(head, text)
            return
        tag = self._catalog.word(head[len(SLASH) :])
        if tag is not None:
            yield from _taken(self._catalog.offers[tag], tail)

    def _words(self, head: str, text: str) -> Iterator[Completion]:
        """Every word this client has that starts with what has been typed so far."""
        word = head[len(SLASH) :]
        for candidate in sorted({*self._catalog.aliases, *self._catalog.offers, *READS, TRAVELLED}):
            if candidate.startswith(word):
                yield Completion(
                    f"{SLASH}{candidate}", start_position=-len(text), display_meta=self._summary(candidate)
                )

    def _summary(self, candidate: str) -> str:
        """What a word does, looked up through the alias that spells it."""
        if candidate in READS:
            return _read_summary(READS[candidate])
        if candidate == TRAVELLED:
            return TRAVEL_NOTE
        tag = self._catalog.word(candidate)
        return "" if tag is None else self._catalog.offers[tag].summary
