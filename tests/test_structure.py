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
from typing import TYPE_CHECKING, get_args

from plyngent import boot, cli, kernel, wire
from plyngent.cli.grammar import ALIASES, build
from plyngent.cli.render import render
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


def _catalog():
    """The client's vocabulary, derived the one way it is ever derived."""
    return build(wire.KernelManifest(commands=wire.command_specs()), session="main")
