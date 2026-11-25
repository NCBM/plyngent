"""Driving a plugin through a real turn: a run that is started, carried on, and stopped.

These are the calls that cost something — a child process, a root of its own, a kernel brought up inside it —
so what they are tested for is what a person does with them: try a plugin, read what happened, fix it, carry
on, and finish. The plugin under test is the example the distribution ships, so a passing run is also the
proof that the shape the guide describes is one that loads (E184).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest

from plyngent.kernel.access import AccessMode
from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.env import Env
from plyngent.kernel.host import Host
from plyngent.kernel.toolresult import Status, Text
from plyngent.kernel.toolspec import Res
from plyngent_creative.tools.drive import DebugArgs, RunArgs, StopArgs, plugin_debug, plugin_stop, plugin_test
from plyngent_creative.tools.runs import RUNS
from tests.test_creative_checks import EXTRAS

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.cancel import Scope


@dataclass
class _Session:
    """As much of a session as a grant needs: a table, and its own id."""

    grants: dict[Path, AccessMode] = field(default_factory=dict)
    session_id: str = "s1"

    async def grant(self, path: Path, need: AccessMode) -> bool:
        if path in self.grants:
            return False
        self.grants[path] = need
        return True


@dataclass
class _Ctx:
    """The parts of a turn these calls touch: the host they resolve through, and the session they grant to."""

    host: Host
    session: _Session = field(default_factory=_Session)

    def scope(self, *_segments: str) -> Scope:
        return CancelToken().at("tool-1")


def _env(root: Path) -> Env:
    return Env(cast("Any", _Ctx(Host(root))), Res.PATHS | Res.EXEC, ("tool-1",), tool="probe", tools=cast("Any", None))


def _body(result: object) -> str:
    """What a model reads under the summary: the body's text parts, in the order they were laid out."""
    parts = cast("Any", result).body
    return "\n".join(part.value for part in parts if isinstance(part, Text))


@pytest.fixture(autouse=True)
async def _no_run_left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A temporary plugin directory, and no run left behind by a test that failed halfway."""
    monkeypatch.setattr("plyngent_creative.tools.plugins.user_data_path", lambda _name: tmp_path)
    yield
    await RUNS.close_all()
    assert RUNS.numbers() == (), "a run outlived the test that started it"


def _plugin(root: Path, name: str) -> Path:
    """The example plugin, put where the plugin directory holds it."""
    directory = root / "creative_plugins"
    directory.mkdir(parents=True, exist_ok=True)
    found = directory / f"{name}.py"
    _ = shutil.copy(EXTRAS / "examples" / "greets.py", found)
    return found


#: What a run of the example plugin does: a message, a model that asks for the group, the group arriving, the
#: plugin's tool answering, and the turn ending. It is written the way an author writes one — static JSON,
#: three keys a step, and each step's expectation looking at the events after the step before it.
_SCRIPT: list[dict[str, object]] = [
    {
        "model": {"text": "", "calls": [{"tool": "tool_require", "arguments": {"group": "greets"}}]},
        "send": "hello",
        "expect": {"tag": "tool-called", "fields": {"tool": "tool_require"}},
    },
    {"expect": {"tag": "tool-returned", "fields": {"tool": "tool_require", "status": "ok"}}},
    {
        "model": {"text": "", "calls": [{"tool": "greet", "arguments": {"name": "you"}}]},
        "expect": {"tag": "tool-called", "fields": {"tool": "greet"}},
    },
    {"expect": {"tag": "tool-returned", "fields": {"tool": "greet", "status": "ok"}}},
    {"model": {"text": "I greeted you"}, "expect": {"tag": "turn-finished", "fields": {"turn": 1, "reason": ""}}},
]


async def test_a_plugin_is_driven_through_a_real_turn_and_stopped(tmp_path: Path) -> None:
    """The whole point of the distribution: a plugin of somebody's, answering a model, in a kernel of its own.

    The events the run is checked against are the wire the kernel speaks, so a passing script is a statement
    about the contract and not about a rendering of it. ``greet`` being called is the plugin's own tool
    arriving, being asked for, and answering in one turn.
    """
    _ = _plugin(tmp_path, "greets")

    started = await plugin_test(_env(tmp_path), RunArgs(name="greets", steps=_SCRIPT))

    assert started.status is Status.OK, started.summary
    handle = int(started.summary.split("run ")[1].split(":")[0])
    told = _body(started)
    assert "step 5: ok" in told
    answered = [line for line in told.splitlines() if line.startswith("  tool-returned")]
    assert any("tool='greet'" in line and "status='ok'" in line for line in answered), told

    stopped = await plugin_stop(_env(tmp_path), StopArgs(handle=handle))

    assert stopped.status is Status.OK
    assert "stopped run" in stopped.summary
    assert "is gone" in _body(stopped)


async def test_a_step_that_expected_something_else_says_what_arrived(tmp_path: Path) -> None:
    """A failing step is a sentence, and the run stays open: what an author fixes is one step at a time.

    What arrived is printed beside what was expected, because "the tool was not called" is not actionable and
    "these are the four events that came instead" is.
    """
    _ = _plugin(tmp_path, "greets")
    wrong = [
        {"model": {"text": "I will greet nobody"}},
        {"send": "hello"},
        {"expect": {"tag": "tool-called", "fields": {"tool": "not_a_tool"}}},
    ]

    started = await plugin_test(_env(tmp_path), RunArgs(name="greets", steps=wrong))

    assert started.status is Status.FAILED
    assert "expected" in started.summary and "not_a_tool" in started.summary
    handle = int(started.summary.split("run ")[1].split(":")[0])
    assert "what arrived after that step was" in _body(started)

    carried = await plugin_debug(_env(tmp_path), DebugArgs(handle=handle, steps=[]))

    assert carried.status is Status.OK
    assert "nothing new" in carried.summary
    _ = await plugin_stop(_env(tmp_path), StopArgs(handle=handle))


async def test_a_run_that_was_stopped_is_not_one_to_carry_on_with(tmp_path: Path) -> None:
    """A handle names a process, so after the process has gone the number names nothing.

    Both calls that take a handle say the same thing about it, and the sentence points at the call that
    starts a run, because "stop it again" is not what a person wants next.
    """
    _ = _plugin(tmp_path, "greets")
    started = await plugin_test(_env(tmp_path), RunArgs(name="greets", steps=[]))
    handle = int(started.summary.split("run ")[1].split(":")[0])
    _ = await plugin_stop(_env(tmp_path), StopArgs(handle=handle))

    carried = await plugin_debug(_env(tmp_path), DebugArgs(handle=handle, steps=[]))
    again = await plugin_stop(_env(tmp_path), StopArgs(handle=handle))

    assert carried.status is Status.FAILED and "no run is open" in carried.summary
    assert again.status is Status.FAILED and "no run is open" in again.summary


async def test_a_script_that_is_not_one_is_refused_before_anything_starts(tmp_path: Path) -> None:
    """A step with a word nothing reads is a step that says less than it looks, so it is refused by name.

    Nothing is started for it: a run costs a child process, and a script that is not one is a fact this end
    can see without paying for it.
    """
    _ = _plugin(tmp_path, "greets")

    result = await plugin_test(_env(tmp_path), RunArgs(name="greets", steps=[{"sent": "hello"}]))

    assert result.status is Status.FAILED
    assert "says sent" in result.summary
    assert RUNS.numbers() == (), "nothing was started, so there is no run to stop"
