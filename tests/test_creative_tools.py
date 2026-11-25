"""What ``plugin_check`` says about a plugin: a report, its sentences, and the path to write.

The call is driven the way the executor drives one — with an environment built from its declarations — so
what is asserted is what a model would be answered with: a summary, a body that is a line per fault, and a
path when the file it was told about does not exist yet.
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
from plyngent_creative.tools.check import CheckArgs, plugin_check
from tests.test_creative_checks import EXTRAS

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.cancel import Scope


@dataclass
class _Session:
    """As much of a session as a grant needs, and nothing else: these calls write no rows of their own."""

    grants: dict[Path, AccessMode] = field(default_factory=dict)
    session_id: str = "s1"

    async def grant(self, path: Path, need: AccessMode) -> bool:
        if path in self.grants:
            return False
        self.grants[path] = need
        return True


@dataclass
class _Ctx:
    """The parts of a turn these calls touch: the host they resolve through and the session they grant to."""

    host: Host
    session: _Session = field(default_factory=_Session)

    def scope(self, *_segments: str) -> Scope:
        return CancelToken().at("tool-1")


def _env(root: Path) -> Env:
    return Env(cast("Any", _Ctx(Host(root))), Res.PATHS, ("tool-1",), tool="probe", tools=cast("Any", None))


def _body(result: object) -> str:
    """What a model would read under the summary: the body's text parts, in the order they were laid out."""
    parts = cast("Any", result).body
    return "\n".join(part.value for part in parts if isinstance(part, Text))


@pytest.fixture(autouse=True)
def _elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin directory is a temporary one, so the suite never writes into the person's data root."""
    monkeypatch.setattr("plyngent_creative.tools.plugins.user_data_path", lambda _name: tmp_path)


def _plugin(root: Path, source: Path, name: str) -> Path:
    """One file of the counterexamples, put where the plugin directory holds it."""
    directory = root / "creative_plugins"
    directory.mkdir(parents=True, exist_ok=True)
    found = directory / f"{name}.py"
    _ = shutil.copy(source, found)
    return found


async def test_a_plugin_that_is_loadable_is_answered_with_the_count(tmp_path: Path) -> None:
    """The path comes from the group's own directory, so a call names a plugin and never a path."""
    _ = _plugin(tmp_path, EXTRAS / "examples" / "greets.py", "greets")

    result = await plugin_check(_env(tmp_path), CheckArgs(name="greets"))

    assert result.status is Status.OK
    assert result.summary == "greets is loadable: 4 checks passed"


async def test_a_plugin_with_a_fault_answers_each_sentence_on_its_own_line(tmp_path: Path) -> None:
    """An author fixes one fault at a time, so the report is a line per check and not one paragraph."""
    written = _plugin(tmp_path, EXTRAS / "broken" / "same_group.py", "mine")

    result = await plugin_check(_env(tmp_path), CheckArgs(name="mine"))

    assert result.status is Status.FAILED
    assert "1 of 4 checks failed" in result.summary
    assert str(written) in result.summary, "the answer says which file it read"
    assert _body(result) == "groups: 'fs' is a group this program already has"


async def test_a_name_nothing_has_written_yet_answers_with_the_path_to_write(tmp_path: Path) -> None:
    """The plugin being written is the ordinary case, and the answer is where the file goes.

    The directory is made by the call, which is what lets ``write_file`` take the absolute path it was just
    handed: the grant is this conversation's and it is made before anything is resolved (E184).
    """
    result = await plugin_check(_env(tmp_path), CheckArgs(name="say_hi"))

    assert result.status is Status.FAILED, "nothing was checked, because there is nothing to check"
    going = tmp_path / "creative_plugins" / "say_hi.py"
    assert result.summary == f"no plugin named 'say_hi' yet: {going} is where one goes"
    assert (tmp_path / "creative_plugins").is_dir()
