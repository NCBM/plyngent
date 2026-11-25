"""The plugin directory: where a name lands, and the one grant that makes it reachable.

Two facts are tested here and they are two: a name resolves to a file under one directory and nowhere else,
and the group that resolves it also makes the conversation able to write there. The first is the door
between a string a model wrote and a path this process opens; the second is why no sixth tool has to write
the file (E184).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest

from plyngent.kernel.access import AccessMode, mode_covers
from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.env import Env
from plyngent.kernel.host import Host
from plyngent.kernel.toolspec import Res
from plyngent_creative.tools.plugins import DIRECTORY, PluginNameError, directory, path_of, prepared

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.cancel import Scope


@dataclass
class _Session:
    """As much of a session as a grant needs: the table it writes into, and its own id."""

    granted: list[tuple[Path, AccessMode]] = field(default_factory=list)
    grants: dict[Path, AccessMode] = field(default_factory=dict)
    session_id: str = "s1"

    async def grant(self, path: Path, need: AccessMode) -> bool:
        """The real table's rule: a grant this conversation already holds is answered without a row."""
        held = self.grants.get(path)
        if held is not None and mode_covers(held, need):
            return False
        self.granted.append((path, need))
        self.grants[path] = need
        return True


@dataclass
class _Ctx:
    """The parts of a turn a grant touches: the host it resolves through, and the session it writes to."""

    host: Host
    session: _Session = field(default_factory=_Session)

    def scope(self, *_segments: str) -> Scope:
        return CancelToken().at("tool-1")


def _env(root: Path) -> tuple[Env, _Session]:
    ctx = _Ctx(Host(root))
    return Env(cast("Any", ctx), Res.PATHS, ("tool-1",), tool="probe", tools=cast("Any", None)), ctx.session


def test_a_plugin_is_one_file_under_the_program_s_data_root() -> None:
    """The directory is the data root's, so a plugin outlives one tree and no workspace can be it."""
    assert directory().parts[-1] == DIRECTORY
    assert directory().parent.name == "plyngent"


def test_a_name_resolves_inside_the_directory_and_nowhere_else(tmp_path: Path) -> None:
    """A name is a module name, so the path built from it cannot leave the directory it belongs to.

    The three refused shapes are the ones a shell would resolve somewhere else: a parent, a separator, and
    a name that is not a module's at all.
    """
    assert path_of("say_hi", where=tmp_path) == tmp_path / "say_hi.py"
    assert path_of("say2", where=tmp_path) == tmp_path / "say2.py"

    for refused in ("../say_hi", "a/b", "say-hi", "Say_hi", ""):
        with pytest.raises(PluginNameError):
            _ = path_of(refused, where=tmp_path)


async def test_the_group_makes_the_directory_and_gives_the_conversation_a_write(tmp_path: Path) -> None:
    """Both halves happen on the call that needs them, and both are idempotent, so a second call is free.

    The grant is what makes ``write_file`` with an absolute path work at all: the directory is outside the
    workspace, so without it the path a model was told to write is one its tools refuse.
    """
    env, session = _env(tmp_path)

    found = await prepared(env, where=tmp_path / DIRECTORY)
    again = await prepared(env, where=tmp_path / DIRECTORY)

    assert found == again == tmp_path / DIRECTORY
    assert found.is_dir()
    assert session.granted == [(found, AccessMode.WRITE)]
    assert session.granted.count((found, AccessMode.WRITE)) == 1, "the second call asked once and wrote nothing"
