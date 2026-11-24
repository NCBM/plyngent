"""What is left of the filesystem tools' own tests: the claims no story can make.

The rest of what this module asserted is a story now (:mod:`tests.stories.test_tools`) or a shape
(:mod:`tests.test_structure`). Three things stayed, because none of them has an outside:

* A handler can only reach the resources its tool declared. A story would have to call a handler and
  watch for an ``AttributeError``, which is the one thing a story must not do.
* A missing path and a fixable argument are two different failures with two different answers, and the
  answer — what to do next — is not in anything a result's text carries (§10.3).
* A trash that will not take a path is answered with the one value that goes through it, while an
  installation with no trash is answered with nothing at all: the second is not a fact about the path,
  and a card there would invite an irreversible delete to work around a broken install (E189).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest
import send2trash

from plyngent.kernel import trash
from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.env import Env
from plyngent.kernel.failure import Failure
from plyngent.kernel.host import Host
from plyngent.kernel.tools import fs, session
from plyngent.kernel.toolspec import ArgsError, Res
from plyngent.wire.ask import AskSite, Confirmation

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.access import AccessMode
    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.session import TurnCtx


@dataclass(frozen=True)
class _Session:
    """As much of a session as a path tool needs: no grants, since nothing here was allowed one.

    ``session_id`` is what a temporary workspace is made for and the only owner whose scratch a call may
    reach (E152), so it is here even though none of these tests makes one.
    """

    grants: dict[Path, AccessMode] = field(default_factory=dict)
    session_id: str = "s1"


@dataclass(frozen=True)
class _Ctx:
    """The parts of a turn these tools touch: its host, and the session whose grants it resolves with.

    ``scope`` is one that nothing ever cancels, because none of these tests is about a stop: what a
    call hands to a thread is the work's own business, and the answer is all that is read here.
    ``places`` is every address this call asked to be put under, as the segments it passed, because a
    segment is what a test can check and the joined path it would become is not.
    """

    host: Host
    session: _Session = field(default_factory=_Session)
    places: list[tuple[str, ...]] = field(default_factory=list)

    def scope(self, *segments: str) -> Scope:
        self.places.append(segments)
        return CancelToken().at(*segments)

    def site(self, *segments: str) -> AskSite:
        self.places.append(segments)
        return AskSite(session=self.session.session_id, turn=1, scope="/".join(("1", "r0", *segments)))


#: What these tests let a call reach: everything, because each test is about a tool's behaviour rather
#: than about what it declared. The gate itself is asserted once, by name.
_EVERYTHING = Res.PATHS | Res.EXEC | Res.PTY | Res.NET | Res.HUMAN | Res.HISTORY | Res.DOCUMENT | Res.CLIENT | Res.TOOLS

#: The tool set a call may look at here: nothing, because none of these tests asks.
_VIEW = ToolCatalog().view(())


def _env(root: Path, *, trash_enabled: bool = True) -> Env:
    """A call's world as the executor builds it, around a host, not around a turn."""
    host = Host(root, trash_enabled=trash_enabled)
    return Env(cast("TurnCtx", cast("object", _Ctx(host))), _EVERYTHING, ("tool-1",), tool="probe", tools=_VIEW)


async def test_a_handler_reaches_only_what_its_tool_declared(tmp_path) -> None:
    env = Env(cast("TurnCtx", cast("object", _Ctx(Host(tmp_path)))), Res.PATHS, ("tool-1",), tool="probe", tools=_VIEW)

    assert env.workspace.root == tmp_path.resolve(), "a declared resource answers"
    with pytest.raises(AttributeError, match="pty"):
        _ = env.pty
    with pytest.raises(AttributeError, match="ask"):
        _ = await env.ask(Confirmation(prompt="anything?"))


async def test_a_call_inside_a_call_is_a_path_of_segments(tmp_path) -> None:
    """A program's step is named ``tool-3/s1``, and it is addressed as two segments (E190).

    One string would be a segment holding a slash, which a cancellation address refuses: the step could
    then not be cancelled under the call that runs the program, which is the one thing naming it is for.
    """
    ctx = _Ctx(Host(tmp_path))
    env = Env(cast("TurnCtx", cast("object", ctx)), _EVERYTHING, ("tool-3", "s1"), tool="probe", tools=_VIEW)

    _ = env.scope("gate")
    _ = env.address("gate")

    assert ctx.places == [("tool-3", "s1", "gate"), ("tool-3", "s1", "gate")]


async def test_a_wait_with_nobody_to_ask_still_sleeps(tmp_path: Path) -> None:
    """A run with no terminal waits, and does not ask a no-sink that answers at once.

    The bound is declared in the schema, so a call that would wait an hour is one the model can see is
    wrong before it makes it — and the sleep itself is what "waited" means, which a clock has to say.
    """
    world = _env(tmp_path)
    started = time.monotonic()

    result = await session.wait(world, session.WaitArgs(seconds=0.05, intention="let the build finish"))

    assert result.summary == "waited 0.05s: nobody is at a terminal to say anything"
    assert time.monotonic() - started >= 0.03, "a headless wait is a sleep, not an answer that arrives at once"
    with pytest.raises(ArgsError):
        _ = session.WAIT.decode('{"seconds": 700, "intention": "the rest of the day"}')


async def test_a_missing_path_is_the_worlds_answer_and_a_fixable_parameter_is_the_models(tmp_path) -> None:
    missing = await fs.delete_path(_env(tmp_path), fs.DeleteArgs(path="absent"))

    assert missing.failure is Failure.MISSING
    assert missing.retry.value == "fix_args"


async def test_a_trash_that_refuses_names_the_one_value_that_gets_past_it(tmp_path, monkeypatch) -> None:
    """A volume with no trash directory is a fact about this path, so one argument goes through it."""
    kept = tmp_path / "kept.txt"
    _ = kept.write_text("x")
    _ = monkeypatch.setattr(fs, "under_system_temp", lambda _path: False)

    def refuse(path: str) -> None:
        raise OSError(5, "no trash directory on this volume")

    _ = monkeypatch.setattr(send2trash, "send2trash", refuse)

    result = await fs.delete_path(_env(tmp_path), fs.DeleteArgs(path="kept.txt"))

    assert result.failure is Failure.BLOCKED
    assert "no trash directory" in result.summary
    assert [hint.given for hint in result.hints] == [(("permanent", "true"),)]
    assert kept.exists(), "what the human was told was recoverable is not destroyed"


async def test_an_install_with_no_trash_is_answered_with_nothing_to_try(tmp_path, monkeypatch) -> None:
    """A missing library is a fact about this installation, so no argument the model has changes it."""
    kept = tmp_path / "kept.txt"
    _ = kept.write_text("x")
    _ = monkeypatch.setattr(fs, "under_system_temp", lambda _path: False)
    _ = monkeypatch.setattr(trash, "available", lambda: False)

    result = await fs.delete_path(_env(tmp_path), fs.DeleteArgs(path="kept.txt"))

    assert result.failure is Failure.BLOCKED
    assert "send2trash is not installed" in result.summary
    assert result.hints == (), "a broken install is no reason to offer an irreversible delete"
    assert kept.exists()
