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

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
import send2trash

from plyngent.kernel import trash
from plyngent.kernel.env import Env
from plyngent.kernel.failure import Failure
from plyngent.kernel.host import Host
from plyngent.kernel.tools import fs
from plyngent.kernel.toolspec import Res

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.session import TurnCtx


@dataclass(frozen=True)
class _Ctx:
    """The only part of a turn these tools touch: its host."""

    host: Host


#: What these tests let a call reach: everything, because each test is about a tool's behaviour rather
#: than about what it declared. The gate itself is asserted once, by name.
_EVERYTHING = Res.PATHS | Res.EXEC | Res.PTY | Res.NET | Res.HUMAN | Res.HISTORY | Res.DOCUMENT | Res.CLIENT


def _env(root: Path, *, trash_enabled: bool = True) -> Env:
    """A call's world as the executor builds it, around a host rather than around a turn."""
    return Env(cast("TurnCtx", cast("object", _Ctx(Host(root, trash_enabled=trash_enabled)))), _EVERYTHING, "tool-1")


def test_a_handler_reaches_only_what_its_tool_declared(tmp_path) -> None:
    env = Env(cast("TurnCtx", cast("object", _Ctx(Host(tmp_path)))), Res.PATHS, "tool-1")

    assert env.workspace.root == tmp_path.resolve(), "a declared resource answers"
    with pytest.raises(AttributeError, match="pty"):
        _ = env.pty
    with pytest.raises(AttributeError, match="human"):
        _ = env.human


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
