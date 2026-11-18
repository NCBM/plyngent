"""The store of versions: what a call left, kept so a difference and a way back both exist.

These are mechanism assertions, not stories: a story reaches the store only through a turn, and what is
being held here — that a version is a patch on the one before it, that a removal is an absence rather
than a missing line, that the tree the versions are about is never written to — has no second way in.
The store is a real repository because a fake would have to answer every question git answers, and the
question these tests ask is whether this build asks git the right one.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.history import BRANCH, History, history_path
from tests import repositories

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(repositories.MISSING, reason="the store is built on git, and this machine has none")

#: The scope a version is recorded on. A turn's, in a real run; here, one nothing ever stops, because what
#: these tests ask is what git was asked, not what a stop does to a thread.
_SCOPE = CancelToken().at("turn", "call")


def _store(root: Path) -> History:
    """The store a workspace at ``root`` would have: beside the journal, which is outside the tree.

    ``root.parent`` stands in for the data directory: the journal of a workspace never lives inside it
    (opening the program leaves the tree as it found it), and neither does this.
    """
    return History(root, history_path(root.parent / "data" / "journal.db"))


async def test_a_version_holds_what_the_call_left_behind(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    (root / "b.txt").write_text("two\n", encoding="utf-8")
    store = _store(root)

    revision = await store.record([root / "a.txt", root / "b.txt"], says="write_file", scope=_SCOPE)

    assert revision, "a call that changed something is a version"
    assert await store.head(scope=_SCOPE) == revision, "and it is where the next one continues from"
    assert await store.changed(revision) == ("a.txt", "b.txt")
    assert await store.blob(revision, "a.txt") == b"one\n"


async def test_the_version_before_is_where_the_previous_bytes_are(tmp_path) -> None:
    """The whole reason a version is a patch on the one before it: nothing has to be kept by hand."""
    root = tmp_path / "ws"
    root.mkdir()
    leaf = root / "a.txt"
    leaf.write_text("one\n", encoding="utf-8")
    store = _store(root)

    first = await store.record([leaf], says="write_file", scope=_SCOPE)
    leaf.write_text("two\n", encoding="utf-8")
    second = await store.record([leaf], says="edit_replace", scope=_SCOPE)

    assert await store.parent(second) == first
    assert await store.blob(first, "a.txt") == b"one\n"
    assert await store.blob(second, "a.txt") == b"two\n"
    difference = await store.diff(first, second)
    assert "-one" in difference and "+two" in difference, "the difference is read out of the store"


async def test_a_removed_path_is_an_absence_and_not_a_missing_line(tmp_path) -> None:
    """A file nobody can read back is not the same as a file that was taken away, and a version says which."""
    root = tmp_path / "ws"
    root.mkdir()
    leaf = root / "gone.txt"
    leaf.write_text("here\n", encoding="utf-8")
    store = _store(root)

    first = await store.record([leaf], says="write_file", scope=_SCOPE)
    leaf.unlink()
    second = await store.record([leaf], says="delete_path", scope=_SCOPE)

    assert second != first
    assert await store.blob(second, "gone.txt") is None, "the newer version does not hold it"
    assert await store.blob(first, "gone.txt") == b"here\n", "and the older one still does"
    assert await store.changed(second) == ("gone.txt",), "a removal is a change like any other"
    assert "-here" in await store.diff(first, second)


async def test_nothing_to_keep_writes_no_version(tmp_path) -> None:
    """A read is not a version: a store that grew a commit per call would be a log of nothing happening."""
    root = tmp_path / "ws"
    root.mkdir()
    store = _store(root)

    assert await store.record([], scope=_SCOPE) == ""
    assert await store.head(scope=_SCOPE) == ""
    assert not store.path.exists(), "and nothing is made until there is something to keep"


async def test_only_what_is_inside_the_tree_is_this_stores_to_keep(tmp_path) -> None:
    """The other side of ``Effects.changed``: a path outside the workspace is not a change to record."""
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not ours\n", encoding="utf-8")
    store = _store(root)

    assert await store.record([outside], says="skill_edit", scope=_SCOPE) == ""
    assert not store.path.exists()


async def test_a_version_never_writes_into_the_tree_it_is_about(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    store = _store(root)

    revision = await store.record([root / "a.txt"], says="write_file", scope=_SCOPE)

    assert revision
    assert sorted(item.name for item in root.iterdir()) == ["a.txt"], "the tree holds what it held"
    assert (store.path / "index").exists(), "the index is the store's own file"


async def test_the_branch_a_version_lands_on_is_the_one_this_build_names(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    store = _store(root)

    revision = await store.record([root / "a.txt"], says="write_file", scope=_SCOPE)

    assert BRANCH == "workspace"
    assert (store.path / "refs" / "heads" / BRANCH).read_text(encoding="utf-8").strip() == revision


async def test_a_sweep_finds_what_a_version_does_not_hold(tmp_path) -> None:
    """The one caller that cannot say what it changed: the tree is compared with the version in force."""
    root = tmp_path / "ws"
    root.mkdir()
    leaf = root / "a.txt"
    leaf.write_text("one\n", encoding="utf-8")
    store = _store(root)

    first = await store.record([leaf], says="write_file", scope=_SCOPE)
    assert await store.differing(first, [leaf], scope=_SCOPE) == (), "an unchanged file is not a change"

    leaf.write_text("two\n", encoding="utf-8")
    fresh = root / "b.txt"
    fresh.write_text("new\n", encoding="utf-8")

    changed = await store.differing(first, [leaf, fresh], scope=_SCOPE)

    assert set(changed) == {leaf, fresh}, "one file written back to other bytes, and one that never existed"


async def test_a_sweep_finds_a_path_that_is_gone(tmp_path) -> None:
    """A version holds what a path was; a sweep has to notice that the path is not there any more."""
    root = tmp_path / "ws"
    root.mkdir()
    leaf = root / "a.txt"
    leaf.write_text("one\n", encoding="utf-8")
    store = _store(root)

    first = await store.record([leaf], says="write_file", scope=_SCOPE)
    leaf.unlink()

    assert await store.differing(first, [], scope=_SCOPE) == (leaf,), (
        "a removal is a change even though nothing was walked"
    )


async def test_two_conversations_committing_at_once_are_still_one_line(tmp_path, monkeypatch) -> None:
    """The head, the private index and the branch are one tree's: a store takes one commit at a time.

    Both commits are made to reach the middle of a commit together — a barrier of two, with a short
    timeout that *is* the answer when the other one never arrives. Two roots landing on one branch is
    what that would do without the lock: one commit's ref update finds the branch already moved.
    """
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (root / "b.txt").write_text("b\n", encoding="utf-8")
    store = _store(root)

    together = threading.Barrier(2, timeout=0.05)
    wrote_tree = store._treed

    def meeting(parent: str, changed: list[Path]) -> str:
        with contextlib.suppress(threading.BrokenBarrierError):
            _ = together.wait()
        return wrote_tree(parent, changed)

    monkeypatch.setattr(store, "_treed", meeting)

    first, second = await asyncio.gather(
        store.record([root / "a.txt"], says="first", scope=_SCOPE),
        store.record([root / "b.txt"], says="second", scope=_SCOPE),
    )

    head = await store.head(scope=_SCOPE)
    behind = await store.parent(head)

    assert {head, behind} == {first, second}, "two versions, and the second continues from the first"
    assert await store.blob(head, "a.txt") == b"a\n"
    assert await store.blob(head, "b.txt") == b"b\n", "and the later one holds what the earlier one kept"
