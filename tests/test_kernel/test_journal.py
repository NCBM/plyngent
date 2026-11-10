"""The journal on disk: where a workspace's file lives, and what survives the process that wrote it.

Two of these are mechanism assertions no story can make, and §0.3 of the rewrite records them as the
exceptions they are: a migration step is only visible by interrupting it, and "a second open changes
nothing" is only visible by opening twice. They are shape assertions, kept beside the shape they hold —
which is why they do not go away with the rest of this directory.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

from platformdirs import user_data_path

from plyngent.kernel.journal import MIGRATIONS, SqliteTrail, journal_path
from plyngent.kernel.model import Assistant, User


async def test_a_workspace_names_its_journal_and_creates_nothing(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    path = journal_path(root)

    assert user_data_path("plyngent") in path.parents, "the file lives in the person's data directory"
    assert path.name == "journal.db"
    assert path.parent.name.startswith("project-"), "the directory says whose history it is, for a person"
    assert not path.exists(), "naming a journal does not make one: --help may leave the disk alone"
    assert journal_path(root / ".") == path, "two spellings of one workspace are one journal"
    assert journal_path(tmp_path / "other") != path, "two workspaces are two histories"


async def test_what_one_backend_writes_another_reads(tmp_path) -> None:
    path = tmp_path / "journal.db"
    first = SqliteTrail(path)
    await first.open()
    try:
        said = await first.append("main", User(text="hello"))
        _ = await first.append("main", Assistant(text="hi"), begin_after=said.id)
        assert await first.next_turn("main") == 1
    finally:
        await first.close()

    second = SqliteTrail(path)
    await second.open()
    try:
        head = await second.head("main")
        assert head is not None, "the conversation is still there under a process that never ran it"
        assert head.payload == Assistant(text="hi")
        assert await second.derive("main", head.id) == (User(text="hello"), Assistant(text="hi")), (
            "the chain is read back in the order it was said, beginning at its root"
        )
        assert await second.next_turn("main") == 2, "the turn counter resumes where the last process left it"
    finally:
        await second.close()


async def test_opening_a_journal_again_changes_nothing(tmp_path) -> None:
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    _ = await trail.append("main", User(text="one"))
    await trail.open()
    await trail.close()
    await trail.close()

    again = SqliteTrail(path)
    await again.open()
    try:
        head = await again.head("main")
        assert head is not None and await again.derive("main", head.id) == (User(text="one"),)
        assert _version(path) == len(MIGRATIONS), "the file is at this build's version, once and not twice"
        assert _count(path) == 1
    finally:
        await again.close()


async def test_a_step_that_was_interrupted_runs_again(tmp_path) -> None:
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        _ = await trail.append("main", User(text="one"))
    finally:
        await trail.close()

    # A process that died between creating the schema and recording how far it got: the file looks
    # unmigrated, so the step runs again over objects that are already there.
    with closing(sqlite3.connect(path)) as db:
        _ = db.execute("PRAGMA user_version = 0")

    again = SqliteTrail(path)
    await again.open()
    try:
        assert _version(path) == len(MIGRATIONS)
        assert _count(path) == 1, "the step ran again without losing the row that was already there"
    finally:
        await again.close()


async def test_a_projection_is_cached_and_the_cache_can_be_dropped(tmp_path) -> None:
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.append("main", User(text="hello"))
        _ = await trail.append("main", Assistant(text="hi"), begin_after=said.id)
        head = await trail.head("main")
        assert head is not None
        first = await trail.derive("main", head.id)
        assert await trail.derive("main", head.id) == first, "a second read answers the first one's answer"
    finally:
        await trail.close()

    assert _projections(path) == 1, "and the answer was kept, rather than computed twice"

    with closing(sqlite3.connect(path)) as db:
        _ = db.execute("DELETE FROM projection")

    again = SqliteTrail(path)
    await again.open()
    try:
        head = await again.head("main")
        assert head is not None
        assert await again.derive("main", head.id) == (User(text="hello"), Assistant(text="hi")), (
            "a checkpoint is a cache: dropping every row of it changes the answer not at all"
        )
    finally:
        await again.close()


def _version(path) -> int:
    """How many migration steps this file has been through, read the way another process would."""
    with closing(sqlite3.connect(path)) as db:
        row = db.execute("PRAGMA user_version").fetchone()
    assert row is not None
    return int(row[0])


def _count(path) -> int:
    """How many entries this file holds."""
    with closing(sqlite3.connect(path)) as db:
        row = db.execute("SELECT COUNT(*) FROM entry").fetchone()
    assert row is not None
    return int(row[0])


def _projections(path) -> int:
    """How many projections this file has cached."""
    with closing(sqlite3.connect(path)) as db:
        row = db.execute("SELECT COUNT(*) FROM projection").fetchone()
    assert row is not None
    return int(row[0])
