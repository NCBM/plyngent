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

from plyngent.kernel.journal import MEMORY_JOURNAL, MIGRATIONS, SqliteTrail, journal_path
from plyngent.kernel.model import Assistant, User
from plyngent.kernel.trail import StateChange, Summary, as_message


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


async def test_a_state_change_is_not_where_a_conversation_continues() -> None:
    """Bookkeeping arrives last and is still not the head: a fact about a session is not a branch of it.

    A session that disclosed a group before anybody said anything has a newest row and no conversation,
    so a client that continued from "the newest row" would project to nothing — the one failure this
    distinction exists to prevent (§10.9).
    """
    trail = SqliteTrail(MEMORY_JOURNAL)
    await trail.open()
    try:
        assert await trail.head("main") is None, "a session with nothing in it has no head"
        _ = await trail.append("main", StateChange(key="disclosure", value="{}"))
        assert await trail.head("main") is None, "and bookkeeping does not invent one"
        said = await trail.append("main", User(text="hello"))
        _ = await trail.append("main", StateChange(key="disclosure", value='{"groups": [], "toolset_epoch": 1}'))
        head = await trail.head("main")
        assert head is not None and head.id == said.id, "the person's line is where the conversation goes on"
    finally:
        await trail.close()


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


async def test_a_session_state_is_folded_last_write_wins_and_never_projects(tmp_path) -> None:
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.append("main", User(text="hello"))
        said = await trail.append("main", Assistant(text="hi"), begin_after=said.id)
        _ = await trail.append("main", StateChange(key="disclosure", value='{"groups": []}'))
        _ = await trail.append("main", StateChange(key="disclosure", value='{"groups": ["fs"]}'))

        assert await trail.state_of("main") == {"disclosure": '{"groups": ["fs"]}'}, "the last write wins"
        assert await trail.state_of("elsewhere") == {}, "a session that wrote nothing knows nothing"
    finally:
        await trail.close()

    again = SqliteTrail(path)
    await again.open()
    try:
        assert await again.derive("main", said.id) == (User(text="hello"), Assistant(text="hi")), (
            "a fact about the session is not something a model is sent"
        )
        assert await again.state_of("main") == {"disclosure": '{"groups": ["fs"]}'}, "and it outlives the process"
    finally:
        await again.close()


async def test_a_summary_is_what_a_model_reads_of_what_it_replaced(tmp_path) -> None:
    """A compaction is an ordinary entry, and three things about it are one thing.

    It is where the conversation continues (so a restarted client does not go back to the tip it
    replaced), it is what a model reads of the stretch that went away, and it rewrites nothing: the tip
    it replaced still projects to exactly what it projected to before, because compressing changes the
    future and not the past.
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.append("main", User(text="we should ship on friday"))
        reply = await trail.append("main", Assistant(text="agreed"), begin_after=said.id)
        summary = await trail.append("main", Summary(text="ship on friday"), begin_after=None)

        assert summary.turn == 0, "a compaction belongs to no turn"
        head = await trail.head("main")
        assert head is not None and head.id == summary.id, "a summary is where the conversation goes on"
        assert await trail.derive("main", summary.id) == (
            User(
                text=(
                    "Conversation summary (compacted):\n\n"
                    "ship on friday\n\n"
                    "Continue from this summary. Prefer not to re-ask for information already covered."
                )
            ),
        ), "the summary is the whole projection, in the one wording a model is told that it is one"
        assert await trail.derive("main", reply.id) == (
            User(text="we should ship on friday"),
            Assistant(text="agreed"),
        ), "and the chain it replaced still projects to what it always did"
    finally:
        await trail.close()


async def test_a_summary_can_keep_the_context_before_it() -> None:
    trail = SqliteTrail(MEMORY_JOURNAL)
    await trail.open()
    try:
        first = await trail.append("main", User(text="one"))
        _ = await trail.append("main", Assistant(text="two"), begin_after=first.id)
        kept = await trail.append("main", Summary(text="three"), begin_after=first.id)

        assert await trail.derive("main", kept.id) == (User(text="one"), as_message(Summary(text="three"))), (
            "what the summary continues is kept, and what it does not continue is off the chain"
        )
    finally:
        await trail.close()


async def test_what_a_summary_replaces_is_bookkeeping_and_the_summary_outlives_the_process(tmp_path) -> None:
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        named = await trail.append("a", Summary(text="one", replaces=("e1", "e2")))
        silent = await trail.append("b", Summary(text="one"))
        assert await trail.derive("a", named.id) == await trail.derive("b", silent.id), (
            "the projection never reads replaces, which is what makes it bookkeeping"
        )

        first = await trail.append("c", Summary(text="first"))
        second = await trail.append("c", Summary(text="second"), begin_after=first.id)
        assert await trail.derive("c", second.id) == (
            as_message(Summary(text="first")),
            as_message(Summary(text="second")),
        ), "a summary of a summary is just another entry: no rewriting and no folding rule"
    finally:
        await trail.close()

    again = SqliteTrail(path)
    await again.open()
    try:
        head = await again.head("c")
        assert head is not None and head.id == second.id
        assert await again.derive("c", second.id) == (
            as_message(Summary(text="first")),
            as_message(Summary(text="second")),
        ), "a summary is a row like any other, so a restart reads it back"
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
