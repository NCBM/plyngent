"""The journal on disk: where a workspace's file lives, and what survives the process that wrote it.

What a story can see of this — a conversation read back under a new process, a summary where the
conversation goes on, what a model is told of the stretch a compaction replaced — is asserted there and
not here. What is left is what no story can make: the name of the file, which is a pure function; a
second open; a migration step interrupted between the schema and the version that records it; a
projection, which is only visible as rows beside the entries; a head that bookkeeping does not move; a
summary's `replaces`, which the projection never reads; and an effect's row, whose whole content is the
paths and the version it names.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

from platformdirs import user_data_path

from plyngent.kernel.journal import MEMORY_JOURNAL, MIGRATIONS, SqliteTrail, journal_path
from plyngent.kernel.model import Assistant, Call, Result, User
from plyngent.kernel.trail import Effect, StateChange, Summary, as_message


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


async def test_what_a_call_changed_is_written_beside_its_result(tmp_path) -> None:
    """An effect is a row with no words in it: what changed, and where to read the change back from.

    Nothing reads a body out of this one — the paths and the version are the whole record — so a
    conversation that carries one projects to exactly what it projected to before it arrived. That is
    the property that lets it sit *in* the chain, directly below the result it belongs to, without
    becoming something a later turn grows from differently (contrast ``attempt``, which is a root).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.append("c", User(text="write a file"))
        asked = await trail.append(
            "c",
            Assistant(calls=(Call(id="a1", tool="write_file", arguments="{}"),)),
            begin_after=said.id,
        )
        done = await trail.append(
            "c",
            Result(call="a1", tool="write_file", text="wrote a.txt"),
            begin_after=asked.id,
        )
        effect = await trail.append(
            "c",
            Effect(
                turn=1,
                call="a1",
                tool="write_file",
                revision="0" * 40,
                changed=("a.txt",),
                approvals=("write_file",),
            ),
            begin_after=done.id,
            turn=1,
        )
        assert isinstance(effect.payload, Effect), "the tag a row keeps is enough to read it back"
        assert effect.payload.changed == ("a.txt",)
        assert await trail.derive("c", effect.id) == await trail.derive("c", done.id), (
            "a chain ending on an effect projects to the same conversation as the result above it"
        )
    finally:
        await trail.close()

    again = SqliteTrail(path)
    await again.open()
    try:
        head = await again.head("c")
        assert head is not None and head.id == effect.id, (
            "an effect is not bookkeeping that moves the head: a turn goes on from where the call did"
        )
        read = await again.entry("c", effect.id)
        assert read is not None and read.payload.revision == "0" * 40, "a stored version survives the process"
    finally:
        await again.close()


async def test_the_log_can_be_read_back_whole_and_the_sessions_it_holds_named(tmp_path) -> None:
    """The two reads a forest needs: every row of one conversation, and which conversations there are.

    A projection reads one chain and only what a model understands, so neither question is answered by
    ``derive``: what a person asks is what happened here in what order, bookkeeping included, and what
    else is in this journal at all. Both are answers a *file* has rather than a process, which is why
    they are asserted here — a conversation nobody has opened since is exactly the one somebody is
    looking for.
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.append("main", User(text="hello"))
        _ = await trail.append("main", StateChange(key="disclosure", value="{}"))
        asked = await trail.append("main", Assistant(text="hi"), begin_after=said.id)
        _ = await trail.append("side", User(text="another conversation"))

        read = await trail.entries("main")
        assert [entry.id for entry in read] == [said.id, read[1].id, asked.id], "arrival order, as written"
        assert [entry.begin_after for entry in read] == [None, None, said.id], "and the parents, which are the forest"
        assert isinstance(read[1].payload, StateChange), "bookkeeping is a row here: this is what happened"
        assert await trail.entries("nowhere") == (), "a conversation with nothing in it has no rows"
        assert await trail.sessions() == ("main", "side"), "first arrival, not name order"
    finally:
        await trail.close()

    again = SqliteTrail(path)
    await again.open()
    try:
        assert len(await again.entries("main")) == 3, "the log is the file's, so a restart reads it"
        assert await again.sessions() == ("main", "side")
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
