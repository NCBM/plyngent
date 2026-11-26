"""The journal on disk: where a workspace's file lives, and what survives the process that wrote it.

What a story can see of this — a conversation read back under a new process, a summary where the
conversation goes on, what a model is told of the stretch a compaction replaced — is asserted there and
not here. What is left is what no story can make: the name of the file, which is a pure function; a
second open; a migration step interrupted between the schema and the version that records it; a
projection, which is only visible as rows beside the entries; a head that bookkeeping does not move; a
summary's `replaces`, which the projection never reads; an effect's row, whose whole content is the
paths and the version it names; a notice, which is a message only while it has words; and a snapshot,
which is a table of its own beside the log and nothing a conversation continues from (E177).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

from plyngent.kernel.journal import (
    CHECKPOINTS,
    KEEP_CHECKPOINTS,
    MEMORY_JOURNAL,
    MIGRATIONS,
    PROJECTOR,
    SqliteTrail,
    journal_path,
)
from plyngent.kernel.model import Assistant, Call, Note, Result, User
from plyngent.kernel.trail import (
    BY_BUDGET,
    BY_PERSON,
    NOTICE_MARKER,
    Effect,
    Notice,
    Shape,
    Snapshot,
    StateChange,
    Summary,
    Withheld,
    as_message,
    preview,
)


async def test_a_workspace_names_its_journal_and_creates_nothing(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    path = journal_path(root)

    assert path.parent.parent.name == "workspaces", "the file lives under this program's own data directory"
    assert path.name == "journal.db"
    assert path.parent.name.startswith("project-"), "the directory says whose history it is, for a person"
    assert not path.exists(), "naming a journal does not make one: --help may leave the disk alone"
    assert journal_path(root / ".") == path, "two spellings of one workspace are one journal"
    assert journal_path(tmp_path / "other") != path, "two workspaces are two histories"


async def test_a_tree_that_moved_is_a_new_workspace(tmp_path) -> None:
    """Moving a directory leaves its history behind, because the name of a journal is the path it has.

    Following a move would need something inside the tree saying which journal is its own — a second copy
    would carry the same id, and a model can read whatever is in there — or the inode for a name, and an inode
    is recycled. So a moved tree is a tree nobody has written in yet (E181).
    """
    root = tmp_path / "project"
    root.mkdir()
    path = journal_path(root)

    root.rename(tmp_path / "moved")

    assert journal_path(tmp_path / "moved") != path, "the same files under another name are another history"


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

    assert _projections(path) == 1, "and the answer was kept, and not computed twice"

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


async def test_a_notice_is_a_row_a_model_reads_and_an_empty_one_is_not(tmp_path) -> None:
    """E142: a reminder is on the chain, so the bytes a model is given are the bytes the log keeps.

    Two rows and one projection rule. A notice with words is the harness speaking, at the place it was
    written — so a later turn continues from it and reads it above its own line. A notice with no words is
    a reminder that stopped being true: it stays in the log for whoever reads it afterwards, and a
    projection makes no message of it, because nothing was said to anybody.
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.append("a", User(text="one"))
        told = await trail.append(
            "a",
            Notice(text=f"{NOTICE_MARKER}\n2 things are still on your list", summary="2 things left"),
            begin_after=said.id,
        )
        gone = await trail.append("a", Notice(summary="nothing is left on the list"), begin_after=told.id)
        spoken = Note(text=f"{NOTICE_MARKER}\n2 things are still on your list")

        assert await trail.derive("a", told.id) == (User(text="one"), spoken), "the marker is in the text itself"
        assert await trail.derive("a", gone.id) == (User(text="one"), spoken), "an empty notice adds no message"
        assert preview(gone.payload) == "nothing is left on the list", "and a person reads its summary"
    finally:
        await trail.close()

    again = SqliteTrail(path)
    await again.open()
    try:
        head = await again.head("a")
        assert head is not None and head.id == gone.id, "a notice is a place the next turn continues from"
        assert await again.derive("a", gone.id) == (User(text="one"), spoken), "and it reads back after a restart"
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
    else is in this journal at all. Both are answers a *file* has, not a process, which is why
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


async def test_a_snapshot_is_kept_beside_the_log_and_answers_both_folds() -> None:
    """A snapshot is a table of its own: nothing continues from it, and it says where a fold stopped.

    It is the only reason it exists — reopening a conversation that has grown to hundreds of rows does not
    have to fold them again — so what it holds is exactly what a fold holds, and nothing that is not
    derivable: settings, the forest, and the row it reached (E177).
    """
    trail = SqliteTrail(MEMORY_JOURNAL)
    await trail.open()
    try:
        said = await trail.append("main", User(text="hello"))
        asked = await trail.append("main", Assistant(text="hi"), begin_after=said.id)
        assert await trail.head("main") is not None and await trail.latest("main") is None, (
            "a conversation that was never folded has no snapshot, and is not the less a conversation"
        )

        snapshot = Snapshot(
            seq=asked.seq,
            projector=PROJECTOR,
            settings={"disclosure": '{"groups": ["fs"], "toolset_epoch": 1}'},
            heads=(asked.id,),
            roots=(said.id,),
        )
        await trail.put("main", snapshot)

        kept = await trail.latest("main")
        assert kept is not None
        assert kept.seq == asked.seq, "the row the fold reached"
        assert kept.settings == snapshot.settings
        assert kept.shape == Shape(heads=(asked.id,), roots=(said.id,)), "the forest, as the fold answers it"
        assert await trail.latest("other") is None, "and one conversation's snapshot is not another's"
        assert await trail.head("main") is not None, "nothing about the log moved: a snapshot is not an entry"
    finally:
        await trail.close()


async def test_only_the_newest_snapshots_of_a_conversation_are_kept(tmp_path) -> None:
    """Three, and the newest wins: a fold that reached a row is one fold, and the rows behind it are old news."""
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        for seq in range(1, 6):
            await trail.put("main", Snapshot(seq=seq, projector=PROJECTOR))
        await trail.put("main", Snapshot(seq=5, projector=PROJECTOR, heads=("e5",)))

        latest = await trail.latest("main")
        assert latest is not None and latest.heads == ("e5",), "a fold of the same row replaces the one before it"
        assert _snapshots(path) == KEEP_CHECKPOINTS, "and the oldest ones go, because nothing reads them"
    finally:
        await trail.close()


async def test_a_snapshot_this_build_cannot_vouch_for_is_not_read(tmp_path) -> None:
    """Two answers a later build must not trust, and one payload nobody could: all three are "no snapshot".

    A snapshot is an accelerator, so the only honest answer to one that was written under other rules is
    the fold from the root — which is slower and the same (E177).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        await trail.put("main", Snapshot(seq=1, projector=PROJECTOR + 1))
        assert await trail.latest("main") is None, "a fold made by another projector is not this build's fold"
        await trail.put("main", Snapshot(seq=2, projector=PROJECTOR))
        assert await trail.latest("main") is not None, "the one this build wrote is readable"
    finally:
        await trail.close()

    with closing(sqlite3.connect(path)) as db:
        _ = db.execute("DELETE FROM checkpoint WHERE seq = 1")
        _ = db.execute("UPDATE checkpoint SET version = ?", (CHECKPOINTS + 1,))
        db.commit()
    assert await _readable(path) is None, "a row written by a build with another shape is skipped"

    with closing(sqlite3.connect(path)) as db:
        _ = db.execute("UPDATE checkpoint SET version = ?, payload = ?", (CHECKPOINTS, "not a snapshot"))
        db.commit()
    assert await _readable(path) is None, "and so is one whose payload decodes to nothing"


#: The settings a conversation wrote before the snapshot, and after it: a fold of the whole log is the
#: first one overwritten by the second.
ROUTE = '{"name": "one", "epoch": 1}'
NAME = "kept"


async def _part_way(trail: SqliteTrail) -> Snapshot:
    """Three rows, and the snapshot a fold of them would keep — hand-written, as a fold would."""
    said = await trail.append("main", User(text="hello"))
    route = await trail.append("main", StateChange(key="route", value=ROUTE))
    asked = await trail.append("main", Assistant(text="hi"), begin_after=said.id)
    return Snapshot(
        seq=asked.seq,
        projector=PROJECTOR,
        settings={"route": ROUTE},
        heads=(route.id, asked.id),
        roots=(said.id, route.id),
    )


async def _the_rest(trail: SqliteTrail) -> None:
    """Two more settings after the fold, so the snapshot is behind the log it belongs to."""
    _ = await trail.append("main", StateChange(key="name", value=NAME))
    _ = await trail.append("main", User(text="and another thing"))


async def _folded(trail: SqliteTrail) -> tuple[dict[str, str], Shape]:
    """The two folds of one conversation, as one value to compare."""
    return dict(await trail.state_of("main")), await trail.shape("main")


async def test_a_fold_starts_at_a_snapshot_and_takes_the_rows_after_it() -> None:
    """The invariant a snapshot rests on: folding from it answers what folding from the root answers.

    Every row is a setting or a head, so a snapshot that lost one — or a fold that started at the wrong
    place — would answer with a smaller conversation, and the comparison would show it (E177).
    """
    trail = SqliteTrail(MEMORY_JOURNAL)
    await trail.open()
    try:
        snapshot = await _part_way(trail)
        await _the_rest(trail)
        from_root = await _folded(trail)

        await trail.put("main", snapshot)
        assert await _folded(trail) == from_root, "the snapshot is behind the log, and the rows after it are taken"

        settings, shape = from_root
        assert settings == {"route": ROUTE, "name": NAME}
        assert shape == Shape(heads=("e2", "e3", "e4", "e5"), roots=("e1", "e2", "e4", "e5")), (
            "a continuing row stops its parent being a head, and a root is a root"
        )
    finally:
        await trail.close()


async def test_a_fold_whose_snapshot_was_deleted_answers_the_same(tmp_path) -> None:
    """A snapshot is derived, so deleting it costs the walk and not the answer — the whole of "disposable".

    This is the reason the table can be dropped by any build that distrusts it: nothing outside is a
    cache of the fold it names (E177).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        snapshot = await _part_way(trail)
        await _the_rest(trail)
        await trail.put("main", snapshot)
        kept = await _folded(trail)
    finally:
        await trail.close()

    with closing(sqlite3.connect(path)) as db:
        _ = db.execute("DELETE FROM checkpoint")
        db.commit()

    assert await _folded_of(path) == kept, "the fold from the root, which is the same fold"


async def test_a_snapshot_of_the_newest_row_needs_no_rows_after_it(tmp_path) -> None:
    """A snapshot taken at the end is the whole answer, and the fold adds nothing."""
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        _ = await _part_way(trail)
        await _the_rest(trail)
        whole = await _folded(trail)
        await trail.put(
            "main",
            Snapshot(
                seq=5,
                projector=PROJECTOR,
                settings=whole[0],
                heads=whole[1].heads,
                roots=whole[1].roots,
            ),
        )
        assert await _folded(trail) == whole, "there is nothing after the fold"
    finally:
        await trail.close()


async def test_a_conversation_gathers_a_fold_every_few_rows(tmp_path) -> None:
    """The cadence is what keeps a long conversation cheap to open, and it is a number the program sets.

    Three rows is what this journal was told, so the third append writes the fold of all three, and the
    snapshot answers the same two folds a walk from the root would (E177).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path, checkpoint_rows=3)
    await trail.open()
    try:
        said = await trail.append("main", User(text="one"))
        _ = await trail.append("main", StateChange(key="name", value=NAME))
        assert await trail.latest("main") is None, "two rows are not a cadence"

        asked = await trail.append("main", Assistant(text="two"), begin_after=said.id)
        written = await trail.latest("main")
        assert written is not None and written.seq == asked.seq, "the third row is where the fold is taken"
        assert dict(written.settings) == {"name": NAME}, "the setting that was written before it is in the fold"
        assert written.shape == Shape(heads=("e2", asked.id), roots=(said.id, "e2")), (
            "and the forest a walk from the root would have answered"
        )
    finally:
        await trail.close()


async def test_a_prune_writes_a_fold_at_once(tmp_path) -> None:
    """A prune is the other boundary: what a model is shown just changed, and the next open should not
    walk back over what it no longer shows (E177).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path, checkpoint_rows=1000)
    await trail.open()
    try:
        said = await trail.append("main", User(text="one"))
        _ = await trail.append("main", Withheld(targets=(said.id,), reason="a person said so", by=BY_PERSON))

        written = await trail.latest("main")
        assert written is not None and written.seq == 2, "the prune's own row is the row the fold reached"
    finally:
        await trail.close()


async def test_closing_the_journal_keeps_what_this_process_wrote(tmp_path) -> None:
    """A conversation nobody finished still gets its fold, because closing is the last moment it is open.

    What it saves is the walk a later process would otherwise do over rows this one already had in hand.
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path, checkpoint_rows=1000)
    await trail.open()
    said = await trail.append("main", User(text="one"))
    _ = await trail.append("main", StateChange(key="route", value=ROUTE))
    await trail.close()

    written = await _readable(path)
    assert written is not None and written.seq == 2, "the fold of the two rows this process wrote"
    assert written.settings == {"route": ROUTE}
    assert written.shape == Shape(heads=(said.id, "e2"), roots=(said.id, "e2"))


async def test_a_conversation_says_when_it_was_last_written_to(tmp_path) -> None:
    """The one time this tree keeps: one row per conversation, for a reader that runs with no kernel.

    Every write moves it, and a conversation nothing has written to is the one nobody is using — which is
    the question an offline program asks before it lets one go (E186).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        assert await trail.used("main") is None, "nothing has been written about it yet"
        first = await trail.append("main", User(text="one"))
        written = await trail.used("main")
        assert written is not None, "a row was written, so the conversation is dated"
        _ = await trail.append("main", Assistant(text="two"), begin_after=first.id)
        again = await trail.used("main")
        assert again is not None and again.at >= written.at, "and the newest write is the newest date"
    finally:
        await trail.close()


async def test_a_prune_adds_the_text_it_took_out_to_the_weight(tmp_path) -> None:
    """``aside`` is what pruning moved out of the window, counted from the rows the prune names.

    The pair ``live + aside`` is what a conversation has carried, and pruning only moves text from one to
    the other: a conversation that was pruned does not look small (E186).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.append("main", User(text="x" * 400))
        answer = await trail.append("main", Assistant(text="y" * 400), begin_after=said.id)
        before = await trail.used("main")
        assert before is not None and before.aside == 0

        _ = await trail.append(
            "main", Withheld(targets=(said.id, answer.id), reason="budget", by=BY_BUDGET), begin_after=answer.id
        )
        pruned = await trail.used("main")
        assert pruned is not None and pruned.aside > 0, "eight hundred characters is a couple of hundred tokens"

        _ = await trail.append("main", User(text="three"))
        after = await trail.used("main")
        assert after is not None and after.aside == pruned.aside, "and a later write does not take it back"
    finally:
        await trail.close()


async def test_a_charge_says_how_much_a_conversation_is_carrying(tmp_path) -> None:
    """``live`` is the provider's own count and ``room`` the line pruning starts at; both are replaced.

    They are kept rather than worked out later, because they are what one endpoint said about one payload
    and no reading of the log can reach them again (E186).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        await trail.weighed("main", live=1200, room=8000)
        first = await trail.used("main")
        assert first is not None and (first.live, first.room) == (1200, 8000)

        await trail.weighed("main", live=3400, room=8000)
        second = await trail.used("main")
        assert second is not None and second.live == 3400, "the newest request is the one that describes it"
    finally:
        await trail.close()


async def test_opening_a_conversation_counts_as_using_it(tmp_path) -> None:
    """A conversation nothing was said in is still one somebody has, and the offline reader asks about that.

    No entry arrives, so nothing else in the log would date it (E186).
    """
    path = tmp_path / "journal.db"
    trail = SqliteTrail(path)
    await trail.open()
    try:
        await trail.admitted("empty")
        said = await trail.used("empty")

        assert said is not None, "an open conversation is a used one"
        assert (said.live, said.aside, said.room) == (0, 0, 0), "and it has carried nothing"
    finally:
        await trail.close()


async def test_a_file_from_before_this_dates_what_is_already_in_it(tmp_path) -> None:
    """The step that adds the table dates every conversation already in the log to the instant of the upgrade.

    Nothing old can be collected until a week has passed, which is the conservative half of a delete that
    cannot be undone (E186).
    """
    path = tmp_path / "journal.db"
    with closing(sqlite3.connect(path)) as db:
        _ = db.executescript(MIGRATIONS[0])
        _ = db.executescript(MIGRATIONS[1])
        _ = db.execute("PRAGMA user_version = 2")
        _ = db.execute(
            "INSERT INTO entry (session, seq, turn, parent, tag, payload) VALUES (?, ?, ?, ?, ?, ?)",
            ("old", 1, 1, None, "user", '{"text":"said long ago"}'),
        )
        db.commit()

    trail = SqliteTrail(path)
    await trail.open()
    try:
        said = await trail.used("old")

        assert said is not None, "the step dates what was already there"
        assert (said.live, said.aside, said.room) == (0, 0, 0), "and knows nothing about what it carried"
    finally:
        await trail.close()


async def test_a_space_is_the_last_word_about_each_tree_it_gathered(tmp_path) -> None:
    """A member added, added again and taken out is one row each, and the file says what happened.

    No story reaches this: the rows are what a space *is*, and everything a story can see — a path under
    a member resolving, a greeting naming one tree — is a reader of this answer (E193).
    """
    trail = SqliteTrail(tmp_path / "journal.db")
    await trail.open()
    try:
        assert await trail.members() == (), "a tree nothing gathered reaches itself and nothing else"

        await trail.add_member("/one", "one")
        await trail.add_member("/two", "two")
        assert [one.path for one in await trail.members()] == ["/one", "/two"], "in the order they were added"
        assert [one.label for one in await trail.members()] == ["one", "two"]

        await trail.add_member("/one", "first")
        assert [one.label for one in await trail.members()] == ["first", "two"], "the last word about a tree wins"

        assert await trail.drop_member("/one") is True
        assert [one.path for one in await trail.members()] == ["/two"], "and a tree taken out is out"
        assert await trail.drop_member("/one") is False, "taking out what is not in it says so"
    finally:
        await trail.close()


async def test_a_space_keeps_what_it_used_to_reach(tmp_path) -> None:
    """Withdrawing a tree is a row of its own, so a reader with no kernel still sees it was there."""
    trail = SqliteTrail(tmp_path / "journal.db")
    await trail.open()
    try:
        await trail.add_member("/one", "one")
        _ = await trail.drop_member("/one")
    finally:
        await trail.close()

    with closing(sqlite3.connect(tmp_path / "journal.db")) as db:
        rows = [tuple(row) for row in db.execute("SELECT verb, path FROM space ORDER BY id")]

    assert rows == [("add", "/one"), ("drop", "/one")], "two decisions, and neither rewritten"


async def _folded_of(path) -> tuple[dict[str, str], Shape]:
    """The two folds a journal opened over *path* answers for this conversation."""
    trail = SqliteTrail(path)
    await trail.open()
    try:
        return await _folded(trail)
    finally:
        await trail.close()


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


def _snapshots(path) -> int:
    """How many snapshots of any conversation this file keeps."""
    with closing(sqlite3.connect(path)) as db:
        row = db.execute("SELECT COUNT(*) FROM checkpoint").fetchone()
    assert row is not None
    return int(row[0])


async def _readable(path) -> Snapshot | None:
    """What a journal opened over *path* answers as this conversation's newest snapshot."""
    trail = SqliteTrail(path)
    await trail.open()
    try:
        return await trail.latest("main")
    finally:
        await trail.close()
