"""The offline program: what it says about a store, and what it must not change while saying it.

No kernel runs here, and that is the point of the program: a person asking which trees have a ledger, and
what one of them has carried, is asking a machine where nothing is serving (E185). The assertion that
matters most is the negative one — a read that migrated the file would move the very clock the report is
about, so every case here checks what the file looked like afterwards.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner

from plyngent import admin, boot
from plyngent.admin import FAILED, main
from plyngent.kernel.documents import Documents
from plyngent.kernel.journal import MIGRATIONS, SqliteTrail, label, label_of, workspace_dir
from plyngent.kernel.model import User
from plyngent.kernel.trail import BY_BUDGET, Withheld


async def _ledger(data: Path, root: Path, *, weigh: bool = True, room: int = 8000) -> Path:
    """A ledger for *root* under *data*, holding a conversation, as a kernel would have left one."""
    root.mkdir(parents=True, exist_ok=True)
    journal = workspace_dir(root, data=data) / "journal.db"
    trail = SqliteTrail(journal)
    await trail.open()
    try:
        said = await trail.append("s1", User(text="hello there"))
        _ = await trail.append("s1", Withheld(targets=(said.id,), reason="budget", by=BY_BUDGET), begin_after=said.id)
        await trail.admitted("s2")
        if weigh:
            await trail.weighed("s1", live=1200, room=room)
    finally:
        await trail.close()
    label(journal.parent, root)
    return journal


async def _kept(journal: Path, session: str) -> None:
    """One document a conversation put aside, in the store that lives beside its journal."""
    store = Documents(journal.parent / "documents.db")
    await store.open()
    try:
        _ = await store.put(session, "the text nobody is going to ask for", summary="a note")
    finally:
        await store.close()


def _aged(journal: Path, session: str, said: str) -> None:
    """Move one conversation's last use into the past, which is the only clock this program reads (E186)."""
    with closing(sqlite3.connect(journal)) as db:
        _ = db.execute("UPDATE weight SET at = ? WHERE session = ?", (said, session))
        db.commit()


def _rows(journal: Path, session: str) -> int:
    """How many rows one conversation has in a ledger."""
    with closing(sqlite3.connect(journal)) as db:
        return int(db.execute("SELECT COUNT(id) FROM entry WHERE session = ?", (session,)).fetchone()[0])


def _documents(journal: Path, session: str) -> int:
    """How many documents one conversation holds beside its journal."""
    beside = journal.parent / "documents.db"
    if not beside.exists():
        return 0
    with closing(sqlite3.connect(beside)) as db:
        return int(db.execute("SELECT COUNT(id) FROM document WHERE session = ?", (session,)).fetchone()[0])


def _logged(journal: Path) -> tuple[str, ...]:
    """What was written down beside this ledger about what was let go of it."""
    found = journal.parent / "gc.log"
    return () if not found.exists() else tuple(found.read_text(encoding="utf-8").splitlines())


def _ran(*argv: str) -> tuple[int, str]:
    """What the program came to, and everything it wrote."""
    result = CliRunner().invoke(main, list(argv))
    return result.exit_code, result.output


def _untouched(path: Path) -> tuple[int, int, int]:
    """What a file looks like from outside: its size, its modification time, and its schema version."""
    with closing(sqlite3.connect(path)) as db:
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, version


def test_a_data_directory_with_no_ledgers_says_so(tmp_path: Path) -> None:
    """An empty machine is not a failure: a question that has an empty answer is still answered."""
    code, said = _ran("--data-dir", str(tmp_path), "workspaces")

    assert code == 0
    assert said.strip() == f"no ledgers in {tmp_path}"


async def test_every_ledger_is_listed_with_the_tree_it_is_about(tmp_path: Path) -> None:
    """A ledger's name comes from its directory, and the tree from the one line beside it.

    A ledger nothing has opened since it was made has no such line, and that is said as the fact it is —
    a name nobody can check would be worse than one nobody has written.
    """
    data = tmp_path / "data"
    named = await _ledger(data, tmp_path / "named")
    _ = await _ledger(data, tmp_path / "anonymous")
    (named.parent / "workspace.txt").unlink()

    code, said = _ran("--data-dir", str(data), "workspaces")

    assert code == 0
    rows = said.strip().splitlines()
    assert len(rows) == 2, said
    assert any(str(tmp_path / "anonymous") in row for row in rows), "the ledger that says which tree it is"
    assert any(row.endswith("no tree has opened this ledger since it was made") for row in rows), said


async def test_a_conversation_says_what_it_has_carried(tmp_path: Path) -> None:
    """``sessions`` reads the pair pruning keeps in balance, and one conversation nobody has said a word in.

    ``live + aside`` is the number, and ``room`` is the line a kernel prunes at: the two together are what
    "this conversation is big" means to a reader with no kernel and no clock (E186).
    """
    data = tmp_path / "data"
    tree = tmp_path / "tree"
    _ = await _ledger(data, tree)

    code, said = _ran("--data-dir", str(data), "sessions", str(tree))

    assert code == 0, said
    rows = said.strip().splitlines()
    assert any(row.startswith("s1  2 rows  ") for row in rows), said
    one = next(row for row in rows if row.startswith("s1  "))
    found = re.match(r"s1  2 rows  (\d+) tokens of 8000  last written \S+$", one)
    assert found is not None, one
    assert int(found.group(1)) > 1200, "the text pruning set aside is part of what this conversation carried"
    assert any(re.match(r"s2  0 rows  0 tokens  last written \S+$", row) for row in rows), (
        "an open conversation nothing was said in is still one this ledger holds, and it carries nothing"
    )


async def test_reading_a_ledger_leaves_it_exactly_as_it_was(tmp_path: Path) -> None:
    """``mode=ro`` is the promise: no migration, no write, and not even a moved modification time (E185).

    A reader that opened the file read-write would migrate it, and the migration would set the very clock
    this report is about to the instant of the question.
    """
    data = tmp_path / "data"
    tree = tmp_path / "tree"
    journal = await _ledger(data, tree)
    before = _untouched(journal)

    code, _ = _ran("--data-dir", str(data), "sessions", str(tree))

    assert code == 0
    assert _untouched(journal) == before


async def test_a_ledger_older_than_this_build_is_described_as_what_it_is(tmp_path: Path) -> None:
    """A file with no ``weight`` table is one this build can count rows in and cannot weigh.

    Working the size out is what a kernel would have written there, and reading is not writing: so the
    report says what it cannot say, and the file stays at the version it was (E186).
    """
    data = tmp_path / "data"
    tree = tmp_path / "old"
    tree.mkdir()
    journal = workspace_dir(tree, data=data) / "journal.db"
    journal.parent.mkdir(parents=True)
    with closing(sqlite3.connect(journal)) as db:
        _ = db.executescript(MIGRATIONS[0])
        _ = db.executescript(MIGRATIONS[1])
        _ = db.execute("PRAGMA user_version = 2")
        _ = db.execute(
            "INSERT INTO entry (session, seq, turn, parent, tag, payload) VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", 1, 1, None, "user", '{"text":"said long ago"}'),
        )
        db.commit()
    before = _untouched(journal)

    code, said = _ran("--data-dir", str(data), "sessions", str(tree))

    assert code == 0, said
    assert "this ledger was written before this build can weigh one" in said, said
    assert _untouched(journal) == before, "and asking did not migrate it"


def test_a_tree_with_no_ledger_here_is_a_refusal(tmp_path: Path) -> None:
    """A tree nothing has opened under this data directory is a sentence and the error code, not a blank."""
    code, said = _ran("--data-dir", str(tmp_path / "data"), "sessions", str(tmp_path / "never"))

    assert code == FAILED
    assert "no ledger for" in said


async def test_a_kernel_says_which_tree_a_ledger_is_about(tmp_path: Path) -> None:
    """The line beside a ledger is written by whoever opens the tree, and that is the kernel (E185).

    Nothing else in the suite would notice if it stopped: the offline program is the only reader, and it
    would answer "no tree has opened this ledger" about every one of them.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    installed = await boot.install(tree, journal=tmp_path / "journal.db", versions=False)
    try:
        assert label_of(tmp_path) == str(tree.resolve())
    finally:
        await installed.close()


async def test_a_tree_that_moved_is_not_the_ledger_it_left_behind(tmp_path: Path) -> None:
    """The name of a ledger is the path of the tree, so a move leaves the two disagreeing (E181/E185).

    That is the honest answer: following a move needs an identity inside the tree or the inode for a name,
    and both are worse than a person saying which ledger belongs where.
    """
    data = tmp_path / "data"
    tree = tmp_path / "moved"
    _ = await _ledger(data, tree)
    tree.rename(tmp_path / "elsewhere")

    code, said = _ran("--data-dir", str(data), "sessions", str(tmp_path / "elsewhere"))

    assert code == FAILED
    assert str(tmp_path / "elsewhere") in said


async def test_a_conversation_nobody_has_used_is_reported_and_kept(tmp_path: Path) -> None:
    """The default is a report: a deletion nobody can undo is not decided by a flag left out (E186).

    What a person reads is what it would free — the rows, the documents, the tokens it carried — and the
    ledger still holds every one of them.
    """
    data = tmp_path / "data"
    journal = await _ledger(data, tmp_path / "tree", room=1000)
    _aged(journal, "s1", "2020-01-01T00:00:00Z")

    code, said = _ran("--data-dir", str(data), "gc")

    assert code == 0, said
    assert "s1  2 rows" in said and "last written 2020-01-01T00:00:00Z" in said, said
    assert "nothing was deleted; say --apply to let 1 go" in said, said
    assert _rows(journal, "s1") == 2
    assert _logged(journal) == (), "and nothing was written down, because nothing happened"


async def test_apply_lets_a_conversation_go_and_writes_down_what_it_did(tmp_path: Path) -> None:
    """``--apply`` deletes the rows and the documents, gives the room back, and leaves a line about it.

    The line comes last, so it cannot describe a deletion that did not happen (E186).
    """
    data = tmp_path / "data"
    journal = await _ledger(data, tmp_path / "tree", room=1000)
    _aged(journal, "s1", "2020-01-01T00:00:00Z")
    await _kept(journal, "s1")
    assert _documents(journal, "s1") == 1

    code, said = _ran("--data-dir", str(data), "gc", "--apply")

    assert code == 0, said
    assert "let 1 go, and gave back" in said, said
    assert _rows(journal, "s1") == 0, "every table that named it is done with it"
    assert _documents(journal, "s1") == 0, "and the documents went with it"
    lines = _logged(journal)
    assert any("s1" in line and "2020-01-01T00:00:00Z" in line for line in lines), lines
    assert any("bytes given back" in line for line in lines), lines


async def test_a_conversation_used_just_now_is_not_a_candidate(tmp_path: Path) -> None:
    """The clock is the one the kernel wrote, and what it says about the last week is the whole criterion."""
    data = tmp_path / "data"
    journal = await _ledger(data, tmp_path / "tree", room=1000)

    code, said = _ran("--data-dir", str(data), "gc", "--older-than", "1h")

    assert code == 0, said
    assert "nothing to let go" in said, "written to a moment ago, so it is in use"
    assert _rows(journal, "s1") == 2


async def test_a_conversation_whose_route_never_said_a_window_needs_a_number(tmp_path: Path) -> None:
    """``room = 0`` is "this route never said how much it holds", so there is no line to be over (E186).

    A number given on the command line takes the place of that line, which is the only way such a
    conversation can be weighed at all.
    """
    data = tmp_path / "data"
    journal = await _ledger(data, tmp_path / "tree", room=0)
    _aged(journal, "s1", "2020-01-01T00:00:00Z")

    code, said = _ran("--data-dir", str(data), "gc")
    assert code == 0, said
    assert "nothing to let go" in said, "nothing here can say how big it is"

    code, said = _ran("--data-dir", str(data), "gc", "--bigger-than", "1000")
    assert code == 0, said
    assert "s1  2 rows" in said, "and a number given here is the line it is measured by"


def test_a_length_of_time_that_is_not_one_is_a_usage_error(tmp_path: Path) -> None:
    """``--older-than 7`` is a question this program cannot turn into a moment, and click says so."""
    code, said = _ran("--data-dir", str(tmp_path), "gc", "--older-than", "7")

    assert code == 2
    assert "not a length of time" in said


async def test_a_kernel_that_is_answering_keeps_the_ledgers_to_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers of one ledger is the one thing an offline program cannot do (E185).

    ``--force`` is the person saying they know: the program then works on the files anyway, which is the
    state a machine is in when a kernel was killed rather than stopped.
    """
    data = tmp_path / "data"
    journal = await _ledger(data, tmp_path / "tree", room=1000)
    _aged(journal, "s1", "2020-01-01T00:00:00Z")
    monkeypatch.setattr(admin, "_answering", lambda endpoint: True)

    code, said = _ran("--data-dir", str(data), "gc", "--apply")
    assert code == FAILED
    assert "--force" in said, said
    assert _rows(journal, "s1") == 2, "and nothing was deleted"

    code, said = _ran("--data-dir", str(data), "--force", "gc", "--apply")
    assert code == 0, said
    assert _rows(journal, "s1") == 0


async def test_a_ledger_left_mid_write_is_left_alone(tmp_path: Path) -> None:
    """A write-ahead log with anything in it is a store somebody has open, or one that died mid-write."""
    data = tmp_path / "data"
    journal = await _ledger(data, tmp_path / "tree", room=1000)
    _aged(journal, "s1", "2020-01-01T00:00:00Z")
    (journal.parent / "journal.db-wal").write_bytes(b"not empty")

    code, said = _ran("--data-dir", str(data), "gc", "--apply")

    assert code == 0, said
    assert "left alone" in said and "not empty" in said, said
    assert _rows(journal, "s1") == 2


async def test_adopting_a_ledger_is_saying_which_tree_it_is_now(tmp_path: Path) -> None:
    """A tree that moved kept its ledger, and one command says where it went (E185).

    The three stores are one directory, so the move is one move: the conversation, the documents it kept and
    the versions of the bytes its calls left all arrive together, and the line beside them says which tree
    this is now.
    """
    data = tmp_path / "data"
    tree = tmp_path / "tree"
    journal = await _ledger(data, tree)
    await _kept(journal, "s1")
    tree.rename(tmp_path / "moved")

    code, said = _ran("--data-dir", str(data), "adopt", "--from", str(tree), "--to", str(tmp_path / "moved"))

    assert code == 0, said
    adopted = workspace_dir(tmp_path / "moved", data=data)
    assert adopted.is_dir() and not journal.parent.exists(), "the one directory moved"
    assert label_of(adopted) == str((tmp_path / "moved").resolve())
    assert _rows(adopted / "journal.db", "s1") == 2, "and what it held came with it"
    assert _documents(adopted / "journal.db", "s1") == 1

    code, said = _ran("--data-dir", str(data), "sessions", str(tmp_path / "moved"))
    assert code == 0, said
    assert "s1  2 rows" in said


async def test_adopting_a_tree_that_has_no_ledger_is_a_refusal(tmp_path: Path) -> None:
    """There is nothing to move, and saying so is the whole answer."""
    now = tmp_path / "now"
    now.mkdir()

    code, said = _ran(
        "--data-dir", str(tmp_path / "data"), "adopt", "--from", str(tmp_path / "never"), "--to", str(now)
    )

    assert code == FAILED
    assert "no ledger for" in said


async def test_adopting_onto_a_tree_that_has_its_own_ledger_is_a_refusal(tmp_path: Path) -> None:
    """Two ledgers of one tree is the thing this command exists to prevent, so it never makes one."""
    data = tmp_path / "data"
    _ = await _ledger(data, tmp_path / "tree")
    taken = tmp_path / "taken"
    _ = await _ledger(data, taken)

    code, said = _ran("--data-dir", str(data), "adopt", "--from", str(tmp_path / "tree"), "--to", str(taken))

    assert code == FAILED
    assert "already has a ledger" in said, said


async def test_adopting_a_ledger_onto_itself_says_it_is_already_there(tmp_path: Path) -> None:
    """The same tree twice is not a failure: nothing is moved, and nothing is lost."""
    data = tmp_path / "data"
    tree = tmp_path / "tree"
    journal = await _ledger(data, tree)

    code, said = _ran("--data-dir", str(data), "adopt", "--from", str(tree), "--to", str(tree))

    assert code == 0, said
    assert "already its own" in said, said
    assert _rows(journal, "s1") == 2


async def test_adopting_onto_a_path_that_is_not_a_tree_is_a_refusal(tmp_path: Path) -> None:
    """A ledger is adopted by a tree, and a path that is not a directory is not one."""
    data = tmp_path / "data"
    _ = await _ledger(data, tmp_path / "tree")

    code, said = _ran(
        "--data-dir", str(data), "adopt", "--from", str(tmp_path / "tree"), "--to", str(tmp_path / "gone")
    )

    assert code == FAILED
    assert "is not a directory" in said, said


async def test_adopting_while_a_kernel_answers_needs_the_person_to_insist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same refusal ``gc`` has, and the same word to get past it (E185)."""
    data = tmp_path / "data"
    tree = tmp_path / "tree"
    journal = await _ledger(data, tree)
    tree.rename(tmp_path / "moved")
    monkeypatch.setattr(admin, "_answering", lambda endpoint: True)

    code, said = _ran("--data-dir", str(data), "adopt", "--from", str(tree), "--to", str(tmp_path / "moved"))
    assert code == FAILED
    assert "--force" in said, said
    assert journal.parent.exists(), "and nothing moved"

    code, said = _ran("--data-dir", str(data), "--force", "adopt", "--from", str(tree), "--to", str(tmp_path / "moved"))
    assert code == 0, said
    assert not journal.parent.exists()
