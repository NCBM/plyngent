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

from click.testing import CliRunner

from plyngent import boot
from plyngent.admin import FAILED, main
from plyngent.kernel.journal import MIGRATIONS, SqliteTrail, label, label_of, workspace_dir
from plyngent.kernel.model import User
from plyngent.kernel.trail import BY_BUDGET, Withheld


async def _ledger(data: Path, root: Path, *, weigh: bool = True) -> Path:
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
            await trail.weighed("s1", live=1200, room=8000)
    finally:
        await trail.close()
    label(journal.parent, root)
    return journal


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
