"""The offline program: what is on disk, looked at with no kernel in the way (E185).

A kernel keeps three things per workspace — a journal, the documents its bodies went to, and the version
store its calls left behind — and everything about them is reachable through a connection while the kernel
runs. What is *not* reachable is a question asked about a machine where nothing is running: which trees have
a ledger here, what one of them has carried, and what may be let go. That is this program, and it is why it
is a program of its own.

**It is on the composition root's side of the boundary, and not the frontend's.** It reads the store modules
directly and it may ask :func:`plyngent.boot.endpoint_for` whether something is answering; what it may not
do is import :mod:`plyngent.cli`, which would drag a terminal library into a command a supervisor runs.

**Reading is reading.** Every question is answered from a connection opened ``mode=ro``: no migration, no
write, and not even a changed modification time. That is a promise with a reason — a build that opened the
file read-write would move the very clock the report is about, so the answer would be wrong about the one
thing it exists to say.

**Two exit codes.** ``2`` is a usage error, which click owns; ``1`` is "that could not be done", which is
every refusal here. The numbers are the ones the kernel's own program returns (E174).
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import click
from platformdirs import user_data_path

from plyngent import boot
from plyngent.kernel.journal import label, label_of, workspace_dir

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Where a person's ledgers are when nobody says otherwise: the directory a kernel writes them in.
DEFAULT_DATA = user_data_path("plyngent")

#: What this program returns when it was asked for something it could not do. ``2`` is click's, for usage.
FAILED = 1

#: Which schema version has the table that says what a conversation has carried (E186).
WEIGHED = 3

#: The line beside a ledger that says what was let go of it, and what that gave back (E186).
LOG = "gc.log"


@dataclass(frozen=True, slots=True)
class _Gone:
    """One conversation that may be let go, and what letting it go would free (E186)."""

    session: str
    rows: int
    documents: int
    bytes: int
    tokens: int
    at: str

    def line(self) -> str:
        """The line a person reads about it: what it is called, what it holds, and when it was last used."""
        said = f"{self.session}  {self.rows} rows  {self.documents} documents  {self.tokens} tokens"
        return f"{said}  last written {self.at}"


class _Desk:
    """The data directory this run is about, and the three questions a ledger answers.

    It exists as a value so that no verb has to know how a ledger is spelled twice: ``--data-dir`` decides
    *which* ledgers are in front of this program, and one rule decides the name of each one (E185).
    """

    data: Path
    force: bool

    def __init__(self, data: Path, *, force: bool = False) -> None:
        self.data = data
        self.force = force

    def ledgers(self) -> tuple[Path, ...]:
        """Every ledger directory under this data directory, sorted by name, without opening any of them."""
        root = self.data / "workspaces"
        if not root.is_dir():
            return ()
        return tuple(sorted(one for one in root.iterdir() if one.is_dir()))

    def journal(self, root: Path | str) -> Path:
        """Where the ledger of *root* keeps its journal under this data directory."""
        return workspace_dir(root, data=self.data) / "journal.db"

    def may_write(self) -> None:
        """Refuse to write while a kernel is answering, because two writers of one ledger is not a thing
        an offline program can do safely (E185).

        The question is asked of this person's own address, which is where a kernel serving these ledgers
        would be. Nothing is asked of an address that is not there: no kernel, no lease to take.
        """
        if self.force or not _answering(boot.endpoint_for()):
            return
        _stop(f"a kernel is answering at {boot.endpoint_for()}; say --force to work on the ledgers anyway")


@click.group()
@click.option(
    "--data-dir",
    "data",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="the directory whose ledgers to work on",
)
@click.option("--force", is_flag=True, help="work on the ledgers even though a kernel is answering")
@click.pass_context
def main(ctx: click.Context, data: Path | None, *, force: bool) -> None:
    """Read, and where it is asked to, rewrite the ledgers under one data directory."""
    ctx.obj = _Desk(DEFAULT_DATA if data is None else data, force=force)


@main.command("workspaces")
@click.pass_obj
def workspaces(desk: _Desk) -> None:
    """List the ledgers here, and which tree each one is about.

    Nothing is opened: a ledger's name comes from its directory, and which tree it belongs to from the
    one-line file beside it. A ledger nothing has opened since it was made has no such line, which is said
    as the fact it is.
    """
    found = desk.ledgers()
    if not found:
        click.echo(f"no ledgers in {desk.data}")
        return
    for one in found:
        said = label_of(one) or "no tree has opened this ledger since it was made"
        click.echo(f"{one.name}  {said}")


@main.command("sessions")
@click.argument("root", type=click.Path(file_okay=False, path_type=Path))
@click.pass_obj
def sessions(desk: _Desk, root: Path) -> None:
    """Say what each conversation of one tree has carried, and when it was last written to (E186).

    Two reads of the file and nothing else. A ledger old enough to have no ``weight`` table is reported as
    what it is: its size unknown, because working it out is what a kernel would have written there.
    """
    journal = desk.journal(root)
    if not journal.exists():
        _stop(f"no ledger for {root}: nothing has opened that tree with this data directory")
    for row in _weighed(journal):
        click.echo(row)


@main.command("gc")
@click.option(
    "--older-than",
    "older_than",
    default="7d",
    show_default=True,
    help="how long a conversation must have been left alone, as in 7d, 12h or 30m",
)
@click.option(
    "--bigger-than",
    "bigger_than",
    type=int,
    default=None,
    help="tokens a conversation must have carried, in place of the window the kernel prunes at",
)
@click.option("--apply", is_flag=True, help="delete what is found, which cannot be undone")
@click.pass_obj
def gc(desk: _Desk, older_than: str, bigger_than: int | None, *, apply: bool) -> None:
    """Find, and with ``--apply`` let go, the conversations nobody has used in a while (E186).

    Two facts, and the second is what makes the first worth asking: how long ago a conversation was last
    written to, and how much it has carried — the window a kernel prunes at, unless a number is given. A
    conversation nothing has ever written about is not a candidate, because nothing is known about it.

    Nothing is deleted without ``--apply``, and a deletion is not reversible: the rows and the documents go,
    the space is given back, and what was done is written down beside the ledger.
    """
    horizon = datetime.now(UTC) - _duration(older_than)
    if apply:
        desk.may_write()
    found = False
    for ledger in desk.ledgers():
        journal = ledger / "journal.db"
        if not journal.exists():
            continue
        root = label_of(ledger) or ledger.name
        gone = _candidates(journal, horizon=horizon, bigger=bigger_than)
        if not gone:
            continue
        found = True
        for one in gone:
            click.echo(f"{root}  {one.line()}")
        if not apply:
            click.echo(f"{root}  nothing was deleted; say --apply to let {len(gone)} go")
            continue
        _let_go(journal, gone, root=root)
    if not found:
        click.echo(f"nothing to let go under {desk.data}")


@main.command("adopt")
@click.option(
    "--from",
    "was",
    required=True,
    type=click.Path(path_type=Path),
    help="the path this ledger was made under",
)
@click.option("--to", "now", required=True, type=click.Path(path_type=Path), help="where that tree is now")
@click.pass_obj
def adopt(desk: _Desk, was: Path, now: Path) -> None:
    """Say that a ledger belongs to a tree that moved, which is the half a move leaves behind (E185).

    A ledger's name is its tree's path, so a tree that moved is a tree nothing can find a ledger for. The
    three stores live in one directory — the journal, the documents, and the versions — so moving it is one
    move, and the line beside it is rewritten to say where the tree is now.
    """
    desk.may_write()
    old = desk.journal(was).parent
    new = desk.journal(now).parent
    if not (old / "journal.db").exists():
        _stop(f"no ledger for {was}: nothing has opened that tree with this data directory")
    if old == new:
        click.echo(f"{was} and {now} are one tree, so this ledger is already its own")
        return
    if not Path(now).expanduser().is_dir():
        _stop(f"{now} is not a directory, so there is no tree there to take this ledger")
    if (new / "journal.db").exists():
        _stop(f"{now} already has a ledger, and two ledgers of one tree is the thing this command is for")
    reason = _unclean(old / "journal.db")
    if reason:
        _stop(f"left alone: {reason}")
    _ = shutil.move(str(old), str(new))
    _ = label(new, now)
    click.echo(f"{was} is now {now}: {old.name} says so")


def _candidates(journal: Path, *, horizon: datetime, bigger: int | None) -> tuple[_Gone, ...]:
    """The conversations of one ledger that are idle enough and large enough, with what each would free."""
    with closing(_read_only(journal)) as db:
        if _version(db) < WEIGHED:
            return ()
        rows = dict(_counted(db, WEIGHED))
        documents = _documents_of(journal)
        found: list[_Gone] = []
        for session, at, live, aside, room in _weights(db):
            when = _instant(at)
            carried = live + aside
            line = room if bigger is None else bigger
            if when is None or when > horizon or line <= 0 or carried < line:
                continue
            found.append(
                _Gone(
                    session=session,
                    rows=rows.get(session, 0),
                    documents=documents.get(session, (0, 0))[0],
                    bytes=documents.get(session, (0, 0))[1],
                    tokens=carried,
                    at=at,
                )
            )
        return tuple(found)


def _let_go(journal: Path, gone: tuple[_Gone, ...], *, root: str) -> None:
    """Delete what was found, reclaim the room, and write down what was done (E186).

    The order is the promise: the rows go, the file is made to give the space back, and the line is written
    last — a log entry about a deletion that did not happen would be the one thing here nobody could check.
    A space that cannot be reclaimed is said as the pair of facts it is: the rows are gone, the bytes are
    still on disk.
    """
    reason = _unclean(journal)
    if reason:
        click.echo(f"{root}  left alone: {reason}", err=True)
        return
    before = journal.stat().st_size
    _deleted(journal, gone)
    _reclaimed(journal)
    after = journal.stat().st_size
    for one in gone:
        _logged(journal, f"{_now()}  {root}  {one.line()}")
    _logged(journal, f"{_now()}  {root}  {before - after} bytes given back")
    click.echo(f"{root}  let {len(gone)} go, and gave back {before - after} bytes")


def _deleted(journal: Path, gone: tuple[_Gone, ...]) -> None:
    """Take the rows of these conversations out of every table a kernel keeps them in, and their documents."""
    with closing(sqlite3.connect(journal)) as db:
        for one in gone:
            for table in ("entry", "turn_no", "projection", "checkpoint", "weight"):
                _ = db.execute(f"DELETE FROM {table} WHERE session = ?", (one.session,))
        db.commit()
    beside = journal.parent / "documents.db"
    if not beside.exists():
        return
    with closing(sqlite3.connect(beside)) as db:
        for one in gone:
            _ = db.execute("DELETE FROM document WHERE session = ?", (one.session,))
        db.commit()


def _reclaimed(journal: Path) -> None:
    """Make the file give back the space its rows no longer need (E186).

    ``VACUUM`` needs room the size of the database itself, and a machine without it says so: the rows are
    already gone, and this is the half that can fail without having broken anything.
    """
    with closing(sqlite3.connect(journal)) as db:
        db.isolation_level = None
        try:
            _ = db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _ = db.execute("VACUUM")
        except sqlite3.OperationalError as exc:
            click.echo(f"the rows are gone, and the space is still on disk: {exc}", err=True)


def _logged(journal: Path, line: str) -> None:
    """Write one line about what happened to this ledger, where a person can find it afterwards (E186)."""
    with (journal.parent / LOG).open("a", encoding="utf-8") as said:
        _ = said.write(f"{line}\n")


def _weights(db: sqlite3.Connection) -> tuple[tuple[str, str, int, int, int], ...]:
    """Every conversation's weight in one ledger, in the order they were last written to."""
    rows = db.execute("SELECT session, at, live, aside, room FROM weight ORDER BY at, session").fetchall()
    return tuple((str(row[0]), str(row[1]), int(row[2]), int(row[3]), int(row[4])) for row in rows)


def _documents_of(journal: Path) -> dict[str, tuple[int, int]]:
    """How many documents each conversation holds beside this journal, and how many bytes they are."""
    beside = journal.parent / "documents.db"
    if not beside.exists():
        return {}
    with closing(_read_only(beside)) as db:
        rows = db.execute(
            "SELECT session, COUNT(id), COALESCE(SUM(LENGTH(content)), 0) FROM document GROUP BY session"
        ).fetchall()
    return {str(row[0]): (int(row[1]), int(row[2])) for row in rows}


def _instant(said: str) -> datetime | None:
    """The moment a weight row says, or ``None`` when it says something this build cannot read."""
    try:
        return datetime.fromisoformat(said)
    except ValueError:
        return None


_DURATIONS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400}


def _duration(said: str) -> timedelta:
    """How long ``7d`` means, refusing anything that is not a number and one of four units."""
    number, unit = said[:-1], said[-1:]
    if not number.isdigit() or unit not in _DURATIONS:
        why = f"{said!r} is not a length of time: give a number and one of s, m, h, d"
        raise click.BadParameter(why)
    return timedelta(seconds=int(number) * _DURATIONS[unit])


def _answering(endpoint: Path) -> bool:
    """Whether a kernel is answering at *endpoint* right now (E185)."""
    if not endpoint.exists():
        return False
    return asyncio.run(_pinged(endpoint))


async def _pinged(endpoint: Path) -> bool:
    """Ask an address whether a kernel is behind it, and say goodbye before answering (E185)."""
    try:
        link = await boot.connect(str(endpoint), declare=False)
    except boot.NoKernelError, OSError:
        return False
    link.close()
    await link.finished()
    return True


def _unclean(journal: Path) -> str:
    """Why this ledger may not be written to as it stands, or the empty string (E185)."""
    beside = journal.with_name(journal.name + "-wal")
    if beside.exists() and beside.stat().st_size:
        return f"{beside.name} is not empty, so this store was left mid-write or somebody has it open"
    return ""


def _now() -> str:
    """The instant a line is written, spelled the way a weight row spells one (E186)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _weighed(journal: Path) -> Iterator[str]:
    """One line per conversation in this journal: its rows, and what it has carried (E186)."""
    with closing(_read_only(journal)) as db:
        version = _version(db)
        for session, rows in _counted(db, version):
            if version < WEIGHED:
                yield f"{session}  {rows} rows  (this ledger was written before this build can weigh one)"
                continue
            found = _weight_of(db, session)
            yield f"{session}  {rows} rows  {found}"


def _weight_of(db: sqlite3.Connection, session: str) -> str:
    """What one conversation has carried, as the line a person reads about it (E186).

    ``live + aside`` is the number that matters: pruning moves text from one to the other and leaves the
    sum where it was, which is why a conversation that was pruned does not look small.
    """
    row = db.execute("SELECT at, live, aside, room FROM weight WHERE session = ?", (session,)).fetchone()
    if row is None:
        return "nothing has written about what it carried"
    at, live, aside, room = (str(row[0]), int(row[1]), int(row[2]), int(row[3]))
    room = f" of {room}" if room else ""
    return f"{live + aside} tokens{room}  last written {at}"


def _counted(db: sqlite3.Connection, version: int) -> tuple[tuple[str, int], ...]:
    """Every conversation this ledger holds, with how many rows each has, in the order they arrived.

    The set is the log's and the weights' together: a conversation somebody opened and said nothing in has
    no row at all, and it is still one this ledger holds. A file too old to have a weight table answers from
    its rows alone, which is every conversation it can still prove.
    """
    if version < WEIGHED:
        rows = db.execute("SELECT session, COUNT(id) FROM entry GROUP BY session ORDER BY MIN(id)").fetchall()
    else:
        rows = db.execute(
            "SELECT s.session, COUNT(e.id)"
            "  FROM (SELECT session FROM entry UNION SELECT session FROM weight) AS s"
            "  LEFT JOIN entry AS e ON e.session = s.session"
            " GROUP BY s.session"
            " ORDER BY MIN(e.id), s.session"
        ).fetchall()
    return tuple((str(row[0]), int(row[1])) for row in rows)


def _read_only(journal: Path) -> sqlite3.Connection:
    """A connection that cannot write, migrate, or move the file's modification time (E185).

    ``mode=ro`` is the whole of it, and it is the difference between reading a clock and setting one.
    """
    return sqlite3.connect(f"file:{journal}?mode=ro", uri=True)


def _version(db: sqlite3.Connection) -> int:
    """How many schema steps the file on disk has been through."""
    row = db.execute("PRAGMA user_version").fetchone()
    return 0 if row is None else int(row[0])


def _stop(message: str) -> None:
    """Say why this could not be done, and leave with the code a script reads for it (E185)."""
    click.echo(message, err=True)
    raise SystemExit(FAILED)
