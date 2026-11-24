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

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import click
from platformdirs import user_data_path

from plyngent.kernel.journal import label_of, workspace_dir

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Where a person's ledgers are when nobody says otherwise: the directory a kernel writes them in.
DEFAULT_DATA = user_data_path("plyngent")

#: What this program returns when it was asked for something it could not do. ``2`` is click's, for usage.
FAILED = 1

#: Which schema version has the table that says what a conversation has carried (E186).
WEIGHED = 3


class _Desk:
    """The data directory this run is about, and the three questions a ledger answers.

    It exists as a value so that no verb has to know how a ledger is spelled twice: ``--data-dir`` decides
    *which* ledgers are in front of this program, and one rule decides the name of each one (E185).
    """

    data: Path

    def __init__(self, data: Path) -> None:
        self.data = data

    def ledgers(self) -> tuple[Path, ...]:
        """Every ledger directory under this data directory, sorted by name, without opening any of them."""
        root = self.data / "workspaces"
        if not root.is_dir():
            return ()
        return tuple(sorted(one for one in root.iterdir() if one.is_dir()))

    def journal(self, root: Path | str) -> Path:
        """Where the ledger of *root* keeps its journal under this data directory."""
        return workspace_dir(root, data=self.data) / "journal.db"


@click.group()
@click.option(
    "--data-dir",
    "data",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="the directory whose ledgers to work on",
)
@click.pass_context
def main(ctx: click.Context, data: Path | None) -> None:
    """Read, and where it is asked to, rewrite the ledgers under one data directory."""
    ctx.obj = _Desk(DEFAULT_DATA if data is None else data)


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
