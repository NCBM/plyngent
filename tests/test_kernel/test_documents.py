"""The store behind a document handle: what goes in, what comes back, and who may ask for it.

A handle promises only that it locates content, so most of what it promises has no screen: bytes are
kept as they are, a session cannot see what another one kept, a body bigger than the whole budget is
still kept, an id nobody minted is not an error, and the budget lets the oldest bodies go. Those are
asserted here; the stories assert what a model reads — including a handle outliving the process that
made it, which is the one promise of a handle with a screen of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from plyngent.kernel.documents import (
    MAX_KEPT_BYTES,
    MAX_TEXT_CHARS,
    MEMORY_DOCUMENTS,
    SUMMARY_CHARS,
    Documents,
    TooBigError,
    documents_path,
    refuses,
)

if TYPE_CHECKING:
    from pathlib import Path


async def _opened(path: Path | str) -> Documents:
    """A store that is up, in a file of the test's own."""
    store = Documents(path)
    await store.open()
    return store


async def test_bytes_are_kept_as_they_are(tmp_path: Path) -> None:
    store = await _opened(tmp_path / "documents.db")
    try:
        picture = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        kept = await store.put("main", picture, summary="a picture", media="image/png")

        found = await store.get("main", kept.id)
        assert found is not None
        assert found.data == picture, "bytes are not decoded on the way in or on the way out"
        assert found.kept.chars == len(picture), "the count says how big it is, not how much text it has"
    finally:
        await store.close()


async def test_a_session_cannot_read_another_ones_documents(tmp_path: Path) -> None:
    store = await _opened(tmp_path / "documents.db")
    try:
        kept = await store.put("main", "mine\n", summary="mine")

        assert await store.get("other", kept.id) is None, "content is as private as the conversation it was kept for"
        assert await store.listing("other") == ()
        assert [item.id for item in await store.listing("main")] == [kept.id]
    finally:
        await store.close()


async def test_an_id_that_names_nothing_is_not_an_error(tmp_path: Path) -> None:
    store = await _opened(tmp_path / "documents.db")
    try:
        for made_up in ("d999", "d", "", "e1", "d1x"):
            assert await store.get("main", made_up) is None, f"{made_up!r} is not a handle this store minted"
    finally:
        await store.close()


async def test_a_store_with_no_file_keeps_nothing_after_it_closes() -> None:
    store = await _opened(MEMORY_DOCUMENTS)
    kept = await store.put("main", "for now\n", summary="for now")
    assert await store.get("main", kept.id) is not None
    await store.close()

    again = await _opened(MEMORY_DOCUMENTS)
    try:
        assert await again.get("main", kept.id) is None, "a conversation nobody meant to keep has no content either"
    finally:
        await again.close()


async def test_a_session_over_its_budget_lets_the_oldest_bodies_go(tmp_path: Path) -> None:
    """The store is a cache: writing past the budget makes room at the write, oldest first (§0.6, R10)."""
    store = Documents(tmp_path / "documents.db", budget=20, rows=100)
    await store.open()
    try:
        first = await store.put("main", "x" * 8, summary="first")
        second = await store.put("main", "y" * 8, summary="second")
        assert [item.id for item in await store.listing("main")] == [first.id, second.id]

        third = await store.put("main", "z" * 8, summary="third")

        assert [item.id for item in await store.listing("main")] == [second.id, third.id], "the oldest went"
        assert await store.get("main", first.id) is None, "and its handle names nothing this session has"
        assert await store.get("main", third.id) is not None, "what was just written is never the one dropped"
    finally:
        await store.close()


async def test_a_body_bigger_than_the_whole_budget_is_still_kept(tmp_path: Path) -> None:
    """A budget that dropped everything for one large body would keep nothing at all for it."""
    store = Documents(tmp_path / "documents.db", budget=4, rows=100)
    await store.open()
    try:
        kept = await store.put("main", "x" * 8, summary="large on its own")

        assert await store.get("main", kept.id) is not None, "the newest row stays whatever its size"
    finally:
        await store.close()


async def test_the_count_is_a_budget_too(tmp_path: Path) -> None:
    """Two numbers, because a session can be full of small bodies as easily as of large ones."""
    store = Documents(tmp_path / "documents.db", budget=1_000, rows=2)
    await store.open()
    try:
        ids = [(await store.put("main", f"{index}\n", summary=str(index))).id for index in range(3)]

        assert [item.id for item in await store.listing("main")] == ids[1:], "the oldest of three went"
    finally:
        await store.close()


def test_the_bodies_live_beside_the_journal() -> None:
    assert documents_path("/data/ws-a/journal.db").name == "documents.db"
    assert documents_path("/data/ws-a/journal.db").parent.name == "ws-a", "one workspace's kept bodies stay with it"
    assert documents_path(MEMORY_DOCUMENTS) == MEMORY_DOCUMENTS, "a journal with no file has no directory to use"


async def test_content_past_the_limit_is_refused_with_its_number(tmp_path: Path) -> None:
    """A limit only the polite path honours is not a limit: the store refuses it itself (§0.6, R10)."""
    store = await _opened(tmp_path / "documents.db")
    try:
        for content, cap, unit in (
            ("x" * (MAX_TEXT_CHARS + 1), MAX_TEXT_CHARS, "characters"),
            (b"\x00" * (MAX_KEPT_BYTES + 1), MAX_KEPT_BYTES, "bytes"),
        ):
            assert refuses(content) is not None, "the predicate a writer asks first says no"
            with pytest.raises(TooBigError) as refused:
                _ = await store.put("main", content, summary="too big")
            assert f"{len(content)} {unit}" in str(refused.value), "the sentence names the size"
            assert f"({cap})" in str(refused.value), "and the number it passed"
        assert await store.listing("main") == (), "nothing was kept on the way to finding out"
    finally:
        await store.close()


async def test_the_same_bytes_are_kept_once_in_a_session(tmp_path: Path) -> None:
    """A handle a model can hold: reading one window twice is one document, not two (§0.6, R10)."""
    store = await _opened(tmp_path / "documents.db")
    try:
        first = await store.put("main", "same\n", summary="first time")
        again = await store.put("main", "same\n", summary="second time")

        assert again.id == first.id, "the same bytes are the same document"
        assert again.summary == "first time", "a handle names content, and says where it first came from"
        assert [item.id for item in await store.listing("main")] == [first.id], "one row, not two"
        other = await store.put("other", "same\n", summary="somebody else's")
        assert other.id != first.id, "and sessions never share a row"
        as_picture = await store.put("main", "same\n", summary="pretending", media="image/png")
        assert as_picture.id != first.id, "a row is content and media: the same bytes are not the same picture"
    finally:
        await store.close()


async def test_a_file_from_the_older_shape_keeps_its_content_and_gains_the_column(tmp_path: Path) -> None:
    """The first step of the ladder: add the column, index it, lose nothing (R10)."""
    path = tmp_path / "documents.db"
    older = await aiosqlite.connect(str(path), isolation_level=None)
    try:
        _ = await older.executescript(
            """
            CREATE TABLE document (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                summary TEXT NOT NULL,
                media TEXT NOT NULL,
                chars INTEGER NOT NULL,
                created INTEGER NOT NULL,
                content BLOB NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )
        _ = await older.execute(
            "INSERT INTO document (session, summary, media, chars, created, content) "
            "VALUES ('main', 'from before', 'text/plain', 6, 0, ?)",
            (b"kept\n",),
        )
    finally:
        await older.close()

    store = await _opened(path)
    try:
        listed = await store.listing("main")
        assert [item.summary for item in listed] == ["from before"], "the row written by the older build is still here"
        found = await store.get("main", listed[0].id)
        assert found is not None and found.data == b"kept\n"
        again = await store.put("main", "kept\n", summary="now")
        assert again.id != listed[0].id, "a row from before deduplication has no hash, so nothing matches it"
    finally:
        await store.close()


async def test_a_summary_is_one_line(tmp_path: Path) -> None:
    """The line that stands in for content is a line, whatever the writer handed over (E54)."""
    store = await _opened(tmp_path / "documents.db")
    try:
        kept = await store.put("main", "body\n", summary="s" * (SUMMARY_CHARS + 300))

        assert len(kept.summary) == SUMMARY_CHARS, "cut to the limit, not past it"
        assert kept.summary.endswith("…"), "and marked, so nobody takes the cut for the whole sentence"
        found = await store.get("main", kept.id)
        assert found is not None and found.kept.summary == kept.summary, "what came back is what was written"
    finally:
        await store.close()
