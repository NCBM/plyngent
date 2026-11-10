"""The store behind a document handle: what goes in, what comes back, and who may ask for it.

A handle promises only that it locates content, so most of what it promises has no screen: an id is
the same id in the next process, content is kept whole whatever its size, and a session cannot see
what another one kept. Those are asserted here; the stories assert what a model reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.documents import MEMORY_DOCUMENTS, TEXT_MEDIA, Documents, documents_path

if TYPE_CHECKING:
    from pathlib import Path


async def _opened(path: Path | str) -> Documents:
    """A store that is up, in a file of the test's own."""
    store = Documents(path)
    await store.open()
    return store


async def test_what_is_kept_comes_back_whole(tmp_path: Path) -> None:
    store = await _opened(tmp_path / "documents.db")
    try:
        kept = await store.put("main", "one\ntwo\n", summary="two lines")

        assert kept.id.startswith("d"), "a handle is a document, and says so"
        assert (kept.media, kept.chars, kept.summary) == (TEXT_MEDIA, 8, "two lines")
        found = await store.get("main", kept.id)
        assert found is not None
        assert found.data == b"one\ntwo\n", "text is kept as the bytes it encodes to, and comes back as they were"
        assert found.kept == kept
    finally:
        await store.close()


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


async def test_a_handle_outlives_the_process_that_made_it(tmp_path: Path) -> None:
    path = tmp_path / "documents.db"
    first = await _opened(path)
    kept = await first.put("main", "kept\n", summary="kept")
    await first.close()

    second = await _opened(path)
    try:
        found = await second.get("main", kept.id)
        assert found is not None and found.data == b"kept\n", "a restart is not a reason for a handle to stop working"
    finally:
        await second.close()


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


def test_the_bodies_live_beside_the_journal() -> None:
    assert documents_path("/data/ws-a/journal.db").name == "documents.db"
    assert documents_path("/data/ws-a/journal.db").parent.name == "ws-a", "one workspace's kept bodies stay with it"
    assert documents_path(MEMORY_DOCUMENTS) == MEMORY_DOCUMENTS, "a journal with no file has no directory to use"
