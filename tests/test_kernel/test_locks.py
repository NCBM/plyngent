"""One lock per path: callers that name the same file take turns, and a read shares it with a read.

These are the only exports of :mod:`plyngent.kernel.locks`, asserted as behaviour rather than as a shape:
the properties below are what the rest of the kernel is allowed to assume, and they are also what a
batch's graph has to agree with — the graph decides what may run, this is what holds.
"""

from __future__ import annotations

import asyncio

from plyngent.kernel.locks import Locks


async def test_two_callers_of_one_path_take_turns() -> None:
    locks = Locks()
    order: list[str] = []

    async def one(tag: str, path: str) -> None:
        async with locks.hold(writes=[path]):
            order.append(f"{tag}-in")
            await asyncio.sleep(0)  # the other caller's chance to cut in, which it must not get
            order.append(f"{tag}-out")

    await asyncio.gather(one("a", "/w/x"), one("b", "/w/x"))

    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    ), "one holder at a time, and nothing in between"


async def test_two_readers_of_one_path_are_in_the_block_together() -> None:
    locks = Locks()
    order: list[str] = []

    async def one(tag: str, path: str) -> None:
        async with locks.hold(reads=[path]):
            order.append(f"{tag}-in")
            await asyncio.sleep(0)
            order.append(f"{tag}-out")

    await asyncio.gather(one("a", "/w/x"), one("b", "/w/x"))

    assert order == ["a-in", "b-in", "a-out", "b-out"], "neither read changes the file, so neither waits"


async def test_a_writer_waits_for_a_reader_and_a_reader_for_a_writer() -> None:
    locks = Locks()
    order: list[str] = []

    async def reader(tag: str, path: str) -> None:
        async with locks.hold(reads=[path]):
            order.append(f"{tag}-in")
            await asyncio.sleep(0)
            order.append(f"{tag}-out")

    async def writer(tag: str, path: str) -> None:
        async with locks.hold(writes=[path]):
            order.append(f"{tag}-in")
            await asyncio.sleep(0)
            order.append(f"{tag}-out")

    await asyncio.gather(writer("w", "/w/x"), reader("a", "/w/x"), reader("b", "/w/x"))

    assert order == ["w-in", "w-out", "a-in", "b-in", "a-out", "b-out"], (
        "the writer went alone, and the readers behind it shared the path once it left"
    )


async def test_a_reader_behind_a_waiting_writer_lets_it_through() -> None:
    locks = Locks()
    order: list[str] = []
    holding = asyncio.Event()

    async def first() -> None:
        async with locks.hold(reads=["/w/x"]):
            holding.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            order.append("first-out")

    async def writer() -> None:
        await holding.wait()
        async with locks.hold(writes=["/w/x"]):
            order.append("writer-in")

    async def late_reader() -> None:
        await holding.wait()
        async with locks.hold(reads=["/w/x"]):
            order.append("late-in")

    await asyncio.gather(first(), writer(), late_reader())

    assert order == ["first-out", "writer-in", "late-in"], (
        "the writer was already waiting when the second reader arrived, so it went first"
    )


async def test_two_callers_of_different_paths_do_not_wait() -> None:
    locks = Locks()
    order: list[str] = []

    async def one(tag: str, path: str) -> None:
        async with locks.hold(writes=[path]):
            order.append(f"{tag}-in")
            await asyncio.sleep(0)
            order.append(f"{tag}-out")

    await asyncio.gather(one("a", "/w/a"), one("b", "/w/b"))

    assert order == ["a-in", "b-in", "a-out", "b-out"], "both were in the block at once"


async def test_two_paths_are_one_acquisition_in_a_stable_order() -> None:
    """Two callers wanting ``{a, b}`` and ``{b, a}`` cannot deadlock: both take them alphabetically."""
    locks = Locks()
    finished: list[str] = []

    async def one(tag: str, writes: list[str], reads: list[str]) -> None:
        async with locks.hold(reads=reads, writes=writes):
            await asyncio.sleep(0)
            finished.append(tag)

    await asyncio.wait_for(
        asyncio.gather(
            one("a", ["/w/a"], ["/w/b"]),
            one("b", ["/w/b"], ["/w/a"]),
        ),
        timeout=5.0,
    )

    assert sorted(finished) == ["a", "b"], "one wrote what the other read, and each took them in key order"


async def test_a_lock_nobody_wants_is_forgotten() -> None:
    locks = Locks()

    async with locks.hold(writes=["/w/a"], reads=["/w/b"]):
        assert locks.count == 2, "a held path has a lock"
    assert locks.count == 0, "and a path nobody holds or wants keeps none"
