"""The fixtures the suite is handed: a person, a terminal, a fresh kernel, and a server to watch.

``TRANSPORTS`` comes from the driver, which takes it from the kernel itself: a story says nothing
about how it reaches the kernel, so every story runs once per transport the kernel can open, and
adding one adds the runs, and no suite (R2.1 D29).

``serve`` is here, because the two tests that watch a
route from the other side of a socket — ``test_route`` and ``test_config`` — are in different directories
and one of them is not under ``tests/test_kernel``. A web fetch has its own server in ``test_net``: it
answers ``GET``, which is a different handler and a different thing to watch.
"""

from __future__ import annotations

import os
import threading
from http.server import ThreadingHTTPServer
from typing import TYPE_CHECKING, cast

import pytest

from plyngent import admin, config
from plyngent.cli import line, profile
from plyngent.cli.offers import TERMINAL
from plyngent.kernel import journal, space
from plyngent_creative.model import ScriptedModel
from tests.loopback import Site, handler
from tests.stories.driver import TRANSPORTS, Story

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from composat import Declaration

    from plyngent.cli.announce import Announcements
    from plyngent.cli.offers import ClientTools
    from plyngent.cli.policy import Policy
    from plyngent.config import Settings
    from plyngent.kernel.lm.route import Route
    from plyngent.kernel.net.web import Engines
    from plyngent.kernel.prompt import PromptPlan
    from plyngent.kernel.pty.registry import Limits
    from plyngent.kernel.skills import Skills
    from plyngent.wire.ask import Grades
    from plyngent_creative.model import Step
    from tests.loopback import Site


@pytest.fixture(autouse=True)
def the_kernel_does_not_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``os.fork`` raise, for every test in the suite.

    The kernel hands its terminals to a process of its own (§9), and this is the assertion that it
    does: anything that forks from inside the kernel — directly, or through something it calls — fails
    on the call, and leaves no hazard to a warning nobody reads. The broker is a process of its
    own, so the one fork this design has still happens; it just does not happen in here.
    """

    def refuse(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("the kernel called os.fork(); terminals fork in the broker")

    monkeypatch.setattr(os, "fork", refuse)


@pytest.fixture(autouse=True)
def nothing_is_kept_in_the_persons_data_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point this program's own data directory at ``tmp_path``, for every test in the suite.

    A kernel with no journal named keeps the conversation under ``platformdirs.user_data_path("plyngent")``,
    which is the directory the program is *run* from — so one test that brings a kernel up without saying
    where its history goes writes a ledger into the environment of whoever ran the suite, and a suite run
    often enough fills that directory with throwaway trees nobody asked for.

    Every module that spells the directory is redirected here, because they are independent readers of one
    fact: the journal, the space pointers, the configuration, the client's own history and the marks it
    remembers. A module left out is a leak, and ``test_structure`` holds the list: every module that reads
    the directory is one of these.

    The answer is also what ``tmp_path`` is for, and it is returned so a test may assert about it.
    """
    home = tmp_path / "data"
    for module in (journal, space, config, profile, line):
        monkeypatch.setattr(module, "user_data_path", lambda name, home=home: home / name)
    monkeypatch.setattr(admin, "DEFAULT_DATA", home / "plyngent")
    return home / "plyngent"


@pytest.fixture(params=TRANSPORTS)
def transport(request: pytest.FixtureRequest) -> str:
    """Which transport this run of a story reaches the kernel over."""
    return cast("str", request.param)


@pytest.fixture
def serve() -> Iterator[Callable[[], Site]]:
    """A factory of servers on the loopback, each on a port the OS picks, all stopped at the end.

    The returned callable starts one and hands back the :class:`~tests.loopback.Site` that answers for it:
    what it says next, and what arrived — the path, the headers and the body, which is the only way "the
    credential became that header" can be an assertion, not a wish.
    """
    running: list[ThreadingHTTPServer] = []

    def start() -> Site:
        site = Site()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(site))
        server.daemon_threads = True
        site.port = int(server.server_address[1])
        running.append(server)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        return site

    try:
        yield start
    finally:
        for server in running:
            server.shutdown()
            server.server_close()


@pytest.fixture
async def story(tmp_path, transport):
    """A person in front of a kernel, in a workspace of their own.

    The kernel is the real one (`boot.install`) and the loop is the real one (`cli.app`); what the
    fixture supplies is the terminal, which is the one thing a test cannot have. Everything the story
    asserts about is outside the program: the lines it wrote, the questions it asked, the code it
    returned.

    Nothing is installed as the model, so this kernel refuses turns — which is its own story. A story
    about a turn says what the model answers, through :func:`story_with`.

    The journal is a file beside the workspace, because the kernel keeps one conversation per workspace
    by default and a test may not write into the person's data directory.
    """
    started = await Story.start(tmp_path / "workspace", transport=transport, journal=tmp_path / "journal.db")
    try:
        yield started
    finally:
        await started.close()


@pytest.fixture
async def story_with(tmp_path, transport):
    """A person in front of a kernel that asks a scripted model: one step per request, in order.

    A story names its replies, so what the model answers is part of the story, not a default
    somewhere: ``story = await story_with(text("hello"))``. Every kernel a test starts this way is closed
    with the fixture, whether or not the story got as far as stopping it.
    """
    started: list[Story] = []

    async def start(
        *script: Step,
        config: Settings | None = None,
        routes: Iterable[Route] | None = None,
        route: str = "",
        engines: Engines | None = None,
        skills: Skills | None = None,
        prompt: PromptPlan | None = None,
        tools: ClientTools = TERMINAL,
        pty: Limits | None = None,
        running: int | None = None,
        rounds: int | None = None,
        soft: bool = False,
        goals: int | None = None,
        depth: int | None = None,
        goal_depth: int | None = None,
        deadline: float | None = None,
        grades: Grades | None = None,
        policy: Policy | None = None,
        announce: Announcements | None = None,
        markdown: bool = False,
        mark: Path | None = None,
        once: str | None = None,
        components: Iterable[Declaration] = (),
    ) -> Story:
        root = tmp_path / f"workspace{len(started)}"
        journal = tmp_path / f"journal{len(started)}.db"
        if config is not None and config.journal is None:
            msg = (
                "a story that runs a settings value puts its journal in the file: without a [journal] path "
                "the kernel keeps one in the person's own data directory"
            )
            raise AssertionError(msg)
        story = await Story.start(
            root,
            transport=transport,
            model=ScriptedModel(script),
            config=config,
            routes=routes,
            route=route,
            engines=engines,
            skills=skills,
            prompt=prompt,
            journal=None if config is not None else journal,
            tools=tools,
            pty=pty,
            running=running,
            rounds=rounds,
            soft=soft,
            goals=goals,
            depth=depth,
            goal_depth=goal_depth,
            deadline=deadline,
            grades=grades,
            policy=policy,
            announce=announce,
            markdown=markdown,
            mark=mark,
            once=once,
            components=components,
        )
        started.append(story)
        return story

    try:
        yield start
    finally:
        for opened in started:
            await opened.close()
