"""The seam through which the kernel asks somebody outside it a question.

The kernel does not read stdin, does not print, and does not import the CLI: asking a human is a
*question* that leaves as a value and comes back as an answer, and a turn asks over the connection that
started it (``TurnCtx.asker``). What is fixed here is the shape of that round trip, not how it travels —
the values themselves are the contract's (:mod:`plyngent.wire.ask`), and how they reach a person is the
link's business (:mod:`plyngent.kernel.link`).

Four rules the shape is built around:

* **Four verbs, because there are four questions.** :meth:`Human.confirm` asks yes or no,
  :meth:`~Human.choose` offers options, :meth:`~Human.text` takes a line, :meth:`~Human.secret` takes
  a line that is not echoed. A caller that wanted "a line, but maybe secretly" would have to decide
  which it meant anyway; saying it in the verb is what makes the answer's type follow from the call.
  :meth:`~Human.put` is those four taken as a **value** — the question as it arrives from a tool — and
  is written once here in terms of them, so that no channel and no caller has to match on the kinds.
* **An answer carries what the question asked for.** ``Agreed`` holds a bool, ``Picked`` the chosen
  value, ``Written`` the text — and ``None`` in the last two is "nobody wrote anything", which is a
  different thing from the empty string.
* **Asking is the tool layer's job, never a policy's.** A policy answers *whether* a human has to be
  asked (:class:`~plyngent.kernel.command.CommandPolicy`, the PTY policy, the workspace's grants);
  the tool that was refused asks, and records the answer back into the policy. A policy that asked
  would be a callback into the CLI, which is the seam this package does not have.
* **Nobody to ask is not the same as an unanswered question.** :class:`NoSink` answers immediately
  and says ``by=NOBODY``: a confirmation is refused whatever its level, the rest are unset. That is
  what a headless run means, and it is why a level's silence fallback is not reachable there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from composat import Component

from plyngent.wire.ask import (
    Agreed,
    AnyAnswer,
    AnyAsk,
    AskSite,
    By,
    Confirmation,
    FreeText,
    Picked,
    Secret,
    Selection,
    Severity,
    Written,
)

if TYPE_CHECKING:
    from plyngent.wire.event import AnyEvent


class Human(ABC):
    """Where a question goes, and how the answer comes back.

    ``emit`` is synchronous and non-blocking by contract: a renderer that is slow, or gone, must not
    be able to stall a step. The four question verbs are the one place the kernel waits on a person,
    and each waits with a deadline, because an unanswered question is a stuck turn.

    ``site`` is where the question belongs, and the caller is the one who knows it: a tool runs under
    a turn and passes that turn's place, so a frontend holding several sessions can show the question
    under the right one. It stays an argument, because the channel is one per host while the place is one
    per call.

    ``timeout`` is how long *this* question is worth waiting for; ``None`` is the connection's own
    deadline. It is an argument for the same reason ``site`` is: "wait a minute" is the caller's
    promise, and the channel is one per host, whatever the number of promises.

    ``severity`` is only on :meth:`confirm`, because that is the one question whose silence means
    something: the level says how long it waits and what an unanswered one becomes, and the channel
    passes it on without deciding either.
    """

    @abstractmethod
    def emit(self, event: AnyEvent) -> None:
        """Hand an event outward, without waiting for anything to happen to it."""

    @abstractmethod
    async def confirm(
        self,
        *,
        prompt: str,
        default: bool | None = None,
        severity: Severity = Severity.CONFIRM,
        executable_code: str = "",
        site: AskSite | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — the caller's deadline for the question, not a wait of this call's
    ) -> Agreed:
        """Ask yes or no; ``default`` is the answer suggested to the human, if any.

        Silence is not consent unless the level says it is: a ``routine`` question falls back to yes
        when its deadline arrives, and the two heavier levels never do.

        ``executable_code`` is the code the person is about to allow, when what is being allowed is code:
        it travels with the question because the channel that shows it decides nothing about it.
        """

    @abstractmethod
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        default: str | None = None,
        multiple: bool = False,
        site: AskSite | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — the caller's deadline for the question, not a wait of this call's
    ) -> Picked:
        """Offer ``options`` and take one — or, with ``multiple``, any number of them.

        ``default`` is a suggested line, not an extra option. Which of the two ways a pick comes back is
        the question's own (:attr:`~plyngent.wire.ask.Selection.multiple`), so it travels with the
        question, and whoever shows the question decides nothing about it.
        """

    @abstractmethod
    async def text(
        self,
        *,
        prompt: str,
        default: str | None = None,
        site: AskSite | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — the caller's deadline for the question, not a wait of this call's
    ) -> Written:
        """Take a line of prose; the empty string is an answer and ``None`` is the absence of one."""

    @abstractmethod
    async def secret(
        self,
        *,
        prompt: str,
        site: AskSite | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — the caller's deadline for the question, not a wait of this call's
    ) -> Written:
        """Take a line the client must not echo, and never offer one back as a suggestion.

        The value goes back in the answer and straight to where it was needed; it is never carried
        into an event, a result or a journal (R2.1 D6).
        """

    async def put(self, question: AnyAsk, *, site: AskSite | None = None) -> AnyAnswer:
        """Put whichever of the four this value is, and answer what came back.

        The value form of the four verbs, because a caller can already be holding a question — a tool
        reaches this through :meth:`~plyngent.kernel.env.Env.ask`. It is concrete here so that every
        channel — the connection's, the headless one, a test double — answers it by implementing the same
        four verbs and nothing else.

        ``timeout`` is the question's own (:attr:`~plyngent.wire.ask.Ask.timeout`), not an argument:
        how long a question is worth waiting for is part of the question, and a second parameter could
        disagree with it.
        """
        match question:
            case Confirmation():
                return await self.confirm(
                    prompt=question.prompt,
                    default=question.default,
                    severity=question.severity,
                    executable_code=question.executable_code,
                    site=site,
                    timeout=question.timeout,
                )
            case Selection():
                return await self.choose(
                    prompt=question.prompt,
                    options=question.options,
                    default=question.default,
                    multiple=question.multiple,
                    site=site,
                    timeout=question.timeout,
                )
            case Secret():
                return await self.secret(prompt=question.prompt, site=site, timeout=question.timeout)
            case FreeText():
                return await self.text(
                    prompt=question.prompt, default=question.default, site=site, timeout=question.timeout
                )


class NoSink(Human):
    """Nobody can be asked, so nothing is agreed to and nothing is entered.

    This is what a headless run and ``--yes`` mean: the questions still have answers
    (:func:`plyngent.wire.ask.unanswered`), so the tool that wanted to ask reports that it was
    blocked and does not hang. Every answer says ``NOBODY``: nothing was waited for, and a level's
    fallback is an answer to a deadline arriving, not to there being nobody.
    """

    @override
    def emit(self, event: AnyEvent) -> None:
        """Drop it: there is no one to show it to."""

    @override
    async def confirm(
        self,
        *,
        prompt: str,
        default: bool | None = None,
        severity: Severity = Severity.CONFIRM,
        executable_code: str = "",
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Agreed:
        """Answer "no", with nothing attached and nobody named."""
        _ = (prompt, default, severity, executable_code, site, timeout)
        return Agreed(by=By.NOBODY)

    @override
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        multiple: bool = False,
        default: str | None = None,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Picked:
        """Answer "nothing was picked"."""
        _ = (prompt, options, default, multiple, site, timeout)
        return Picked(by=By.NOBODY)

    @override
    async def text(
        self,
        *,
        prompt: str,
        default: str | None = None,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Written:
        """Answer "nothing was written"."""
        _ = (prompt, default, site, timeout)
        return Written(by=By.NOBODY)

    @override
    async def secret(
        self,
        *,
        prompt: str,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Written:
        """Answer "nothing was written", and offer nothing to suggest it."""
        _ = (prompt, site, timeout)
        return Written(by=By.NOBODY)


class HumanChannel(Component, Human):
    """A host's way out to everyone watching it, and the one place a frontend joins that audience.

    Two things travel through here and they are no longer one thing:

    * **events fan out.** Everything this host says goes to every connection watching it, because a
      second frontend on one tree is a second view of one conversation and not a competitor for it
      (E154). Watching is :meth:`join` and :meth:`leave`, which is a set — a connection joins while
      another is already there, and a replace that had to read the current value and put it back would be
      a race the framework answers by refusing the second client
      (:class:`~composat.errors.KeyInTransitError`, which is what this used to be).
    * **a question with no turn behind it goes to one watcher.** A turn's own questions never come
      through here: a turn carries the connection that started it (``TurnCtx.asker``, E156), so a
      frontend is asked by its own work, not by whoever attached last. What is left is the lane a
      stimulus uses, and one answer is owed to it, so it is asked of the first watcher in the order the
      watchers joined.

    Nobody watching is not the same as no channel. A question gets an answer, and the answer is "no"
    (:class:`NoSink`), so a tool reports that it was blocked, and no tool hangs — the same answer a
    headless run gives. A question already out when its client went away is refused by the transport
    that carried it (:meth:`plyngent.kernel.link.Link.close`), and the turn it belongs to runs on: a
    client disappearing is a refusal, not a verdict (R2.1 D14).
    """

    #: Everybody watching this host, in the order they joined: a dict because one connection joins once
    #: and the order is what makes "the first watcher answers" a stable answer.
    watchers: dict[Human, None]

    def __init__(self) -> None:
        super().__init__()
        self.watchers = {}

    def join(self, watcher: Human) -> None:
        """Add one connection to the audience: what it hears from here on is this host's events."""
        self.watchers[watcher] = None

    def leave(self, watcher: Human) -> None:
        """Take one connection out; one that never joined leaves nothing behind."""
        _ = self.watchers.pop(watcher, None)

    @override
    def emit(self, event: AnyEvent) -> None:
        """Hand an event to every watcher, or drop it when there is none."""
        for watcher in tuple(self.watchers):
            watcher.emit(event)

    @override
    async def confirm(
        self,
        *,
        prompt: str,
        default: bool | None = None,
        severity: Severity = Severity.CONFIRM,
        executable_code: str = "",
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Agreed:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().confirm(
            prompt=prompt,
            default=default,
            severity=severity,
            executable_code=executable_code,
            site=site,
            timeout=timeout,
        )

    @override
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        multiple: bool = False,
        default: str | None = None,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Picked:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().choose(
            prompt=prompt, options=options, default=default, multiple=multiple, site=site, timeout=timeout
        )

    @override
    async def text(
        self,
        *,
        prompt: str,
        default: str | None = None,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Written:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().text(prompt=prompt, default=default, site=site, timeout=timeout)

    @override
    async def secret(
        self,
        *,
        prompt: str,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Written:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().secret(prompt=prompt, site=site, timeout=timeout)

    def _human(self) -> Human:
        """Who answers a question with no turn behind it: the first watcher, or the answer that is always no."""
        for watcher in self.watchers:
            return watcher
        return NoSink()
