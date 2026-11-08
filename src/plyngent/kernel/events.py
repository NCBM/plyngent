"""The seam through which the kernel asks somebody outside it a question.

The kernel does not read stdin, does not print, and does not import the CLI: asking a human is a
*question* that leaves through ``Host.human`` and comes back as an answer. What is fixed here is the
shape of that round trip, not how it travels — the values themselves are the contract's
(:mod:`plyngent.wire.ask`), and how they reach a person is the link's business
(:mod:`plyngent.kernel.link`).

Four rules the shape is built around:

* **Four verbs, because there are four questions.** :meth:`Human.confirm` asks yes or no,
  :meth:`~Human.choose` offers options, :meth:`~Human.text` takes a line, :meth:`~Human.secret` takes
  a line that is not echoed. A caller that wanted "a line, but maybe secretly" would have to decide
  which it meant anyway; saying it in the verb is what makes the answer's type follow from the call.
* **An answer carries what the question asked for.** ``Agreed`` holds a bool, ``Picked`` the chosen
  value, ``Written`` the text — and ``None`` in the last two is "nobody wrote anything", which is a
  different thing from the empty string.
* **Asking is the tool layer's job, never a policy's.** A policy answers *whether* a human has to be
  asked (:class:`~plyngent.kernel.command.CommandPolicy`, the PTY policy, the workspace's grants);
  the tool that was refused asks, and records the answer back into the policy. A policy that asked
  would be a callback into the CLI, which is the seam this package does not have.
* **Nobody to ask is not the same as an unanswered question.** :class:`NoSink` answers immediately,
  and with :func:`plyngent.wire.ask.unanswered`'s values: a confirmation is refused, the rest are
  unset. That is what headless and ``--yes`` runs mean.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from composat import Component

from plyngent.wire.ask import Agreed, AskSite, Picked, Written

if TYPE_CHECKING:
    from plyngent.wire.event import AnyEvent


class Human(ABC):
    """Where a question goes, and how the answer comes back.

    ``emit`` is synchronous and non-blocking by contract: a renderer that is slow, or gone, must not
    be able to stall a step. The four question verbs are the one place the kernel waits on a person,
    and each waits with a deadline, because an unanswered question is a stuck turn.

    ``site`` is where the question belongs, and the caller is the one who knows it: a tool runs under
    a turn and passes that turn's place, so a frontend holding several sessions can show the question
    under the right one. It stays an argument rather than a field on the channel because the channel
    is one per host while the place is one per call.
    """

    @abstractmethod
    def emit(self, event: AnyEvent) -> None:
        """Hand an event outward, without waiting for anything to happen to it."""

    @abstractmethod
    async def confirm(self, *, prompt: str, default: bool | None = None, site: AskSite | None = None) -> Agreed:
        """Ask yes or no; ``default`` is the answer suggested to the human, if any.

        An unanswered question is a ``False``, so a question nobody answered can never be mistaken
        for consent.
        """

    @abstractmethod
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        default: str | None = None,
        site: AskSite | None = None,
    ) -> Picked:
        """Offer ``options`` and take one; ``default`` is a suggested line, not an extra option."""

    @abstractmethod
    async def text(self, *, prompt: str, default: str | None = None, site: AskSite | None = None) -> Written:
        """Take a line of prose; the empty string is an answer and ``None`` is the absence of one."""

    @abstractmethod
    async def secret(self, *, prompt: str, site: AskSite | None = None) -> Written:
        """Take a line the client must not echo, and never offer one back as a suggestion.

        The value goes back in the answer and straight to where it was needed; it is never carried
        into an event, a result or a journal (R2.1 D6).
        """


class NoSink(Human):
    """Nobody can be asked, so nothing is agreed to and nothing is entered.

    This is what a headless run and ``--yes`` mean: the questions still have answers
    (:func:`plyngent.wire.ask.unanswered`), so the tool that wanted to ask reports that it was
    blocked instead of hanging.
    """

    @override
    def emit(self, event: AnyEvent) -> None:
        """Drop it: there is no one to show it to."""

    @override
    async def confirm(self, *, prompt: str, default: bool | None = None, site: AskSite | None = None) -> Agreed:
        """Answer "no", with nothing attached."""
        _ = (prompt, default, site)
        return Agreed()

    @override
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        default: str | None = None,
        site: AskSite | None = None,
    ) -> Picked:
        """Answer "nothing was picked"."""
        _ = (prompt, options, default, site)
        return Picked()

    @override
    async def text(self, *, prompt: str, default: str | None = None, site: AskSite | None = None) -> Written:
        """Answer "nothing was written"."""
        _ = (prompt, default, site)
        return Written()

    @override
    async def secret(self, *, prompt: str, site: AskSite | None = None) -> Written:
        """Answer "nothing was written", and offer nothing to suggest it."""
        _ = (prompt, site)
        return Written()


class HumanChannel(Component, Human):
    """Where a question leaves the kernel, as a component so that attach and detach are one move.

    The host holds *this* object, never the client that answers: attaching is
    ``assembly.replace(key, Declaration(HumanChannel, args=(client,)))`` and detaching is the same call
    with nobody. Nothing has to be cleared afterwards, and a reader of ``Host.human`` never sees a
    half-attached state — the key answers the old generation until the new one is loaded.

    That is also where "one approval owner" comes from, without an election of ours: the framework
    refuses a second ``replace`` of a key another call is still moving (``KeyInTransitError`` naming
    what it is doing), so a second client is told the truth rather than quietly waiting its turn.

    Nobody attached is not the same as no channel. Questions still get an answer, and the answer is
    "no" (:class:`NoSink`), so a tool reports that it was blocked instead of hanging — the same answer
    a headless run gives. A question already out when its client went away is refused by the transport
    that carried it (:meth:`plyngent.kernel.link.InProcLink.close`), and the turn it belongs to runs
    on: a client disappearing is a refusal, not a verdict (R2.1 D14).
    """

    #: The client that answers, or ``None`` when nobody is attached.
    attached: Human | None

    def __init__(self, attached: Human | None = None) -> None:
        super().__init__()
        self.attached = attached

    @override
    def emit(self, event: AnyEvent) -> None:
        """Hand an event to the attached client, or drop it when there is none."""
        self._human().emit(event)

    @override
    async def confirm(self, *, prompt: str, default: bool | None = None, site: AskSite | None = None) -> Agreed:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().confirm(prompt=prompt, default=default, site=site)

    @override
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        default: str | None = None,
        site: AskSite | None = None,
    ) -> Picked:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().choose(prompt=prompt, options=options, default=default, site=site)

    @override
    async def text(self, *, prompt: str, default: str | None = None, site: AskSite | None = None) -> Written:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().text(prompt=prompt, default=default, site=site)

    @override
    async def secret(self, *, prompt: str, site: AskSite | None = None) -> Written:
        """Ask whoever is attached now, and wait for them within the deadline."""
        return await self._human().secret(prompt=prompt, site=site)

    def _human(self) -> Human:
        """Who answers: the attached client, or the answer that is always no."""
        return self.attached if self.attached is not None else NoSink()
