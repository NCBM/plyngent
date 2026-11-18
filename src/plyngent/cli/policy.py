"""The questions this client answers by itself, and what it leaves on the screen for them (§0.15).

A person who started the program with ``--yolo`` or ``--away`` has said what to do about being
interrupted, and that is a fact about *this* frontend rather than about the kernel: nothing below
``cli/`` knows either word, and a kernel that did would be a kernel with a mode. What a policy may
answer is deliberately narrow — a yes-or-no question, at a level it was told about, and never a
password — because everything else a question can ask is a person's words rather than a permission.
The one exception is a choice that already says what its own silence picks: answering it with that
option is answering it the way an empty line would, and a question with no default stays the person's
(E150).

Two names for two promises:

* ``--yolo`` answers the light and the ordinary question with a yes, and stops there. It is not a
  promise to run programs nobody was asked about.
* ``--away`` answers by level: the light question is allowed, the ordinary one is answered as its own
  silence would have answered it, and the heaviest is refused. ``--away-danger`` is a second switch
  that allows the heaviest too, because "I am not here" and "I am not here and you may run anything"
  are different things to have said — and it says so once when the program starts.

Both may be on at once, and then the first one with an answer gives it: ``--yolo`` is consulted first
because it is the one that says yes. Whatever a policy answers, the screen says so — ``away: refused``
— because a program that answers questions a person did not see has to be a program that says which
ones it answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from plyngent.wire.ask import SILENCE, Agreed, AnyAsk, By, Confirmation, Picked, Selection, Severity

if TYPE_CHECKING:
    from plyngent.wire.ask import AnyAnswer

#: What each flag is called in the record: the answer carries one of these, so "which policy said yes
#: to a ``danger`` question?" is a question to the record rather than an inference.
YOLO = "yolo"
AWAY = "away"

#: The second away switch as a *mode* rather than as an answer: a question it answers is still answered
#: by ``away``, but a run that was started with it is a different run to be in, and the status line says
#: which one this is.
AWAY_DANGER = "away-danger"


@dataclass(frozen=True, slots=True, kw_only=True)
class Answered:
    """One answer a policy gave: whether it agreed, which policy it was, and the option it picked.

    ``picked`` is filled only where the question was a choice and it named its own default — the one
    shape of a choice a policy may answer (E150). A yes-or-no question leaves it empty, and the answer
    is then the bool it always was.
    """

    accepted: bool
    policy: str
    picked: str = ""

    def reply(self, question: AnyAsk) -> AnyAnswer:
        """The answer to send, in the shape of the question it answers."""
        if isinstance(question, Selection):
            return Picked(id=question.id, value=self.picked or None, by=By.POLICY, policy=self.policy)
        return Agreed(id=question.id, accepted=self.accepted, by=By.POLICY, policy=self.policy)


@dataclass(frozen=True, slots=True)
class Policy:
    """Which of the kernel's questions this client answers itself, and in which direction."""

    yolo: bool = False
    away: bool = False
    away_danger: bool = False

    def answers(self, ask: AnyAsk) -> Answered | None:
        """What this client answers, or ``None`` when the person in front of the machine decides.

        A yes-or-no question is answered by level, and a secret never: what neither name knows is words,
        and both were promises about permission. A choice is answered only where the question already
        says what its own silence would pick — the option in front of the person is the same one an
        empty line gives, so answering it is not a policy choosing on somebody's behalf, which is why a
        choice with no default stays the person's (E150).
        """
        if isinstance(ask, Selection):
            return self._chosen(ask)
        if not isinstance(ask, Confirmation):
            return None
        return self._confirmed(ask)

    def _confirmed(self, ask: Confirmation) -> Answered | None:
        """A yes or a no, answered by the level the question says it is at — or not answered at all."""
        if self.yolo and ask.severity is not Severity.DANGER:
            return Answered(accepted=True, policy=YOLO)
        if not self.away:
            return None
        if ask.severity is Severity.DANGER:
            return Answered(accepted=self.away_danger, policy=AWAY)
        if ask.severity is Severity.CONFIRM:
            # "Answer it now, as its own silence would have": the two levels differ in what waiting
            # means, and an away policy is not a second opinion about that.
            return Answered(accepted=SILENCE[ask.severity], policy=AWAY)
        return Answered(accepted=True, policy=AWAY)

    def _chosen(self, ask: Selection) -> Answered | None:
        """The default of a choice, answered by whichever switch is on — or nobody's answer at all."""
        if ask.default is None or ask.default not in ask.options:
            return None
        if self.yolo:
            return Answered(accepted=True, policy=YOLO, picked=ask.default)
        if self.away:
            return Answered(accepted=True, policy=AWAY, picked=ask.default)
        return None

    @property
    def speaking(self) -> bool:
        """Whether anything is being answered here at all — a program with no policy says nothing."""
        return self.yolo or self.away

    @property
    def mode(self) -> str:
        """What to call this run's answering, for a status line: the switches, in the order they read.

        It is a word about *this* frontend and not about the kernel, which is why it is named here and
        why the screen may say it: a person who left ``--yolo`` on three hours ago should be able to see
        it without reading their own shell history.
        """
        named = [name for name, on in ((YOLO, self.yolo), (AWAY, self.away), (AWAY_DANGER, self.away_danger)) if on]
        return " ".join(named)

    def opening(self) -> str:
        """What the program says about itself when it is started with the loudest switch.

        Once, at the start, because a person who typed ``--away-danger`` needs to know it took effect
        — and it is the one line that says what every later ``away: allowed`` on a heavy question means.
        """
        return "away: the heaviest questions are answered with a yes too"

    def said(self, answered: Answered) -> str:
        """The line one answered question leaves: which policy, and which way it went."""
        return f"{answered.policy}: {'allowed' if answered.accepted else 'refused'}"


@dataclass
class Tally:
    """What the policies answered in this stretch, so the end of it can be one line rather than none.

    The lines above are what a person reads while it happens; this is what they read when it stops,
    because a run that answered forty questions a person did not see is a run whose shape is only
    visible as a count.
    """

    _said: dict[str, list[bool]] = field(default_factory=dict)

    def keep(self, answered: Answered) -> None:
        """Count one answer under the policy that gave it."""
        self._said.setdefault(answered.policy, []).append(answered.accepted)

    def said(self) -> str:
        """The one line that ends the stretch, or nothing at all when no policy answered anything."""
        parts = [
            f"{name}: {len(said)} answered ({sum(said)} allowed, {len(said) - sum(said)} refused)"
            for name, said in self._said.items()
        ]
        return "; ".join(parts)
