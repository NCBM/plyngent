"""What a tool call comes back with, for the three parties that read it.

A result is read by the model, by the person watching the terminal, and by the journal. One string
cannot serve all three: the model wants the useful part and nothing else, the terminal wants fields
it can lay out, and the journal wants what actually changed. So a result is a value with those
parts named, and nobody parses anybody else's prose.

Three rules the shape is built around:

* **`status`, `summary` and the retry are made together.** The same failure reads the same way every
  time it happens, because it is rendered from one closed set (:class:`~plyngent.kernel.failure.
  Failure`) through one table, and nowhere spelled out at each raise site.
* **A next step is data, not a sentence.** ``Hint`` names a tool and the arguments that fix the
  problem, and it is a part of the body so that the model reads it beside what went wrong; the
  renderer decides whether that becomes a phrase, and the kernel never writes a tool name into a
  message about a tool it cannot see.
* **Cancelling is not a failure of the world.** A turn that was stopped reports ``LIMIT`` and is not
  retried, so the journal can tell "it was stopped" from "it went wrong".

A body that is too long to send is not thrown away and not cut in place: it goes to the document
store and the result carries a handle plus the summary that stands in for it. That handle is also
what makes programmatic tool calling cheap — a program can pull the parts it needs, and the whole
thing never enters the conversation, so one reader serves both the tool that truncated and the
program that is curious.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from msgspec import Struct

from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.workspace import OutsideWorkspace, PathRefused
from plyngent.wire.ask import Agreed, By

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class Status(StrEnum):
    """How a call ended, in the four ways that are not the same kind of thing.

    ``refused`` never ran: policy said no, so nothing happened and nothing is owed.
    ``failed`` ran and could not finish. ``partial`` did some of what it was asked and then stopped —
    the effects that happened still happened, which is why they are recorded separately.
    """

    OK = "ok"
    REFUSED = "refused"
    FAILED = "failed"
    PARTIAL = "partial"


class DenialKind(StrEnum):
    """Why a call that never ran was refused, as one word (§0.15, E118).

    A refusal used to be one word plus prose, so a frontend could not tell "the person said no" from
    "the deadline passed" without parsing English. The answers a question can get give four of these,
    and the fifth is the rule this build has that no question and no grant lifts.

    It is not the whole of refusal, and it is not meant to be: a path outside the root and a budget that
    was reached are ``BLOCKED`` and ``LIMIT`` with a hint attached, because those are fixable and a
    fixable refusal is not a denial. ``RULE`` is the one denial that is still fixable — a rule of this
    build refusing a call the message says how to change — and ``NEVER_ALLOWED`` is the one no argument
    and no question lifts.
    """

    HUMAN = "human"
    POLICY = "policy"
    TIMEOUT = "timeout"
    NOBODY = "nobody"
    RULE = "rule"
    NEVER_ALLOWED = "never-allowed"


#: What each kind says, with ``{about}`` the thing that was refused and ``{policy}`` the policy's name.
#:
#: One table because the word and the sentence are the same decision read by two readers: a model reads
#: the sentence, and a client, a benchmark or a log reads the kind.
DENIAL_SAYS: Mapping[DenialKind, str] = {
    DenialKind.HUMAN: "the human did not allow {about}",
    DenialKind.POLICY: "this program's {policy} policy did not allow {about}",
    DenialKind.TIMEOUT: "nobody answered in time, so {about} did not happen",
    DenialKind.NOBODY: "there was nobody to ask, so {about} did not happen",
    DenialKind.RULE: "{about} is refused by this build's own rules",
    DenialKind.NEVER_ALLOWED: "{about} is on a list this build never allows",
}


def kind_of(answer: Agreed) -> DenialKind:
    """Which refusal an answer amounts to when it is not a yes: its author is the kind."""
    match answer.by:
        case By.HUMAN:
            return DenialKind.HUMAN
        case By.POLICY:
            return DenialKind.POLICY
        case By.TIMEOUT:
            return DenialKind.TIMEOUT
        case By.NOBODY:
            return DenialKind.NOBODY


def denial_says(kind: DenialKind, *, about: str, policy: str = "") -> str:
    """The sentence a refusal of this kind says; ``about`` is what was refused, in the caller's words."""
    return DENIAL_SAYS[kind].format(about=about, policy=policy or "approval")


#: What a status and a retry are, given the kind of failure — the one place either is decided.
#:
#: ``ARGS`` fails, because ``refused`` means "you may not", and "you wrote it wrong" is a different
#: answer. ``BUSY`` and ``TIMEOUT`` are the two the world can fix by itself; everything else asks for
#: different arguments or for nothing at all.
OUTCOMES: dict[Failure, tuple[Status, Retry]] = {
    Failure.ARGS: (Status.FAILED, Retry.FIX_ARGS),
    Failure.DENIED: (Status.REFUSED, Retry.NEVER),
    Failure.BLOCKED: (Status.REFUSED, Retry.FIX_ARGS),
    Failure.MISSING: (Status.FAILED, Retry.FIX_ARGS),
    Failure.PERMISSION: (Status.FAILED, Retry.NEVER),
    Failure.BUSY: (Status.FAILED, Retry.SAME_CALL),
    Failure.LIMIT: (Status.FAILED, Retry.FIX_ARGS),
    Failure.TIMEOUT: (Status.FAILED, Retry.SAME_CALL),
    Failure.PROTOCOL: (Status.FAILED, Retry.FIX_ARGS),
    Failure.NEXT_TURN: (Status.REFUSED, Retry.SAME_CALL),
    Failure.CLIENT: (Status.FAILED, Retry.NEVER),
    Failure.BUG: (Status.FAILED, Retry.NEVER),
}


class Part(Struct, frozen=True, kw_only=True, tag_field="part"):
    """One piece of a result's body, in the parts that are read differently.

    A body is several parts, because three readers want different things out of it and one string cannot
    be all of them: a program's output is fields somebody wants to lay out
    (exit code, whether it timed out, which argv), a document is a handle, and prose is prose.
    ``kernel/render.py`` turns any of them into the one text the model reads — that projection happens
    in exactly one place, so "what the model sees" can be tested on its own.
    """


class Text(Part, tag="text", frozen=True, kw_only=True):
    """A body small enough to send as it is."""

    value: str = ""


class Document(Part, tag="document", frozen=True, kw_only=True):
    """A body kept outside the conversation, with the summary that stands in for it.

    ``document_id`` is what a continuation reads from: the same handle a truncated result hands back,
    so there is one thing to resume from and one reader that resumes it. The identifier promises only
    that it locates the content — not its length, its order, its readability or its lifetime. Those
    differences are what ``read_document`` / ``search_document`` / ``document_info`` exist to hide.
    """

    document_id: str = ""
    summary: str = ""
    chars: int = 0


class ProcessControl(Part, tag="process", frozen=True, kw_only=True):
    """One command a batch actually started, as the fields that are not prose.

    ``errout`` is err+out: when the caller asked for the two streams to be merged at the OS level
    there is only one of them, and it is this field, so nothing downstream has to guess whether
    ``stderr`` was empty because the program was quiet or because it was never separate.

    ``depth`` and ``via`` are how it got here — ``then`` / ``else`` / ``pipe`` of an earlier step —
    which is the only way the model can tell a nested command from a top-level one in a flat list.
    """

    argv: tuple[str, ...] = ()
    cwd: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    fed: bool = False
    depth: int = 0
    via: str = ""
    mix_stderr: bool = False
    #: Whether the deadline ended this one and something survived the kill — a process that left the group
    #: and still held the pipes, so this end of them was closed and anything it says from now on goes
    #: nowhere. Nothing portable can say whether it is still running (E161).
    escaped: bool = False
    stdout: str = ""
    stderr: str = ""
    errout: str = ""
    #: How many bytes went into the next program of this command's own chain. Those bytes are not in
    #: ``stdout`` — they were the next program's input — and the count is bytes rather than characters
    #: because a pipe carries bytes (§2 R6). A command with no pipe hands nothing on, so it says nothing.
    handed: int | None = None


#: A body is a sequence of parts; one part is a body of length one.
type Body = tuple[Part, ...]
#: What a caller may hand over where a body is wanted: one part, or several.
type BodyLike = Part | Iterable[Part]


def parts(body: BodyLike | None) -> Body:
    """One part or several, as the tuple a body always is."""
    if body is None:
        return ()
    if isinstance(body, Part):
        return (body,)
    return tuple(body)


class Hint(Part, tag="hint", frozen=True, kw_only=True):
    """A next step that would change the outcome, as data.

    A hint is a part of the body, because it is meant for the model and the body is what the model
    reads: a hint outside the body is a hint nobody can be given. So
    ``hints`` on a result is a filter over the body, and there is no second place one could hide.

    ``given`` is an *argument patch*: the values this failed call should have carried instead
    (``write_file(overwrite=true)``). It is not the tool's parameter list, it does not overwrite the
    arguments that were already given, and it does not promise to be a complete call — a hint either
    changes one value in this call or says nothing at all. That is why a tree-shaped package never
    has to be expressible here. Nothing else promises the step works: a hint is a suggestion.
    """

    tool: str = ""
    given: tuple[tuple[str, str], ...] = ()
    why: str = ""


@dataclass(frozen=True, slots=True)
class Effects:
    """What the call changed, recorded as it happens.

    ``changed`` is not "these files were written": it is "these paths are no longer what you last
    saw", which is why a delete and a move's source are in it too — a journal built from this alone
    must not make a deleted file look like one that never existed.

    A program that runs several tool calls and then fails still moved those files and started those
    children; a result assembled only at the end would lose them.
    """

    changed: tuple[Path, ...] = ()
    external: tuple[str, ...] = ()
    approvals: tuple[str, ...] = ()
    #: The version these bytes were kept under, or empty when nothing was kept — a host with no store,
    #: a call that changed no path, or a store that could not write. It is filled in by
    #: :class:`~plyngent.kernel.executor.ToolExecutor` once the call has returned, because the version
    #: has to be taken while the paths are still the ones this call left (the executor's own lock is
    #: what says so) and a handler cannot know it exists.
    revision: str = ""
    #: How many files the tree held when this call's own paths could not be found *and* it was too big to
    #: compare, or ``0`` when nothing was given up on. It is the other half of the empty ``revision``:
    #: "there was nothing to keep" and "this call was too big to look at" read the same without it.
    unswept: int = 0


@dataclass(frozen=True, slots=True)
class Skips:
    """How much a listing kept back, split the way the cost of knowing it differs.

    Hidden *files* can be counted exactly: one entry, one directory read. Hidden *directories* can
    only be counted as directories — what is inside one is unknown without walking into it, which is
    exactly what hiding it was meant to avoid. So the contents of a skipped directory are in neither
    number, and "2 dirs, 3 files" is not a count of everything under the base.
    """

    dirs: int = 0
    files: int = 0


@dataclass(frozen=True, slots=True)
class Meta:
    """What the call cost and how much it left out, for the log and not for the model."""

    elapsed_ms: int = 0
    chars: int = 0
    truncated: bool = False
    skipped: Skips = field(default_factory=Skips)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One call's outcome: a status, one line, and the parts that are not the line.

    There is no ``hints`` field, and no ``hints`` argument on the constructor either: a hint is a part
    of the body (it is meant for the model, and the body is what the model reads), so the two factories
    that take hints put them there and :attr:`hints` filters the body back out. A part therefore lives
    in exactly one place, which is what lets ``render`` be the only thing that decides what any of it
    means.
    """

    status: Status
    summary: str
    body: Body = ()
    effects: Effects = field(default_factory=Effects)
    meta: Meta = field(default_factory=Meta)
    failure: Failure | None = None
    #: Why, when the failure is :attr:`~plyngent.kernel.failure.Failure.DENIED` — and never otherwise:
    #: the two are one fact told to two readers, so a result that has one and not the other is a bug
    #: (`__post_init__` refuses it), so nobody classifies it by reading a sentence.
    denial: DenialKind | None = None

    def __post_init__(self) -> None:
        stopped = self.status in {Status.REFUSED, Status.FAILED}
        if stopped and self.failure is None:
            msg = f"{self.status}: a result that did not finish has to say what kind of failure it was"
            raise ValueError(msg)
        if not stopped and self.failure is not None:
            msg = f"{self.status}: a result that finished cannot also carry {self.failure}"
            raise ValueError(msg)
        if (self.failure is Failure.DENIED) != (self.denial is not None):
            msg = "a denial is a kind, not a sentence: Failure.DENIED and DenialKind are set together"
            raise ValueError(msg)
        object.__setattr__(self, "body", parts(self.body))
        if self.meta.chars == 0 and self.body:
            # ``chars`` is the length of what the model will read, not of what the tool produced, so
            # it is measured through the projection. The import is
            # local because the projection speaks of these parts: it is one direction or the other,
            # and this is the direction that keeps the renderer free to grow.
            from plyngent.kernel.render import render_body

            object.__setattr__(self, "meta", replace(self.meta, chars=len(render_body(self.body))))

    @property
    def hints(self) -> tuple[Hint, ...]:
        """The hints this result carries — the ones in its body, in the order they were appended."""
        return tuple(part for part in self.body if isinstance(part, Hint))

    @property
    def retry(self) -> Retry:
        """What the caller should do next — from the table, never from the raise site."""
        return Retry.NEVER if self.failure is None else OUTCOMES[self.failure][1]

    @property
    def handles(self) -> tuple[str, ...]:
        """The documents this result's body kept, in the order they were laid out.

        A handle is what a request acts on later — a body too long to send is continued from it, and a
        picture is fetched through it — and the body's parts are where those handles live, so this is the
        one place that reads them out.
        """
        return tuple(part.document_id for part in self.body if isinstance(part, Document))

    @classmethod
    def ok(
        cls,
        summary: str,
        *,
        body: BodyLike | None = None,
        effects: Effects | None = None,
        meta: Meta | None = None,
    ) -> ToolResult:
        """The call did what it was asked."""
        return cls(
            Status.OK,
            summary,
            body=parts(body),
            effects=effects if effects is not None else Effects(),
            meta=meta if meta is not None else Meta(),
        )

    @classmethod
    def of(
        cls,
        failure: Failure,
        summary: str,
        *,
        body: BodyLike | None = None,
        hints: tuple[Hint, ...] = (),
        effects: Effects | None = None,
        meta: Meta | None = None,
    ) -> ToolResult:
        """The call did not finish in the way it was meant to, for *failure*'s reasons.

        The status is the table's, so a tool never decides whether its trouble counts as a refusal.
        """
        return cls(
            OUTCOMES[failure][0],
            summary,
            body=parts(body) + hints,
            effects=effects if effects is not None else Effects(),
            meta=meta if meta is not None else Meta(),
            failure=failure,
        )

    @classmethod
    def partial(
        cls,
        summary: str,
        *,
        effects: Effects,
        body: BodyLike | None = None,
        hints: tuple[Hint, ...] = (),
        meta: Meta | None = None,
    ) -> ToolResult:
        """Some of it landed and the rest did not; ``effects`` says which.

        A partial result is never retried as a whole: running it again would do the parts that worked
        a second time. ``hints`` says where to continue instead.
        """
        return cls(
            Status.PARTIAL,
            summary,
            body=parts(body) + hints,
            effects=effects,
            meta=meta if meta is not None else Meta(),
        )

    @classmethod
    def denied(
        cls,
        kind: DenialKind,
        summary: str,
        *,
        body: BodyLike | None = None,
        hints: tuple[Hint, ...] = (),
        effects: Effects | None = None,
        meta: Meta | None = None,
    ) -> ToolResult:
        """The call was refused and never ran; ``kind`` says by what.

        The only door to :attr:`~plyngent.kernel.failure.Failure.DENIED`, which is why the kind cannot be
        forgotten: every refusal in this tree comes through here, and a bare ``DENIED`` with a sentence
        beside it is a shape the result itself refuses to be. The sentence stays the caller's, because a
        refusal has to name what was refused and only the caller knows which.
        """
        return cls(
            Status.REFUSED,
            summary,
            body=parts(body) + hints,
            effects=effects if effects is not None else Effects(),
            meta=meta if meta is not None else Meta(),
            failure=Failure.DENIED,
            denial=kind,
        )

    @classmethod
    def unagreed(cls, answer: Agreed, *, about: str) -> ToolResult:
        """A refusal built from a question's answer: the author of the answer is the kind of refusal.

        ``about`` is what was asked for — ``"this call"``, ``"running 'rm'"`` — and the sentence is
        :data:`DENIAL_SAYS`'. A question whose silence is a yes never gets here, so silence reaching this
        point was a no.
        """
        kind = kind_of(answer)
        return cls.denied(kind, denial_says(kind, about=about, policy=answer.policy))


def refusal(exc: PathRefused) -> ToolResult:
    """A path refusal as a result: refused, and — for the liftable kind — what would lift it.

    The tools travel as hints, never inside the summary: the kernel does not name tools in prose, and
    a caller that composes the message can decide how loud to make the suggestion. ``Denied`` says
    nothing about tools, because nothing lifts it — that refusal carries a ``never-allowed`` kind, while
    an ``OutsideWorkspace`` comes with hints.
    """
    if isinstance(exc, OutsideWorkspace):
        return ToolResult.of(Failure.BLOCKED, str(exc), hints=hints_for(exc))
    return ToolResult.denied(DenialKind.NEVER_ALLOWED, str(exc))


def hints_for(exc: PathRefused) -> tuple[Hint, ...]:
    """What would lift a refusal, as data: which tool, and what a refusal can fill in for it.

    Nothing for a denial, because nothing lifts it; the tools come off the refusal with their patches
    already attached, which is what keeps a hint from naming an argument its tool does not have.
    """
    if not isinstance(exc, OutsideWorkspace):
        return ()
    return tuple(Hint(tool=lift.tool, given=lift.given) for lift in exc.related_tools)
