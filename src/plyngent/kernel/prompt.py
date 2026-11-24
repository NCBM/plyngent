"""The prompt a model is sent: five sections, in one order, and the last of them is about the tool set.

A request carries one string that is not the conversation, and until now that string was one line: which
tool groups this session has not been shown. It is still one string — a dialect knows how to put a system
message on its wire, and nothing here changes that — but it is made of parts, because a model told *how*
to use a tool uses it, and the how is not one sentence about what is missing.

The sections are fixed, in this order: persona, the tool playbook, one section per group whose tools this
session can reach, the skills catalog, and the index line that ends it.
The first two are defaults a program may replace — who the model is and how it is told to work are the
two things a different product changes without touching this tree — so the plan carries both as fields
and every section is the plan's answer, never a constant read at compose time.
The order is a constant and not a function of state, because the prompt is the front of a provider's
cached prefix (§10.5): two requests that differ only in what a turn did must differ only in what the turn
touched, and a section that moved because a dictionary iterated differently would be a miss we caused and
could not account for.

One of the sections is a fact about *how the program was started*, and it is read once, here, at install:
the skills this host can see. Discovery caches nothing — it re-reads its roots on every question, on
purpose — so a section computed per request would be a prefix that changes whenever somebody writes a
skill file, with no counter moving to say so. The price is that a skill made during a session joins the
catalog at the next start.

What is deliberately not a section: a workspace's ``AGENTS.md``. A file that happens to lie in the tree is
the model's to read, with the same ``read`` call as any other path, and only when it matters. The moment
the kernel read it for the model, the prompt would stop being what the program chose to say and become
whatever was lying in a directory somebody pointed a socket at — and nothing in the conversation would
show which of the two had spoken.

What is deliberately not here either: a section for MCP (there is no MCP — nothing to say), and any
per-request nag or reminder (a request-level sentence belongs to the request, and no writer in this tree
puts one in a history: the one exception is a compaction's summary, which continues the conversation as a
person's own turn because that is what it stands for).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.kernel.toolspec import Group
from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from plyngent.kernel.catalog import ToolView
    from plyngent.kernel.skills import Skills

#: Who the model is, and the first section. A program may install its own (``boot.install(prompt=…)``),
#: and an empty one is a program saying it has no persona — the section disappears, like every empty one.
PERSONA = "You are a professional coding agent in a workspace-bound tool environment."

#: The tool playbook that is always there: how to use the tools a session starts with, compressed to what
#: a call gets wrong. The groups' own sections arrive with their groups (§10.4, E64), so this is the part
#: that has to stand on its own for a conversation that has asked for nothing yet. It is also the default
#: for :attr:`PromptPlan.directives` — a program that says how its tools are to be used says it there.
CORE = """\
How to work here:
- Paths are inside the workspace unless a tool says otherwise, and a refusal says which call would lift
  it: change the argument it names rather than making the same call again.
- `read` shows a window of a file and numbers the lines; `edit_replace` and `edit_lineno` change what is
  already there, and `write_file` writes a whole file — it refuses to replace one without `overwrite=true`.
- `execute` takes an argv list and never a line for a shell to parse; a shell or an interpreter is a call
  a person has to agree to, so name the program you mean.
- The todo stack is the plan: `todo_push` / `todo_update` / `todo_pop`, where the numbers a listing prints
  are positions and not ids. `tool_require(group=…)` shows the groups the last line says are missing.
- `wait` is how a turn gives a person time: it asks for a line with a deadline and says which of the three
  things happened. Use it when the next step needs somebody to have read something, not to pass a moment.
- When one answer needs several calls that do not touch each other, ask for them together. A result too
  long to send is kept whole and handed back as a handle; read it rather than running the call again."""

#: The network section, written for the tools a kernel actually serves: a tool nobody has is not
#: described, and a name a model cannot read is a name it cannot use. ``engines`` is therefore both a
#: description and the value domain of ``web_search``'s argument — the only place a model learns them.
NETWORK_CLOSE = (
    "A denial is a decision rather than a transient failure — answer it by changing what you ask for, not "
    "by asking again."
)


def _network(served: Collection[str], engines: Sequence[str]) -> str:
    """What the network section says for one session's tools, and which engines a search may name.

    ``client_fetch`` asks from this machine and is always there; ``web_search`` and ``web_fetch`` are there
    only when an engine is, so a program configured without one gets a section about the tool it has. The
    engine names are listed in the order a search tries them, because that is what leaving the argument out
    does.
    """
    found = ["`client_fetch` asks from the person's own machine"]
    if "web_search" in served:
        found.append("`web_search` finds pages")
    if "web_fetch" in served:
        found.append("`web_fetch` reads one as the readable text somebody else's network made of it")
    said = f"The network: {'; '.join(found)}."
    if "web_search" in served and engines:
        said += (
            f" A search may say which engine answers — {', '.join(engines)} — and leaving that out asks them "
            "in that order, stopping at the first that answers."
        )
    return f"{said} {NETWORK_CLOSE}"


#: What each group's tools add, and the mistake each one invites. Filled for every :class:`Group`: a
#: group is a disclosure unit, so its playbook is a unit too — the section arrives when the group is shown
#: and with the tools it describes, which is what keeps the standing prompt at one compressed block, and
#: not 1.5k of instructions for calls a session cannot make yet (§10.4).
GROUP_PLAYBOOK: Mapping[Group, str] = {
    Group.DOCUMENT: """\
The rest of a document: a body too long for an answer is kept rather than cut, and `documents` lists what
is kept while `search` looks inside it. `read` takes a line window, so a big file is read in pieces.
`search` takes one file, a kept handle, or a directory: a directory is every file under it that the
rules show, and the answer ends with what it did not look at — a picture, a file too big to hold, one the
ignore rules keep back.""",
    Group.FS: """\
Finding things: `listdir` shows one directory, `tree` shows a subtree, and `glob_paths` finds by glob
under one directory. `copy_path`, `move_path` and `delete_path` change the tree; a delete goes to the
human's trash unless it is given `permanent=true`. A path outside the workspace is refused, and the
refusal names the two calls that lift it: `request_directory_access` asks a person for this
conversation, and `new_temporary_workspace` makes a scratch directory the host removes again.""",
    Group.META: """\
The tool set itself: the last line of this prompt says which groups this conversation has not been shown,
and `tool_require(group=…)` shows one. Ask for a group as soon as a task needs it rather than working
around it — a call naming a tool nobody was shown is refused, and the refusal is one wasted round.""",
    Group.PROCESS: """\
Running things: `execute` takes an argv list and never a line for a shell to parse, and what it runs
decides whether a person is asked — a shell or an interpreter always is, so name the program you mean.
An interactive program is a terminal: `open_pty` to start one, `read_pty` and `write_pty` to talk to it,
`write_pty_keys` for control sequences rather than their bytes, and `ask_into_pty` for a secret, which is
never echoed back into this conversation. A `pipe` feeds one command's output to the next, so what comes
back is the count of bytes it handed over: redirect into a file when you want to read them.""",
    Group.NET: _network(frozenset({"client_fetch"}), ()),
    Group.SESSION: """\
The plan a turn is worked through: `todo_push` opens a group of steps, `todo_update` moves one,
`todo_pop` closes a group, and `todo_list` shows the stack. The numbers a listing prints are positions in
that listing, not names, and they move when the stack does. Nothing is cleared for you: a finished group
stays until it is popped. A task that cannot be done now is `paused` (`todo_update` with that status): it
leaves the open ones alone, stays in the listing marked `[!]`, and the harness records it at the next turn
it starts itself — so ask once, and spend that turn on what else is open. A group holding a paused task
cannot be popped, and a stack holding one cannot be cleared, until it is `done` or `cancelled`: nobody
else can unlock it. `wait` is the other call here:
it hands the turn over to a person for a while and answers what they did — wrote a line, pressed enter, or
said nothing at all.""",
    Group.SKILLS: """\
Skills are instruction directories somebody installed for this machine. Read the one you mean with
`skill_read` before following it, and reach for `skill_list` or `skill_search` when the catalog above is
not enough. `skill_create` and `skill_edit` write outside the workspace, so every call asks a person. A
skill written during this conversation joins the catalog at the next start.""",
    Group.VCS: """\
Version control: ask the repository instead of a shell running the same program. This group answers what
the checkout is, what its state and history are, and what changed, in a shape this conversation can read,
and none of its calls writes.""",
}

#: What a compaction is asked for, when the program did not install its own instruction. It says what the
#: summary is *for*: the row it becomes stands in for the stretch below it, so a summary that leaves out a
#: decision or a path makes that turn worse.
COMPACT = """\
Write a summary of the conversation above, which later turns will be given instead of it. Keep what a
later reader needs to continue the work: what was asked for, what was decided and why, which paths and
files it touched, what is still open, and anything the next turn must not repeat or undo. Write it as
prose in the third person, in the present tense, and say nothing about being a summary — this text is
what the conversation becomes."""

#: What the harness says when it starts a turn nobody asked for (E146), and the plan is not finished.
#:
#: It is a *directive* and not a message: nothing new was said, so a row saying "carry on" would be words
#: the person did not say and the harness cannot pretend to. It goes in the prompt of that one request,
#: which is also what keeps the conversation itself untouched by a mode that only decides when to ask.
CONTINUING = """\
Carrying on:
- Nothing new was said. The plan below is not finished, so take the next step of it yourself — the same
  way you would if the person had just said "go on".
- What is left of the plan: {left}
- If there is nothing you can do about what is left, say so plainly and stop; a turn that waits for the
  person is the right answer to a task that needs one."""

#: What a compaction is told when its own input had to be cut to fit the route it is being sent to. The
#: summary then stands for a conversation whose beginning nobody read, so it has to say so — a summary that
#: describes words that were never sent is worse than a short one.
DROPPED = """\
{dropped} of this conversation did not fit what this route can hold, so they are not in what you were
given: it begins partway through. Say so where it matters in the summary, and do not write about them as if
you had read them."""


@dataclass(frozen=True, slots=True)
class PromptPlan:
    """The sections of a prompt that are decided once, and the one call that assembles them with a view.

    ``persona``, ``directives`` and ``skills`` are the standing part: they are the same text for every
    request of every session on this host, so they are made once (at install) and handed in. ``skills`` is
    the empty string for a host with no skills, and an empty section is not an empty line — it is not there
    at all.

    ``persona`` and ``directives`` are the two sections a program is expected to replace: the first is
    who the model is, the second is how it is told to use what it was given, and
    both are questions about the product, not about this kernel. ``directives`` defaults to
    :data:`CORE`, and an empty one is a program that has no playbook — the section disappears like any
    other empty one (§0.11, E79).

    A plan is a value: no lifecycle, nothing to close, and :meth:`compose` is a pure function of it and one
    session's tool view, which is what makes "the same prompt" something a test can compare as bytes.
    """

    persona: str = PERSONA
    directives: str = CORE
    skills: str = ""
    #: What a compaction asks the model for. Replaced by a program the same way ``directives`` is: what a
    #: summary is worth keeping is a question about the product, not about this kernel.
    compact: str = COMPACT
    #: The engines this program can search with, in the order it asks them — the value domain of
    #: ``web_search``'s ``engine`` argument, and the order leaving it out falls back through.
    engines: tuple[str, ...] = ()

    @classmethod
    def of(cls, skills: Skills | None, *, engines: Sequence[str] = ()) -> PromptPlan:
        """The plan this host's skills make, read once — the ordinary way to get one."""
        return cls(skills=catalog_of(skills), engines=tuple(engines))

    def compose(self, view: ToolView, *, continuing: str = "") -> str:
        """The whole system message for one request, as the five sections in their one order.

        ``view`` is what this session can reach right now, which decides two things: the group sections that
        travel with the groups it has been shown, and the index line that ends the prompt saying what is
        still missing. Nothing else here depends on the session, which is why one plan serves every
        conversation on a host.

        ``continuing`` is the one section that depends on *why* this request is being made, and it is
        therefore the only one a turn hands in: a turn the kernel started itself says so, and the index line
        still ends the prompt because what a session cannot reach is a fact about the session, not
        about this request (E146).
        """
        sections = (
            self.persona,
            self.directives,
            *self._arrived(view),
            self.skills,
            continuing,
            view.index(),
        )
        return "\n\n".join(section.strip() for section in sections if section.strip())

    def carrying_on(self, left: str) -> str:
        """What this program says when it starts a turn itself: the plan, and that nothing new was said."""
        return CONTINUING.format(left=left)

    def summarize(self, asks: str, *, dropped: int = 0) -> tuple[str, str]:
        """The system message and the instruction for one compaction, in that order.

        A compaction's request is not a conversation's: it carries no tools, no group sections and no index
        line, because nothing it says is going to be *used* by a turn — it is going to be read as the text
        the next turn starts from. What it keeps is the one part of a prompt that says who is writing: the
        persona.

        ``asks`` is what the operator wanted kept, and it is appended to the plan's own instruction: the
        closing rule of a summary (what a later reader needs) is not something a
        person should have to remember to type. ``dropped`` is how many earlier entries did not fit the
        route the request is going to, and it opens the instruction when there are any, because the summary
        would otherwise describe a conversation it only read the end of.
        """
        system = self.persona.strip()
        instruction = self.compact.strip()
        if dropped:
            instruction = f"{DROPPED.format(dropped=counted(dropped, 'entry'))}\n\n{instruction}"
        if asks.strip():
            instruction = f"{instruction}\n\nWhat the person asked to keep: {asks.strip()}"
        return system, instruction

    def _arrived(self, view: ToolView) -> tuple[str, ...]:
        """One section per group this session has been shown, in name order, if it has tools here.

        Two conditions and not one: a group is described only once it has been asked for, so a session
        that has made no request is not carrying instructions for tools it cannot call; and a group
        somebody asked for in another program's build may have nothing reachable in this one, which is a
        section about nothing. The section itself is written for the tools that *are* reachable: the net
        group's depends on whether this kernel has an engine at all, which is why it is not one paragraph
        in the table (§0.9, E26).
        """
        reached = {group for group, members in view.groups if any(name in view.specs for name in members)}
        served = frozenset(view.specs)
        return tuple(
            _described(group, served, self.engines)
            for group in sorted(Group)
            if group in reached and group.value in view.disclosed
        )


def _described(group: Group, served: Collection[str], engines: Sequence[str]) -> str:
    """What one group says about itself, worded for the tools this session actually has from it."""
    if group is Group.NET:
        return _network(served, engines)
    return GROUP_PLAYBOOK[group]


def catalog_of(skills: Skills | None) -> str:
    """The skills this host can see, as a section — or the empty string when there is nothing to list.

    One line per name, because the whole point of the section is that a model need not call
    ``skill_list`` to know a skill exists: what it can afford is a name and what the skill is for, and
    ``skill_read`` is the call that fetches the instructions. No store and an empty store read the same
    way here — nothing to say — since the skill *tools* are the index line's business (§0.10, E36).
    """
    if skills is None:
        return ""
    rows = [f"- {skill.name}: {skill.description or '(no description)'}" for skill in skills.visible()]
    if not rows:
        return ""
    return "Skills installed here — read one with `skill_read` before following it:\n" + "\n".join(rows)
