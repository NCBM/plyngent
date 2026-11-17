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

if TYPE_CHECKING:
    from collections.abc import Mapping

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

#: What each group's tools add, and the mistake each one invites. Filled for every :class:`Group`: a
#: group is a disclosure unit, so its playbook is a unit too — the section arrives when the group is shown
#: and with the tools it describes, which is what keeps the standing prompt at one compressed block
#: instead of 1.5k of instructions for calls a session cannot make yet (§10.4).
GROUP_PLAYBOOK: Mapping[Group, str] = {
    Group.DOCUMENT: """\
The rest of a document: a body too long for an answer is kept rather than cut, and `documents` lists what
is kept while `search` looks inside it. `read` takes a line window, so a big file is read in pieces.""",
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
never echoed back into this conversation.""",
    Group.NET: """\
The network: `client_fetch` asks from the person's own machine, `web_search` finds pages, and
`web_fetch` reads one as markdown. A denial is a decision rather than a transient failure — answer it by
changing what you ask for, not by asking again.""",
    Group.SESSION: """\
The plan a turn is worked through: `todo_push` opens a group of steps, `todo_update` moves one,
`todo_pop` closes a group, and `todo_list` shows the stack. The numbers a listing prints are positions in
that listing, not names, and they move when the stack does. Nothing is cleared for you: a finished group
stays until it is popped. `wait` is the other call here: it hands the turn over to a person for a while
and answers what they did — wrote a line, pressed enter, or said nothing at all.""",
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


@dataclass(frozen=True, slots=True)
class PromptPlan:
    """The sections of a prompt that are decided once, and the one call that assembles them with a view.

    ``persona``, ``directives`` and ``skills`` are the standing part: they are the same text for every
    request of every session on this host, so they are made once (at install) and handed in. ``skills`` is
    the empty string for a host with no skills, and an empty section is not an empty line — it is not there
    at all.

    ``persona`` and ``directives`` are the two sections a program is expected to replace rather than
    inherit: the first is who the model is, the second is how it is told to use what it was given, and
    both are questions about the product rather than about this kernel. ``directives`` defaults to
    :data:`CORE`, and an empty one is a program that has no playbook — the section disappears like any
    other empty one (§0.11, E79).

    A plan is a value: no lifecycle, nothing to close, and :meth:`compose` is a pure function of it and one
    session's tool view, which is what makes "the same prompt" something a test can compare as bytes.
    """

    persona: str = PERSONA
    directives: str = CORE
    skills: str = ""

    @classmethod
    def of(cls, skills: Skills | None) -> PromptPlan:
        """The plan this host's skills make, read once — the ordinary way to get one."""
        return cls(skills=catalog_of(skills))

    def compose(self, view: ToolView) -> str:
        """The whole system message for one request, as the five sections in their one order.

        ``view`` is what this session can reach right now, which decides two things: the group sections that
        travel with the groups it has been shown, and the index line that ends the prompt saying what is
        still missing. Nothing else here depends on the session, which is why one plan serves every
        conversation on a host.
        """
        sections = (self.persona, self.directives, *self._arrived(view), self.skills, view.index())
        return "\n\n".join(section.strip() for section in sections if section.strip())

    def _arrived(self, view: ToolView) -> tuple[str, ...]:
        """One section per group this session has been shown, in name order, if it has tools here.

        Two conditions and not one: a group is described only once it has been asked for, so a session
        that has made no request is not carrying instructions for tools it cannot call; and a group
        somebody asked for in another program's build may have nothing reachable in this one, which is a
        section about nothing.
        """
        reached = {group for group, members in view.groups if any(name in view.specs for name in members)}
        return tuple(
            GROUP_PLAYBOOK[group] for group in sorted(Group) if group in reached and group.value in view.disclosed
        )


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
