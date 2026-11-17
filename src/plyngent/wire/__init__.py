"""The contract between the kernel and whatever is talking to it.

Nothing here knows about a terminal, a tool, or a turn's internals; nothing in the kernel's own
packages is importable from here. That is the whole reason this package exists separately: a client —
the CLI, a daemon's other end, a test — may import it and nothing else, and a test can assert exactly
that with a directory walk instead of a review.

Four kinds of traffic, and the difference between them is what a transport may do with each:

============================  ==============  ===========================================
what                          direction       what a transport may do
============================  ==============  ===========================================
:class:`Command`              client → kernel never dropped, never reordered
:class:`Event`                kernel → client may be coalesced into a :class:`Batch`
:class:`AnyAsk` / :class:`AnyAnswer`  both            never dropped, correlated, has a deadline
:class:`Invocation` / :class:`Returned`  both         the same lane and the same rules, for a tool
:class:`Ack`                  client → kernel never dropped; it *is* the resume point
:class:`Query` / :class:`Reply`  both            correlated by a number, and answered once
============================  ==============  ===========================================

A question and an invocation travel on one lane because they are one thing asked of two different
people, and the rule they share is the rule that matters: neither may be dropped. What they carry is
not shared, which is why they are two pairs of values and not one.

A **query** is the same shape a third time, in the one direction that had none: a client asking the
kernel to *tell* it something. It belongs on its own lane rather than on the event lane, because the
event lane is allowed to coalesce what it carries and a read whose answer was folded into a batch is a
frontend waiting for an answer that will never come (§5, E134).

Deliberately absent: a ``Fork`` command. Branching is "the next ``begin_after`` points somewhere
else", which is a client's view state, so the kernel has nothing to execute. Also absent: any query
*command* — a read is a method on the link (:mod:`plyngent.wire.link`) and travels as a ``Query``, which
is why no command needs a reply id.
"""

from __future__ import annotations

from plyngent.wire.ask import DEFAULT_ASK_TIMEOUT_SECONDS as DEFAULT_ASK_TIMEOUT_SECONDS
from plyngent.wire.ask import SEVERITIES as SEVERITIES
from plyngent.wire.ask import SILENCE as SILENCE
from plyngent.wire.ask import Agreed as Agreed
from plyngent.wire.ask import AnyAnswer as AnyAnswer
from plyngent.wire.ask import AnyAsk as AnyAsk
from plyngent.wire.ask import Ask as Ask
from plyngent.wire.ask import AskSite as AskSite
from plyngent.wire.ask import By as By
from plyngent.wire.ask import Confirmation as Confirmation
from plyngent.wire.ask import FreeText as FreeText
from plyngent.wire.ask import Grades as Grades
from plyngent.wire.ask import Invocation as Invocation
from plyngent.wire.ask import Picked as Picked
from plyngent.wire.ask import Returned as Returned
from plyngent.wire.ask import Secret as Secret
from plyngent.wire.ask import Selection as Selection
from plyngent.wire.ask import Severity as Severity
from plyngent.wire.ask import Written as Written
from plyngent.wire.ask import answer_kind as answer_kind
from plyngent.wire.ask import unanswered as unanswered
from plyngent.wire.command import AnyCommand as AnyCommand
from plyngent.wire.command import Cancel as Cancel
from plyngent.wire.command import Close as Close
from plyngent.wire.command import Command as Command
from plyngent.wire.command import Compact as Compact
from plyngent.wire.command import Export as Export
from plyngent.wire.command import Goal as Goal
from plyngent.wire.command import ListSessions as ListSessions
from plyngent.wire.command import Offer as Offer
from plyngent.wire.command import OfferedTool as OfferedTool
from plyngent.wire.command import Open as Open
from plyngent.wire.command import Prune as Prune
from plyngent.wire.command import Resume as Resume
from plyngent.wire.command import Rollback as Rollback
from plyngent.wire.command import SetRoute as SetRoute
from plyngent.wire.command import Submit as Submit
from plyngent.wire.command import Ungrant as Ungrant
from plyngent.wire.event import AnswerCut as AnswerCut
from plyngent.wire.event import AnyEvent as AnyEvent
from plyngent.wire.event import AssistantText as AssistantText
from plyngent.wire.event import AssistantThought as AssistantThought
from plyngent.wire.event import Batch as Batch
from plyngent.wire.event import Changed as Changed
from plyngent.wire.event import Compacted as Compacted
from plyngent.wire.event import CompactRefused as CompactRefused
from plyngent.wire.event import DocumentStored as DocumentStored
from plyngent.wire.event import Entry as Entry
from plyngent.wire.event import Event as Event
from plyngent.wire.event import ExportRefused as ExportRefused
from plyngent.wire.event import GoalMode as GoalMode
from plyngent.wire.event import GoalRefused as GoalRefused
from plyngent.wire.event import GoalSet as GoalSet
from plyngent.wire.event import GrantRevoked as GrantRevoked
from plyngent.wire.event import KernelLeaving as KernelLeaving
from plyngent.wire.event import ListedSession as ListedSession
from plyngent.wire.event import Noticed as Noticed
from plyngent.wire.event import OfferRefused as OfferRefused
from plyngent.wire.event import Pruned as Pruned
from plyngent.wire.event import PruneRefused as PruneRefused
from plyngent.wire.event import Refusal as Refusal
from plyngent.wire.event import ResumeRefused as ResumeRefused
from plyngent.wire.event import RollbackRefused as RollbackRefused
from plyngent.wire.event import RolledBack as RolledBack
from plyngent.wire.event import RouteRefused as RouteRefused
from plyngent.wire.event import Seeded as Seeded
from plyngent.wire.event import SeedRefused as SeedRefused
from plyngent.wire.event import SessionsListed as SessionsListed
from plyngent.wire.event import ToolCalled as ToolCalled
from plyngent.wire.event import ToolReturned as ToolReturned
from plyngent.wire.event import TurnFinished as TurnFinished
from plyngent.wire.event import TurnKind as TurnKind
from plyngent.wire.event import TurnRefused as TurnRefused
from plyngent.wire.event import TurnStarted as TurnStarted
from plyngent.wire.event import Usage as Usage
from plyngent.wire.flow import Ack as Ack
from plyngent.wire.link import KernelLink as KernelLink
from plyngent.wire.link import KernelSide as KernelSide
from plyngent.wire.manifest import CLIENT_OFFER as CLIENT_OFFER
from plyngent.wire.manifest import PROTOCOL_VERSION as PROTOCOL_VERSION
from plyngent.wire.manifest import CommandSpec as CommandSpec
from plyngent.wire.manifest import KernelManifest as KernelManifest
from plyngent.wire.manifest import command_specs as command_specs
from plyngent.wire.query import AnyQuery as AnyQuery
from plyngent.wire.query import AnyReply as AnyReply
from plyngent.wire.query import Forest as Forest
from plyngent.wire.query import Grants as Grants
from plyngent.wire.query import GrantTable as GrantTable
from plyngent.wire.query import LiveTurn as LiveTurn
from plyngent.wire.query import LiveTurns as LiveTurns
from plyngent.wire.query import Query as Query
from plyngent.wire.query import Reached as Reached
from plyngent.wire.query import Reply as Reply
from plyngent.wire.query import ShadowedSkill as ShadowedSkill
from plyngent.wire.query import SkillRow as SkillRow
from plyngent.wire.query import Skills as Skills
from plyngent.wire.query import SkillTable as SkillTable
from plyngent.wire.query import Trajectory as Trajectory
from plyngent.wire.query import Turns as Turns
from plyngent.wire.scope import CancelReason as CancelReason
from plyngent.wire.scope import CancelScope as CancelScope
