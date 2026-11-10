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
:class:`Ack`                  client → kernel never dropped; it *is* the resume point
============================  ==============  ===========================================

Deliberately absent: a ``Fork`` command. Branching is "the next ``begin_after`` points somewhere
else", which is a client's view state, so the kernel has nothing to execute. Also absent: any query
command — reads are methods on the link (:mod:`plyngent.wire.link`), which is why no command needs a
reply id.
"""

from __future__ import annotations

from plyngent.wire.ask import DEFAULT_ASK_TIMEOUT_SECONDS as DEFAULT_ASK_TIMEOUT_SECONDS
from plyngent.wire.ask import Agreed as Agreed
from plyngent.wire.ask import AnyAnswer as AnyAnswer
from plyngent.wire.ask import AnyAsk as AnyAsk
from plyngent.wire.ask import Ask as Ask
from plyngent.wire.ask import AskSite as AskSite
from plyngent.wire.ask import Confirmation as Confirmation
from plyngent.wire.ask import FreeText as FreeText
from plyngent.wire.ask import Picked as Picked
from plyngent.wire.ask import Secret as Secret
from plyngent.wire.ask import Selection as Selection
from plyngent.wire.ask import Written as Written
from plyngent.wire.ask import answer_kind as answer_kind
from plyngent.wire.ask import unanswered as unanswered
from plyngent.wire.command import AnyCommand as AnyCommand
from plyngent.wire.command import Cancel as Cancel
from plyngent.wire.command import Close as Close
from plyngent.wire.command import Command as Command
from plyngent.wire.command import Compact as Compact
from plyngent.wire.command import Export as Export
from plyngent.wire.command import ListSessions as ListSessions
from plyngent.wire.command import Open as Open
from plyngent.wire.command import Prune as Prune
from plyngent.wire.command import Resume as Resume
from plyngent.wire.command import SetRoute as SetRoute
from plyngent.wire.command import Submit as Submit
from plyngent.wire.event import AnyEvent as AnyEvent
from plyngent.wire.event import ArtifactStored as ArtifactStored
from plyngent.wire.event import AssistantText as AssistantText
from plyngent.wire.event import Batch as Batch
from plyngent.wire.event import Entry as Entry
from plyngent.wire.event import Event as Event
from plyngent.wire.event import KernelLeaving as KernelLeaving
from plyngent.wire.event import ListedSession as ListedSession
from plyngent.wire.event import SessionsListed as SessionsListed
from plyngent.wire.event import ToolCalled as ToolCalled
from plyngent.wire.event import ToolReturned as ToolReturned
from plyngent.wire.event import TurnFinished as TurnFinished
from plyngent.wire.event import TurnKind as TurnKind
from plyngent.wire.event import TurnRefused as TurnRefused
from plyngent.wire.event import TurnStarted as TurnStarted
from plyngent.wire.flow import Ack as Ack
from plyngent.wire.link import KernelLink as KernelLink
from plyngent.wire.link import KernelSide as KernelSide
from plyngent.wire.manifest import PROTOCOL_VERSION as PROTOCOL_VERSION
from plyngent.wire.manifest import CommandSpec as CommandSpec
from plyngent.wire.manifest import KernelManifest as KernelManifest
from plyngent.wire.manifest import command_specs as command_specs
from plyngent.wire.scope import CancelReason as CancelReason
from plyngent.wire.scope import CancelScope as CancelScope
