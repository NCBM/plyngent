# The plyngent wire, version 1

This is the contract two programs agree on when one of them is this kernel. It is written for the
program on the other side — a frontend with buttons, a plugin that replaces somebody else's kernel with
this one — and not for a reader of this repository: the shapes live in
`src/plyngent/wire/*.py` and are the reference, while what follows is the part no struct can state on its
own.

**Version.** `KernelManifest.protocol_version` and `Bind.protocol_version` both carry it. Two ends
compare and neither adapts: a shape this document does not describe is not one either side may read.

## Frames

A stream carries bytes, not messages, so a value says where it ends:

```
[4 bytes: length, big-endian][1 byte: lane][body]        length counts the lane and the body
```

`body` is msgpack. The lane is a byte of its own rather than a field of the body, because the two ends
have to agree about *where* a value goes before either can read what it says. Lane numbers are fixed
forever: a new lane is a new number, never a renumbering.

| byte | lane | direction | what a transport may do with it |
|---|---|---|---|
| 0 | `hello` | kernel → client, once | the first frame on the stream, and never repeated |
| 10 | `bind` | client → kernel, once | the client's first frame, before any command |
| 1 | `command` | client → kernel | never dropped, never reordered |
| 2 | `ack` | client → kernel | never dropped; it is the resume point and the heartbeat |
| 3 | `answer` | client → kernel | never dropped; one per question |
| 4 | `returned` | client → kernel | never dropped; one per call |
| 5 | `event` | kernel → client | may be folded, and dropped if the client is presumed gone |
| 6 | `question` | kernel → client | never folded, never dropped |
| 7 | `invocation` | kernel → client | the same lane's rules, for a tool the client offered |
| 8 | `query` | client → kernel | correlated by a number, answered once |
| 9 | `reply` | kernel → client | the answer to one query, carrying its number |

## The handshake

1. **The kernel greets.** The first frame it writes is `hello`, whose body is the manifest.
2. **The client binds.** The first frame *it* writes is `bind`, whose body says which version of this
   contract it speaks.
3. **A mismatch ends the connection.** No error value crosses the wire, because an error would have to be
   written in the contract and this is the one case where the two ends have not agreed about what the
   contract is. The kernel that is sent a version it does not speak serves none of that connection; the
   client that is greeted with one does not send `bind` at all. Each side's sentence is its own.
4. **Then everything else**, in any order the two lanes allow.

The manifest is what a client builds itself from, and it carries: `protocol_version`, `kernel`, `epoch`
(which run of this kernel this is), `workspace` (the tree it edits, resolved), `commands` and `queries`
(one row each: a tag, a one-line summary, and the JSON Schema of the arguments), `caps` (advertised
capabilities), `ack_interval` and `dead_after` (the two numbers liveness is measured with).

A client must not treat `workspace` as an allowlist — it is where *this* kernel works — and must not
assume the table it read is complete for another connection: it is what this kernel offers, read once.

## Commands

A command changes the world and has **no reply**: what came of it arrives on the event lane, including
every refusal. `session` names the conversation, and an empty one means the caller's default.

| word | what it asks for |
|---|---|
| `open` | start a conversation, or copy another one's text into a new one |
| `close` | stop serving a conversation; its entries stay on disk |
| `submit` | say something in a conversation; the kernel answers with a turn |
| `retry` | ask the model again from the end of the chain, without repeating a call |
| `cancel` | stop one piece of work, or everything a conversation is doing |
| `set-route` | point a conversation at another model route, or back at the default |
| `goal` | read or move the switch that lets the kernel continue a plan on its own |
| `compact` | write one summary entry standing in for a stretch of the conversation |
| `prune` | stop showing a set of entries to the model, reversibly |
| `resume` | continue in the store a conversation was left in |
| `list-sessions` | list the conversations this kernel holds |
| `export` | write this conversation out as a document, and answer with its handle |
| `offer` | say what tools this client can run |
| `rollback` | put a path back to the version an earlier entry recorded |
| `ungrant` | take back one path a conversation was allowed to reach |

`submit.kind` is a `TurnKind` — `chat`, `compact`, `aside`, `goal`, `retry` — and three of the five are not
a client's to send: `goal` is started by the kernel when an unfinished plan says the work is not done
(`no-goal`), `aside` is declared and not implemented (`no-aside`), and `retry` is asked for with the command
of that name (`no-retry`). Each is refused by name, which is a refusal a client can be written against.

`retry` asks one more request against the chain as it stands: the tool calls already on it stay where they
are, with their results, and none of them runs again. `begin_after` names the entry to continue, and an
empty one is the end of the chain — a conversation with nothing in it has nothing to ask again
(`nothing-to-retry`).

`open.seed_from` is `(session, entry)` — the last entry of the chain to copy — and only the projected text
travels: no pointer, no grants, no todo, no route, no name. `open.inherit` is an `Inherit`, `none` (the
default) or `approvals`, and it is the one thing a client may ask to carry beside the text: a conversation
whose predecessor was just folded into it should not put the same questions to a person again. Nothing
else is inherited, and `approvals` has to be said explicitly.

## Reads

| word | answer | what it asks |
|---|---|---|
| `grants` | `grant-table` | which paths outside the workspace a conversation may reach, and how far on each |
| `trajectory` | `forest` | every row of a conversation, in the order it was written |
| `skills` | `skill-table` | which skills this kernel can see, and what was wrong with the rest |
| `turns` | `live-turns` | which turns of a conversation are running, right now |

A read is a question and one answer, and it is on its own lane because the event lane is allowed to fold
what it carries: a folded answer is a frontend waiting forever. `id` is the connection's number: a client
leaves it at `0` and the link stamps it. A read is never answered by an event, and its answer is the same
kind as the question — never an envelope with a status in it.

`turns` is a snapshot and not a wait. A client that wants to know when a conversation went idle asks
again after the next event; the kernel does not notify.

## Events

`seq` is the kernel's number: it is assigned at emission, never reused, and is a *transport* number (how
far a client has rendered). It is not a history selector — that is `begin_after`.

- **A turn**: `turn-started`, `turn-finished` (`reason` says why it ended, `ends_at` is the entry a client
  may continue from), `turn-refused`, `answer-cut`, `usage`.
- **An answer arriving**: `assistant-text`, `assistant-delta`, `assistant-thought`, `tool-called`,
  `tool-returned`.
- **The bookkeeping a person asked for**: `sessions-listed`, `compacted`, `compact-refused`, `pruned`,
  `prune-refused`, `export-refused`, `document-stored`, `seeded`, `seed-refused`, `rolled-back`,
  `rollback-refused`, `grant-revoked`, `goal-set`, `goal-refused`, `routed`, `route-refused`, `changed`,
  `noticed`, `offer-refused`.
- **The connection itself**: `kernel-leaving` (the stream ends after it), `batch`.

A refusal carries one word from a closed set; the sentence a person reads is the frontend's to write.

**Events may be folded**: a `batch` carries items that keep their own `seq`, so an acknowledgement of a
batch covers all of them. Folding changes how many messages arrive and nothing else.

### What a client must adopt, not just read

Two events move where a client stands, and a client that ignored them would grow a branch that never saw
them:

- **`compacted`** names the summary entry. What was written after the compaction continues past it, so the
  client's next `begin_after` is `compacted.entry`.
- **`pruned`** names the prune entry. What is written after a prune must not continue from before it, so
  the client's next `begin_after` is `pruned.entry` — otherwise a checkpoint is silently undone by the
  next message.

## Entries and the cursor

An `Entry` is one row: `id`, `begin_after` (the row it continues, or `null` for a root), `kind` (the
record's own tag), `session`, `turn`, `preview` (one line, for picking a place), and `children` (how many
rows continue it — the row nothing continues is the head).

`forest.entries` is the whole conversation in the order it was written, parents included, so a client
draws the tree without a second read. `forest.unfinished` names the turns that have rows and no row saying
they ended — a value, not a verdict: only `turns` can say which of them is still running.

**Where a client stands is the client's.** The kernel keeps no current branch. A client says where the
next message goes with `begin_after`, and the kernel validates only that the entry exists.

## Asking a person

Four questions, three answers, and a question's kind says which answer it admits:

| question | answer | unanswered |
|---|---|---|
| `confirmation` | `agreed` (`accepted: bool`) | the level's fallback |
| `selection` | `picked` (`value` or `values`) | nothing picked |
| `free-text` | `written` (`text`) | nothing written |
| `secret` | `written` | nothing written |

- **`severity`** (`routine`, `confirm`, `danger`) is how heavy it is. `timeout_default` is what a
  yes-or-no question answers if the deadline arrives — the light level falls back to *yes*, the two
  heavier ones never do — and it travels with the question so a client can say what silence will mean.
- **`timeout`** is how long the question gets, already resolved by the connection: the level's grade and
  the caller's own number, whichever is shorter, with `null` meaning the connection's clock.
- **`by`** on every answer names who spoke: `human`, `policy`, `timeout`, or `nobody`. It has no default,
  because "no" from a person is final, "no" from a deadline is worth another try, and "nobody" is never a
  yes.
- **`label`** is what this answer is called when several are gathered into one form; empty means the
  question stands alone.
- **`session`, `turn`, `scope`** say where the question belongs, so a frontend holding several
  conversations can draw it under the right one.
- An answer of the wrong kind does not end the wait: the question is still outstanding and its deadline
  still applies.

`secret` means: read without echo, never carry into an event, a result or a journal, and suggest nothing.

## A client's own tools

A client may offer tools of its own, and only where the manifest carries the `client-offer` capability —
a kernel that cannot run them must not be told about them. The offer is a command; the kernel then sends
`invocation` (`tool`, `arguments` as the model's own JSON, plus where it belongs) and waits for
`returned` (`ok`, `text`).

Two rules a client must be written against. `returned.text` is untrusted: it reaches the model as prose
and nothing in it is read as a status, a path, or a next step. And a call in flight is *not* released by a
disconnect the way a question is — a dropped connection is often the gap before a reconnect, so the call
keeps its own deadline instead.

## Reconnecting, and being gone

`ack` carries `seq` and `resume`.

- **`seq`** is a claim about the present: everything up to it has been rendered. It is also the heartbeat.
- **`resume: true`** is a claim about a connection that is *gone*, and is answered once per connection:
  the kernel replays what it said since that number, from a window of the most recent 512 events. Anything
  older is the history's business (`trajectory`), not the stream's.
- A number beyond anything this kernel said is not answered — replaying a stranger's count would be
  replaying somebody else's words.
- **`epoch`** guards across runs: a number remembered from another run of a kernel is another count, so a
  client compares `epoch` first and claims nothing when it differs.
- **Liveness**: a client that is behind the backlog *and* has said nothing for longer than `dead_after`
  is presumed gone. Its questions are refused, its connection dropped, and the work it started is not.

## What this document does not say

The shape of each value — every field, every default — is the Python package beside this file, which
ships with the kernel and is the reference implementation of this contract. A frontend written in
another language reads it once; a frontend written against it must not depend on anything in the kernel
beyond these values and the two rules that keep them honest: no in-process shortcut, and no field that
only one frontend knows how to use.
