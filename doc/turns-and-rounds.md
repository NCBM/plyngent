# Turns and Rounds

Two nested units of work carry most of the loop's state. A **round** is one
request→response exchange with the provider plus the tool calls it asked for; a
**turn** is everything between a user message and the explicit stop that ends it.
A session is a stored sequence of turns.

```
turn ─┬─ round ─┬─ request  (hosted tools, history, system prompt)
      │         ├─ response (one assistant message)
      │         └─ tool batch (0..n calls, executed before the next round)
      ├─ round …
      └─ explicit stop (a round with no tool calls)
```

## Round — one model round trip

A round is one completion request and the single assistant message it produced,
followed by the tool calls that message asked for (one batch). The next round
starts only after that batch has reported results.

- **Identity.** One `UsageEvent` per round: `ChatAgent` counts them into
  `last_turn_rounds` (`agent/chat.py`). `AssistantMessageEvent` carries the
  response, `ToolCallEvent`/`ToolResultEvent` the batch. The transcript's
  `rounds=N` label counts the rounds that asked for tools (every round but the
  final one), so `/history` and `last_turn_rounds` can differ by that stop round.
- **Ends.** In one of four ways: with tool calls (the loop continues), with no
  tool calls (natural stop → the turn ends), at the output limit
  (`TruncatedEvent`, `finish_reason=length`: answer and thinking stay, that
  round's tool calls are dropped and a notice is appended; up to
  `max_truncation_continues` more rounds of the same turn carry on from there),
  or in an error (a retry re-runs the round).
- **Bounds.** `max_rounds` per turn (default `DEFAULT_MAX_ROUNDS = 32`,
  `agent/loop.py`). Reaching the allowance yields `MaxRoundsEvent(rounds,
  continued=False)` — `rounds` is the allowance that ran out — unless the
  `on_limit` hook answers yes (CLI: `y`/`n`/`yyy` prompt in `cli/limits.py`):
  then the event says `continued=True` and another `max_rounds` are granted.
- **Parallelism.** Calls of one round run concurrently (`parallel_tools`), but
  calls naming the same file serialize through per-file write locks, and the
  soft confirm for a round is decided once, before the batch runs
  (`ToolRegistry.prepare_batch_confirm`).
- **Round-scoped work.** Request assembly, context reduction / tool-result
  truncation over `max_context_tokens`, provider-tool (hosted search) blocks,
  and the usage sample are all per round.
- **Not durable on its own.** Only committed prefixes are persisted as they
  happen — a complete tool batch with its results, plus trailing developer
  notices. A final text answer is committed at turn end, so a turn that dies
  mid-round leaves its side effects in history but not a half-written answer.

## Turn — one user message to the explicit stop

A turn begins when the user message is appended (`ChatAgent.run`, which also
persists the message immediately) and ends at the explicit stop.

- **Explicit stop.** The first round whose assistant message has no tool calls.
  A turn also ends at a boundary: `max_rounds` without continuing,
  truncation-continues exhausted (left for `/retry`), cancellation (Ctrl+C), or
  an unrecovered error — the tail is rolled back to the committed prefix and the
  user message stays so `/retry` can continue.
- **Rounds per turn.** Zero (a failure before the first response) to many.
  `retry` continues the *same* turn — it reuses the user message instead of
  appending a new one — so retries never create another turn.
- **Turn-scoped state** (reset at the start of the next turn):
  - read-before-edit: line-numbered `read_file` authorizes `edit_lineno` for that
    path (mtime-checked) for this turn only (`reset_lineno_tracker`);
  - `yyy` limit auto-continue: cleared before and after
    `run_turn_with_retries`, so a retry of the same turn keeps it and the next
    user turn prompts again;
  - `/yolo once`: expires when the turn finishes (`expire_yolo_once`), so it
    covers exactly one turn;
  - todo nagging: `todo_stack.begin_turn()` clears the "touched" flag; any
    `todo_*` tool sets it again, and `needs_review()` at the stop decides whether
    a stack untouched this turn gets the review reminder;
  - usage totals: `last_turn_usage` and `last_turn_rounds` sum the turn's rounds
    (`last_request_usage` keeps the last round's sample).
- **Not turn-scoped.** Session identity and stored rows, the todo stack's
  contents, soft-confirm trust (path grants, access/skill grants, `TRUSTABLE`
  tool grants), and directive-reminder bands outlive the turn.

## Session

A session is the stored sequence of turns; the transcript groups rows by turn,
and `/history N` addresses a turn by number. Compaction rewrites the history sent
in one request and never moves a turn boundary. A side turn (`run_aside`) is a
turn whose history is forked and never written back.

## Where it lives

| unit | begins | ends | counters / records |
| --- | --- | --- | --- |
| round | provider request | response assembled, tool batch done | `UsageEvent`, `last_turn_rounds`, `MaxRoundsEvent`, `TruncatedEvent` |
| turn | `ChatAgent.run` appends the user message | a round with no tool calls (or a boundary) | session rows, `last_turn_usage` |
| session | first turn | process exit / session switch | store tables, transcript |

Loop mechanics: `agent/loop.py` (`run_chat_loop`); turn bookkeeping:
`agent/chat.py` (`ChatAgent.run` / `retry` / `_run_from_user_message`); retry and
turn-scoped flags: `cli/retry.py`, `cli/limits.py`.
