"""The terminal tools: open a child on a PTY, read it, type into it, close it, ask a human for it.

Six verbs over one registry, and every one of them answers the same three questions in the same order:

* **may I?** — the command policy for a new child (with the human deciding for a denylisted program),
  and the PTY policy for a session that is still alive. A session that is already gone is not a
  question at all, which is why closing one is not "ask every time".
* **is there room?** — the sweep runs before the limit is reported, so a full registry of finished
  sessions is not a stuck one.
* **what does the caller see?** — the registry's own fields, as fields. Child output is sent as text
  and made safe by the renderer, never here.

A secret asked for with ``ask_into_pty`` goes from the human to the terminal and nowhere else: it is not
in the summary, not in the body, not in the effects, and not in the journal.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode
from plyngent.kernel.command import NeedsHuman
from plyngent.kernel.failure import Failure
from plyngent.kernel.pty.keys import KeySyntaxError, decode_write_data
from plyngent.kernel.pty.policy import room_for_session
from plyngent.kernel.pty.registry import DEFAULT_PTY_READ_BYTES, PtyError, PtyReadResult
from plyngent.kernel.toolresult import Effects, Meta, Text, ToolResult
from plyngent.kernel.tools import lifted
from plyngent.kernel.toolspec import Group, Mutation, Posture, Res, Role, ToolArgs, ToolSpec
from plyngent.kernel.workspace import PathRefused

if TYPE_CHECKING:
    from plyngent.kernel.env import Env
    from plyngent.kernel.pty.registry import PtySession

#: How long one read may wait, whatever a call asks for: a mis-typed argument must not stall a turn.
MAX_READ_WAIT_SECONDS = 120.0


class OpenPtyArgs(ToolArgs, kw_only=True):
    """Open a PTY session running ``command`` (argv) under the workspace.

    Use this for a program that expects a terminal — a REPL, an editor, anything that asks questions —
    and then ``read_pty`` to see what it says and ``write_pty`` to answer. ``cwd`` is relative to the
    workspace. A command that cannot start still opens the session: the failure arrives with the first
    read, as output and an exit code, which is what a terminal is like.
    """

    command: list[str]
    cwd: str = "."


class ReadPtyArgs(ToolArgs, kw_only=True):
    """Read what a PTY session wrote, with its state.

    Without ``until``, waits up to ``timeout`` seconds for any data; with it, polls until that substring
    appears, the child dies, the deadline passes, or the session's output budget is spent. Empty data
    with ``alive=true`` means nothing was ready — not that the child has finished. Only one read per
    session at a time: two would split the byte stream between them.
    """

    session_id: int
    max_bytes: int = DEFAULT_PTY_READ_BYTES
    timeout: float = 0.2
    until: str | None = None


class WritePtyArgs(ToolArgs, kw_only=True):
    """Write literal text to a PTY session's input, exactly as given.

    A newline is not added: write one (``\\n`` inside the JSON string) when the program expects Enter.
    For a key rather than text — Ctrl-C, an arrow key — use ``write_pty_keys`` instead.
    """

    session_id: int
    data: str


class WritePtyKeysArgs(ToolArgs, kw_only=True):
    """Write to a PTY session after expanding ``<key>`` markers (never for plain typing).

    Markers are written between angle brackets: ``<esc>`` ``<enter>`` ``<tab>`` ``<s-tab>``
    ``<space>`` ``<bs>`` ``<del>`` ``<ins>`` ``<home>`` ``<end>`` ``<up>`` ``<down>`` ``<left>``
    ``<right>`` ``<pgup>`` ``<pgdn>`` ``<f1>`` … ``<f12>``, a modifier in front where a key has one
    (``<c-up>``, ``<s-home>``, ``<c-s-f5>``), ``<up*3>`` to repeat, and ``<<`` for a literal ``<``. On
    a plain character Ctrl is its control byte (``<c-a>``) and Alt an ESC before it (``<m-x>``).

    Escapes are the characters the string carries, so they are doubled inside JSON: ``\\n`` ``\\r``
    ``\\t`` ``\\e``/``\\E`` (ESC) ``\\0`` ``\\xHH`` (a byte) ``\\uHHHH`` (a code point). Anything else is
    literal. A marker that names no key is an error, not text typed into the terminal.
    """

    session_id: int
    data: str


class ClosePtyArgs(ToolArgs, kw_only=True):
    """Close a PTY session: terminate it, then kill it after a short grace period.

    A session that has already exited is closed without a question — there is no process left to change
    — and the exit code it left is reported either way.
    """

    session_id: int


class AskIntoPtyArgs(ToolArgs, kw_only=True):
    """Ask the **human** for a value and write it into a PTY session (local only).

    Use this for what a model must never see: a password, a passphrase, a one-time code. ``message`` is
    shown to the human and says what to enter — it is not the secret, and must not be. ``secret=true``
    reads without echoing. ``submit=true`` appends a newline after the answer. An empty answer writes
    nothing and cancels.
    """

    session_id: int
    message: str
    secret: bool = False
    submit: bool = True


async def open_pty(env: Env, args: OpenPtyArgs) -> ToolResult:
    """Start a child on a terminal, once the command policy and the registry have both agreed."""
    workspace = env.workspace
    try:
        cwd = workspace.resolve(args.cwd, AccessMode.EXEC)
    except PathRefused as exc:
        return lifted(exc)
    if not cwd.is_dir():
        return ToolResult.of(Failure.MISSING, f"not a directory: {args.cwd}")

    refused = await _allowed(env, args.command)
    if isinstance(refused, ToolResult):
        return refused
    room = await room_for_session(env.pty)
    if room is not None:
        return ToolResult.of(Failure.LIMIT, room)
    try:
        session = env.pty.open(args.command, cwd=cwd, owner=env.session)
    except PtyError as exc:
        return ToolResult.of(exc.failure, str(exc))
    command = shlex.join(session.command)
    return ToolResult.ok(
        f"opened PTY session {session.session_id}",
        body=Text(value=f"{_state(session.session_id, alive=True, exit_code=None)}\ncmd={command}"),
        effects=Effects(
            external=(f"pty session {session.session_id}: {command}",),
            approvals=refused,
        ),
    )


async def read_pty(env: Env, args: ReadPtyArgs) -> ToolResult:
    """Read what the child wrote, and say what the child is doing."""
    try:
        result = await env.pty.read(
            args.session_id,
            max_bytes=args.max_bytes,
            wait_seconds=min(max(0.0, args.timeout), MAX_READ_WAIT_SECONDS),
            until=args.until,
        )
    except PtyError as exc:
        return ToolResult.of(exc.failure, str(exc))
    return ToolResult.ok(
        f"read {len(result.data)} chars from session {result.session_id}",
        body=Text(value=_read(result)),
        meta=_read_meta(result),
    )


async def write_pty(env: Env, args: WritePtyArgs) -> ToolResult:
    """Write literal text to a session, once this session has been agreed to."""
    return await _write(env, args.session_id, args.data)


async def write_pty_keys(env: Env, args: WritePtyKeysArgs) -> ToolResult:
    """Write the keys the markers name, once this session has been agreed to."""
    try:
        payload = decode_write_data(args.data)
    except KeySyntaxError as exc:
        return ToolResult.of(Failure.ARGS, str(exc))
    return await _write(env, args.session_id, payload)


async def close_pty(env: Env, args: ClosePtyArgs) -> ToolResult:
    """Close a session — a question only while there is still a process in it."""
    session = env.pty.get(args.session_id)
    if session is not None:
        refused = await _consent(env, session)
        if refused is not None:
            return refused
    result = await env.pty.close(args.session_id)
    if session is not None:
        env.pty_policy.forget(args.session_id)
    if not result.closed:
        return ToolResult.of(Failure.MISSING, f"no such PTY session: {args.session_id}")
    body = _state(result.session_id, alive=False, exit_code=result.exit_code)
    effects = Effects(external=(f"closed pty session {result.session_id}",))
    return ToolResult.ok(f"closed PTY session {result.session_id}", body=Text(value=body), effects=effects)


async def ask_into_pty(env: Env, args: AskIntoPtyArgs) -> ToolResult:
    """Ask the human, write the answer, and keep it out of everything that is reported."""
    if env.pty.get(args.session_id) is None:
        return ToolResult.of(Failure.MISSING, f"no such PTY session: {args.session_id}")
    prompt = args.message.strip() or ("Secret" if args.secret else "Input")
    where = env.site("answer")
    written = await (
        env.human.secret(prompt=prompt, site=where) if args.secret else env.human.text(prompt=prompt, site=where)
    )
    if written.text is None:
        return ToolResult.of(Failure.BLOCKED, "the human gave no answer, so nothing was written")
    payload = written.text + ("\n" if args.submit else "")
    try:
        env.pty.write(args.session_id, payload)
    except PtyError as exc:
        return ToolResult.of(exc.failure, str(exc))
    body = Text(
        value="\n".join(
            (
                f"session_id={args.session_id}",
                "wrote=true",
                f"secret={'true' if args.secret else 'false'}",
                f"submit={'true' if args.submit else 'false'}",
                "source=human",
            )
        )
    )
    return ToolResult.ok(f"wrote the human's answer to session {args.session_id}", body=body)


async def _write(env: Env, session_id: int, payload: str) -> ToolResult:
    """The half of both write verbs that touches the terminal."""
    session = env.pty.get(session_id)
    if session is not None:
        refused = await _consent(env, session)
        if refused is not None:
            return refused
    try:
        env.pty.write(session_id, payload)
    except PtyError as exc:
        return ToolResult.of(exc.failure, str(exc))
    body = Text(value=f"session_id={session_id}\nwrote={len(payload)} chars")
    return ToolResult.ok(f"wrote {len(payload)} chars to session {session_id}", body=body)


async def _consent(env: Env, session: PtySession) -> ToolResult | None:
    """Ask before changing a live session, once per session; ``None`` when nothing has to be asked."""
    if not env.pty_policy.needs_consent(session):
        return None
    agreed = await env.human.confirm(
        prompt=f"change PTY session {session.session_id}, which is still running?", site=env.site("consent")
    )
    if not agreed.accepted:
        return ToolResult.of(Failure.DENIED, f"the human did not allow changing PTY session {session.session_id}")
    env.pty_policy.approve(session.session_id)
    return None


async def _allowed(env: Env, command: list[str]) -> ToolResult | tuple[str, ...]:
    """Whether this argv may start, from the policy and — when it asks — from the human.

    The answer is the programs the human allowed, so the caller can record them as this call's effects:
    the policy remembers them for the life of the host, and the effect record is about this call.
    """
    verdict = env.workspace.commands.verdict(command)
    if not isinstance(verdict, NeedsHuman):
        return ()
    approvals: list[str] = []
    for program in verdict.programs:
        agreed = await env.human.confirm(
            prompt=f"run {program!r} on a terminal? it is on the command denylist", site=env.site("gate")
        )
        if not agreed.accepted:
            return ToolResult.of(Failure.DENIED, f"the command denylist refused {program!r}")
        env.workspace.commands.approve(program)
        approvals.append(program)
    return tuple(approvals)


def _state(session_id: int, *, alive: bool, exit_code: int | None) -> str:
    """The four fields a caller asks about a session, in the order they ask them."""
    return "\n".join(
        (
            f"session_id={session_id}",
            f"alive={'true' if alive else 'false'}",
            f"exit_code={'' if exit_code is None else exit_code}",
        )
    )


def _read(result: PtyReadResult) -> str:
    """A read, as its state and then the bytes the child wrote."""
    flags = "\n".join(
        (
            f"matched={'true' if result.matched else 'false'}",
            f"truncated={'true' if result.truncated else 'false'}",
            f"budget_exhausted={'true' if result.budget_exhausted else 'false'}",
        )
    )
    state = _state(result.session_id, alive=result.alive, exit_code=result.exit_code)
    return f"{state}\n{flags}\n--- data ---\n{result.data}"


def _read_meta(result: PtyReadResult) -> Meta:
    """What the read cost and whether the session's budget is what stopped it."""
    return Meta(chars=len(result.data), truncated=result.truncated)


OPEN = ToolSpec(
    name="open_pty",
    group=Group.PROCESS,
    args=OpenPtyArgs,
    run=open_pty,
    # `PATHS` too: the call resolves its own directory against the workspace, and a path tool that
    # could not resolve one would be a group of declarations that contradict each other.
    uses=Res.EXEC | Res.PTY | Res.PATHS,
    mutates=Mutation.HOST,
    roles={"cwd": Role.CWD},
)

READ = ToolSpec(
    name="read_pty",
    group=Group.PROCESS,
    args=ReadPtyArgs,
    run=read_pty,
    uses=Res.PTY,
)

WRITE = ToolSpec(
    name="write_pty",
    group=Group.PROCESS,
    args=WritePtyArgs,
    run=write_pty,
    uses=Res.PTY,
    mutates=Mutation.HOST,
    # The baseline would be `ONCE(SESSION)`, and the handler already asks exactly that — while the
    # session is still alive, which is a runtime fact the approval gate cannot see. Asking twice for one
    # write would be one question too many, so the gate is told not to, and the handler's own ask stands.
    posture=Posture.AUTO,
)

WRITE_KEYS = ToolSpec(
    name="write_pty_keys",
    group=Group.PROCESS,
    args=WritePtyKeysArgs,
    run=write_pty_keys,
    uses=Res.PTY,
    mutates=Mutation.HOST,
    posture=Posture.AUTO,
)

CLOSE = ToolSpec(
    name="close_pty",
    group=Group.PROCESS,
    args=ClosePtyArgs,
    run=close_pty,
    uses=Res.PTY,
    mutates=Mutation.HOST,
    # A session that has already exited changes nothing when it is closed, so the baseline is `AUTO`
    # and the PTY policy raises it to `ONCE(session)` while there is still a process in there.
    posture=Posture.AUTO,
)

ASK = ToolSpec(
    name="ask_into_pty",
    group=Group.PROCESS,
    args=AskIntoPtyArgs,
    run=ask_into_pty,
    uses=Res.HUMAN | Res.PTY,
    mutates=Mutation.HOST,
)

SPECS: tuple[ToolSpec, ...] = (OPEN, READ, WRITE, WRITE_KEYS, CLOSE, ASK)
