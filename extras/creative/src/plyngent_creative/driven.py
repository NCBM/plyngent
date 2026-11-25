"""One plugin driven through a real kernel, in a process of its own, and what arrived.

A check reads a file; this runs one. The plugin is loaded into a kernel of its own — a root of its own, a
data directory of its own, a journal of its own — a conversation is opened, and the steps somebody wrote are
carried out in order. What comes back is the events the kernel said, because that is what an expectation is
matched against: the contract a frontend reads, and not a rendering of it (E184).

**Two channels and no third.** Steps arrive on the child's **stdin**, one JSON object a line, which makes
carrying on from where a run stopped the same thing as having sent more. What happened goes to its
**stdout**, also one JSON object a line: an event as the kernel says it, and then the answer to one step. The
error stream is kept as a tail by whoever started this: a plugin that printed to it printed something nobody
scripted, and the tail is where an author reads it.

**Nothing here is a sandbox.** The child has this process's rights, which is what an installed plugin has
anyway (E174). What the separate process buys is the failure and not the boundary: a plugin that kills its
interpreter kills one of these, not the conversation that asked it anything.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent_creative.model import QueueModel
from plyngent_creative.session import Run
from plyngent_creative.steps import ScriptError, arrived, parse, tag_of, values

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.wire.event import AnyEvent
    from plyngent_creative.steps import Step

#: How long one step waits for what it expected. It is this child's own bound, because nothing in the parent
#: is watching while a step is carried out: a step that never matched would hold the whole run still.
STEP_SECONDS = 20.0


def main() -> None:
    """Read steps from stdin, carry them out, and write what happened as one JSON object a line."""
    asyncio.run(_served(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))


async def _served(root: Path, journal: Path, plugin: Path) -> None:
    """The whole of this child's life: open the run, take steps, answer each, and stop when stdin closes."""
    from plyngent.plugins import from_file

    model = QueueModel()
    run = await Run.open(
        root,
        model=model,
        journal=journal,
        components=from_file(str(plugin)),
        on_event=lambda one: tell(said(one)),
    )
    try:
        tell({"kind": "hello", "session": run.session})
        async for line in _lines():
            await _carried(run, model, line)
    finally:
        await run.close()


async def _carried(run: Run, model: QueueModel, line: str) -> None:
    """One step, from the line it arrived as to the line it is answered with.

    An answer is ``wrong: null`` when the step was done, and the sentence when it was not — what the model
    could not answer, what was expected against what arrived, what the plugin raised. A step that went wrong
    is a fact about the plugin and leaves the run where it is: the next step is carried out as if this one had
    been carried out, which is what makes "fix it and ask again" one call (E184).
    """
    try:
        (step,) = parse([_object(line)])
        wrong = await _step(run, model, step)
    except ScriptError as exc:
        wrong = str(exc)
    except Exception as exc:  # noqa: BLE001 — what a step raised is the sentence, not a traceback
        wrong = f"{type(exc).__name__}: {exc}"
    tell({"kind": "step", "wrong": wrong})


async def _step(run: Run, model: QueueModel, step: Step) -> str | None:
    """Carry out one step in the order its three keys are written in, and say what went wrong.

    The model's answer is handed over before the line is sent, because a turn takes the next queued answer the
    moment it asks: a message sent first would have the model asked with nothing scripted for it. That is the
    one order this end fixes, and it is why a step that says both is written ``model`` then ``send``.

    An expectation looks from where the last one matched, not from where this step started: an event that
    arrived while nothing was waiting is still this script's to see, and a step that matched nothing says
    what arrived since the last match (E184).
    """
    at = run.cursor
    if step.model is not None:
        model.schedule(step.model.scripted())
    if step.send is not None:
        run.send(step.send)
    if step.expect is None:
        return None
    try:
        _ = await run.expect(step.expect.wants, patience=STEP_SECONDS)
    except TimeoutError:
        return f"expected {step.expect.says()}, and what arrived after that step was:\n{arrived(run.since(at))}"
    return None


def _object(line: str) -> object:
    """One line of the parent's script as the value it is, or the refusal saying it is not one."""
    try:
        found: object = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"a step is one JSON object a line: {exc}"
        raise ScriptError(msg) from exc
    return found


async def _lines() -> AsyncIterator[str]:
    """The parent's lines, read as they arrive, and nothing once it closes the stream."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        said = line.strip()
        if said:
            yield said


def tell(what: dict[str, object]) -> None:
    """One line to the parent, flushed, so that it can be read before the next one is written."""
    _ = sys.stdout.write(json.dumps(what) + "\n")
    _ = sys.stdout.flush()


def said(event: AnyEvent) -> dict[str, object]:
    """One event as the parent reads it: the tag, and the fields the wire carries."""
    return {"kind": "event", "tag": tag_of(event), "fields": values(event)}


if __name__ == "__main__":
    main()
