"""The half of a check that runs in a process of its own.

A plugin is a file somebody else wrote, and loading one executes it: an import that starts a server, that
reads a file, that raises after printing — all of it happens the moment the loader touches the file. So the
loader runs where a program runs, in a child of this process with its own argv, and the answer travels back
as one line of JSON. What the kernel's own process keeps is a sentence (E184).

Nothing here is a sandbox: the child has this process's rights, which is what an installed plugin has anyway
(E174). What the separate process buys is the *failure*, not the boundary — a plugin that kills its
interpreter kills one of these and not the conversation that asked.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import cast

from plyngent_creative.tools import checks

#: How long a check's child is given. It reads one file and asks its providers for their tools, so this is
#: generous by two orders of magnitude; what it is really for is a plugin that never returns.
CHECK_SECONDS = 30.0

#: How much of a child's error stream is kept. The tail and not a log, for the reason a spawned program's
#: stderr is kept anywhere: what explains a crash is the last thing it said (E184).
TAIL_CHARS = 2048

#: How many things one report is on the wire: the check's name, and what it said was wrong.
_PAIR = 2


class Unchecked(Exception):  # noqa: N818 — a check that produced no answer, said as the sentence it is
    """The child said nothing this end could read, and here is why."""


def main() -> None:
    """Read one plugin's path from argv, check it, and say what came back as one line of JSON."""
    reports = asyncio.run(checks.run(_subject(Path(sys.argv[1]))))
    _ = sys.stdout.write(json.dumps({"reports": [[one.check, one.wrong] for one in reports]}) + "\n")
    _ = sys.stdout.flush()


async def checked(path: Path) -> tuple[checks.Report, ...]:
    """Run the checks over one plugin in a child process, and answer what it said.

    The child is started with ``-I`` so that nothing about this process's import path changes what the plugin
    sees, and ``-m`` names the module above rather than a string of code: a bootstrap written as a line is a
    line nobody type-checks and nobody tests.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-m",
        "plyngent_creative.tools.remote",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        said, wrong = await asyncio.wait_for(process.communicate(), CHECK_SECONDS)
    except TimeoutError:
        process.kill()
        _ = await process.wait()
        msg = f"checking it took longer than {CHECK_SECONDS:.0f}s, so this program stopped it"
        raise Unchecked(msg) from None
    if process.returncode != 0:
        raise Unchecked(_tail(wrong) or f"the check stopped with {process.returncode}")
    try:
        payload: object = json.loads(said.decode("utf-8", errors="replace") or "null")
    except json.JSONDecodeError as exc:
        msg = f"the check answered something this program cannot read: {exc}"
        raise Unchecked(msg) from exc
    return _reports(payload)


def _subject(path: Path) -> checks.Subject:
    """The plugin as the child checks it, with what *it* answers to — a child has the same program installed."""
    groups, tools = checks.taken()
    return checks.Subject(plugin=path, taken_groups=groups, taken_tools=tools)


def _reports(payload: object) -> tuple[checks.Report, ...]:
    """The child's answer as reports, or the sentence saying that it did not answer with any."""
    if not isinstance(payload, dict):
        msg = "the check answered nothing"
        raise Unchecked(msg)
    rows = cast("dict[str, object]", payload).get("reports")
    if not isinstance(rows, list):
        msg = "the check answered without reports"
        raise Unchecked(msg)
    answers: list[checks.Report] = []
    for row in cast("list[object]", rows):
        if not isinstance(row, list) or len(cast("list[object]", row)) != _PAIR:
            continue
        name, wrong = cast("list[object]", row)
        answers.append(checks.Report(str(name), None if wrong is None else str(wrong)))
    return tuple(answers)


def _tail(wrong: bytes) -> str:
    """The end of what a child wrote to its error stream, stripped, and nothing when it was quiet."""
    return wrong.decode("utf-8", errors="replace")[-TAIL_CHARS:].strip()


if __name__ == "__main__":
    main()
