"""The shape a commit message keeps, checked before a commit is written.

    tools/commit_message.py <message file>...

`prek.toml` runs this at the `commit-msg` stage: a subject of `scope: statement` whose statement is one sentence, a
blank line, and one unwrapped paragraph of reason of at most five sentences, where a clause past fifteen words or CJK
characters reads as a sentence of its own. Wording that sets one reading against another is a break as well, since
`AGENTS.md` keeps it out of a message. Every break is reported with the line it is on, and the exit status is 1.
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

#: A mark that ends a sentence whatever stands after it.
FULL_WIDTH = "。？！；．"
#: A mark that ends a sentence where a space or the line's end follows it, so `java.lang.Object` stays one word.
HALF_WIDTH = ".?!;"
#: One scope: a part of the tree, a slash joining a name to the part inside it.
SCOPE = r"[a-z_][a-z0-9_]*(?:/[a-z_][a-z0-9_]*)*"
#: A subject opens with one or more of them, a comma joining a set, then a colon and a space.
SCOPED = re.compile(SCOPE + "(?:, " + SCOPE + ")*: ")
#: How many sentences the paragraph of reason may hold.
SENTENCES = 5
#: How many words, or CJK characters, one clause between commas may hold before it counts as a sentence.
CLAUSE = 15
#: Where a clause ends.
COMMAS = re.compile("[,，]")
#: What counts as one word, and what counts one apiece in a script written without spaces.
WORD = re.compile(r"[A-Za-z0-9_][\w’'.-]*")
CJK = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
#: Wording that sets one reading against another.
AGAINST = re.compile(r"\b(?:although|rather than|instead of)\b|\bnot\b[^.;?!]*\bbut\b", re.IGNORECASE)
#: What `git commit -v` puts between the message and the diff it shows below it.
SCISSORS = "# ------------------------ >8 ------------------------"


def terminators(text: str) -> list[int]:
    """Where a sentence ends in `text`."""
    found: list[int] = []
    for at, char in enumerate(text):
        if char in FULL_WIDTH or (char in HALF_WIDTH and (at + 1 == len(text) or text[at + 1].isspace())):
            found.append(at)
    return found


def clause_size(clause: str) -> int:
    """How many words `clause` holds, a CJK character counting one apiece."""
    return len(WORD.findall(clause)) + len(CJK.findall(clause))


def long_clauses(text: str) -> list[str]:
    """The clauses between commas that hold more than a sentence's worth of words, one per sentence mark."""
    bounds = [-1, *terminators(text), len(text)]
    return [
        clause.strip()
        for start, end in itertools.pairwise(bounds)
        for clause in COMMAS.split(text[start + 1 : end])
        if clause_size(clause) > CLAUSE
    ]


def sentences(text: str) -> int:
    """How many sentences `text` reads as: one per mark, and one more for each clause past its limit."""
    return len(terminators(text)) + len(long_clauses(text))


def message_text(raw: str) -> list[str]:
    """The message as its writer wrote it: git's comments and what stands below the scissors line are git's."""
    lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith(SCISSORS):
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[0]:
        del lines[0]
    while lines and not lines[-1]:
        del lines[-1]
    return lines


def faults(lines: list[str]) -> list[tuple[int, str]]:
    """Every way the message breaks the shape, each with the line it is on."""
    if not lines:
        return [(1, "the message is empty")]
    broken = [*_subject(lines[0]), *_against(lines)]
    blocks = _blocks(lines[1:])
    paragraph = next((block for block in blocks if block), None)
    if paragraph is None:
        return [*broken, (1, "the reason follows in one paragraph, below a blank line")]
    return sorted(broken + _paragraph(blocks, paragraph))


def _subject(subject: str) -> list[tuple[int, str]]:
    """What the first line gets wrong: the scope it opens with, and the one sentence that follows it."""
    scoped = SCOPED.match(subject)
    if scoped is None or not subject[scoped.end() :].strip():
        return [(1, "the subject reads `scope: statement`, the scope lowercase and the statement after it")]
    found = terminators(subject[scoped.end() :])
    if not found:
        return []
    return [(1, f"the statement is one sentence, and the mark at column {scoped.end() + found[0] + 1} ends it")]


def _against(lines: list[str]) -> list[tuple[int, str]]:
    """Where the message sets one reading against another."""
    broken: list[tuple[int, str]] = []
    for at, line in enumerate(lines):
        found = AGAINST.search(line)
        if found:
            broken.append((at + 1, f"the message says `{found.group(0)}`, and that wording stays out"))
    return broken


def _paragraph(blocks: list[list[tuple[int, str]]], paragraph: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """What the reason gets wrong: how it is set out, and how many sentences it reads as."""
    broken: list[tuple[int, str]] = []
    if blocks[0]:
        broken.append((2, "a blank line stands between the subject and the paragraph"))
    broken.extend(
        (block[0][0] + 2, "the reason is one paragraph, and a blank line starts another")
        for block in blocks
        if block and block is not paragraph
    )
    if len(paragraph) > 1:
        broken.append(
            (paragraph[1][0] + 2, "the paragraph is one line, since a wrap is the writer's and a viewer's is not")
        )
    count = sentences(" ".join(line for _at, line in paragraph))
    if count > SENTENCES:
        broken.append(
            (
                paragraph[0][0] + 2,
                f"the paragraph gives its reason in at most {SENTENCES} sentences, and this one reads as {count}, "
                f"a clause past {CLAUSE} words counting one",
            )
        )
    return broken


def _blocks(lines: list[str]) -> list[list[tuple[int, str]]]:
    """The runs of text among `lines`, each line with its offset, a blank line closing one run and opening the next."""
    blocks: list[list[tuple[int, str]]] = [[]]
    for at, line in enumerate(lines):
        if line:
            blocks[-1].append((at, line))
        else:
            blocks.append([])
    return blocks


def _complain(line: str) -> None:
    """Write one line of report to the error stream."""
    _ = sys.stderr.write(line + "\n")


def main(argv: list[str]) -> int:
    """Report every fault of each message the command line names, and pass only when there is none."""
    if not argv:
        _complain("usage: commit_message.py <message file>...")
        return 2
    status = 0
    for name in argv:
        broken = faults(message_text(Path(name).read_text(encoding="utf-8", errors="replace")))
        if broken:
            status = 1
            _complain(f"{name}: {len(broken)}")
            for at, what in broken:
                _complain(f"  line {at}: {what}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
