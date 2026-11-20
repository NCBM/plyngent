# AGENTS.md

plyngent is an LLM chat and agent toolkit (Python 3.14+, PDM): a kernel that talks to a model, a client that
draws it, and a wire between them that another program may speak.

This file is the map. It carries the commands, the rules for what a sentence says, the rules for what a
commit is, and where every other claim lives; a claim restated here is a second copy that can disagree, so
a restatement is a bug in this file.

## Commands

```bash
pdm install                                    # first time; pdm sync after a pull
pdm run pytest -q -o faulthandler_timeout=120  # the whole suite, about two minutes
pdm run pytest -q tests/test_kernel            # one bucket while working
pdm run ruff check src tests
pdm run ruff format --check src tests
pdm run basedpyright src tests
```

`prek install` wires those three as the commit gateway (`prek.toml`), `prek run --all-files` runs them on
demand.

## Writing

Docs, docstrings, code comments and commit messages carry statements: the subject and the claim first, the
reason after. An invariant may be stated negatively (`a field set after the attach is never read`).

Wording that sets one reading against another stays out — `although`, `rather than`, `instead of`,
`not X but Y`. What the tree refused is named where the refusal is made (a `Failure` member, a `Refusal`,
the sentence in `cli/render.py`), and a claim about the code stands on its own.

A fact is said by the kernel — which refusal, which entry, which route — and the sentence is written where
the reader is. `cli/render.py` holds one line per `Refusal`, and `set(_REFUSALS) == set(Refusal)` is
asserted so the table cannot fall behind the set; a `Failure` travels with the message the tool wrote, and
the frontend does not re-word it.

## Commits

A message reads `scope: statement`, then one paragraph giving the reason, in the state the change leaves the
tree. The exploration that produced it does not appear, and neither does a correction of something the same
change introduced. One coherent change per commit.

Work stays unstaged while it is worked out: staging, amending, rebasing and resetting are how a change comes
to have the commits it should, and a commit is settled once its content is the content to keep. Unpushed
history is rewritten freely. A pushed commit is rewritten when the change itself was wrong, and the remote is
brought to the rewritten history in the same breath.

## Where the truth is

| What you need to know | Where it is written |
|---|---|
| what the program does, and why that is the specification | `tests/stories/` — a story drives the real program through a real connection, and every story runs on each transport the platform can open (`KERNEL_TRANSPORTS`) |
| the contract a second frontend speaks | `src/plyngent/wire/SPEC.md`, `wire/manifest.py` |
| what no story can express — import rules, exhaustive unions, declared schemas, help text | `tests/test_structure.py`, `tests/cli/` |
| what landed, in order, and why | `git log` — the message is the record; this tree keeps no changelog |
| the decisions and the refusals behind them, and what is still owed | `docs/rewrite-requirements.md` — a local working doc, excluded from git, so a fresh clone does not have it; §5 is the `E<n>` a code docstring cites |
| how the program is assembled and started | `src/plyngent/__main__.py` (`run`), `src/plyngent/boot.py`, `config.py` |
| lint, type and test rules | `pyproject.toml`, `prek.toml` |
