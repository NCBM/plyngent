"""The three rules a plugin author reads before writing one.

## 1. A plugin is one function and a tuple

The file answers `enable()` with declarations — the value `boot.install(components=…)` takes — and the only
declaration a plugin usually needs is a tool set::

    from composat import Declaration
    from plyngent.kernel.providing import ToolProvider, ToolSet

    class Greets(ToolProvider):
        group = "greets"          # what a session asks for
        summary = "a greeting"    # the phrase the index line prints
        guide = "`greet` says hello."
        timeout = 5.0             # how long it may take to come up

        async def materialise(self):
            return (GREET,)       # the tools, once somebody asks

        async def close(self):
            return

    def enable():
        return (Declaration(ToolSet, instance="greets", kwargs={"providers": (Greets(),)}),)

The file is one file, and it goes in this program's plugin directory — the path `plugin_check` answers with.

## 2. A tool's name is a constant

A name is written into history: it is the `Effect` row a call leaves, and the `tool-called` event a person
reads. Renaming a tool therefore renames a fact in every conversation that has the name in it, and there is
no migration for that. Pick the name before the first call and treat it as a constant of the plugin.

The same goes for the group: a session asks for it by name, and the name is what a conversation remembers
afterwards.

## 3. The shortest way to try a plugin out

`plugin_check` reads the file and says what is wrong with what it declares. When that passes, `plugin_test`
runs the steps in a kernel of its own::

    {"name": "greets", "steps": [
      {"send": "hello"},
      {"model": {"text": "I will greet"}},
      {"model": {"text": "", "calls": [{"tool": "tool_require", "arguments": {"group": "greets"}}]}},
      {"expect": {"tag": "tool-returned", "fields": {"tool": "tool_require"}}},
      {"model": {"text": "", "calls": [{"tool": "greet", "arguments": {"name": "you"}}]}},
      {"expect": {"tag": "tool-returned", "fields": {"tool": "greet", "status": "ok"}}}
    ]}

The answer carries a handle, and `plugin_debug(handle, steps)` carries on from where the run stopped. A step
that failed leaves the handle valid: fix the file, then ask again.

A step is a linear sequence of three keys, and a step may hold all three in this order — the model answers,
the line is sent, what arrives is looked for:

* `{"model": {"text": "…", "calls": [...]}}` — what the scripted model answers. `{"fail": "timeout"}` is an
  endpoint that could not be asked, and `{"text": "…", "cut_after": 2, "failure": "timeout"}` is an answer
  that broke after two pieces.
* `{"send": "…"}` — a line a person types. `{"send": {"cmd": "compact", "text": "…"}}` is a command, in the
  shape the contract already has.
* `{"expect": {"tag": "tool-called", "fields": {...}}}` — the first event after the previous step whose tag
  and fields match. `tag` is exact, `fields` is a subset, and a scalar or a list is equal.

What an expectation names is the wire the kernel speaks — `tool-called`, `tool-returned`, `noticed`,
`turn-finished` — and not a line a screen drew. That is the same contract a second frontend reads, so a test
written here says something true of every frontend.

## Trying one without this program

A plugin is a Python file and the checks are a library, so a test of your own can run them::

    # /// script
    # requires-python = ">=3.14"
    # dependencies = ["plyngent-creative"]
    # ///
    from pathlib import Path

    from plyngent_creative.tools.checks import Subject, run, summary

    reports = asyncio.run(run(Subject(plugin=Path("say_hi.py"))))
    print(summary(reports, plugin="say_hi"))

`uv run that_file.py` installs what it names and runs it, which is the whole of setting up an environment
for one check.
