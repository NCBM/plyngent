# Writing a plugin

A plugin is a Python file, and this program loads one when a person names it. This page is what to know
before writing the first one: the shape, the two rules a schema cannot state, and the shortest way to try
out what you wrote.

## 1. A plugin is one function and a tuple

The file answers `enable()` with declarations — the value `boot.install(components=…)` takes. What a plugin
usually brings is one tool set, and a tool set is a tuple of providers:

    from composat import Declaration
    from plyngent.kernel.providing import ToolProvider, ToolSet

    class Greets(ToolProvider):
        group = "greets"          # what a session asks for
        summary = "a greeting"    # the phrase the index line prints
        guide = "`greet` says hello to one name and changes nothing."
        timeout = 5.0             # how long it may take to come up

        async def materialise(self):
            return (GREET,)       # the tools, once somebody asks for the group

        async def close(self):
            return

    def enable():
        return (Declaration(ToolSet, instance="greets", kwargs={"providers": (Greets(),)}),)

A tool is a `ToolSpec`: a name, an argument package, the handler, and what the handler may reach. The group
is not in the table until a session asks for it, which is what makes a provider a provider.

The file is one file, and it goes in this program's plugin directory — the path `plugin_check` answers with
when there is nothing there yet.

## 2. A tool's name is a constant

A name is written into history: it is the effect row a call leaves, and the `tool-called` event a person
reads. Renaming a tool therefore renames a fact in every conversation that has the name in it, and there is
no migration for that. Pick the name before the first call, and treat it as a constant of the plugin.

The same goes for the group. A session asks for it by name, and the name is what the conversation remembers
afterwards.

## 3. The shortest way to try a plugin out

`plugin_check(name)` reads the file and says what is wrong with what it declares: one file, four checks,
about a second, and nothing is started. When that passes, `plugin_test` runs a script against it in a kernel
of its own:

    [
      {"model": {"text": "", "calls": [{"tool": "tool_require", "arguments": {"group": "greets"}}]},
       "send": "hello",
       "expect": {"tag": "tool-called", "fields": {"tool": "tool_require"}}},
      {"expect": {"tag": "tool-returned", "fields": {"tool": "tool_require", "status": "ok"}}},
      {"model": {"text": "", "calls": [{"tool": "greet", "arguments": {"name": "you"}}]},
       "expect": {"tag": "tool-returned", "fields": {"tool": "greet", "status": "ok"}}},
      {"model": {"text": "said hello"},
       "expect": {"tag": "turn-finished", "fields": {"turn": 1, "reason": ""}}}
    ]

The answer carries a handle, and `plugin_debug(handle, steps)` carries on from where the run stopped. A step
that failed leaves the handle valid: fix the file, then ask again. `plugin_stop(handle)` finishes and takes
the run's temporary root away.

A script is a linear sequence of steps, and a step says three things, in this order:

* `{"model": {"text": "…", "calls": [...]}}` — what the scripted model answers. `{"fail": "timeout"}` is an
  endpoint that could not be asked, and `{"text": "…", "cut_after": 2, "failure": "timeout"}` is an answer
  that broke after two pieces. The answer is handed over before the line is sent, because a turn takes the
  next answer the moment it asks.
* `{"send": "…"}` — a line a person types. `{"send": {"cmd": "compact", "text": "…"}}` is a command, in the
  shape the contract already has.
* `{"expect": {"tag": "tool-called", "fields": {…}}}` — the first event after the last match whose tag is
  that word and whose fields carry those values. `fields` is a subset all the way down, and a scalar or a
  list is equal.

What an expectation names is the wire the kernel speaks — `tool-called`, `tool-returned`, `noticed`,
`turn-finished` — and a step that matched nothing answers with what arrived in place of it. That is the same
contract a second frontend reads, so what a script says is true of every frontend.

## Trying one without this program

The checks are a library as well as a call, so a test of your own can run them. One file that names what it
needs is the whole of the setup:

    # /// script
    # requires-python = ">=3.14"
    # dependencies = ["plyngent-creative"]
    # ///
    import asyncio
    from pathlib import Path

    from plyngent_creative.tools.checks import Subject, run, summary, taken

    groups, tools = taken()
    plugin = Path("say_hi.py")
    reports = asyncio.run(run(Subject(plugin=plugin, taken_groups=groups, taken_tools=tools)))
    print(summary(reports, plugin=plugin.stem))

`uv run that_file.py` installs what it names and runs it.
