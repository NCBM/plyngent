# Skills

A **skill** is a directory of instructions the agent can find, read, and follow:
a `SKILL.md` (frontmatter + Markdown body) plus whatever bundled files it needs
— scripts, references, templates. Plyngent discovers skills from your own user
directory and from the directories other harnesses install into, so one skill
folder can serve several agents.

## Layout

```text
<root>/
  pdf-processing/
    SKILL.md            # required: frontmatter + instructions
    scripts/extract.sh  # optional bundled files
    references/forms.md
```

`SKILL.md`:

```markdown
---
name: pdf-processing
description: Extract text and tables from PDFs; fill forms.
version: 1
allowed-tools: [run_argv, read_file]
---

# PDF processing

Use `scripts/extract.sh <file>` for the text layer, then ...
```

## Roots and discovery

Roots are scanned in priority order; the **first** skill with a given name wins
and the copies it shadows stay listed (`skill_list`, `/skills`) instead of
disappearing:

1. `[skills].paths` — explicit roots (highest priority)
2. our own user directory: `<user config>/skills` (`plyngent config path` shows
   the directory that contains `plyngent.toml`)
3. other harnesses: `~/.claude/skills`, and `<workspace>/.claude/skills` when
   `include_project_roots` is on (`discover = ["plyngent", "claude"]` by default)

A root may also *be* a skill directory (its own `SKILL.md`), which is handy when
you point `paths` at a single skill. Missing roots are not an error; `/skills`
lists the ones with no directory.

```toml
[skills]
enabled = true            # false = no skill tools and no prompt catalog
paths = ["~/team-skills"] # explicit roots; `~` expands to your home
discover = ["plyngent", "claude"]
include_project_roots = true
inject_catalog = true     # fold the catalog into the system prompt
max_catalog_skills = 50
allow_write = false       # true = standing grant for skill writes
```

## Frontmatter

The reader covers the YAML subset those files use — top-level `key: value`
scalars (quoted, plain, `true`/`false`, integers), inline `[a, b]` lists, block
`- item` lists, and `|` / `>` blocks. A plain scalar folds its indented
continuation lines. A nested mapping (say a `metadata:` block) is skipped rather
than guessed at, and a structurally broken block degrades to "no metadata" with
a note: the skill stays listable instead of vanishing.

| Field | Meaning |
|-------|---------|
| `name` | Skill name; must be lowercase letters, digits, and single hyphens (max 64). A name that cannot be used falls back to the directory name, with a note. |
| `description` | One line shown in the catalog and in `skill_list`. Missing → the first non-heading paragraph of the body. |
| `version`, `license`, `allowed-tools`, … | Kept and shown by `skill_list`; `allowed-tools` is **recorded, not enforced** today. |

## What the agent sees

- **Catalog** (system prompt, `inject_catalog`): names, sources, descriptions —
  never the bodies, so a turn only pays for the skill it actually reads.
- `skill_list` — what exists, where it came from, shadowed copies, and the notes
  from repaired frontmatter.
- `skill_read NAME [file]` — the `SKILL.md` body (ranges, line numbers, and a
  continuation token like any other read) or one bundled file, resolved inside
  the skill directory only.
- `skill_search PATTERN [skill]` — regex over every visible skill; hits come back
  as `<skill>/<file>:<line>: content`.

## Grants: reading vs writing

- **Reading needs no grant.** The host registers the roots as read-only roots, so
  `read_file`, `grep_files`, and `listdir` see skill files too. Writes are denied
  by the path policy — a skill directory never becomes a write grant.
- **Writing needs a grant.** `skill_create` / `skill_edit` ask the human for the
  skill directory: `o` once, `s` for the rest of the chat, anything else (or a
  timeout) denies. The prompt is independent of YOLO: `--yes` / `/yolo` never
  skips it. `[skills].allow_write = true` is the standing grant, and the path
  denylist (`[agent].path_denylist`) still wins over everything.

`skill_create` writes a new `SKILL.md` (name, description, body) plus any
`files` you list; `skill_edit` changes one file by literal replacement
(`old_string` → `new_string`, all matches with `replace_all`), by `append`, or
by rewriting it. A rewritten `SKILL.md` is re-parsed, and the reply says so when
the description (or the whole frontmatter) is gone.

## Care and safety

- A skill is **data, not code**: nothing in a skill body bypasses the confirm
  flow, the path denylist, or the command denylist. Bundled scripts run through
  the normal process tools, so they still need whatever grant those require.
- Third-party skills (other harnesses' directories, shared repos) are untrusted
  input. Read before you follow, keep `paths` pointed at directories you trust,
  and prefer `allow_write = false` + per-write prompts when a skill comes from
  outside.
- Keep skills small and specific; long bodies are better split into a short
  `SKILL.md` plus reference files the model reads on demand.
