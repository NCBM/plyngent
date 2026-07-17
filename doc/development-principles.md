# Development Principles

## If you are new to project

### Have a general overview of the project

You should at least have a clear view of the general architecture. This aims for:

1. keeping the style of codebase consistent.
2. using or extending existing objects or functions instead of writing new but duplicate code, since the latter is usually a kind of **SHIT**.

For a general architecture reference, see [here](./architecture.md).

This does not mean you should immediately grasp the whole codebase, but **you should not make changes until you have a clear view of the general architecture**.

### Commit messages

This project uses scoped commit message. Examples:

- `project: init`: initializing a new project.
- `deps: add fastapi`: add/upgrade/removing a dependency.
- `deps/dev: upgrade ruff`: add/upgrade/removing a dev dependency.
- `core/mq: fix incorrect message sending`, `router: add routing service for xxx`: add/change/fix/refactor/removing functional code.
- `test/webserver: change test client to niquests`: add/change/fix/refactor/removing tests.
- `scripts/i18n: implement dumping translation keys`: add/change/fix/refactor/removing general scripts.
- `ci/lint: run ruff style check`, `ci/build: fix changelog generation`: add/change/fix/refactor/removing CI workflows.

Once conventional commit was good, until redundant forms like `fix: fix ...`, `refactor: refactor ...` came up. So we use scoped commit message here.

If you are still not quite familiar to this, check the git log in any way you prefer.

## General information and suggestions

### Local development environment

Syncing packages: `pdm install` the first time, then `pdm sync`. The former can fully initialize the dependencies, and the latter can continuously keep consistent dependencies.

For other commands like adding or removing packages, see [PDM Documentation](https://pdm-project.org/en/latest/usage/dependency/). These commands usually updates lock file automatically by default, so `pdm sync` should only be used when after `git pull/fetch/checkout`.

### Type checking and linting

Firstly, we prefer using `pdm run <command/script>` to directly running commands in venv (whether in activated environment or not).

Then for every projects, check the dev dependency group where you can find tools used for type checking and linting.

Type checking: `pdm run basedpyright .`
Linting: `pdm run ruff check .`
Formatting (apply): `pdm run ruff format .`
Formatting (CI check): `pdm run ruff format --check .`

CI (`.github/workflows/ci.yml`) runs lint, **format check**, typecheck, then tests. Always run `pdm run ruff format .` before push so `ruff format --check` does not fail on `main` after a release tag has already published.

<!-- Contents in comments are for projects with long-period developments, but this is a new project.

### Branches

In most cases, we do not need and should not make bare commits on the main branch.

### Tests

...

-->
