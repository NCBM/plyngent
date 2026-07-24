"""Shared pytest fixtures.

Ephemeral ``@tool`` tests should pass ``register=False`` so they do not pollute
the process catalog. Do not autouse an empty ``catalog_scope`` here: it would
hide builtins from ``default_tool_definitions`` during tests.
"""
