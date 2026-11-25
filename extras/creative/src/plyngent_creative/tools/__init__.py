"""The tools this distribution serves: what a plugin is checked with, and what it is driven with.

They are a subpackage because they reach further than the top of the package may: a tool needs the kernel's
own declarations — what a tool spec is, what an environment grants — and the top is held to the three doors
a client has (`tests/test_structure.py`). So the reach is the group's, and it is one group's: nothing outside
this package imports anything from here.
"""
