"""The CLI: the adapter that owns the terminal.

Modules here read what a human typed, render what the kernel reported, and turn a keystroke into a
command. They import ``plyngent.wire`` and nothing else of ours — the contract is the only thing a
client may know — so the rules of the terminal (width, colour, the cursor, grapheme clusters) stop at
this package and the kernel's rules stop at ``plyngent.wire``.

Four modules, and each owns one thing: ``grammar`` what a line means, ``render`` what a person reads,
``line`` who holds the cursor, ``app`` the three tasks and the exit. ``plyngent/__main__.py`` is the
only place that knows both halves, and it does nothing but hand one to the other.
"""
