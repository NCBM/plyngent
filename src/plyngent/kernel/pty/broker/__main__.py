"""``python -m plyngent.kernel.pty.broker`` — how the kernel starts the process that forks.

An entry point rather than a second program: it takes the channel it was handed (a descriptor on POSIX,
a pipe name on Windows) and the epoch it is, and it serves until the kernel goes away. Nothing here is
meant to be run by hand, and nothing is read from the environment.
"""

from plyngent.kernel.pty.broker.child import main

if __name__ == "__main__":
    raise SystemExit(main())
