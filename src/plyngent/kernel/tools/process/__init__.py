"""The tool group that runs things: programs under the workspace, and children on a terminal.

The two halves share the word "process" and nothing else. ``execution`` runs commands to completion and
reports what they said; ``pty`` owns children that keep running, are read a piece at a time, and are
typed into. Keeping them apart is what keeps the command tool from having to know about sessions and the
terminal tools from having to know about exit codes.
"""
