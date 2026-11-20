"""PTY sessions as the kernel owns them: a registry per host, and one broker per process.

Five parts, deliberately apart: ``broker`` is the process that has a terminal to give (and the only
part that forks), ``registry`` knows which children *this host* has and belongs to it, ``errors``
says which kind of refusal each of them is, ``keys`` turns ``<c-c>`` into the
bytes a terminal would send, and ``policy`` answers whether a person has to agree to changing a session
— an answer, never a question.

``backend`` is the platform layer under the broker: how a child is given a terminal and how its bytes
come back. Nothing but the broker's own process imports it.
"""
