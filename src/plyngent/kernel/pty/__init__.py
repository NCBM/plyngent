"""PTY sessions as the kernel owns them: a registry per host, bytes as bytes.

Four parts, deliberately apart: the platform layer (``backend``) knows how a child gets a terminal, the
registry (``registry``) knows which children exist and belongs to one ``Host`` rather than to the
process, ``keys`` turns ``<c-c>`` into the bytes a terminal would send, and ``policy`` answers whether
a person has to agree to changing a session — an answer, never a question.
"""
