"""Why a call did not finish, as a closed set, and the one place an OS error becomes one of them.

A tool does not write a sentence about what went wrong, and it does not decide whether trying again
is worth it. It says *what kind of thing* went wrong, and the table in
:mod:`plyngent.kernel.toolresult` turns that into the status the model reads and the retry the caller
acts on. One wording, one answer, wherever the same failure happens.

The set is closed on purpose. A new kind of trouble gets a member here, next to the answer it implies,
and not a new string at a raise site — which is how ``retryable: bool`` used to be filled in by
whoever happened to be writing the handler.

``of`` is the only errno table in the tree: a tool catches ``OSError`` and asks for its kind, so
``write_file`` and ``copy_path`` cannot disagree about what ``EACCES`` means. Platform differences
belong here too, for the same reason.
"""

from __future__ import annotations

import errno
from enum import StrEnum


class Failure(StrEnum):
    """What went wrong, in kinds that change what the caller should do about it.

    ``ARGS`` is the model's mistake — the arguments are not what this tool takes, and ``refused`` is
    reserved for "you may not", not for "you wrote it wrong". ``DENIED`` is a policy that no grant
    lifts. ``BLOCKED`` is a missing step — a grant, a parameter, a permission to overwrite — that a
    hint can name. The rest are the world's answers, and they differ only in whether asking again
    could possibly help.
    """

    ARGS = "args"
    DENIED = "denied"
    BLOCKED = "blocked"
    MISSING = "missing"
    PERMISSION = "permission"
    BUSY = "busy"
    LIMIT = "limit"
    TIMEOUT = "timeout"
    PROTOCOL = "protocol"
    #: A tool the client offered was called and never answered: the client is gone, or the deadline
    #: passed. What it did is unknown on purpose — the effect happened in somebody else's process, so
    #: the call is neither "refused" nor "nothing happened" (§10.8).
    CLIENT = "client"
    BUG = "bug"

    @classmethod
    def of(cls, exc: OSError) -> Failure:
        """The kind of failure an OS error is, the same way wherever it was raised.

        An errno nobody classified is reported as ``BUG``: the table is the
        whole point, and a silent fallback into "some other error" would put the guessing back at the
        call site.
        """
        if exc.errno is None:
            return cls.BUG
        return _LOOKUP.get(exc.errno, cls.BUG)


class Retry(StrEnum):
    """What a caller should do next, said in the three ways that are actually different.

    ``bool`` could only say "again" or "not again", which conflates "the world was busy" with "your
    arguments are wrong" — two completely different next moves for whoever is reading the result.
    """

    NEVER = "never"
    SAME_CALL = "same_call"
    FIX_ARGS = "fix_args"


def _codes(*names: str) -> frozenset[int]:
    """The numbers this platform gives *names*; a name it does not have is simply not in the set."""
    return frozenset(value for name in names if (value := getattr(errno, name, None)) is not None)


#: The errno values that mean the same kind of trouble, grouped the way the answers are. Names, because
#: the numbers differ per platform and some names do not exist on every one.
_BY_ERRNO: dict[Failure, frozenset[int]] = {
    Failure.MISSING: _codes("ENOENT", "ENOTDIR"),
    Failure.PERMISSION: _codes("EACCES", "EPERM", "EROFS"),
    Failure.BLOCKED: _codes("EEXIST"),
    Failure.BUSY: _codes("EBUSY", "EAGAIN", "EWOULDBLOCK", "ETXTBSY", "ENOTEMPTY"),
    Failure.LIMIT: _codes("ENOSPC", "EDQUOT", "EFBIG", "ENAMETOOLONG", "EMFILE"),
    Failure.TIMEOUT: _codes("ETIMEDOUT"),
    Failure.PROTOCOL: _codes("ECONNREFUSED", "ECONNRESET", "EHOSTUNREACH", "ENETUNREACH"),
}

_LOOKUP: dict[int, Failure] = {code: failure for failure, codes in _BY_ERRNO.items() for code in codes}
