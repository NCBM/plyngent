"""The kernel: what a turn is, and what it is allowed to touch.

Everything here is transport-agnostic, and the boundary is a directory rather than a habit:

* nothing under ``plyngent.kernel`` imports ``plyngent.cli``, ``click``, ``rich``,
  ``prompt_toolkit``, ``readline`` or ``termios``;
* nothing here touches ``sys.stdin`` / ``sys.stdout``, and no event carries ANSI, colour or a
  terminal width;
* every request for a human goes through ``Host.human``, and every outcome leaves as an event.

Two owners sit below this package and are not its business: the withdrawal of an already
accepted side effect (a subprocess, an fd, a remote lease) is explicit and offline, so those
resources live on ``Host``, which does not get replaced within a process. What a replacement
may carry is only what stops working when nobody holds it — connections, subscriptions,
in-memory projections — and that is what ``Session`` is for.
"""
