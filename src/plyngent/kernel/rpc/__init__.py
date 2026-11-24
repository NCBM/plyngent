"""A conversation with a child process, in the parts that are not about what it is told.

Three parts, and the split is what keeps the next protocol cheap:

* ``framing`` is where a message ends and how much of one is read: one JSON object per line, bounded.
* ``peer`` is the conversation above it — ids, answers, notifications, and the one request that arrives
  from the wrong direction.
* ``child`` is the process: how it is started, how it is stopped, and what happens when it dies.

Nothing here knows what a method is called. What is written above this package is one protocol's
vocabulary, and a second protocol — a language server, a database's own client — is a second peer over
the same two modules.
"""
