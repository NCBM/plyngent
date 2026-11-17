"""What this program does with a network, split into three things that are tested by replacing one.

A request has three separable halves and this package keeps them apart:

* :mod:`plyngent.kernel.net.policy` — what a URL is, what the name behind it resolves to, and whether the
  request leaves the machine at all. Pure but for one lookup, and it never asks a person.
* :mod:`plyngent.kernel.net.client` — one request, hop by hop, with a gate asked before each connection.
  It knows HTTP and nothing about who may say yes.
* :mod:`plyngent.kernel.net.web` — the web half of the tools: a table of engines and the two verbs a model
  can reach through one. :mod:`~plyngent.kernel.net.shipped` is what this build puts in that table, and
  ``deepseek`` and ``bing`` are the implementations it names.
* :mod:`plyngent.kernel.tools.net` — the tools: the arguments, the question a person is asked, and the
  result the model reads. That is where a reason becomes a sentence somebody agrees to.

The seam between the last two is the whole point: **the engine does not know there are people.** A gate
that answers "no" ends the request, and a gate that answers "yes" leaves no trace in the engine — so the
same bytes are fetched the same way whether a person agreed once for a private host or a request happened
to need no agreement at all (§0.9, E27/E34).
"""
