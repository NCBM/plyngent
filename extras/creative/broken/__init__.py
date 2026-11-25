"""Four plugins, each with one thing wrong, that the checks are held against.

A checker nobody has seen refuse anything may be a checker that checks nothing, so these files exist for
the same reason a test asserts its own failure: the suite is run over them and the sentences are held by
``tests/test_creative_checks.py``. Each file has one fault and not two, because a run reports the first
sentence a check answers and one fault is what a sentence proves.

They are also the shape of a plugin written out — a tuple of declarations, each a tool set of providers —
so an author who copies one gets the shape right and learns the rest from what the checks say.
"""
