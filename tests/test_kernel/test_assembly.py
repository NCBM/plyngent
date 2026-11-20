"""What a launcher adds, which no story can say.

A component is declared by the program that *starts* this one, so nothing a person types and nothing a
model asks can make one appear: the only observer of "it was declared" is the assembly, and the only
caller of that is a launcher's ``run``. What the stories therefore cannot pin is the seam itself —
that a declaration handed to the kernel reaches the table instead of being silently dropped — and that
is the whole of what E143 asks this file to hold.
"""

from __future__ import annotations

from composat import Component, Declaration

from plyngent import boot


class Marker(Component):
    """A component nothing reaches for: it exists only to be counted in the table."""


async def test_a_component_a_launcher_brings_is_declared_beside_this_process_own(tmp_path) -> None:
    """The program this tree ships with declares nothing of its own but the broker; a launcher's arrives.

    Declaring is not building: nothing requires ``Marker``, so nothing constructs it, which is exactly
    the promise a launcher is given — what a component may do is what its declaration was constructed
    with, and it is built by this assembly when something asks for it (E143).
    """
    declared = Declaration(Marker)

    installed = await boot.install(tmp_path, journal=tmp_path / "journal.db", components=(declared,))
    try:
        assert installed.assembly.declaration(declared.key) == declared
    finally:
        await installed.close()
