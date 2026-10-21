# Minimal stubs for ``send2trash`` (a required dependency).
# Only the surface ``plyngent.tools.trash`` uses is declared; the real package is
# pure Python over ctypes (freedesktop trash spec / Finder trash / Recycle Bin).

from collections.abc import Iterable

class TrashPermissionError(PermissionError): ...

def send2trash(paths: str | Iterable[str]) -> None: ...
