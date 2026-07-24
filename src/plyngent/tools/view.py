"""Transactional path-scoped views over durable / publishable trees."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar, cast, overload, override

U = TypeVar("U")


class ViewStore(Protocol):
    """Backend for a root PersistentDataView tree."""

    async def load(self) -> object: ...

    async def store(self, root: object) -> None: ...


class MemoryViewStore:
    """In-memory root store (tests / process-only state)."""

    def __init__(self, initial: object | None = None) -> None:
        self._root: object = {} if initial is None else initial

    async def load(self) -> object:
        return self._root

    async def store(self, root: object) -> None:
        self._root = root


@dataclass
class _TxnState:
    """Private buffer for an open root transaction."""

    root: object
    dirty: bool = False
    depth: int = 0


class PersistentDataView[T](AbstractAsyncContextManager["PersistentDataView[T]"]):
    """Path-scoped view over a durable/publishable tree.

    ``T`` is the type of data referenced at this path (for load/store/typed()).
    The view **is** the transaction context manager: ``async with view``.
    Mutations outside an open txn raise. Nested ``async with`` joins the same
    root txn (savepoint-lite: depth counter; full rollback on any exception).
    """

    _store: ViewStore
    _path: tuple[str | int, ...]
    _bound_type: type[T] | None
    _txn: _TxnState | None
    _domain: dict[tuple[str | int, ...], object]
    _child_cache: dict[tuple[str | int, ...], PersistentDataView[Any]]

    def __init__(
        self,
        store: ViewStore,
        *,
        path: tuple[str | int, ...] = (),
        bound_type: type[T] | None = None,
        _txn: _TxnState | None = None,
        _domain: dict[tuple[str | int, ...], object] | None = None,
        _child_cache: dict[tuple[str | int, ...], PersistentDataView[Any]] | None = None,
    ) -> None:
        self._store = store
        self._path = path
        self._bound_type = bound_type
        self._txn = _txn
        self._domain = _domain if _domain is not None else {}
        self._child_cache = _child_cache if _child_cache is not None else {}

    @property
    def path(self) -> tuple[str | int, ...]:
        return self._path

    def _require_txn(self) -> _TxnState:
        if self._txn is None or self._txn.depth < 1:
            msg = "PersistentDataView mutation/read of live buffer requires an open transaction (async with view)"
            raise RuntimeError(msg)
        return self._txn

    def _navigate(self, root: object) -> object:
        current: object = root
        for key in self._path:
            if isinstance(current, dict):
                current = cast("dict[object, object]", current).get(key)
            elif isinstance(current, list) and isinstance(key, int):
                lst = cast("list[object]", current)
                current = lst[key] if 0 <= key < len(lst) else None
            else:
                return None
        return current

    def __getitem__(self, key: str | int) -> PersistentDataView[Any]:
        child_path = (*self._path, key)
        cached = self._child_cache.get(child_path)
        if cached is not None:
            return cached
        child: PersistentDataView[Any] = PersistentDataView(
            self._store,
            path=child_path,
            bound_type=None,
            _txn=self._txn,
            _domain=self._domain,
            _child_cache=self._child_cache,
        )
        self._child_cache[child_path] = child
        return child

    def load(self) -> T:
        """Materialize the value at this path (from txn buffer or by reading store root snapshot)."""
        if self._txn is not None and self._txn.depth >= 1:
            if self._path in self._domain:
                return cast("T", self._domain[self._path])
            value = self._navigate(self._txn.root)
            return cast("T", value)
        # Outside txn: only allowed as a frozen read via store is async — sync load
        # outside txn is not supported for remote stores; require txn.
        msg = "load() requires an open transaction"
        raise RuntimeError(msg)

    def store(self, value: T) -> None:
        txn = self._require_txn()
        self._domain[self._path] = value
        self._write_at_path(txn, self._path, value)
        txn.dirty = True

    def _write_at_path(self, txn: _TxnState, path: tuple[str | int, ...], value: object) -> None:
        """Write *value* into the txn buffer at *path* (may be a live domain object)."""
        if not path:
            txn.root = value
            return
        parent = self._ensure_parent_for(txn.root, path)
        last = path[-1]
        if isinstance(parent, dict):
            cast("dict[object, object]", parent)[last] = value
        elif isinstance(parent, list) and isinstance(last, int):
            lst = cast("list[object]", parent)
            while len(lst) <= last:
                lst.append(None)
            lst[last] = value
        else:
            msg = f"cannot store at path {path!r}"
            raise TypeError(msg)

    def _ensure_parent_for(self, root: object, path: tuple[str | int, ...]) -> object:
        """Ensure dict parents along *path* exist; return parent container for last key."""
        if not path:
            return root
        current: object = root
        for key in path[:-1]:
            if not isinstance(current, dict):
                msg = f"cannot navigate non-dict parent at {key!r}"
                raise TypeError(msg)
            mapping = cast("dict[object, object]", current)
            child: object | None = mapping.get(key)
            if child is None:
                child = cast("object", {})
                mapping[key] = child
            current = child
        return current

    @staticmethod
    def _serialize_value(value: object) -> object:
        """Prefer domain ``to_raw()`` so stores receive JSON-ish trees, not live objects."""
        to_raw = getattr(value, "to_raw", None)
        if callable(to_raw):
            return to_raw()
        return value

    def _flush_domain(self, txn: _TxnState) -> None:
        """Rewrite domain objects into the buffer in serializable form before store."""
        for path, value in self._domain.items():
            self._write_at_path(txn, path, self._serialize_value(value))

    def _materialize_domain(self, typ: type[Any], current: object) -> object:
        if current is not None and isinstance(current, typ):
            self._domain[self._path] = current
            return current

        # Hybrid domain objects (e.g. TodoStack): reconstruct from durable raw.
        from_raw = getattr(typ, "from_raw", None)
        if current is not None and callable(from_raw):
            converted: object = from_raw(current)
            self._domain[self._path] = converted
            self.store(cast("T", converted))
            return converted

        ctor = cast("Any", typ)
        if current is None:
            try:
                created: object = ctor()
            except TypeError:
                created = None
            self._domain[self._path] = created
            if created is not None:
                self.store(cast("T", created))
            return created
        try:
            converted = ctor(current)
        except TypeError:
            converted = current
        self._domain[self._path] = converted
        self.store(cast("T", converted))
        return converted

    @overload
    def typed(self, typ: None = None) -> T: ...

    @overload
    def typed(self, typ: type[U]) -> U: ...

    def typed(self, typ: type[U] | None = None) -> T | U:
        """Return the live bound data at this path (not a view).

        - ``typed()`` → ``T`` when bound
        - ``typed(U)`` → rebind / convert as ``U``
        """
        txn = self._require_txn()
        if self._path in self._domain:
            current = self._domain[self._path]
        else:
            current = self._navigate(txn.root)
            if current is not None:
                self._domain[self._path] = current

        if typ is not None:
            return cast("U", self._materialize_domain(typ, current))

        if self._bound_type is not None and current is None:
            return cast("T", self._materialize_domain(self._bound_type, current))
        return cast("T", current)

    async def save(self) -> None:
        """Flush the root buffer to the store without closing the txn."""
        txn = self._require_txn()
        if txn.dirty:
            self._flush_domain(txn)
            await self._store.store(txn.root)
            txn.dirty = False

    @override
    async def __aenter__(self) -> Self:
        if self._txn is None:
            # Root view owns the txn state.
            root = await self._store.load()
            self._txn = _TxnState(root=root, dirty=False, depth=1)
            # Propagate txn to cached children created before enter (rare).
            for child in self._child_cache.values():
                child._txn = self._txn
            return self
        self._txn.depth += 1
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        txn = self._txn
        if txn is None:
            return False
        txn.depth -= 1
        if txn.depth > 0:
            # Nested exit: on exception, mark for full rollback at root.
            if exc_type is not None:
                txn.dirty = False
                # Reload root discarded on root exit via exception path.
                txn.root = await self._store.load()
                self._domain.clear()
            return False
        # Root exit
        try:
            if exc_type is None and txn.dirty:
                self._flush_domain(txn)
                await self._store.store(txn.root)
        finally:
            self._txn = None
            self._domain.clear()
            # Drop child txn links
            for child in self._child_cache.values():
                child._txn = None
            self._child_cache.clear()
        return False


def session_data_view(
    initial: dict[str, object] | None = None,
    *,
    store: ViewStore | None = None,
) -> PersistentDataView[dict[str, object]]:
    """Convenience root view for session documents (todo, grants, …)."""
    backend: ViewStore = store if store is not None else MemoryViewStore(initial if initial is not None else {})
    return PersistentDataView(backend, path=(), bound_type=dict)
