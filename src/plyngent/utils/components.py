import typing

from typing_extensions import TypeForm

if typing.TYPE_CHECKING:
    from collections.abc import Callable


class Forward[T]:
    target: str
    attr: str

    def __init__(self, target: str, attr: str | None = None) -> None:
        self.target = target
        if attr:
            self.attr = attr

    def __get__(self, obj: object | type[object], tp: type[object] | None = None) -> T:
        return typing.cast("T", getattr(typing.cast("object", getattr(obj, self.target)), self.attr))

    def __set__(self, obj: object | type[object], val: object) -> None:
        setattr(typing.cast("object", getattr(obj, self.target)), self.attr, val)

    def __delete__(self, obj: object | type[object]) -> None:
        delattr(typing.cast("object", getattr(obj, self.target)), self.attr)

    def __set_name__(self, owner: type[object], name: str) -> None:
        if hasattr(self, "attr"):
            return
        self.attr = name


@typing.overload
def forward[T](target: str, attr: str | None = None, hint: type[T] | None = None) -> Forward[T]: ...


@typing.overload
def forward[**PS, R](
    target: str, attr: Callable[typing.Concatenate[object, PS], R], hint: None = None
) -> Forward[Callable[PS, R]]: ...


def forward[T, **PS, R](
    target: str, attr: str | Callable[typing.Concatenate[object, PS], R] | None = None, hint: TypeForm[T] | None = None
) -> Forward[T] | Forward[Callable[PS, R]]:
    """Forwarding specified attribute to its composition field."""
    del hint
    return Forward(target=target, attr=attr.__name__ if callable(attr) else attr)
