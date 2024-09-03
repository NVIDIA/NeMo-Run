import dataclasses as dc
from functools import wraps
from typing import (TYPE_CHECKING, Any, Callable, Optional, Set, Type, TypeVar,
                    Union, overload)

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc

from nemo_run.io.capture import _CaptureContext
from nemo_run.io.registry import _ConfigRegistry

if TYPE_CHECKING:
    from nemo_run.config import Config

_T = TypeVar("_T")
_IO_REGISTRY = _ConfigRegistry()


class capture:
    def __init__(self, cls_to_ignore: Optional[Set[Type]] = None):
        self.cls_to_ignore = cls_to_ignore
        self._context: Optional[_CaptureContext] = None

    @overload
    def __call__(self, func: Callable[..., _T]) -> Callable[..., _T]:
        ...

    @overload
    def __call__(self) -> "capture":
        ...

    def __call__(self, func: Optional[Callable[..., _T]] = None) -> Union[Callable[..., _T], "capture"]:
        if func is None:
            return self

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self) -> None:
        self._context = _CaptureContext(get, register, self.cls_to_ignore)
        return self._context.__enter__()

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[Any]) -> Optional[bool]:
        if self._context:
            return self._context.__exit__(exc_type, exc_value, traceback)
        return None


def register(instance: _T, cfg: "Config[_T]") -> None:
    """
    Registers a configuration for a given instance in the global registry.

    Args:
        instance (_T): The instance to associate with the configuration.
        cfg (Config[_T]): The configuration object to register.

    Returns:
        None

    Example:
        >>> cfg = SomeConfig()
        >>> instance = SomeClass()
        >>> register(instance, cfg)
    """
    if dc.is_dataclass(instance):
        return

    _IO_REGISTRY.register(instance, cfg)


def get(obj: _T) -> "Config[_T]":
    """
    Retrieves the configuration for a given object from the global registry.

    Args:
        obj (_T): The object to retrieve the configuration for.

    Returns:
        Config[_T]: The configuration associated with the object.

    Raises:
        ObjectNotFoundError: If no configuration is found for the given object.

    Example:
        >>> instance = SomeClass()
        >>> cfg = get(instance)
    """
    if dc.is_dataclass(obj):
        return fdl_dc.convert_dataclasses_to_configs(obj, allow_post_init=True)
    return _IO_REGISTRY.get(obj)


def reinit(obj: _T) -> _T:
    """
    Reinitializes an object using its stored configuration.

    Args:
        obj (_T): The object to reinitialize.

    Returns:
        _T: A new instance of the object created from its configuration.

    Example:
        >>> import nemo_sdk as sdk
        >>> instance = sdk.build(sdk.Config(SomeClass, a=1, b=2))
        >>> new_instance = reinit(instance)
    """
    return fdl.build(get(obj))
