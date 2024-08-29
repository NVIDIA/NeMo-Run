import dataclasses as dc
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional, Set, Type, TypeVar, Union, overload

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc

from nemo_run.io.capture import _CaptureContext
from nemo_run.io.registry import _ConfigRegistry

if TYPE_CHECKING:
    from nemo_run.config import Config

_T = TypeVar("_T")
_IO_REGISTRY = _ConfigRegistry()


@overload
def capture(func: Callable[..., _T]) -> Callable[..., _T]: ...


@overload
def capture(*, cls_to_ignore: Optional[Set[Type]] = None) -> contextmanager: ...


def capture(
    func: Optional[Callable[..., _T]] = None, *, cls_to_ignore: Optional[Set[Type]] = None
) -> Union[Callable[..., _T], contextmanager]:
    """
    A decorator and context manager for capturing run.Config objects for instantiated objects.

    This function can be used in three ways:
    1. As a decorator on a function to capture configurations for objects instantiated within that function.
    2. As a context manager to capture configurations for objects instantiated within a specific block of code.
    3. As a decorator or context manager with a set of classes to ignore during capture.

    Args:
        func: The function to be decorated (optional).
        cls_to_ignore: A set of classes to ignore during capture (optional).

    Returns:
        Union[Callable[..., _T], contextmanager]: When used as a decorator, returns a wrapped function.
                                                  When used as a context manager, returns a context manager.

    Example:
        .. code-block:: python

            import nemo_run as run

            # Usage as a decorator
            @run.io.capture
            def create_objects():
                obj1 = SomeClass(param1=1, param2="test")
                return obj1

            # Usage as a context manager
            with run.io.capture():
                obj2 = AnotherClass(value=3.14)

            # Usage as a decorator with classes to ignore
            @run.io.capture(cls_to_ignore={ClassToIgnore})
            def create_more_objects():
                obj3 = YetAnotherClass(value=2.71)
                obj4 = ClassToIgnore()  # This instantiation will not be captured
                return obj3, obj4

            # Usage as a context manager with classes to ignore
            with run.io.capture(cls_to_ignore={ClassToIgnore}):
                obj5 = YetAnotherClass(value=1.41)
                obj6 = ClassToIgnore()  # This instantiation will not be captured
    """
    if func is not None:

        @wraps(func)
        def wrapper(*args, **kwargs):
            with _CaptureContext(get, register, cls_to_ignore):
                return func(*args, **kwargs)

        return wrapper

    @contextmanager
    def capture_context():
        with _CaptureContext(get, register, cls_to_ignore):
            yield

    return capture_context()


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
