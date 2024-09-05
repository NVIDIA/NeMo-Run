# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses as dc
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Set, Type, TypeVar, Union, overload

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc

from nemo_run.io.capture import _CaptureContext
from nemo_run.io.registry import _ConfigRegistry

if TYPE_CHECKING:
    from nemo_run.config import Config

_T = TypeVar("_T")
_IO_REGISTRY = _ConfigRegistry()


class capture:
    """
    A decorator and context manager for capturing object configurations.

    This class provides functionality to automatically capture and register configurations
    of objects created within its scope. It can be used as a decorator on functions or as
    a context manager.

    Args:
        cls_to_ignore (Optional[Set[Type]]): A set of classes to ignore during capture.

    Examples:
        As a decorator:
        >>> @capture()
        ... def create_object():
        ...     return SomeClass(42)
        >>> obj = create_object()
        >>> cfg: run.Config[SomeClass] = get(obj)  # Configuration is automatically captured

        As a context manager:
        >>> with capture():
        ...     obj = SomeClass(42)
        >>> cfg: run.Config[SomeClass] = get(obj)  # Configuration is automatically captured

        With classes to ignore:
        >>> @capture(cls_to_ignore={IgnoredClass})
        ... def create_objects():
        ...     obj1 = SomeClass(1)
        ...     obj2 = IgnoredClass(2)
        ...     return obj1, obj2
        >>> obj1, obj2 = create_objects()
        >>> cfg1: run.Config[SomeClass] = get(obj1)  # Works
        >>> cfg2: run.Config[IgnoredClass] = get(obj2)  # Raises ObjectNotFoundError

    Notes:
        - Nested captures are supported.
        - Exceptions within the capture scope do not prevent object registration.
        - Dataclasses are automatically converted to configs without registration.
        - Complex arguments (lists, dicts, callables) are supported in captured configs.
        - Unsupported types may raise ValueError during capture.
    """

    def __init__(self, cls_to_ignore: Optional[Set[Type]] = None):
        self.cls_to_ignore = cls_to_ignore
        self._context: Optional[_CaptureContext] = None

    @overload
    def __call__(self, func: Callable[..., _T]) -> Callable[..., _T]: ...

    @overload
    def __call__(self) -> "capture": ...

    def __call__(
        self, func: Optional[Callable[..., _T]] = None
    ) -> Union[Callable[..., _T], "capture"]:
        """
        Allows the capture class to be used as a decorator.

        If called without arguments, returns the capture instance for use as a context manager.
        If called with a function argument, returns a wrapped version of the function that
        executes within a capture context.

        Args:
            func (Optional[Callable[..., _T]]): The function to be wrapped.

        Returns:
            Union[Callable[..., _T], "capture"]: Either the wrapped function or the capture instance.
        """
        if func is None:
            return self

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self) -> None:
        """
        Enters the capture context.

        This method is called when entering a `with` block or at the start of a decorated function.
        It sets up the capture context for automatic configuration registration.

        Returns:
            None
        """
        self._context = _CaptureContext(get, register, self.cls_to_ignore)
        return self._context.__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> Optional[bool]:
        """
        Exits the capture context.

        This method is called when exiting a `with` block or at the end of a decorated function.
        It ensures that the capture context is properly closed, even if an exception occurred.

        Args:
            exc_type (Optional[Type[BaseException]]): The type of the exception that occurred, if any.
            exc_value (Optional[BaseException]): The exception instance that occurred, if any.
            traceback (Optional[Any]): The traceback object for the exception, if any.

        Returns:
            Optional[bool]: Returns the result of the context's __exit__ method, if applicable.
        """
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
