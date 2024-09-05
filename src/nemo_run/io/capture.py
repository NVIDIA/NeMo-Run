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

import sys
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from fiddle._src.config import ordered_arguments


def process_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func: Callable,
    get_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    """
    Process both positional and keyword arguments for a given function.

    This function handles the processing of all arguments passed to a function,
    ensuring that each argument is properly processed using the provided get_fn.

    Args:
        args (tuple[Any, ...]): Positional arguments.
        kwargs (dict[str, Any]): Keyword arguments.
        func (Callable): The function for which arguments are being processed.
        get_fn (Callable[[Any], Any]): Function to process individual arguments.

    Returns:
        Dict[str, Any]: A dictionary containing all processed arguments.
    """
    # Process positional arguments
    processed_args = [process_single_arg(arg, get_fn) for arg in args]

    # Process keyword arguments
    processed_kwargs = {k: process_single_arg(v, get_fn) for k, v in kwargs.items()}

    # Combine processed positional and keyword arguments
    result = dict(enumerate(processed_args))
    result.update(processed_kwargs)
    return result


def process_single_arg(v: Any, get_fn: Callable[[Any], Any]) -> Any:
    """
    Process a single argument, handling various data types.

    This function recursively processes complex data structures and applies
    special handling for certain types like Path objects and callables.

    Args:
        v (Any): The argument to process.
        get_fn (Callable[[Any], Any]): Function to process non-primitive types.

    Returns:
        Any: The processed argument.
    """
    from nemo_run.config import Config

    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    elif isinstance(v, Path):
        return Config(Path, str(v))
    elif isinstance(v, (list, tuple)):
        return [process_single_arg(item, get_fn) for item in v]
    elif isinstance(v, dict):
        return {key: process_single_arg(value, get_fn) for key, value in v.items()}
    elif (
        callable(v)
        or isinstance(v, type)
        or (isinstance(v, set) and all(isinstance(item, type) for item in v))
    ):
        return v
    else:
        try:
            return get_fn(v)
        except Exception:
            return v  # If we can't process it, return the original value


def wrap_init(frame: FrameType, capture_context: "_CaptureContext"):
    """
    Wrap the __init__ method of a class to capture its arguments.

    This function is called when an object is instantiated within a capture context.
    It processes the arguments passed to the __init__ method and creates a Config
    object representing the instantiated class.

    Args:
        frame (FrameType): The current stack frame.
        capture_context (_CaptureContext): The current capture context.
    """
    cls = frame.f_locals.get("self").__class__
    if cls not in capture_context.cls_to_ignore:
        # Capture arguments for the current class
        args = frame.f_locals.copy()
        del args["self"]
        if "__class__" in args:
            del args["__class__"]  # Remove __class__ attribute
        capture_context.arg_stack.append((cls, args))

        # If we've reached the top of the inheritance chain, create the Config
        if len(capture_context.arg_stack) == len(cls.__mro__) - 1:  # -1 to exclude 'object'
            from nemo_run.config import Config

            combined_args = {}
            for captured_cls, captured_args in reversed(capture_context.arg_stack):
                combined_args.update(captured_args)

            # Use ordered_arguments to get all arguments, including defaults
            cfg = Config(cls)
            all_args = ordered_arguments(cfg, include_defaults=True)

            # Update all_args with the actually provided arguments
            all_args.update(combined_args)

            # Process all arguments before creating the final Config
            processed_args = {
                name: process_single_arg(value, capture_context.get)
                for name, value in all_args.items()
            }

            # Create the Config with all processed arguments
            cfg = Config(cls, **processed_args)

            if capture_context.register:
                capture_context.register(frame.f_locals.get("self"), cfg)

            capture_context.arg_stack.clear()


class _CaptureContext:
    """
    A context manager for capturing object configurations during instantiation.

    This class sets up a profiling function to intercept object instantiations
    and capture their configurations. It's used internally by the `capture` decorator.

    Attributes:
        get (Callable): Function to retrieve configurations.
        register (Callable): Function to register captured configurations.
        cls_to_ignore (Set[Type]): Set of classes to ignore during capture.
        old_profile (Optional[Callable]): The previous profiling function.
        arg_stack (List[Tuple[Type, Dict[str, Any]]]): Stack to store captured arguments.
    """

    def __init__(
        self, get_fn: Callable, register_fn: Callable, cls_to_ignore: Optional[Set[Type]] = None
    ):
        """
        Initialize the _CaptureContext.

        Args:
            get_fn (Callable): Function to retrieve configurations.
            register_fn (Callable): Function to register captured configurations.
            cls_to_ignore (Optional[Set[Type]]): Set of classes to ignore during capture.
        """
        self.get = get_fn
        self.register = register_fn
        self.cls_to_ignore = cls_to_ignore or set()
        self.old_profile = None
        self.arg_stack: List[Tuple[Type, Dict[str, Any]]] = []

    def __enter__(self):
        """
        Enter the capture context, setting up the profiling function.
        """
        self.old_profile = sys.getprofile()
        sys.setprofile(self._profile_func)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the capture context, restoring the previous profiling function.
        """
        sys.setprofile(self.old_profile)

    def _profile_func(self, frame: FrameType, event: str, arg: Any):
        """
        Profiling function that intercepts object instantiations.

        This function is called for every function call while the context is active.
        It specifically looks for __init__ calls to capture object configurations.

        Args:
            frame (FrameType): The current stack frame.
            event (str): The type of event (e.g., 'call', 'return').
            arg (Any): Event-specific argument.

        Returns:
            Optional[Callable]: The previous profiling function, if any.
        """
        if event == "call" and frame.f_code.co_name == "__init__":
            wrap_init(frame, self)
        return self.old_profile
