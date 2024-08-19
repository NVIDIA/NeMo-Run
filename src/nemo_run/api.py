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

import importlib.util
import inspect
import os
from functools import cache, wraps
from typing import (
    Any,
    Callable,
    Concatenate,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import catalogue
import fiddle as fdl
import importlib_metadata as metadata
from fiddle.experimental import auto_config as _auto_config
from rich.pretty import Pretty
from rich.table import Table

from nemo_run.config import NEMORUN_HOME, Config, Partial
from nemo_run.core.execution.base import Executor
from nemo_run.core.frontend.console.api import CONSOLE, CustomConfigRepr

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
P = ParamSpec("P")
C = TypeVar("C", bound=Callable)

ROOT_TASK_NAMESPACE = "nemo_run.task"
ROOT_TASK_FACTORY_NAMESPACE = "nemo_run.task_factory"
ROOT_TYPER_NAMESPACE = "nemo_run.typer"
DEFAULT_NAME = "default"

AUTOBUILD_CLASSES = (Executor,)


@cache
def _load_entrypoints():
    try:
        entrypoints = metadata.entry_points().select(group="run.factories")
        for ep in entrypoints:
            ep.load()
    except Exception:
        ...


def _search_workspace_file() -> str | None:
    current_dir = os.getcwd()
    file_names = [
        "workspace_private.py",
        "workspace.py",
        os.path.join(NEMORUN_HOME, "workspace.py"),
    ]

    while True:
        for file_name in file_names:
            workspace_file_path = os.path.join(current_dir, file_name)
            if os.path.exists(workspace_file_path):
                return workspace_file_path

        # Go up one directory level
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Root directory
            break
        current_dir = parent_dir

    return None


def _load_workspace_file(path):
    spec = importlib.util.spec_from_file_location("workspace", path)
    assert spec
    workspace_module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(workspace_module)


@cache
def _load_workspace():
    workspace_file_path = _search_workspace_file()

    if workspace_file_path:
        return _load_workspace_file(workspace_file_path)


def task(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    namespace=None,
    help: Optional[str] = None,
    parse_partial: Optional[Callable[[T, list[str]], Partial[T]]] = None,
    **kwargs,
) -> F | Callable[[F], F]:
    from nemo_run.cli.dynamic_cli import Task

    if namespace:
        _namespace = namespace
    else:
        caller = inspect.stack()[1]
        _module = inspect.getmodule(caller[0])
        assert _module, "Module is None for task."
        _namespace = _module.__name__

    def wrapper(f: F) -> F:
        task = Task(
            f,
            name=name,
            namespace=_namespace,
            help_str=help,
            parse_partial=parse_partial,
        )
        parts = _namespace.split(".")
        task_namespace = (ROOT_TASK_NAMESPACE, *parts, f.__name__)
        catalogue._set(task_namespace, task)

        return f

    if fn is None:
        return wrapper

    return wrapper(fn)


def list_tasks(
    namespace: Optional[str] = None,
) -> dict[str, dict[str, F]] | dict[str, F]:
    """List all tasks for a given or root namespace."""
    _load_entrypoints()
    _load_workspace()

    full_namespace = (ROOT_TASK_NAMESPACE,)
    if namespace:
        full_namespace += (namespace,)
    response = catalogue._get_all(full_namespace)
    output = {}
    for key, fn in response.items():
        parts = fn.path.split(".")
        current_level = output
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[-1]] = fn

    return output


def default_autoconfig_buildable(
    fn: Callable[P, T],
    cls: Type[Union[Partial, Config]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Config[T] | Partial[T]:
    def exemption_policy(cfg):
        return cfg in [Partial, Config] or getattr(cfg, "__auto_config__", False)

    return fdl.cast(
        cls,
        _auto_config.auto_config(
            fn,
            experimental_allow_control_flow=False,
            experimental_allow_dataclass_attribute_access=True,
            experimental_exemption_policy=exemption_policy,
        ).as_buildable(*args, **kwargs),
    )


@overload
def autoconvert(
    fn: Callable[P, Config[T]],
    *,
    partial: bool = False,
) -> Callable[P, Config[T]]: ...


@overload
def autoconvert(  # type: ignore
    fn: Callable[P, Partial[T]],
    *,
    partial: bool = False,
) -> Callable[P, Partial[T]]: ...


@overload
def autoconvert(
    fn: Callable[P, T],
    *,
    partial: bool = False,
) -> Callable[P, Config[T]]: ...


@overload
def autoconvert(
    *,
    partial: Literal[True] = ...,
) -> Callable[
    [Callable[P, T] | Callable[P, Config[T]] | Callable[P, Partial[T]]],
    Callable[P, Partial[T]],
]: ...


@overload
def autoconvert(
    *,
    partial: Literal[False] = False,
) -> Callable[
    [Callable[P, T] | Callable[P, Config[T]] | Callable[P, Partial[T]]],
    Callable[P, Config[T]],
]: ...


def autoconvert(
    fn: Optional[Callable[P, T] | Callable[P, Config[T]] | Callable[P, Partial[T]]] = None,
    *,
    partial: bool = False,
    to_buildable_fn: Callable[
        Concatenate[Callable[P, T], Type[Union[Partial, Config]], P],
        Config[T] | Partial[T],
    ] = default_autoconfig_buildable,
) -> (
    Callable[P, Config[T] | Partial[T]]
    | Callable[
        [Callable[P, T] | Callable[P, Config[T]] | Callable[P, Partial[T]]],
        Callable[P, Config[T] | Partial[T]],
    ]
):
    """
    The autoconvert function is a powerful and flexible decorator for Python functions that can
    modify the behavior of the function it decorates by converting the returned object in a nested manner to:
    run.Config (when partial is False) or run.Partial (when partial is True).
    This conversion is done by a provided conversion function `to_buildable_fn`, which defaults to `default_autoconfig_buildable`.
    Under the hood, it uses `fiddle's autoconfig <https://fiddle.readthedocs.io/en/latest/api_reference/autoconfig.html>`_ to parse the function's AST and convert objects to their run.Config/run.Partial counterparts.

    You can use it in two different ways:

    - Directly as a decorator for a function you define:

      .. code-block:: python

        @autoconvert
        def my_func(param1: int, param2: str) -> MyType:
            return MyType(param1=param1, param2=param2)

      This will return `run.Config(MyType, param1=param1, param2=param2)` when called, assuming that
      `partial=False` (otherwise, it would be a run.Partial instance).

    - Indirectly, as a way to convert an existing function:

      .. code-block:: python

        def my_func(param1: int, param2: str) -> MyType:
            return MyType(param1=param1, param2=param2)

        my_new_func = autoconvert(partial=True)(my_func)

      Now, calling `my_new_func` will actually return `run.Partial(MyType, param1=param1, param2=param2)` rather
      than a `MyType` instance.

    Parameters:

    - fn:
        The function to be decorated. This parameter is optional, and if not provided,
        `autoconvert` acts as a decorator factory. Defaults to None.

    - partial:
        A boolean flag that indicates whether the return type of `fn` should be converted
        to Partial[T] (if True) or Config[T] (if False). Defaults to False.

    - to_buildable_fn:
        The conversion function to be used for the desired output type. This function
        takes another function and any positional and keyword arguments and returns an
        instance of either Config[T] or Partial[T]. By default, it uses
        `default_autoconfig_buildable`.
    """

    def wrapper(
        fn: Callable[P, T] | Callable[P, Config[T]] | Callable[P, Partial[T]],
    ) -> Callable[P, Config[T] | Partial[T]]:
        @wraps(fn)
        def autobuilder(*args: P.args, **kwargs: P.kwargs) -> Config[T] | Partial[T]:
            return to_buildable_fn(
                cast(Callable[P, T], fn),
                Partial if partial else Config,
                *args,
                **kwargs,
            )

        autobuilder.wrapped = fn  # type: ignore
        autobuilder.__auto_config__ = True  # type: ignore
        return autobuilder

    return wrapper if fn is None else wrapper(fn)


def dryrun_fn(
    configured_fn: Partial,
    executor: Optional[Executor] = None,
    build: bool = False,
) -> None:
    if not isinstance(configured_fn, (Config, Partial)):
        raise TypeError(f"Need a run Partial for dryrun. Got {configured_fn}.")

    fn = configured_fn.__fn_or_cls__
    console = CONSOLE
    console.print(f"[bold cyan]Dry run for task {fn.__module__}:{fn.__name__}[/bold cyan]")

    table_resolved_args = Table(show_header=True, header_style="bold magenta")
    table_resolved_args.add_column("Argument Name", style="dim", width=20)
    table_resolved_args.add_column("Resolved Value", width=60)

    for arg_name in dir(configured_fn):
        repr = CustomConfigRepr(getattr(configured_fn, arg_name))
        table_resolved_args.add_row(arg_name, Pretty(repr))

    console.print("[bold green]Resolved Arguments[/bold green]")
    console.print(table_resolved_args)

    if executor:
        console.print("[bold green]Executor[/bold green]")
        table_executor = Table(show_header=False, header_style="bold magenta")
        table_executor.add_column("Executor")
        table_executor.add_row(Pretty(CustomConfigRepr(executor)))
        console.print(table_executor)

    if build:
        fdl.build(configured_fn)


@runtime_checkable
class AutoConfigProtocol(Protocol[C]):
    def __auto_config__(self) -> bool: ...

    __call__: C
