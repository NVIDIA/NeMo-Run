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
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar, get_args, overload, ParamSpec
import functools

import importlib_metadata as metadata
import typer
from rich.logging import RichHandler
from typer import rich_utils
import catalogue

from nemo_run.cli.core.factory import register_factory, get_return_type
from nemo_run.cli.core.workspace import load_entrypoints, load_workspace, delete_cache, cache_factory_metadata, cache_entrypoint_metadata
from nemo_run.cli.core.cli import add_global_options, configure_logging, add_typer_nested, get_or_add_typer, GeneralCommand
from nemo_run.cli.core.entrypoint import CommandDefaults, Entrypoint, RunContext

from nemo_run.cli import devspace as devspace_cli
from nemo_run.cli import experiment as experiment_cli
from nemo_run.cli.core.help import render_help
from nemo_run.cli.sync import sync
from nemo_run.config import Config, Partial
from nemo_run.core.execution import LocalExecutor, SkypilotExecutor, SlurmExecutor
from nemo_run.core.execution.base import Executor
from nemo_run.lazy import LazyEntrypoint
from nemo_run.run.plugin import ExperimentPlugin as Plugin
from nemo_run.core.frontend.console.api import CONSOLE

F = TypeVar("F", bound=Callable[..., Any])
Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")
T = TypeVar("T")

ROOT_ENTRYPOINT_NAMESPACE = "nemo_run.cli.entrypoints"
ROOT_FACTORY_NAMESPACE = "nemo_run.cli.factories"
DEFAULT_NAME = "default"
EXECUTOR_CLASSES = [Executor, LocalExecutor, SkypilotExecutor, SlurmExecutor]
PLUGIN_CLASSES = [Plugin, List[Plugin]]
NEMORUN_SKIP_CONFIRMATION: Optional[bool] = None
POPULATE_CACHE: bool = False

INCLUDE_WORKSPACE_FILE = os.environ.get("INCLUDE_WORKSPACE_FILE", "true").lower() == "true"
NEMORUN_PRETTY_EXCEPTIONS = os.environ.get("NEMORUN_PRETTY_EXCEPTIONS", "false").lower() == "true"

logger = logging.getLogger(__name__)
MAIN_ENTRYPOINT = None


def entrypoint(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    help: Optional[str] = None,
    skip_confirmation: bool = False,
    enable_executor: bool = True,
    default_factory: Optional[Callable] = None,
    default_executor: Optional[Config[Executor]] = None,
    default_plugins: Optional[List[Plugin]] = None,
    entrypoint_cls: Optional[Type["Entrypoint"]] = None,
    type: Literal["task", "experiment"] = "task",
    run_ctx_cls: Optional[Type["RunContext"]] = None,
) -> F | Callable[[F], F]:
    """
    Decorator to register a function as a CLI entrypoint in the NeMo Run framework.

    This decorator transforms a function into a CLI entrypoint, allowing it to be
    executed from the command line with various options and arguments. It supports
    a rich set of Pythonic CLI syntax for argument passing and configuration.

    Args:
        fn (Optional[F]): The function to be decorated. If None, returns a decorator.
        name (Optional[str]): Custom name for the entrypoint. Defaults to the function name.
        namespace (Optional[str]): Custom namespace for the entrypoint. Defaults to the module name.
        help (Optional[str]): Help text for the entrypoint, displayed in CLI help messages.
        skip_confirmation (bool): If True, skips user confirmation before execution. Defaults to False.
        enable_executor (bool): If True, enables executor functionality for the entrypoint. Defaults to True.
        default_factory (Optional[Callable]): A custom default factory to use for this entrypoint.
        default_executor (Optional[Config[Executor]]): A custom default executor to use for this entrypoint.
        default_plugins (Optional[List[Plugin]]): A custom default plugins to use for this entrypoint.
        entrypoint_cls (Optional[Type["Entrypoint"]]): A custom entrypoint class to use for this entrypoint.
        type (Literal["task", "experiment"]): The type of entrypoint. Defaults to "task".
        run_ctx_cls (Type[RunContext]): The class to use for the run context. Defaults to RunContext.

    Returns:
        F | Callable[[F], F]: The decorated function or a decorator function.

    CLI Syntax and Features:
    1. Basic argument passing:
       python script.py arg1=value1 arg2=value2

    2. Nested attribute setting:
       python script.py model.learning_rate=0.01 data.batch_size=32

    3. List and dictionary arguments:
       python script.py list_arg=[1,2,3] dict_arg={'key':'value'}

    4. Operations on arguments:
       python script.py counter+=1 rate*=2 flags|=0x1

    5. Factory function usage:
       python script.py model=create_model(hidden_size=256)

    6. Type casting:
       python script.py int_arg=42 float_arg=3.14 bool_arg=true

    7. None and null values:
       python script.py optional_arg=None

    8. Executor specification:
       python script.py executor=local_executor executor.num_gpus=2

    Example:
        @run.cli.entrypoint
        def train_model(model: MyModel, learning_rate: float = 0.01, epochs: int = 10):
            # Training logic here
            pass

        # CLI usage:
        # python train_script.py model=create_large_model learning_rate=0.001 epochs=20

    Notes:
        - The decorated function becomes accessible via the CLI with rich argument parsing.
        - Use the `help` parameter to provide detailed information about the entrypoint's purpose and usage.
        - Custom parsers can be provided for advanced configuration handling.
        - The `enable_executor` flag allows for integration with different execution environments.
        - The CLI supports various Python-like operations and nested configurations.

    See Also:
        - run.cli.factory: For creating factory functions usable in CLI arguments.
        - run.Config: For configuring complex objects within entrypoints.
        - run.Executor: For understanding different execution backends.
    """
    _namespace = None
    if isinstance(namespace, str):
        _namespace = namespace
    else:
        caller = inspect.stack()[1]
        _module = inspect.getmodule(caller[0])
        if _module:
            _namespace = _module.__name__

    _entrypoint_cls = entrypoint_cls or Entrypoint

    def wrapper(f: F) -> F:
        if default_factory:
            factory(default_factory, target=f, is_target_default=True)

        _entrypoint = _entrypoint_cls(
            f,
            name=name,
            namespace=_namespace,
            help_str=help,
            skip_confirmation=skip_confirmation,
            default_factory=default_factory,
            default_executor=default_executor,
            default_plugins=default_plugins,
            enable_executor=enable_executor,
            type=type,
            run_ctx_cls=run_ctx_cls or RunContext,
        )

        if isinstance(_namespace, str):
            parts = _namespace.split(".")
            task_namespace = (ROOT_ENTRYPOINT_NAMESPACE, *parts, f.__name__)
            catalogue._set(task_namespace, _entrypoint)

        f.cli_entrypoint = _entrypoint

        if POPULATE_CACHE:
            cache_entrypoint_metadata(f, _entrypoint)

        return f

    if fn is None:
        return wrapper

    return wrapper(fn)

def main(
    fn: F,
    default_factory: Optional[Callable] = None,
    default_executor: Optional[Config[Executor]] = None,
    default_plugins: Optional[List[Config[Plugin]] | Config[Plugin]] = None,
    cmd_defaults: Optional[CommandDefaults] = None,
    **kwargs,
):
    """
    Execute the main CLI entrypoint for the given function.

    Args:
        fn (F): The function to be executed as a CLI entrypoint.
        default_factory (Optional[Callable]): A custom default factory to use in the context of this main function.
        default_executor (Optional[Config[Executor]]): A custom default executor to use in the context of this main function.
        **kwargs: Additional keyword arguments to pass to the entrypoint decorator.

    Example:
        @entrypoint
        def my_cli_function():
            # CLI logic here
            pass

        if __name__ == "__main__":
            main(my_cli_function, default_factory=my_custom_defaults)
    """
    lazy_cli = os.environ.get("LAZY_CLI", "false").lower() == "true"

    if not isinstance(fn, EntrypointProtocol):
        # Wrap the function with the default entrypoint
        fn = entrypoint(**kwargs)(fn)
    if getattr(fn, "__is_lazy__", False):
        if lazy_cli:
            fn = fn._import()
        else:
            app = typer.Typer()
            RunContext.cli_command(
                app,
                sys.argv[1] if len(sys.argv) > 1 else "default",
                LazyEntrypoint(" ".join(sys.argv)),
                type="task",
                default_factory=default_factory,
                default_executor=default_executor,
                default_plugins=default_plugins,
            )
            return app(standalone_mode=False)

    _original_default_factory = fn.cli_entrypoint.default_factory
    if default_factory:
        fn.cli_entrypoint.default_factory = default_factory

    _original_default_executor = fn.cli_entrypoint.default_executor
    if default_executor:
        fn.cli_entrypoint.default_executor = default_executor

    _original_default_plugins = fn.cli_entrypoint.default_plugins
    if default_plugins:
        if isinstance(default_plugins, list):
            if not all(isinstance(p, Config) for p in default_plugins):
                raise ValueError("default_plugins must be a list of Config objects")
        else:
            if not isinstance(default_plugins, Config):
                raise ValueError("default_plugins must be a Config object")
        fn.cli_entrypoint.default_plugins = default_plugins

    if lazy_cli:
        global MAIN_ENTRYPOINT
        MAIN_ENTRYPOINT = fn.cli_entrypoint
        return

    fn.cli_entrypoint.main(cmd_defaults)

    fn.cli_entrypoint.default_factory = _original_default_factory
    fn.cli_entrypoint.default_executor = _original_default_executor
    fn.cli_entrypoint.default_plugins = _original_default_plugins


@overload
def factory(
    fn: Optional[F] = None,
    target: Optional[Callable] = None,
    *,
    target_arg: Optional[str] = None,
    is_target_default: bool = False,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> F: ...


@overload
def factory(
    fn: Optional[F] = None,
    target: Optional[Type] = None,
    *,
    target_arg: Optional[str] = None,
    is_target_default: bool = False,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> F: ...


def factory(
    fn: Optional[F] = None,
    target: Optional[Type] = None,
    *,
    target_arg: Optional[str] = None,
    is_target_default: bool = False,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Callable[[F], F] | F:
    """
    A decorator that registers factories for commonly used arguments and types in NeMo Run entrypoints.

    Factories are crucial components that work in conjunction with the @entrypoint decorator.
    They allow for easy configuration and instantiation of complex objects directly from the CLI,
    enhancing the flexibility and usability of NeMo Run entrypoints.

    There are two main ways to use this decorator:

    1. Without any arguments:
        This registers the function to its annotated return type.
        The registry for this method can be thought of as a dict of the form:

        {
            TypeA: [...list of factories for TypeA],
            TypeB: [...list of factories for TypeB],
            TypeC: [...list of factories for TypeC],
        }

        Note: This does not automatically handle inheritance. Factories registered against
        a parent type won't be resolvable under child types.

    2. By specifying the parent and/or arg name:
        This registers the factory under the parent type (if arg is absent) or under the arg name in parent.
        If the parent is a function, the arg name from its signature will be used.
        If the parent is a class, the arg name from its __init__ signature will be used.

    Args:
        fn (Optional[F]): The function to use as a factory.
        target (Optional[Type]): Parent type to register the factory under.
        target_arg (Optional[str]): Specific arg name under the parent to register the factory.
        is_target_default (bool): If True, the factory will be the default choice for its type/parent.
        name (Optional[str]): Name of the factory, defaults to the function name.
        namespace (Optional[str]): Custom namespace to register the factory under.

    Returns:
        Callable[[F], F] | F: The decorated factory function.

    Example:
        @run.cli.entrypoint
        def train_model(model: MyModel):
            # Training logic here
            pass

        @run.cli.factory
        def default_model_config() -> run.Config[MyModel]:
            return run.Config(MyModel, param1=value1, param2=value2)

        # Usage in CLI:
        # nemorun train_model model=default_model_config

    Notes:
        - Factories registered with this decorator can be used as arguments in @entrypoint decorated functions.
        - When a factory is specified in the CLI, it's used to create and configure the corresponding object.
        - Factories can be chained and nested for complex configurations.
        - The @factory decorator uses `fdl.auto_config` under the hood for configuration management.

    See Also:
        - run.cli.entrypoint: For creating CLI entrypoints that can use these factories.
        - run.Config: For configuring complex objects within factories.
    """
    if target_arg and not target:
        raise ValueError("`target_arg` cannot be used without specifying a `target`.")

    def wrapper(fn: Callable[Params, T]) -> Callable[Params, T]:
        if not target and not hasattr(fn, "__auto_config__"):
            return_type = get_return_type(fn)
            if not (
                isinstance(return_type, (Config, Partial))
                or (
                    hasattr(return_type, "__origin__")
                    and issubclass(return_type.__origin__, (Config, Partial))
                )
            ):
                raise ValueError(
                    f"Factory function {fn} has a return type which is not a subclass of Config or Partial. "
                    "`factory` is not supported for this, please use `factory` instead. "
                    "For automatic conversion, use `run.autoconvert`."
                )

        @functools.wraps(fn)
        def as_factory(*args: Params.args, **kwargs: Params.kwargs) -> T:
            return fn(*args, **kwargs)

        as_factory.wrapped = fn
        as_factory.__factory__ = True

        registration = register_factory(
            as_factory,
            target=target,
            target_arg=target_arg,
            name=name,
            namespace=namespace,
            is_target_default=is_target_default,
        )

        if POPULATE_CACHE:
            cache_factory_metadata(as_factory, registration)

        return as_factory

    return wrapper if fn is None else wrapper(fn)


def resolve_factory(
    target: Type[T] | str,
    name: str,
) -> Callable[..., Config[T] | Partial[T]]:
    """
    Helper function to resolve the factory for the give type or namespace.

    .. note::
        This automatically loads the entrypoints defined under `run.factories`.
        So all factories in the module specified in entrypoints will be automatically imported.

    Examples
    --------
    .. code-block:: python

        cfg: run.Config[SomeType] = run.resolve(SomeType, "factory_registered_under_some_type")

        obj: SomeType = run.resolve(SomeType, "factory_registered_under_some_type", build=True)

    """
    load_entrypoints()
    load_workspace()

    fn: Optional[FactoryProtocol] = None

    if isinstance(target, str):
        fn = catalogue._get((target, name))
    else:
        types = get_underlying_types(target)
        num_missing = 0
        for t in types:
            _namespace = get_type_namespace(t)
            try:
                fn = catalogue._get((_namespace, name))
                break
            except catalogue.RegistryError:
                num_missing += 1
        if num_missing == len(types):
            raise ValueError(f"No factory found for {target} under {name}")

    assert isinstance(fn, FactoryProtocol), f"{fn.__qualname__} is not a registered factory."

    return fn


def list_entrypoints(
    namespace: Optional[str] = None,
) -> dict[str, dict[str, F]] | dict[str, F]:
    """List all tasks for a given or root namespace."""
    load_entrypoints()
    load_workspace()

    full_namespace = (ROOT_ENTRYPOINT_NAMESPACE,)
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


def list_factories(type_or_namespace: Type | str) -> list[Callable]:
    """
    Lists all factories for a given type or namespace.
    """
    load_entrypoints()
    load_workspace()

    _namespace = (
        get_type_namespace(type_or_namespace)
        if isinstance(type_or_namespace, type)
        else type_or_namespace
    )
    response = catalogue._get_all([_namespace])
    return list(response.values())


def create_cli(
    add_verbose_callback: bool = False,
    nested_entrypoints_creation: bool = True,
) -> typer.Typer:
    is_lazy = "--lazy" in sys.argv
    is_help_request = "--help" in sys.argv or "-h" in sys.argv

    if is_help_request:
        global POPULATE_CACHE
        POPULATE_CACHE = True
    
    # Check for --no-cache flag
    is_no_cache = "--no-cache" in sys.argv
    if is_no_cache:
        # Delete cache file and remove the flag from sys.argv
        delete_cache()
        sys.argv = [arg for arg in sys.argv if arg != "--no-cache"]
    
    app: typer.Typer = typer.Typer(add_completion=not is_lazy, pretty_exceptions_enable=False)
    if is_lazy:
        if len(sys.argv) > 1 and sys.argv[1] in ["devspace", "experiment"]:
            raise ValueError("Lazy CLI does not support devspace and experiment commands.")

        # remove --lazy from sys.argv
        sys.argv = [arg for arg in sys.argv if arg != "--lazy"]

        RunContext.cli_command(
            app,
            sys.argv[1],
            LazyEntrypoint(" ".join(sys.argv)),
            type="task",
        )

        return app

    if is_help_request and not is_no_cache and sys.argv[1] not in ["devspace", "experiment"]:
        try:
            return render_help(sys.argv)
        except Exception as e:
            # If there's an error with the cache, continue with normal CLI creation
            pass

    entrypoints = metadata.entry_points().select(group="nemo_run.cli")
    for ep in entrypoints:
        get_or_add_typer(app, name=ep.name)

    if not nested_entrypoints_creation or (
        len(sys.argv) > 1 and sys.argv[1] in entrypoints.names
    ):
        add_typer_nested(app, list_entrypoints())

    app.add_typer(
        devspace_cli.create(),
        name="devspace",
        help="[Module] Manage devspaces",
        cls=GeneralCommand,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    app.add_typer(
        experiment_cli.create(),
        name="experiment",
        help="[Module] Manage Experiments",
        cls=GeneralCommand,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    app.command(name="sync")(sync)

    if add_verbose_callback:
        add_global_options(app)

    return app


if __name__ == "__main__":
    app = create_cli()
    app()
