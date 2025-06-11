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
import typing
from dataclasses import dataclass, field
from functools import cache, wraps
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
    overload,
    runtime_checkable,
)

import catalogue
import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import importlib_metadata as metadata
import typer
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from typer import Option, Typer, rich_utils
from typer.core import TyperCommand, TyperGroup
from typer.models import OptionInfo
from typing_extensions import NotRequired, ParamSpec, TypedDict

from nemo_run.cli import devspace as devspace_cli
from nemo_run.cli import experiment as experiment_cli
from nemo_run.cli.cli_parser import parse_cli_args, parse_factory
from nemo_run.cli.config import ConfigSerializer
from nemo_run.cli.lazy import LazyEntrypoint
from nemo_run.config import (
    RECURSIVE_TYPES,
    Config,
    Partial,
    get_nemorun_home,
    get_type_namespace,
)
from nemo_run.core.execution import LocalExecutor, SkypilotExecutor, SlurmExecutor
from nemo_run.core.execution.base import Executor
from nemo_run.core.frontend.console.styles import BOX_STYLE, TABLE_STYLES
from nemo_run.run.experiment import Experiment
from nemo_run.run.plugin import ExperimentPlugin as Plugin


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

        return f

    if fn is None:
        return wrapper

    return wrapper(fn)


class CommandDefaults(TypedDict, total=False):
    direct: NotRequired[bool]
    dryrun: NotRequired[bool]
    load: NotRequired[str]
    yaml: NotRequired[str]
    repl: NotRequired[bool]
    detach: NotRequired[bool]
    skip_confirmation: NotRequired[bool]
    tail_logs: NotRequired[bool]
    rich_exceptions: NotRequired[bool]
    rich_traceback: NotRequired[bool]
    rich_locals: NotRequired[bool]
    rich_theme: NotRequired[str]
    verbose: NotRequired[bool]


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
            return_type = _get_return_type(fn)
            if not (
                isinstance(return_type, (Config, Partial))
                or (hasattr(return_type, "__origin__") and issubclass(return_type.__origin__, (Config, Partial)))
            ):
                raise ValueError(
                    f"Factory function {fn} has a return type which is not a subclass of Config or Partial. "
                    "`factory` is not supported for this, please use `factory` instead. "
                    "For automatic conversion, use `run.autoconvert`."
                )

        @wraps(fn)
        def as_factory(*args: Params.args, **kwargs: Params.kwargs) -> T:
            return fn(*args, **kwargs)

        as_factory.wrapped = fn
        as_factory.__factory__ = True

        _register_factory(
            as_factory,
            target=target,
            target_arg=target_arg,
            name=name,
            namespace=namespace,
            is_target_default=is_target_default,
        )

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
    _load_entrypoints()
    _load_workspace()

    fn: Optional[FactoryProtocol] = None

    if isinstance(target, str):
        fn = catalogue._get((target, name))
    else:
        types = extract_constituent_types(target)
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
    _load_entrypoints()
    _load_workspace()

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
    _load_entrypoints()
    _load_workspace()

    _namespace = get_type_namespace(type_or_namespace) if isinstance(type_or_namespace, type) else type_or_namespace
    response = catalogue._get_all([_namespace])
    return list(response.values())


def create_cli(
    add_verbose_callback: bool = False,
    nested_entrypoints_creation: bool = True,
) -> Typer:
    app: Typer = Typer(pretty_exceptions_enable=False)
    entrypoints = metadata.entry_points().select(group="nemo_run.cli")
    metadata.entry_points().select(group="nemo_run.cli")
    for ep in entrypoints:
        _get_or_add_typer(app, name=ep.name)

    # Check for --lazy flag
    is_lazy = "--lazy" in sys.argv

    app: Typer = Typer(add_completion=not is_lazy)
    if is_lazy:
        if len(sys.argv) > 1 and sys.argv[1] in ["devspace", "experiment"]:
            raise ValueError("Lazy CLI does not support devspace and experiment commands.")

        # Set LAZY_CLI environment variable
        os.environ["LAZY_CLI"] = "true"

        # remove --lazy from sys.argv but keep export flags
        export_flags = {}
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] == "--lazy":
                sys.argv.pop(i)
            elif sys.argv[i] in ["--to-yaml", "--to-toml", "--to-json"] and i + 1 < len(sys.argv):
                export_flags[sys.argv[i]] = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        # Add command with the LazyEntrypoint
        RunContext.cli_command(
            app,
            sys.argv[1],
            LazyEntrypoint(" ".join(sys.argv)),
            type="task",
        )
    else:
        entrypoints = metadata.entry_points().select(group="nemo_run.cli")
        metadata.entry_points().select(group="nemo_run.cli")
        for ep in entrypoints:
            _get_or_add_typer(app, name=ep.name)

        if not nested_entrypoints_creation or (len(sys.argv) > 1 and sys.argv[1] in entrypoints.names):
            _add_typer_nested(app, list_entrypoints())

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

    if add_verbose_callback:
        add_global_options(app)

    return app


@runtime_checkable
class EntrypointProtocol(Protocol):
    def cli_entrypoint(self) -> "Entrypoint": ...


@runtime_checkable
class FactoryProtocol(Protocol):
    @property
    def wrapped(self) -> Callable: ...

    @property
    def __factory__(self) -> bool: ...


def add_global_options(app: Typer):
    @app.callback()
    def global_options(
        verbose: bool = Option(False, "-v", "--verbose"),
        rich_exceptions: bool = typer.Option(
            False, "--rich-exceptions/--no-rich-exceptions", help="Enable rich exception formatting"
        ),
        rich_traceback: bool = typer.Option(
            False,
            "--rich-traceback-short/--rich-traceback-full",
            help="Control traceback verbosity",
        ),
        rich_locals: bool = typer.Option(
            True,
            "--rich-show-locals/--rich-hide-locals",
            help="Toggle local variables in exceptions",
        ),
        rich_theme: Optional[str] = typer.Option(None, "--rich-theme", help="Color theme (dark/light/monochrome)"),
    ):
        _configure_global_options(app, rich_exceptions, rich_traceback, rich_locals, rich_theme, verbose)

    return global_options


def _configure_global_options(
    app: Typer,
    rich_exceptions=False,
    rich_traceback=True,
    rich_locals=True,
    rich_theme=None,
    verbose=False,
):
    configure_logging(verbose)

    app.pretty_exceptions_enable = rich_exceptions
    app.pretty_exceptions_short = False if rich_exceptions else rich_traceback
    app.pretty_exceptions_show_locals = True if rich_exceptions else rich_locals

    if rich_theme:
        from rich.traceback import Traceback

        Traceback.theme = rich_theme


def configure_logging(verbose: bool):
    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
    )

    logger = logging.getLogger("torchx")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.addHandler(handler)


def _add_typer_nested(typer: Typer, to_add: dict):
    for key, value in to_add.items():
        if isinstance(value, dict):
            nested = _get_or_add_typer(typer, name=key)
            _add_typer_nested(nested, value)  # type: ignore
        elif hasattr(value, "cli"):
            value.cli(typer)
        else:
            raise ValueError(f"Cannot add {value} to typer")


def _get_or_add_typer(typer: Typer, name: str, help=None, **kwargs):
    for r in typer.registered_groups:
        if name == r.name:
            return r.typer_instance

    help = help or name
    help = f"[Module] {help}"

    output = Typer(pretty_exceptions_enable=False)
    typer.add_typer(
        output,
        name=name,
        help=help,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        **kwargs,
    )

    return output


@dataclass
class FactoryRegistration:
    namespace: str
    name: str


def _register_factory(
    fn: Callable,
    target: Optional[Type],
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    target_arg: Optional[str] = None,
    is_target_default: bool = False,
) -> FactoryRegistration:
    _name: str = name or fn.__name__

    if namespace:
        _namespace = namespace
    elif target:
        _namespace = get_type_namespace(target)
    else:
        _return_type = _get_return_type(fn)
        if isinstance(_return_type, (Config, Partial)) or (
            hasattr(_return_type, "__origin__") and issubclass(_return_type.__origin__, (Config, Partial))
        ):
            _return_type = get_args(_return_type)[0]

        _namespace = get_type_namespace(_return_type)

    if target_arg:
        assert target, "target_arg cannot be used without specifying a parent."
        _namespace = f"{_namespace}.{target_arg}"

    catalogue._set((_namespace, _name), fn)
    if is_target_default:
        catalogue._set((_namespace, DEFAULT_NAME), fn)

    return FactoryRegistration(namespace=_namespace, name=_name)


def _get_return_type(fn: Callable) -> Type:
    return_type = inspect.signature(fn).return_annotation
    if return_type is inspect.Signature.empty:
        raise TypeError(f"Missing return type annotation for function '{fn}'")

    # Handle forward references
    if isinstance(return_type, str):
        return_type = eval(return_type)

    return return_type


@cache
def _load_entrypoints():
    entrypoints = metadata.entry_points().select(group="nemo_run.cli")
    for ep in entrypoints:
        try:
            ep.load()
        except Exception as e:
            print(f"Couldn't load entrypoint {ep.name}: {e}")


def _search_workspace_file() -> str | None:
    if not INCLUDE_WORKSPACE_FILE:
        return None

    current_dir = os.getcwd()
    file_names = [
        "workspace_private.py",
        "workspace.py",
        os.path.join(get_nemorun_home(), "workspace.py"),
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


@dataclass(kw_only=True)
class RunContext:
    """
    Represents the context for executing a run in the NeMo Run framework.
    """

    name: str
    direct: bool = False
    dryrun: bool = False
    factory: Optional[str] = None
    load: Optional[str] = None
    repl: bool = False
    detach: bool = False
    skip_confirmation: bool = False
    tail_logs: bool = False
    yaml: Optional[str] = None

    to_yaml: Optional[str] = None
    to_toml: Optional[str] = None
    to_json: Optional[str] = None

    executor: Optional[Executor] = field(init=False)
    plugins: List[Plugin] = field(init=False)

    @classmethod
    def cli_command(
        cls,
        parent: typer.Typer,
        name: str,
        fn: Callable | LazyEntrypoint,
        default_factory: Optional[Callable] = None,
        default_executor: Optional[Executor] = None,
        default_plugins: Optional[List[Plugin]] = None,
        type: Literal["task", "experiment"] = "task",
        command_kwargs: Dict[str, Any] = {},
        is_main: bool = False,
        cmd_defaults: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a CLI command for the given function.

        Args:
            parent (typer.Typer): The parent Typer instance to add the command to.
            name (str): The name of the command.
            fn (Callable): The function to create a command for.
            type (Literal["task", "experiment"]): The type of the command.
            command_kwargs (Dict[str, Any]): Additional keyword arguments for the command.

        Returns:
            Callable: The created command function.
        """

        @parent.command(
            name,
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
            **command_kwargs,
        )
        def command(
            run_name: str = typer.Option(None, "--name", "-n", help="Name of the run"),
            direct: bool = typer.Option(False, "--direct/--no-direct", help="Execute the run directly"),
            dryrun: bool = typer.Option(False, "--dryrun", help="Print the scheduler request without submitting"),
            factory: Optional[str] = typer.Option(None, "--factory", "-f", help="Predefined factory to use"),
            load: Optional[str] = typer.Option(None, "--load", "-l", help="Load a factory from a directory"),
            yaml: Optional[str] = typer.Option(None, "--yaml", "-y", help="Path to a YAML file to load"),
            repl: bool = typer.Option(False, "--repl", "-r", help="Enter interactive mode"),
            detach: bool = typer.Option(False, "--detach", help="Detach from the run"),
            skip_confirmation: bool = typer.Option(
                False, "--yes", "-y", "--no-confirm", help="Skip confirmation before execution"
            ),
            tail_logs: bool = typer.Option(False, "--tail-logs/--no-tail-logs", help="Tail logs after execution"),
            verbose: bool = Option(False, "-v", "--verbose", help="Enable verbose logging"),
            rich_exceptions: bool = typer.Option(
                False,
                "--rich-exceptions/--no-rich-exceptions",
                help="Enable rich exception formatting",
            ),
            rich_traceback: bool = typer.Option(
                False,
                "--rich-traceback-short/--rich-traceback-full",
                help="Control traceback verbosity",
            ),
            rich_locals: bool = typer.Option(
                True,
                "--rich-show-locals/--rich-hide-locals",
                help="Toggle local variables in exceptions",
            ),
            rich_theme: Optional[str] = typer.Option(None, "--rich-theme", help="Color theme (dark/light/monochrome)"),
            to_yaml: Optional[str] = typer.Option(None, "--to-yaml", help="Export config to YAML file"),
            to_toml: Optional[str] = typer.Option(None, "--to-toml", help="Export config to TOML file"),
            to_json: Optional[str] = typer.Option(None, "--to-json", help="Export config to JSON file"),
            ctx: typer.Context = typer.Context,
        ):
            _cmd_defaults = cmd_defaults or {}
            self = cls(
                name=run_name or name,
                direct=direct or _cmd_defaults.get("direct", False),
                dryrun=dryrun or _cmd_defaults.get("dryrun", False),
                factory=factory or default_factory,
                load=load or _cmd_defaults.get("load", None),
                yaml=yaml or _cmd_defaults.get("yaml", None),
                repl=repl or _cmd_defaults.get("repl", False),
                detach=detach or _cmd_defaults.get("detach", False),
                skip_confirmation=skip_confirmation or _cmd_defaults.get("skip_confirmation", False),
                tail_logs=tail_logs or _cmd_defaults.get("tail_logs", False),
                to_yaml=to_yaml or _cmd_defaults.get("to_yaml", None),
                to_toml=to_toml or _cmd_defaults.get("to_toml", None),
                to_json=to_json or _cmd_defaults.get("to_json", None),
            )

            print("Configuring global options")
            _configure_global_options(
                parent,
                rich_exceptions or _cmd_defaults.get("rich_exceptions", False),
                rich_traceback or _cmd_defaults.get("rich_traceback", True),
                rich_locals or _cmd_defaults.get("rich_locals", True),
                rich_theme or _cmd_defaults.get("rich_theme", None),
                verbose or _cmd_defaults.get("verbose", False),
            )

            if default_executor:
                self.executor = default_executor

            if default_plugins:
                self.plugins = default_plugins

            _load_workspace()
            if isinstance(fn, LazyEntrypoint):
                self.execute_lazy(fn, sys.argv, name)
                return

            try:
                if not is_main:
                    _load_entrypoints()
                self.cli_execute(fn, ctx.args, type)
            except RunContextError as e:
                if not verbose:
                    typer.echo(f"Error: {str(e)}", err=True, color=True)
                    raise typer.Exit(code=1)
                raise  # Re-raise the exception for verbose mode
            except Exception as e:
                if not verbose:
                    typer.echo(f"Unexpected error: {str(e)}", err=True, color=True)
                    raise typer.Exit(code=1)
                raise  # Re-raise the exception for verbose mode

        return command

    def launch(self, experiment: Experiment, sequential: bool = False):
        """
        Launch the given experiment, respecting the RunContext settings.

        This method launches the experiment, taking into account the various options
        set in the RunContext, such as dryrun, detach, direct execution, and log tailing.

        Args:
            experiment (Experiment): The experiment to launch.
            sequential (bool): If True, run the experiment sequentially.

        Note:
            - If self.dryrun is True, it will only perform a dry run of the experiment.
            - The method respects self.detach for detached execution.
            - It uses direct execution if self.direct is True or if no executor is set.
            - Log tailing behavior is controlled by self.tail_logs.
        """
        if self.dryrun:
            experiment.dryrun()
        else:
            experiment.run(
                sequential=sequential,
                detach=self.detach,
                direct=self.direct or self.executor is None,
                tail_logs=self.tail_logs,
            )

    def cli_execute(self, fn: Callable, args: List[str], entrypoint_type: Literal["task", "experiment"] = "task"):
        """
        Execute the given function as a CLI command.

        Args:
            fn (Callable): The function to execute.
            args (List[str]): The command-line arguments.
            entrypoint_type (Literal["task", "experiment"]): The type of entrypoint.

        Raises:
            ValueError: If an unknown entrypoint type is provided.
        """
        _, run_args, filtered_args = _parse_prefixed_args(args, "run")
        self.parse_args(run_args)

        if self.load:
            raise NotImplementedError("Load is not implemented yet")

        if entrypoint_type == "task":
            self._execute_task(fn, filtered_args)
        elif entrypoint_type == "experiment":
            self._execute_experiment(fn, filtered_args)
        else:
            raise ValueError(f"Unknown entrypoint type: {entrypoint_type}")

    def _execute_task(self, fn: Callable, task_args: List[str]):
        """
        Execute a task.

        Args:
            fn (Callable): The task function to execute.
            task_args (List[str]): The arguments for the task.
        """
        import nemo_run as run

        console = Console()
        task = self.parse_fn(fn, task_args)

        # If any export flag is used, export the configuration and exit
        if any([self.to_yaml, self.to_toml, self.to_json]):
            _serialize_configuration(
                task,
                self.to_yaml,
                self.to_toml,
                self.to_json,
                console=console,
                verbose=True,
            )
            console.print("[bold cyan]Export complete. Skipping execution.[/bold cyan]")
            return

        # Continue with normal execution flow
        def run_task():
            nonlocal task
            run.dryrun_fn(task, executor=self.executor)

            if self.dryrun:
                console.print(f"[bold cyan]Dry run for {self.name}:[/bold cyan]")
                return

            if self._should_continue(self.skip_confirmation):
                console.print(f"[bold cyan]Launching {self.name}...[/bold cyan]")
                run.run(
                    fn_or_script=task,
                    name=self.name,
                    executor=self.executor,
                    plugins=self.plugins,
                    direct=self.direct or self.executor is None,
                    detach=self.detach,
                )
            else:
                console.print("[bold cyan]Exiting...[/bold cyan]")

        if self.repl:
            from IPython import embed

            console.print("[bold cyan]Entering interactive mode...[/bold cyan]")
            console.print("Use 'task' to access and modify the Partial object.")
            console.print("Use 'run_task()' to execute the task when ready.")

            embed(colors="neutral")
            return

        run_task()

    def execute_lazy(self, entrypoint: LazyEntrypoint, args: List[str], name: str):
        console = Console()

        import nemo_run as run

        if self.dryrun:
            raise ValueError("Dry run is not supported for lazy execution")

        if self.repl:
            raise ValueError("Interactive mode is not supported for lazy execution")

        if self.direct:
            raise ValueError("Direct execution is not supported for lazy execution")

        # Parse run args, filtering out CLI flags
        _, run_args, args = _parse_prefixed_args(args, "run")
        filtered_run_args = [arg for arg in run_args if not arg.startswith("--")]
        self.parse_args(filtered_run_args, lazy=True)

        # Filter out --lazy and export flags from args
        cmd, cmd_args, i_self = "", [], 0
        for i, arg in enumerate(sys.argv):
            if arg == "--lazy" or arg.startswith("--to-"):
                continue
            if arg == name:
                i_self = i
            if i_self == 0:
                cmd += f" {arg}"
            elif "=" not in arg and not arg.startswith("--"):
                cmd += f" {arg}"
            elif "=" in arg and not arg.startswith("--"):
                cmd_args.append(arg)

        # Use the yaml file if provided
        yaml_file = self.yaml
        to_run = LazyEntrypoint(cmd, factory=self.factory, yaml=yaml_file)

        # Filter out CLI flags from cmd_args
        filtered_cmd_args = [arg for arg in cmd_args if not arg.startswith("--")]
        to_run._add_overwrite(*filtered_cmd_args)

        # If any export flag is used, export the configuration and exit
        if any([self.to_yaml, self.to_toml, self.to_json]):
            _serialize_configuration(to_run, self.to_yaml, self.to_toml, self.to_json, is_lazy=True, console=console)
            console.print("[bold cyan]Export complete. Skipping execution.[/bold cyan]")
            return

        # Continue with normal execution flow
        if self._should_continue(self.skip_confirmation):
            console.print(f"[bold cyan]Launching {self.name}...[/bold cyan]")
            run.run(
                fn_or_script=to_run,
                name=self.name,
                executor=self.executor,
                plugins=self.plugins,
                direct=False,
                detach=self.detach,
            )
        else:
            console.print("[bold cyan]Exiting...[/bold cyan]")

    def _execute_experiment(self, fn: Callable, experiment_args: List[str]):
        """
        Execute an experiment.

        Args:
            fn (Callable): The experiment function to execute.
            experiment_args (List[str]): The arguments for the experiment.
        """
        import nemo_run as run

        console = Console()
        partial = self.parse_fn(fn, experiment_args, ctx=self)

        # If any export flag is used, export the configuration and exit
        if any([self.to_yaml, self.to_toml, self.to_json]):
            _serialize_configuration(
                partial,
                self.to_yaml,
                self.to_toml,
                self.to_json,
                is_lazy=False,  # The helper function will check LAZY_CLI env var
                console=console,
            )
            console.print("[bold cyan]Export complete. Skipping execution.[/bold cyan]")
            return

        # Continue with normal execution flow
        run.dryrun_fn(partial, executor=self.executor)

        if self._should_continue(self.skip_confirmation):
            fdl.build(partial)()

    def _should_continue(self, skip_confirmation: bool) -> bool:
        """
        Check if the execution should continue based on user confirmation.

        Args:
            skip_confirmation (bool): Whether to skip user confirmation.

        Returns:
            bool: True if execution should continue, False otherwise.
        """
        global NEMORUN_SKIP_CONFIRMATION

        # If we're running under torchrun, always continue
        if _is_torchrun():
            logger.info("Detected torchrun environment. Skipping confirmation.")
            return True

        if NEMORUN_SKIP_CONFIRMATION is not None:
            return NEMORUN_SKIP_CONFIRMATION

        # If skip_confirmation is True or user confirms, continue
        if skip_confirmation or typer.confirm("Continue?"):
            NEMORUN_SKIP_CONFIRMATION = True
            return True

        # Otherwise, don't continue
        NEMORUN_SKIP_CONFIRMATION = False
        return False

    def parse_fn(self, fn: T, args: List[str], **default_kwargs) -> Partial[T]:
        """
        Parse the given function and arguments into a Partial object.

        Args:
            fn (T): The function to parse.
            args (List[str]): The arguments to parse.
            **default_kwargs: Default keyword arguments.

        Returns:
            Partial[T]: A Partial object representing the parsed function and arguments.
        """
        lazy = LazyEntrypoint(
            fn,
            factory=self.factory,
            yaml=self.yaml,
            overwrites=args,
        )

        # Resolve exactly once and always pass the current RunContext
        # NOTE: `LazyEntrypoint.resolve` calls `parse_factory` if
        # `lazy._factory_` is a string.  `parse_cli_args` that follows inside
        # `resolve` used to see the **same** string and call `parse_factory`
        # a second time.  We temporarily clear `_factory_` right after the
        # first resolution so that it cannot be triggered again.

        _orig_factory = lazy._factory_
        try:
            result = lazy.resolve(ctx=self)
        finally:
            # Restore for potential further use
            lazy._factory_ = _orig_factory

        return result

    def _parse_partial(self, fn: Callable, args: List[str], **default_args) -> Partial[T]:
        """
        Parse the given function and arguments into a Partial object.

        Args:
            fn (Callable): The function to parse.
            args (List[str]): The arguments to parse.
            **default_args: Default arguments.

        Returns:
            Partial[T]: A Partial object representing the parsed function and arguments.
        """
        config = parse_cli_args(fn, args, output_type=Partial)
        for key, value in default_args.items():
            setattr(config, key, value)
        return config

    def parse_args(self, args: List[str], lazy: bool = False):
        """
        Parse the given arguments and update the RunContext accordingly.

        Args:
            args (List[str]): The arguments to parse.
        """
        executor_name, executor_args, args = _parse_prefixed_args(args, "executor")
        plugin_name, plugin_args, args = _parse_prefixed_args(args, "plugins")

        if executor_name:
            self.executor = self.parse_executor(executor_name, *executor_args)
        else:
            if hasattr(self, "executor"):
                parse_cli_args(self.executor, args, self.executor)
            else:
                self.executor = None
        if plugin_name:
            plugins = self.parse_plugin(plugin_name, *plugin_args)
            if not isinstance(plugins, list):
                plugins = [plugins]
            self.plugins = plugins
        else:
            if hasattr(self, "plugins"):
                parse_cli_args(self.plugins, args)
                if not isinstance(self.plugins, list):
                    self.plugins = [self.plugins]
            else:
                self.plugins = []

        if self.executor:
            self.executor = fdl.build(self.executor)

        if self.plugins:
            self.plugins = fdl.build(self.plugins)

        if not lazy and args:
            parse_cli_args(self, args, self)

        return args

    def parse_executor(self, name: str, *args: str) -> Partial[Executor]:
        """
        Parse the executor configuration.

        Args:
            name (str): The name of the executor.
            *args (str): Additional arguments for the executor.

        Returns:
            Partial[Executor]: A Partial object representing the parsed executor.

        Raises:
            ValueError: If the specified executor is not found.
        """
        for cls in EXECUTOR_CLASSES:
            try:
                executor = parse_factory(self.__class__, "executor", cls, name)
                executor = parse_cli_args(executor, args)
                return executor
            except ValueError:
                continue

        raise ValueError(f"Executor {name} not found")

    def parse_plugin(self, name: str, *args: str) -> Optional[Partial[Plugin]]:
        """
        Parse the plugin configuration.

        Args:
            name (str): The name of the plugin.
            *args (str): Additional arguments for the plugin.

        Returns:
            Optional[Partial[Plugin]]: A Partial object representing the parsed plugin, or None.

        Raises:
            ValueError: If the specified plugin is not found.
        """
        for cls in PLUGIN_CLASSES:
            try:
                plugins = parse_factory(self.__class__, "plugins", cls, name)
                plugins = parse_cli_args(plugins, args)
                return plugins
            except ValueError:
                continue

        raise ValueError(f"Plugin {name} not found")

    def to_config(self) -> Config:
        """
        Convert this RunContext object to a Config object.

        Returns:
            Config: A Config object representing this RunContext.
        """
        return fdl.cast(Config, fdl_dc.convert_dataclasses_to_configs(self, allow_post_init=True))

    @classmethod
    def get_help(cls) -> str:
        """
        Get the help text for this class.

        Returns:
            str: The help text extracted from the class docstring.
        """
        return cls.__doc__ or "No help available."


class Entrypoint(Generic[Params, ReturnType]):
    """
    Represents an entrypoint for a CLI command in the NeMo Run framework.

    This class encapsulates the functionality required to create and manage
    CLI commands, including parsing arguments, executing functions, and
    handling different types of entrypoints (tasks or experiments).

    Args:
        fn (Callable[Params, ReturnType]): The function to be executed.
        namespace (str): The namespace for the entrypoint.
        env (Optional[Dict]): Environment variables for the entrypoint.
        name (Optional[str]): The name of the entrypoint.
        help_str (Optional[str]): Help string for the entrypoint.
        enable_executor (bool): Whether to enable executor functionality.
        skip_confirmation (bool): Whether to skip user confirmation before execution.
        type (Literal["task", "experiment"]): The type of entrypoint.
        run_ctx_cls (Type[RunContext]): The RunContext class to use.

    Raises:
        ValueError: If the function signature is invalid for the given entrypoint type.
    """

    def __init__(
        self,
        fn: Callable[Params, ReturnType],
        namespace: str,
        default_factory: Optional[Callable] = None,
        default_executor: Optional[Config[Executor]] = None,
        default_plugins: Optional[List[Plugin]] = None,
        env=None,
        name=None,
        help_str=None,
        enable_executor: bool = True,
        skip_confirmation: bool = False,
        type: Literal["task", "experiment"] = "task",
        run_ctx_cls: Type[RunContext] = RunContext,
    ):
        if type == "task":
            if "executor" in inspect.signature(fn).parameters:
                raise ValueError("The function cannot have an argument named `executor` as it is a reserved keyword.")
        elif type in ("sequential_experiment", "parallel_experiment"):
            if "ctx" not in inspect.signature(fn).parameters:
                raise ValueError(
                    "The function must have an argument named `ctx` as it is a required argument for experiments."
                )

        self.fn = fn
        self.arg_types = {}
        self.env = env or {}
        self.name = name or fn.__name__
        self.help_str = self.name
        self.run_ctx_cls = run_ctx_cls
        if help_str:
            self.help_str += f"\n{help_str}"
        elif fn.__doc__:
            self.help_str += f"\n\n {fn.__doc__.split('Args:')[0].strip()}"
        self.namespace = namespace
        self._configured_fn = None
        self.enable_executor = enable_executor
        self.skip_confirmation = skip_confirmation
        self.type = type
        self.default_factory = default_factory
        self.default_plugins = default_plugins
        self.default_executor = default_executor

    def __call__(self, *args: Params.args, **kwargs: Params.kwargs) -> ReturnType:
        return self.fn(*args, **kwargs)

    def configure(self, **fn_kwargs: dict[str, fdl.Config | str | Callable]):
        self._configured_fn = Config(self.fn, **fn_kwargs)

    def parse_partial(self, args: List[str], **default_args) -> Partial[T]:
        """
        Parse the given arguments into a Partial object.

        Args:
            args (List[str]): The arguments to parse.
            **default_args: Default arguments to include.

        Returns:
            Partial[T]: A Partial object representing the parsed arguments.
        """

        config = parse_cli_args(self.fn, args, output_type=Partial)
        for key, value in default_args.items():
            setattr(config, key, value)
        return config

    def cli(self, parent: typer.Typer):
        self._add_command(parent)

    def _add_command(
        self,
        typer_instance: typer.Typer,
        is_main: bool = False,
        cmd_defaults: Optional[Dict[str, Any]] = None,
    ):
        if self.enable_executor:
            self._add_executor_command(typer_instance, is_main=is_main, cmd_defaults=cmd_defaults)
        else:
            self._add_simple_command(typer_instance, is_main=is_main)

    def _add_simple_command(self, typer_instance: typer.Typer, is_main: bool = False):
        @typer_instance.command(
            self.name,
            help=self.help_str,
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        )
        def cmd_cli(ctx: typer.Context):
            console = Console()
            try:
                _load_entrypoints()
                _load_workspace()
                self._execute_simple(ctx.args, console)
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                sys.exit(1)

    def _add_executor_command(
        self,
        parent: typer.Typer,
        is_main: bool = False,
        cmd_defaults: Optional[Dict[str, Any]] = None,
    ):
        help = self.help_str
        colored_help = None
        if help:
            colored_help = f"[Entrypoint] {help}"

        class CLITaskCommand(EntrypointCommand):
            _entrypoint = self

        return self.run_ctx_cls.cli_command(
            parent,
            self.name,
            self.fn,
            type=self.type,
            default_factory=self.default_factory,
            default_executor=self.default_executor,
            default_plugins=self.default_plugins,
            command_kwargs=dict(
                help=colored_help,
                cls=CLITaskCommand,
            ),
            is_main=is_main,
            cmd_defaults=cmd_defaults,
        )

    def _add_options_to_command(self, command: Callable):
        """Add Typer options to the command based on RunContext annotated attributes."""
        for attr_name, type_hint in get_type_hints(self.run_ctx_cls, include_extras=True).items():
            if hasattr(type_hint, "__metadata__"):
                option = type_hint.__metadata__[0]
                if isinstance(option, OptionInfo):
                    option(command, attr_name)

    def _execute_simple(self, args: List[str], console: Console):
        config = parse_cli_args(self.fn, args, Partial)
        fn = fdl.build(config)
        fn.func.__io__ = config
        fn()

    def main(self, cmd_defaults: Optional[Dict[str, Any]] = None):
        app = typer.Typer(help=self.help_str, pretty_exceptions_enable=NEMORUN_PRETTY_EXCEPTIONS)
        self._add_command(app, is_main=True, cmd_defaults=cmd_defaults)
        app(standalone_mode=False)

    def help(self, console=Console(), with_docs: bool = True):
        import nemo_run as run

        run.help(self.fn, console=console, with_docs=with_docs)

    @property
    def path(self):
        return ".".join([self.namespace, self.fn.__name__])


class EntrypointCommand(TyperCommand):
    _entrypoint: Entrypoint

    def format_usage(self, ctx, formatter) -> None:
        pieces = self.collect_usage_pieces(ctx) + ["[ARGUMENTS]"]
        formatter.write_usage(ctx.command_path, " ".join(pieces))

    def format_help(self, ctx, formatter):
        from nemo_run.help import class_to_str

        out = rich_utils.rich_format_help(
            obj=self,
            ctx=ctx,
            markup_mode=self.rich_markup_mode,
        )

        # TODO: Check if args are passed in to provide help for
        # print(sys.argv[1:])

        console = rich_utils._get_rich_console()

        box_style = getattr(box, BOX_STYLE, None)
        table = Table(
            highlight=True,
            show_header=False,
            expand=True,
            box=box_style,
            **TABLE_STYLES,
        )
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="magenta")
        if self._entrypoint.default_factory:
            table.add_row("factory", class_to_str(self._entrypoint.default_factory))
        if self._entrypoint.default_executor:
            table.add_row("executor", class_to_str(self._entrypoint.default_executor))
        if self._entrypoint.default_plugins:
            table.add_row("plugins", class_to_str(self._entrypoint.default_plugins))
        if table.row_count > 0:
            console.print(
                Panel(
                    table,
                    title="Defaults",
                    border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
                    title_align=rich_utils.ALIGN_OPTIONS_PANEL,
                )
            )

        self._entrypoint.help(console, with_docs=sys.argv[-1] in ("--docs", "-d"))

        return out


class GeneralCommand(TyperGroup):
    def format_usage(self, ctx, formatter) -> None:
        pieces = self.collect_usage_pieces(ctx) + ["[ARGUMENTS]"]
        formatter.write_usage(ctx.command_path, " ".join(pieces))

    def format_help(self, ctx, formatter):
        out = rich_utils.rich_format_help(
            obj=self,
            ctx=ctx,
            markup_mode=self.rich_markup_mode,
        )

        # TODO: Check if args are passed in to provide help for
        # print(sys.argv[1:])

        return out


def _parse_prefixed_args(args: List[str], prefix: str) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Parse arguments to separate prefixed args from others.

    Args:
        args (List[str]): List of command-line arguments.
        prefix (str): The prefix to look for in arguments.

    Returns:
        Tuple[Optional[str], List[str], List[str]]: A tuple containing:
            - The value of the prefixed argument (if any)
            - List of arguments specific to the prefix
            - List of other arguments

    Example:
        For prefix "executor":
        executor=local executor.gpus=2 other_arg=value
        Returns: ("local", ["gpus=2"], ["other_arg=value"])
    """
    prefixed_arg_value, prefixed_args, other_args = None, [], []
    for arg in args:
        if arg.startswith(prefix):
            if arg.startswith(f"{prefix}="):
                prefixed_arg_value = arg.split("=")[1]
            else:
                if not arg.startswith(f"{prefix}.") and not arg.startswith(f"{prefix}["):
                    raise ValueError(f"{prefix.capitalize()} overwrites must start with '{prefix}.'. Got {arg}")
                if arg.startswith(f"{prefix}."):
                    prefixed_args.append(arg.replace(f"{prefix}.", ""))
                elif arg.startswith(f"{prefix}["):
                    prefixed_args.append(arg.replace(prefix, ""))
        else:
            other_args.append(arg)
    return prefixed_arg_value, prefixed_args, other_args


class RunContextError(Exception):
    """Base exception for RunContext related errors."""


class InvalidOptionError(RunContextError):
    """Raised when an invalid option is provided."""


class MissingRequiredOptionError(RunContextError):
    """Raised when a required option is missing."""


def _is_torchrun() -> bool:
    """Check if running under torchrun."""
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def _serialize_configuration(
    config: Any,
    to_yaml: Optional[str] = None,
    to_toml: Optional[str] = None,
    to_json: Optional[str] = None,
    is_lazy: bool = False,
    console: Optional[Console] = None,
    verbose: bool = False,
) -> None:
    """
    Serialize configuration to specified file formats.

    Args:
        config: The configuration object to serialize
        to_yaml: Path to export YAML configuration
        to_toml: Path to export TOML configuration
        to_json: Path to export JSON configuration
        is_lazy: Whether to use lazy serialization
        console: Console instance for printing messages
        verbose: Whether to show detailed output with syntax highlighting

    Raises:
        ValueError: If no output format is specified
    """
    if not any([to_yaml, to_toml, to_json]):
        raise ValueError("At least one output format must be provided")

    console = console or Console() if verbose else None
    serializer = ConfigSerializer()

    def _export_config(output_path: str, format: Optional[str] = None) -> None:
        """Export configuration to specified format and path."""
        if not output_path:
            return

        format_name = format.upper() if format else "YAML"

        try:
            # Handle section extraction from path
            section = None
            file_path = output_path
            if ":" in output_path:
                file_path, section = output_path.split(":", 1)

            # Create appropriate section message for display
            section_msg = f" (section: {section})" if section else ""

            # Handle section extraction for Config objects
            section_config = config
            if section:
                # For Config objects, use getattr to access the section
                if hasattr(config, section):
                    section_config = getattr(config, section)
                else:
                    raise ValueError(f"Section '{section}' not found in configuration")

            # Now serialize the extracted section
            if is_lazy:
                # For lazy configs, create a nested structure
                if section:
                    # Section already extracted, just serialize it
                    serializer.dump(section_config, file_path)
                else:
                    # Standard lazy config handling
                    config_dict = {
                        "_target_": config._target_path_ if hasattr(config, "_target_path_") else str(config._target_),
                    }
                    if config._factory_:
                        config_dict["_factory_"] = str(config._factory_)

                    # Convert _args_ to nested structure
                    for path_arg, op, value in config._args_:
                        current = config_dict
                        parts = path_arg.split(".")

                        # Handle nested paths
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]

                        # Handle the final value
                        current[parts[-1]] = value

                    serializer.dump_dict(config_dict, file_path, format=format)
            else:
                # For regular configs, we've already extracted the section if needed
                serializer.dump(section_config, file_path)

            if verbose and console:
                format_ext = format.lower() if format else "yaml"
                # Read the file to display its contents
                with open(file_path, "r") as f:
                    file_content = f.read()

                # Display the content with syntax highlighting
                console.print(
                    f"[bold green]Configuration exported to {format_name}{section_msg}:[/bold green] {file_path}"
                )
                console.print("[bold cyan]File contents:[/bold cyan]")
                console.print(
                    Panel(
                        Syntax(
                            code=file_content,
                            lexer=format_ext,
                            theme="default",
                            line_numbers=True,
                            word_wrap=True,
                            background_color="default",
                        ),
                        title=f"[bold]{file_path}[/bold]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )
            elif verbose:
                print(f"Configuration exported to {format_name}{section_msg}: {file_path}")

        except Exception as e:
            if verbose and console:
                console.print(
                    f"[bold red]Failed to export configuration to {format_name}{section_msg}:[/bold red] {str(e)}"
                )
            else:
                print(f"Failed to export configuration to {format_name}{section_msg}: {str(e)}")
            raise  # Re-raise the exception so test cases can catch it

    # Export to each requested format
    if to_yaml:
        _export_config(to_yaml, format="yaml")
    if to_toml:
        _export_config(to_toml, format="toml")
    if to_json:
        _export_config(to_json, format="json")


def extract_constituent_types(type_hint: Any) -> Set[Type]:
    """
    Extract all constituent types from a type hint, including generics and their type arguments.
    This function recursively traverses complex type hints to find all underlying types.

    For example:
    - For Union[int, str] -> {int, str}
    - For List[int] -> {list, int}
    - For Dict[str, Optional[int]] -> {dict, str, int}
    - For Callable[..., int] -> {Callable}
    - For TypeVar('T') -> {TypeVar}
    - For Annotated[List[int], "metadata"] -> {list, int}
    - For Optional[List[int]] -> {list, int}
    - For a class MyClass -> {MyClass}

    The function handles:
    - Basic types (int, str, etc.)
    - Generic types (List, Dict, etc.)
    - Union and Optional types
    - Annotated types (extracts the underlying type)
    - TypeVars and ForwardRefs
    - Callable types
    - Custom classes

    Args:
        type_hint: A type hint to analyze. Can be any valid Python type annotation,
            including complex nested types.

    Returns:
        A set of all constituent types found in the type hint. NoneType is excluded
        from the results.

    Note:
        This function is particularly useful for type checking and reflection,
        where you need to know all the possible types that could be involved
        in a type annotation.
    """
    # Special case for functions and classes - return the type itself
    if inspect.isfunction(type_hint) or inspect.isclass(type_hint):
        return {type_hint}

    # Handle older style type hints (_GenericAlias)
    if hasattr(typing, "_GenericAlias") and isinstance(type_hint, typing._GenericAlias):  # type: ignore
        # Correctly handle Annotated by getting the first argument (the actual type)
        if str(type_hint).startswith("typing.Annotated") or str(type_hint).startswith("typing_extensions.Annotated"):
            # Recurse on the actual type, skipping metadata
            return extract_constituent_types(type_hint.__args__[0])
        else:
            origin = type_hint.__origin__

        if origin in RECURSIVE_TYPES:
            types = set()
            for arg in type_hint.__args__:
                # Add check to skip NoneType here as well
                if arg is not type(None):
                    types.update(extract_constituent_types(arg))
            return types
        # If not a recursive type handled above, treat it like a concrete generic
        # Collect types from arguments
        result = set()
        for arg in type_hint.__args__:
            if arg is not type(None):  # Also skip NoneType here for generics like list[Optional[int]]
                result.update(extract_constituent_types(arg))
        # Add the origin itself (e.g., list, dict)
        if isinstance(origin, type):
            result.add(origin)
        # Add the original type_hint if it's a specific generic instantiation (and not a Union/Optional)
        if origin is not None and origin not in RECURSIVE_TYPES:
            result.add(type_hint)  # type_hint is the _GenericAlias itself
        return result  # Return collected types

    # Handle Python 3.9+ style type hints
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    # Base case: no origin or args means it's a simple type
    if origin is None:
        if type_hint is type(None):
            return set()
        if isinstance(type_hint, type):
            return {type_hint}
        return {type_hint}  # Return the hint itself if not a type (e.g., TypeVar)

    # Handle Annotated for Python 3.9+
    if origin is Annotated:
        # Recurse on the actual type argument, skipping metadata
        return extract_constituent_types(args[0])

    # Union type (including Optional)
    if origin is Union:
        result = set()
        for arg in args:
            if arg is not type(None):  # Skip NoneType in Unions
                result.update(extract_constituent_types(arg))
        return result

    # List, Dict, etc. - collect types from arguments
    result = set()
    for arg in args:
        result.update(extract_constituent_types(arg))

    # Include the origin type itself if it's a class
    # This handles both typing module types and Python 3.9+ built-in generic types
    if isinstance(origin, type):
        result.add(origin)

    # Add the original type_hint if it's a specific generic instantiation (and not a Union/Annotated)
    if origin not in (None, Union, Annotated):
        # type_hint is the original parameterized generic, e.g., List[int]
        # Add it only if it's indeed a generic (origin of type_hint itself is not None)
        if typing.get_origin(type_hint) is not None:
            result.add(type_hint)

    # If no types were added, return the original type hint to preserve behavior
    if not result:  # This covers cases like type_hint being a TypeVar that resulted in an empty set initially
        return {type_hint}

    return result


if __name__ == "__main__":
    app = create_cli()
    app()
