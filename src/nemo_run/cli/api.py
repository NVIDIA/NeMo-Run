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
from dataclasses import dataclass, field
from functools import cache, wraps
from typing import (Any, Callable, Generic, List, Literal, Optional, Protocol,
                    Tuple, Type, TypeVar, Union, get_args, overload,
                    runtime_checkable)

import catalogue
import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import importlib_metadata as metadata
import typer
from rich.console import Console
from rich.logging import RichHandler
from typer import Option, Typer, rich_utils
from typer.core import TyperCommand, TyperGroup
from typing_extensions import ParamSpec

from nemo_run.cli import devspace as devspace_cli
from nemo_run.cli import experiment as experiment_cli
from nemo_run.config import (NEMORUN_HOME, Config, Partial, Script,
                             get_type_namespace, get_underlying_types)
from nemo_run.core.cli_parser import parse_cli_args, parse_factory
from nemo_run.core.execution import (LocalExecutor, SkypilotExecutor,
                                     SlurmExecutor)
from nemo_run.core.execution.base import Executor
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


def entrypoint(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    help: Optional[str] = None,
    require_confirmation: bool = True,
    enable_executor: bool = True,
    entrypoint_cls: Optional[Type["Entrypoint"]] = None,
    type: Literal["task", "experiment"] = "task"
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
        require_confirmation (bool): If True, requires user confirmation before execution. Defaults to True.
        enable_executor (bool): If True, enables executor functionality for the entrypoint. Defaults to True.
        type (Literal["task", "experiment"]): The type of entrypoint. Defaults to "task".

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
    if namespace:
        _namespace = namespace
    else:
        caller = inspect.stack()[1]
        _module = inspect.getmodule(caller[0])
        if _module:
            _namespace = _module.__name__

    _entrypoint_cls = entrypoint_cls or Entrypoint

    def wrapper(f: F) -> F:
        _entrypoint = _entrypoint_cls(
            f,
            name=name,
            namespace=_namespace,
            help_str=help,
            require_confirmation=require_confirmation,
            enable_executor=enable_executor,
            type=type
        )

        if _namespace:
            parts = _namespace.split(".")
            task_namespace = (ROOT_ENTRYPOINT_NAMESPACE, *parts, f.__name__)
            catalogue._set(task_namespace, _entrypoint)

        f.cli_entrypoint = _entrypoint

        return f

    if fn is None:
        return wrapper

    return wrapper(fn)


def main(fn: F):
    """
    Execute the main CLI entrypoint for the given function.

    This function is used to run the CLI entrypoint associated with a decorated function.
    It checks if the function is properly decorated as an entrypoint and then executes its main CLI logic.

    Args:
        fn (F): The function decorated with the @entrypoint decorator.

    Raises:
        ValueError: If the provided function is not decorated as an entrypoint.

    Example:
        @entrypoint
        def my_cli_function():
            # CLI logic here
            pass

        if __name__ == "__main__":
            main(my_cli_function)
    """
    if not isinstance(fn, EntrypointProtocol):
        raise ValueError("The function is not an entrypoint.")

    fn.cli_entrypoint.main()


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
                isinstance(return_type, (Config, Partial)) or
                (hasattr(return_type, "__origin__") and issubclass(return_type.__origin__, (Config, Partial)))
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
            is_target_default=is_target_default
        )

        return as_factory

    return wrapper if fn is None else wrapper(fn)


def resolve_factory(
    target: Type[T] | str, name: str,
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

    _namespace = (
        get_type_namespace(type_or_namespace)
        if isinstance(type_or_namespace, type)
        else type_or_namespace
    )
    response = catalogue._get_all([_namespace])
    return list(response.values())


def create_cli(
    add_verbose_callback: bool = True,
    nested_entrypoints_creation: bool = True,
) -> Typer:
    app: Typer = Typer()
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
        app.callback()(global_options)

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


def global_options(verbose: bool = Option(False, "-v", "--verbose")):
    configure_logging(verbose)


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

    output = Typer()
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
        if (
            isinstance(_return_type, (Config, Partial)) or
            (hasattr(_return_type, "__origin__") and issubclass(_return_type.__origin__, (Config, Partial)))
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
    try:
        entrypoints = metadata.entry_points().select(group="nemo_run.cli")
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


@dataclass(kw_only=True)
class RunContext:
    name: str
    direct: bool = False
    dryrun: bool = False
    factory: Optional[str] = None
    load: Optional[str] = None
    repl: bool = False
    sequential: bool = True
    detach: bool = False
    require_confirmation: bool = True
    tail_logs: bool = False

    experiment: Experiment = field(init=False)
    executor: Optional[Executor] = field(init=False)
    plugins: List[Plugin] = field(init=False)

    def add(
        self,
        fn_or_script: Union[Partial, Script] | list[Union[Partial, Script]],
        executor: Executor | list[Executor] | None = None,
        name: str = "",
        plugins: Optional[list[Plugin]] = None,
        tail_logs: bool = False,
    ):
        self.experiment.add(
            fn_or_script, executor=executor, name=name, plugins=plugins, tail_logs=tail_logs
        )

    def run(
        self,
        fn: Callable,
        args: List[str],
        entrypoint_type: Literal["task", "experiment"] = "task"
    ):
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
        import nemo_run as run

        console = Console()
        task = self.parse_fn(fn, task_args)

        def run_task():
            nonlocal task
            run.dryrun_fn(task, executor=self.executor)

            if self.dryrun:
                console.print(f"[bold cyan]Dry run for {self.name}:[/bold cyan]")
                return

            if self._should_continue(self.require_confirmation):
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

    def _execute_experiment(self, fn: Callable, experiment_args: List[str]):
        import nemo_run as run

        with run.Experiment(title=self.name) as exp:
            self.experiment = exp
            partial = self.parse_fn(fn, experiment_args, ctx=self)

            run.dryrun_fn(partial, executor=self.executor)

            if self._should_continue(self.require_confirmation):
                fdl.build(partial)()

                if not exp.tasks:
                    raise ValueError(
                        "No tasks found in experiment, please add tasks using the `ctx.add` method."
                    )

                if self.dryrun:
                    exp.dryrun()
                else:
                    exp.run(
                        sequential=self.sequential,
                        detach=self.detach,
                        direct=self.direct or self.executor is None,
                        tail_logs=self.tail_logs
                    )

    def _should_continue(self, require_confirmation: bool) -> bool:
        return not require_confirmation or typer.confirm("Continue?")

    def parse_fn(self, fn: T, args: List[str], **default_kwargs) -> Partial[T]:
        if self.factory:
            output = parse_factory(fn, "factory", fn, self.factory)
        else:
            output = self._parse_partial(fn, args, **default_kwargs)

        return output

    def _parse_partial(self, fn: Callable, args: List[str], **default_args) -> Partial[T]:
        config = parse_cli_args(fn, args, output_type=Partial)
        for key, value in default_args.items():
            setattr(config, key, value)
        return config

    def parse_args(self, args: List[str]):
        executor_name, executor_args, args = _parse_prefixed_args(args, "executor")
        plugin_name, plugin_args, args = _parse_prefixed_args(args, "plugins")

        if executor_name:
            self.executor = self.parse_executor(executor_name, *executor_args)
        else:
            self.executor = None
        if plugin_name:
            plugins = self.parse_plugin(plugin_name, *plugin_args)
            if not isinstance(plugins, list):
                plugins = [plugins]
            self.plugins = plugins
        else:
            self.plugins = []

        if args:
            parse_cli_args(self, args, self)

    def parse_executor(self, name: str, *args: str) -> Partial[Executor]:
        for cls in EXECUTOR_CLASSES:
            try:
                executor = parse_factory(self.__class__, "executor", cls, name)
                executor = parse_cli_args(executor, args)
                return executor
            except ValueError:
                continue

        raise ValueError(f"Executor {name} not found")

    def parse_plugin(self, name: str, *args: str) -> Optional[Partial[Plugin]]:
        for cls in PLUGIN_CLASSES:
            try:
                plugins = parse_factory(self.__class__, "plugins", cls, name)
                plugins = parse_cli_args(plugins, args)
                return plugins
            except ValueError:
                continue

        return None

    def to_config(self) -> Config:
        """
        Converts this RunContext object to a run.Config object.

        Returns:
            Config: A Config object representing this plugin.
        """
        return fdl.cast(Config, fdl_dc.convert_dataclasses_to_configs(self, allow_post_init=True))


class Entrypoint(Generic[Params, ReturnType]):
    run_ctx_cls: Type[RunContext] = RunContext

    def __init__(
        self,
        fn: Callable[Params, ReturnType],
        namespace: str,
        env=None,
        name=None,
        help_str=None,
        enable_executor: bool = True,
        require_confirmation: bool = True,
        type: Literal["task", "experiment"] = "task"
    ):
        if type == "task":
            if "executor" in inspect.signature(fn).parameters:
                raise ValueError("The function cannot have an argument named `executor` as it is a reserved keyword.")
        elif type in ("sequential_experiment", "parallel_experiment"):
            if "ctx" not in inspect.signature(fn).parameters:
                raise ValueError("The function must have an argument named `ctx` as it is a required argument for experiments.")

        self.fn = fn
        self.arg_types = {}
        self.env = env or {}
        self.name = name or fn.__name__
        self.help_str = self.name
        if help_str:
            self.help_str += f"\n{help_str}"
        elif fn.__doc__:
            self.help_str += f"\n\n {fn.__doc__.split('Args:')[0].strip()}"
        self.namespace = namespace
        self._configured_fn = None
        self.enable_executor = enable_executor
        self.require_confirmation = require_confirmation
        self.type = type

    def __call__(self, *args: Params.args, **kwargs: Params.kwargs) -> ReturnType:
        return self.fn(*args, **kwargs)

    def configure(self, **fn_kwargs: dict[str, fdl.Config | str | Callable]):
        self._configured_fn = Config(self.fn, **fn_kwargs)

    def parse_partial(self, args: List[str], **default_args) -> Partial[T]:
        config = parse_cli_args(self.fn, args, output_type=Partial)
        for key, value in default_args.items():
            setattr(config, key, value)
        return config

    def cli(self, parent: typer.Typer):
        self._add_command(parent)

    def _add_command(self, typer_instance: typer.Typer):
        if self.enable_executor:
            self._add_executor_command(typer_instance)
        else:
            self._add_simple_command(typer_instance)

    def _add_simple_command(self, typer_instance: typer.Typer):
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

    def _add_executor_command(self, parent: typer.Typer):
        help = self.help_str
        colored_help = None
        if help:
            colored_help = f"[Entrypoint] {help}"

        class CLITaskCommand(EntrypointCommand):
            _entrypoint = self

        @parent.command(
            self.name,
            help=colored_help,
            cls=CLITaskCommand,
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        )
        def cmd_task(
            ctx: typer.Context,
            load: Optional[str] = typer.Option(
                None,
                "--load",
                "-l",
                help="Load a factory to a directory.",
            ),
            factory: Optional[str] = typer.Option(
                None,
                "--factory",
                "-f",
                help="Predefined factory to use for the task.",
            ),
            detach: bool = typer.Option(
                False,
                "--detach",
                "-d",
                help="Detach from the run.",
            ),
            dryrun: bool = typer.Option(
                False,
                help="Does not actually submit the app, just prints the scheduler request.",
            ),
            run_name: Optional[str] = typer.Option(
                None,
                "--name",
                "-n",
                help="Name of the run.",
            ),
            direct: bool = typer.Option(
                False,
                "--direct",
                "-d",
                help="Execute the run directly.",
            ),
            interactive: bool = typer.Option(
                False,
                "--repl",
                "-r",
                help="Enter interactive mode in a ipython-shell to construct and visualize the run.",
            ),
        ):
            console = Console()
            run_ctx = self.run_ctx_cls(
                name=run_name or self.name,
                direct=direct,
                dryrun=dryrun,
                factory=factory,
                load=load,
                repl=interactive,
                detach=detach,
                require_confirmation=self.require_confirmation,
            )
            try:
                _load_entrypoints()
                _load_workspace()

                run_ctx.run(self.fn, ctx.args, self.type)
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                sys.exit(1)

    def _execute_simple(self, args: List[str], console: Console):
        config = parse_cli_args(self.fn, args, Partial)
        fn = fdl.build(config)
        fn.func.__io__ = config
        fn()

    def main(self):
        app = typer.Typer(help=self.help_str)
        self._add_command(app)
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
        out = rich_utils.rich_format_help(
            obj=self,
            ctx=ctx,
            markup_mode=self.rich_markup_mode,
        )

        # TODO: Check if args are passed in to provide help for
        # print(sys.argv[1:])

        console = rich_utils._get_rich_console()
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


if __name__ == "__main__":
    app = create_cli()
    app()
