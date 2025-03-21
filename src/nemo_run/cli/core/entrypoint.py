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

import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
import functools
from typing import (
    Any, Callable, Dict, Generic, List, Literal, Optional, Protocol, 
    Tuple, Type, TypeVar, get_args, get_type_hints, runtime_checkable
)

import catalogue
import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer import Option, Typer
from typer.core import TyperCommand, TyperGroup
from typer.models import OptionInfo
from typing_extensions import NotRequired, ParamSpec, TypedDict

from nemo_run.cli import devspace as devspace_cli
from nemo_run.cli import experiment as experiment_cli
from nemo_run.cli.core.parser import parse_cli_args, parse_factory
from nemo_run.cli.cache import cache_entrypoint_metadata
from nemo_run.config import Config, Partial, get_type_namespace
from nemo_run.core.execution import LocalExecutor, SkypilotExecutor, SlurmExecutor
from nemo_run.core.execution.base import Executor
from nemo_run.core.frontend.console.styles import BOX_STYLE, TABLE_STYLES
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.lazy import LazyEntrypoint
from nemo_run.run.experiment import Experiment
from nemo_run.run.plugin import ExperimentPlugin as Plugin
from nemo_run.cli.core.workspace import load_entrypoints, load_workspace

F = TypeVar("F", bound=Callable[..., Any])
Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")
T = TypeVar("T")

ROOT_ENTRYPOINT_NAMESPACE = "nemo_run.cli.entrypoints"
ROOT_FACTORY_NAMESPACE = "nemo_run.cli.factories"
NEMORUN_SKIP_CONFIRMATION: Optional[bool] = None
POPULATE_CACHE: bool = False
INCLUDE_WORKSPACE_FILE = os.environ.get("INCLUDE_WORKSPACE_FILE", "true").lower() == "true"
NEMORUN_PRETTY_EXCEPTIONS = os.environ.get("NEMORUN_PRETTY_EXCEPTIONS", "false").lower() == "true"

logger = logging.getLogger(__name__)
MAIN_ENTRYPOINT = None

EXECUTOR_CLASSES = [Executor, LocalExecutor, SkypilotExecutor, SlurmExecutor]
PLUGIN_CLASSES = [Plugin, List[Plugin]]

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

@dataclass
class EntrypointOption:
    """Configuration for a CLI option."""
    default: Any          # Default value for the option
    help: str             # Help text for the option
    flags: Optional[List[str]] = None  # CLI flags (e.g., ["--name", "-n"])

    def __call__(self) -> typer.Option:
        """Return a configured typer.Option instance when called."""
        return typer.Option(
            self.default,
            *self.flags if self.flags else [],
            help=self.help,
        )


entrypoint_options = {
    "run_name": EntrypointOption(None, "Name of the run", ["--name", "-n"]),
    "direct": EntrypointOption(False, "Execute the run directly", ["--direct/--no-direct"]),
    "dryrun": EntrypointOption(False, "Print scheduler request without submitting", ["--dryrun"]),
    "factory": EntrypointOption(None, "Predefined factory to use", ["--factory", "-f"]),
    "load": EntrypointOption(None, "Load factory from directory", ["--load", "-l"]),
    "yaml": EntrypointOption(None, "Path to YAML file", ["--yaml", "-y"]),
    "repl": EntrypointOption(False, "Enter interactive mode", ["--repl", "-r"]),
    "detach": EntrypointOption(False, "Detach from the run", ["--detach"]),
    "skip_confirmation": EntrypointOption(False, "Skip confirmation", ["--yes", "-y"]),
    "tail_logs": EntrypointOption(False, "Tail logs after execution", ["--tail-logs/--no-tail-logs"]),
    "verbose": EntrypointOption(False, "Enable verbose logging", ["--verbose", "-v"]),
    "rich_exceptions": EntrypointOption(False, "Enable rich exception formatting", ["--rich-exceptions/--no-rich-exceptions"]),
    "rich_traceback": EntrypointOption(False, "Control traceback verbosity", ["--rich-traceback-short/--rich-traceback-full"]),
    "rich_locals": EntrypointOption(True, "Toggle local variables in exceptions", ["--rich-show-locals/--rich-hide-locals"]),
    "rich_theme": EntrypointOption(None, "Color theme (dark/light/monochrome)", ["--rich-theme"]),
    "to_yaml": EntrypointOption(None, "Export config to YAML file", ["--to-yaml"]),
    "to_toml": EntrypointOption(None, "Export config to TOML file", ["--to-toml"]),
    "to_json": EntrypointOption(None, "Export config to JSON file", ["--to-json"]),
}


@dataclass(kw_only=True)
class RunContext:
    """
    Represents the context for executing a run in the NeMo Run framework.

    This class encapsulates various options and settings that control how a run
    is executed, including execution mode, logging, and plugin configuration.

    Attributes:
        name (str): Name of the run.
        direct (bool): If True, execute the run directly without using a scheduler.
        dryrun (bool): If True, print the scheduler request without submitting.
        factory (Optional[str]): Name of a predefined factory to use.
        load (Optional[str]): Path to load a factory from a directory.
        repl (bool): If True, enter interactive mode.
        detach (bool): If True, detach from the run after submission.
        skip_confirmation (bool): If True, skip user confirmation before execution.
        tail_logs (bool): If True, tail logs after execution.
        executor (Optional[Executor]): The executor to use for the run.
        plugins (List[Plugin]): List of plugins to use for the run.
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
            run_name: str = entrypoint_options["run_name"](),
            direct: bool = entrypoint_options["direct"](),
            dryrun: bool = entrypoint_options["dryrun"](),
            factory: Optional[str] = entrypoint_options["factory"](),
            load: Optional[str] = entrypoint_options["load"](),
            yaml: Optional[str] = entrypoint_options["yaml"](),
            repl: bool = entrypoint_options["repl"](),
            detach: bool = entrypoint_options["detach"](),
            skip_confirmation: bool = entrypoint_options["skip_confirmation"](),
            tail_logs: bool = entrypoint_options["tail_logs"](),
            verbose: bool = entrypoint_options["verbose"](),
            rich_exceptions: bool = entrypoint_options["rich_exceptions"](),
            rich_traceback: bool = entrypoint_options["rich_traceback"](),
            rich_locals: bool = entrypoint_options["rich_locals"](),
            rich_theme: Optional[str] = entrypoint_options["rich_theme"](),
            to_yaml: Optional[str] = entrypoint_options["to_yaml"](),
            to_toml: Optional[str] = entrypoint_options["to_toml"](),
            to_json: Optional[str] = entrypoint_options["to_json"](),
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
                skip_confirmation=skip_confirmation
                or _cmd_defaults.get("skip_confirmation", False),
                tail_logs=tail_logs or _cmd_defaults.get("tail_logs", False),
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

            if isinstance(fn, LazyEntrypoint):
                self.execute_lazy(fn, sys.argv, name)
                return

            try:
                if not is_main:
                    load_entrypoints()
                load_workspace()
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

    def cli_execute(
        self, fn: Callable, args: List[str], entrypoint_type: Literal["task", "experiment"] = "task"
    ):
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

        _, run_args, args = _parse_prefixed_args(args, "run")
        self.parse_args(run_args, lazy=True)

        cmd, cmd_args, i_self = "", [], 0
        for i, arg in enumerate(sys.argv):
            if arg == name:
                i_self = i
            if i_self == 0:
                cmd += f" {arg}"

            elif "=" not in arg and not arg.startswith("--"):
                cmd += f" {arg}"
            elif "=" in arg and not arg.startswith("--"):
                cmd_args.append(arg)

        to_run = LazyEntrypoint(cmd, factory=self.factory)
        to_run._add_overwrite(*cmd_args)

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

        partial = self.parse_fn(fn, experiment_args, ctx=self)

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
        output = LazyEntrypoint(fn, factory=self.factory, yaml=self.yaml)
        if args:
            output._add_overwrite(*args)

        return output.resolve()

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
                raise ValueError(
                    "The function cannot have an argument named `executor` as it is a reserved keyword."
                )
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
                load_entrypoints()
                load_workspace()
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