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

from __future__ import annotations

import logging
import os
import sys
from typing import Callable, Generic, List, Optional, TypeVar

import catalogue
import fiddle as fdl
import typer
from rich.console import Console
from typer import rich_utils
from typer.core import TyperCommand, TyperGroup
from typer.models import CommandInfo
from typing_extensions import ParamSpec

from nemo_run.api import ROOT_TASK_FACTORY_NAMESPACE, _load_entrypoints, _load_workspace
from nemo_run.config import Config, Partial, set_value
from nemo_run.core.lark_parser import parse_args
from nemo_run.run.task import dryrun_fn

Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")
T = TypeVar("T")


class CLI(Generic[Params, ReturnType]):
    def __init__(
        self,
        fn: Callable[Params, ReturnType],
        cmd: Optional[CommandInfo] = None,
        namespace=None,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.fn = fn
        self.namespace = namespace
        self.cmd = cmd or CommandInfo(**kwargs)
        self.name = name or self.fn.__name__

    def __call__(self, *args: Params.args, **kwargs: Params.kwargs) -> ReturnType:
        return self.fn(*args, **kwargs)

    def cli(self, parent: typer.Typer):
        help = self.name
        colored_help = None
        if help:
            colored_help = f"[CLI] {help}"

        class CLITaskCommand(TaskCommand):
            _task = self

        @parent.command(
            self.name,
            help=colored_help,
            cls=CLITaskCommand,
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        )
        def cmd_cli(
            ctx: typer.Context,
        ):
            _load_entrypoints()
            _load_workspace()

            parsed_kwargs, parsed_overrides = parse_args(ctx.args)
            config = Partial(self.fn, **parsed_kwargs)
            for key, value in parsed_overrides.items():
                set_value(config, key, value)

            fn = fdl.build(config)
            fn.func.__io__ = config
            fn()

    def help(self, console=Console(), with_docs: bool = True):
        import nemo_run as run

        return run.help(self.fn, console=console, with_docs=with_docs)

    @property
    def path(self):
        return ".".join([self.namespace, self.fn.__name__])


class Task(Generic[Params, ReturnType]):
    def __init__(
        self,
        fn: Callable[Params, ReturnType],
        namespace: str,
        env=None,
        name=None,
        help_str=None,
        parse_partial: Optional[Callable[[T, List[str]], Partial[T]]] = None,
    ):
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
        self.parse_partial = parse_partial or _default_parse_partial

    def __call__(self, *args: Params.args, **kwargs: Params.kwargs) -> ReturnType:
        return self.fn(*args, **kwargs)

    def configure(self, **fn_kwargs: dict[str, fdl.Config | str | Callable]):
        self._configured_fn = Config(self.fn, **fn_kwargs)

    def cli(self, parent: typer.Typer):
        help = self.help_str
        colored_help = None
        if help:
            colored_help = f"[Task] {help}"

        class CLITaskCommand(TaskCommand):
            _task = self

        @parent.command(
            self.name,
            help=colored_help,
            cls=CLITaskCommand,
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        )
        def cmd_task(
            ctx: typer.Context,
            executor: str = typer.Option(
                "default",
                "--executor",
                "-e",
                help="Name of the executor to use.",
            ),
            load: str = typer.Option(
                None,
                "--load",
                "-l",
                help="Predefined factory to use for the task.",
            ),
            dryrun: bool = typer.Option(
                False,
                help="Does not actually submit the app, just prints the scheduler request.",
            ),
            # wait: bool = typer.Option(
            #     False,
            #     "--wait",
            #     "-w",
            #     help="Wait for the task to finish before exiting.",
            # ),
            # log: bool = typer.Option(
            #     False,
            #     "--log",
            #     "-l",
            #     help="Stream logs while waiting for app to finish.",
            # ),
            run_name: str = typer.Option(
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
            strict: bool = typer.Option(
                False,
                "--strict",
                "-s",
                help="Throw an exception if unknown arguments are passed in.",
            ),
            # workspace: str = typer.Option(
            #     f"file://{Path.cwd()}",
            #     "--workspace",
            #     "-w",
            #     help="Local workspace to build/patch.",
            # ),
            # parent_run_id: str = typer.Option(
            #     None, help="Optional parent run ID that this run belongs to."
            # ),
        ):
            import nemo_run as run

            _load_entrypoints()
            _load_workspace()

            if strict:
                self.env["NEMO_TASK_STRICT"] = "1"

            console = Console()

            if load:
                parts = self.namespace.split(".")
                task_namespace = (
                    ROOT_TASK_FACTORY_NAMESPACE,
                    *parts,
                    self.fn.__name__,
                    "factory",
                    load,
                )
                task_factory = catalogue._get(task_namespace)
                partial = task_factory()
            else:
                partial: Partial = self.parse_partial(self.fn, ctx.args)

            logging.warning(
                "Resolving executors via the CLI is not supported at the moment, defaulting to run.LocalExecutor"
            )
            executor = run.LocalExecutor()

            dryrun_fn(partial, executor=executor)

            if dryrun:
                return

            run_name = run_name or self.name
            run_env = f"NEMORUN_CLI_{run_name}"
            if run_env in os.environ:
                confirmed = True
            else:
                confirmed = typer.confirm("Continue?")

            if confirmed:
                os.environ[run_env] = "1"

                console.print(f"[bold cyan]Launcing run {run_name}...[/bold cyan]")
                run.run(
                    fn_or_script=partial,
                    name=run_name,
                    dryrun=dryrun,
                    direct=direct,
                    executor=executor,
                )
            else:
                console.print("[bold cyan]Exiting...[/bold cyan]")

    def help(self, console=Console(), with_docs: bool = True):
        import nemo_run as run

        return run.help(self.fn, console=console, with_docs=with_docs, namespace=self.namespace)

    @property
    def path(self):
        return ".".join([self.namespace, self.fn.__name__])


class TaskCommand(TyperCommand):
    _task: Task

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
        self._task.help(console, with_docs=sys.argv[-1] in ("--docs", "-d"))

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


def _default_parse_partial(fn: T, args: List[str]) -> Partial[T]:
    parsed_kwargs, parsed_overrides = parse_args(args)
    config = Partial(fn, **parsed_kwargs)
    for key, value in parsed_overrides.items():
        set_value(config, key, value)

    return config
