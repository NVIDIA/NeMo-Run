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

from typing import Optional, Union

import fiddle as fdl
from invoke.context import Context
from rich.pretty import Pretty
from rich.table import Table

from nemo_run.config import Config, Partial, Script
from nemo_run.core.execution.base import Executor
from nemo_run.core.frontend.console.api import CONSOLE, CustomConfigRepr


def dryrun_fn(
    configured_task: Union[Partial, Script],
    executor: Optional[Executor] = None,
    build: bool = False,
) -> None:
    if not isinstance(configured_task, (Config, Partial)):
        raise TypeError(f"Need a run Partial for dryrun. Got {configured_task}.")

    fn = configured_task.__fn_or_cls__
    # TODO: Move this to run/frontend
    console = CONSOLE
    console.print(f"[bold cyan]Dry run for task {fn.__module__}:{fn.__name__}[/bold cyan]")

    table_resolved_args = Table(show_header=True, header_style="bold magenta")
    table_resolved_args.add_column("Argument Name", style="dim", width=20)
    table_resolved_args.add_column("Resolved Value", width=60)

    for arg_name in dir(configured_task):
        repr = CustomConfigRepr(getattr(configured_task, arg_name))
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
        fdl.build(configured_task)


def direct_run_fn(task: Partial | Script, dryrun: bool = False):
    if hasattr(task, "__is_lazy__"):
        task = task.resolve()

    if not isinstance(task, (Partial, Script)):
        raise TypeError(f"Need a configured run.Partial or run.Script. Got {task}.")

    if dryrun:
        dryrun_fn(task, build=True)
        return

    if isinstance(task, Script):
        ctx = Context()
        ctx.run(" ".join(task.to_command(with_entrypoint=True)))
        return

    built_fn = fdl.build(task)
    built_fn()
