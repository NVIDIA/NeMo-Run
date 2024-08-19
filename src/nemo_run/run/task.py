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
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union, cast

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
from invoke.context import Context
from rich.pretty import Pretty
from rich.table import Table
from torchx.specs.api import AppDef, AppState, is_terminal

import nemo_run.exceptions
from nemo_run.config import Config, Partial, Script
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.frontend.console.api import CONSOLE, CustomConfigRepr
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.logs import get_logs
from nemo_run.run.plugin import ExperimentPlugin
from nemo_run.run.torchx_backend.launcher import launch, wait_and_exit
from nemo_run.run.torchx_backend.packaging import merge_executables, package
from nemo_run.run.torchx_backend.runner import Runner
from nemo_run.run.torchx_backend.schedulers.api import get_executor_str


def dryrun_fn(
    configured_fn_or_script: Union[Partial, Script],
    executor: Optional[Executor] = None,
    build: bool = False,
) -> None:
    if not isinstance(configured_fn_or_script, (Config, Partial)):
        raise TypeError(f"Need a run Partial for dryrun. Got {configured_fn_or_script}.")

    fn = configured_fn_or_script.__fn_or_cls__
    # TODO: Move this to run/frontend
    console = CONSOLE
    console.print(f"[bold cyan]Dry run for task {fn.__module__}:{fn.__name__}[/bold cyan]")

    table_resolved_args = Table(show_header=True, header_style="bold magenta")
    table_resolved_args.add_column("Argument Name", style="dim", width=20)
    table_resolved_args.add_column("Resolved Value", width=60)

    for arg_name in dir(configured_fn_or_script):
        repr = CustomConfigRepr(getattr(configured_fn_or_script, arg_name))
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
        fdl.build(configured_fn_or_script)


def direct_run_fn(fn_or_script: Partial | Script, dryrun: bool = False):
    if not isinstance(fn_or_script, (Partial, Script)):
        raise TypeError(f"Need a configured run.Partial or run.Script. Got {fn_or_script}.")

    if dryrun:
        dryrun_fn(fn_or_script, build=True)
        return

    if isinstance(fn_or_script, Script):
        ctx = Context()
        ctx.run(" ".join(fn_or_script.to_command(with_entrypoint=True)))
        return

    built_fn = fdl.build(fn_or_script)
    built_fn()


class _TaskMixin:
    def to_config(self) -> Config:
        return fdl.cast(Config, fdl_dc.convert_dataclasses_to_configs(self, allow_post_init=True))

    def _repr_svg_(self):
        return self.to_config()._repr_svg_()


@dataclass
class ExperimentTask(_TaskMixin):
    id: str
    fn_or_script: Union[Partial, Script]
    executor: Executor
    handle: str = ""
    launched: bool = False
    state: AppState = AppState.UNSUBMITTED
    plugins: Optional[list[ExperimentPlugin]] = None
    tail_logs: bool = False

    def serialize(self) -> tuple[str, str]:
        cfg = self.to_config()
        fn_or_script_cfg = cfg.fn_or_script
        cfg.fn_or_script = None
        serializer = ZlibJSONSerializer()
        return serializer.serialize(cfg), serializer.serialize(fn_or_script_cfg)

    def status(self, runner: Runner) -> AppState:
        if not self.launched or not self.handle:
            return self.state

        state = None
        try:
            status = runner.status(self.handle)
            state = status.state if status else None
        except Exception:
            ...
        finally:
            return state or self.state

    def logs(self, runner: Runner, regex: str | None = None):
        get_logs(
            sys.stderr,
            identifier=self.handle,
            should_tail=False,
            runner=runner,
            regex=regex,
        )

    def launch(
        self,
        wait: bool,
        runner: Runner,
        dryrun: bool = False,
        direct: bool = False,
    ):
        if not isinstance(self.fn_or_script, (Partial, Config, Script)):
            raise TypeError(f"Need a configured Buildable or run.Script. Got {self.fn_or_script}.")

        executable = package(
            self.id, self.fn_or_script, executor=self.executor, serialize_to_file=True
        )

        executor_str = get_executor_str(self.executor)

        if direct:
            direct_run_fn(self.fn_or_script, dryrun=dryrun)
            self.launched = True
            self.handle = f"{executor_str}://nemo_run/{self.id}_direct_run"
            self.state = AppState.SUCCEEDED
            return

        if dryrun:
            launch(
                executable=executable,
                executor_name=executor_str,
                executor=self.executor,
                dryrun=True,
                wait=wait,
                log=self.tail_logs,
                runner=runner,
            )
            return

        self.handle, status = launch(
            executable=executable,
            executor_name=executor_str,
            executor=self.executor,
            dryrun=False,
            wait=wait,
            log=self.tail_logs,
            runner=runner,
        )
        self.state = status.state if status else AppState.UNKNOWN
        self.launched = True

    def wait(self, runner: Runner | None = None):
        try:
            status = wait_and_exit(
                app_handle=self.handle,
                log=self.tail_logs,
                runner=runner,
            )
            self.state = status.state
        except nemo_run.exceptions.UnknownStatusError:
            self.state = AppState.UNKNOWN

    def cancel(self, runner: Runner):
        if not self.handle:
            return

        runner.cancel(self.handle)

    def cleanup(self):
        if not self.handle or not is_terminal(self.state):
            return

        try:
            self.executor.cleanup(self.handle)
        except Exception as e:
            CONSOLE.log(f"Exception while cleaning up for Task {self.id}: {e}")
            CONSOLE.log(*traceback.format_exception(e))


@dataclass
class ExperimentTaskGroup(_TaskMixin):
    SUPPORTED_EXECUTORS = [SlurmExecutor]

    id: str
    fn_or_scripts: list[Union[Partial, Script]]
    executors: Union[Executor, list[Executor]]
    handles: list[str] = field(default_factory=list)
    launched: bool = False
    states: list[AppState] = field(default_factory=list)
    plugins: Optional[list[ExperimentPlugin]] = None
    tail_logs: bool = False

    def __post_init__(self):
        executors = [self.executors] if isinstance(self.executors, Executor) else self.executors
        assert len(executors) in [
            1,
            len(self.fn_or_scripts),
        ], f"Invalid number of executors. Got {len(executors)} for {len(self.fn_or_scripts)} tasks."
        executor_types = set()
        for exec in executors:
            executor_types.add(exec.__class__)

        assert len(executor_types) == 1, "All executors must be of the same type."
        executor_type = list(executor_types)[0]
        assert executor_type in self.SUPPORTED_EXECUTORS, "Unsupported executor type."
        if executor_type == SlurmExecutor:
            self._merge = True
            self.executors = SlurmExecutor.merge(
                cast(list[SlurmExecutor], executors), num_tasks=len(self.fn_or_scripts)
            )
        else:
            self._merge = False
            if len(executors) == 1:
                self.executors = executors * len(self.fn_or_scripts)

    @property
    def state(self) -> AppState:
        if not self.launched or not self.handles:
            return AppState.UNSUBMITTED

        return self.states[0]

    @property
    def handle(self) -> str:
        if not self.handles:
            return ""

        return self.handles[0]

    def serialize(self) -> tuple[str, str]:
        cfg = self.to_config()
        fn_or_scripts_cfg = cfg.fn_or_scripts
        cfg.fn_or_scripts = None
        serializer = ZlibJSONSerializer()
        return serializer.serialize(cfg), serializer.serialize(fn_or_scripts_cfg)

    def status(self, runner: Runner) -> AppState:
        if not self.launched or not self.handles:
            return self.state

        new_states = []
        for handle in self.handles:
            state = None
            try:
                status = runner.status(handle)
                state = status.state if status else None
            except Exception:
                ...
            finally:
                if not state:
                    state = AppState.UNKNOWN
                new_states.append(state)

        self.states = new_states
        return self.state

    def logs(self, runner: Runner, regex: str | None = None):
        assert len(self.handles) == 1, "Only one handle is supported for task groups currently."
        get_logs(
            sys.stderr,
            identifier=self.handles[0],
            should_tail=False,
            runner=runner,
            regex=regex,
        )

    def launch(
        self,
        wait: bool,
        runner: Runner,
        dryrun: bool = False,
    ):
        for fn_or_script in self.fn_or_scripts:
            if not isinstance(fn_or_script, (Partial, Config, Script)):
                raise TypeError(f"Need a configured Buildable or run.Script. Got {fn_or_script}.")

        executables: list[tuple[AppDef, Executor]] = []
        for i, fn_or_script in enumerate(self.fn_or_scripts):
            executor = self.executors if self._merge else self.executors[i]  # type: ignore
            assert isinstance(executor, Executor)
            executable = package(
                f"{self.id}-{i}",
                fn_or_script,
                executor=executor,
                serialize_to_file=False,
            )
            executables.append((executable, executor))

        if self._merge:
            executable = merge_executables(map(lambda x: x[0], executables), self.id)
            executables = [(executable, executables[0][1])]

        for executable, executor in executables:
            executor_str = get_executor_str(executor)

            if dryrun:
                launch(
                    executable=executable,
                    executor_name=executor_str,
                    executor=executor,
                    dryrun=True,
                    wait=wait,
                    log=self.tail_logs,
                    runner=runner,
                )
            else:
                handle, status = launch(
                    executable=executable,
                    executor_name=executor_str,
                    executor=executor,
                    dryrun=False,
                    wait=wait,
                    log=self.tail_logs,
                    runner=runner,
                )
                self.handles.append(handle)
                self.states.append(status.state if status else AppState.UNKNOWN)
                self.launched = True

    def wait(self, runner: Runner | None = None):
        assert len(self.handles) == 1, "Only one handle is supported for task groups currently."
        try:
            status = wait_and_exit(
                app_handle=self.handles[0],
                log=self.tail_logs,
                runner=runner,
            )
            self.states = [status.state]
        except nemo_run.exceptions.UnknownStatusError:
            self.states = [AppState.UNKNOWN]

    def cancel(self, runner: Runner):
        if not self.handles:
            return

        for handle in self.handles:
            runner.cancel(handle)

    def cleanup(self):
        if not self.handles or not is_terminal(self.state):
            return

        executors: list[Executor] = []
        for i in range(len(self.fn_or_scripts)):
            executor = self.executors if self._merge else self.executors[i]  # type: ignore
            assert isinstance(executor, Executor)
            executors.append(executor)

        for i, handle in enumerate(self.handles):
            try:
                executor = executors[i]
                executor.cleanup(handle)
            except Exception as e:
                CONSOLE.log(f"Exception while cleaning up for Task {self.id}: {e}")
                CONSOLE.log(*traceback.format_exception(e))
