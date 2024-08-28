import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union, cast

from torchx.specs.api import AppDef, AppState, is_terminal

import nemo_run.exceptions
from nemo_run.config import Config, ConfigurableMixin, Partial, Script
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.logs import get_logs
from nemo_run.run.plugin import ExperimentPlugin
from nemo_run.run.task import direct_run_fn
from nemo_run.run.torchx_backend.launcher import launch, wait_and_exit
from nemo_run.run.torchx_backend.packaging import merge_executables, package
from nemo_run.run.torchx_backend.runner import Runner
from nemo_run.run.torchx_backend.schedulers.api import get_executor_str


@dataclass
class Job(ConfigurableMixin):
    id: str
    task: Union[Partial, Script]
    executor: Executor
    handle: str = ""
    launched: bool = False
    state: AppState = AppState.UNSUBMITTED
    plugins: Optional[list[ExperimentPlugin]] = None
    tail_logs: bool = False

    def serialize(self) -> tuple[str, str]:
        cfg = self.to_config()
        task_cfg = cfg.task
        cfg.task = None
        serializer = ZlibJSONSerializer()
        return serializer.serialize(cfg), serializer.serialize(task_cfg)

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
        if not isinstance(self.task, (Partial, Config, Script)):
            raise TypeError(f"Need a configured Buildable or run.Script. Got {self.task}.")

        executable = package(self.id, self.task, executor=self.executor, serialize_to_file=True)

        executor_str = get_executor_str(self.executor)

        if direct:
            direct_run_fn(self.task, dryrun=dryrun)
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
class JobGroup(ConfigurableMixin):
    SUPPORTED_EXECUTORS = [SlurmExecutor]

    id: str
    tasks: list[Union[Partial, Script]]
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
            len(self.tasks),
        ], f"Invalid number of executors. Got {len(executors)} for {len(self.tasks)} tasks."
        executor_types = set()
        for exec in executors:
            executor_types.add(exec.__class__)

        assert len(executor_types) == 1, "All executors must be of the same type."
        executor_type = list(executor_types)[0]
        assert executor_type in self.SUPPORTED_EXECUTORS, "Unsupported executor type."
        if executor_type == SlurmExecutor:
            self._merge = True
            self.executors = SlurmExecutor.merge(
                cast(list[SlurmExecutor], executors), num_tasks=len(self.tasks)
            )
        else:
            self._merge = False
            if len(executors) == 1:
                self.executors = executors * len(self.tasks)

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
        tasks_cfg = cfg.tasks
        cfg.tasks = None
        serializer = ZlibJSONSerializer()
        return serializer.serialize(cfg), serializer.serialize(tasks_cfg)

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
        for task in self.tasks:
            if not isinstance(task, (Partial, Config, Script)):
                raise TypeError(f"Need a configured Buildable or run.Script. Got {task}.")

        executables: list[tuple[AppDef, Executor]] = []
        for i, task in enumerate(self.tasks):
            executor = self.executors if self._merge else self.executors[i]  # type: ignore
            assert isinstance(executor, Executor)
            executable = package(
                f"{self.id}-{i}",
                task,
                executor=executor,
                serialize_to_file=True,
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
