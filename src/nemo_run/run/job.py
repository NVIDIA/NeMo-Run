import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union, cast

from torchx.specs.api import AppDef, AppState, is_terminal

import nemo_run.exceptions
from nemo_run.config import Config, ConfigurableMixin, Partial, Script
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.docker import DockerExecutor
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
    """
    A Job represents a single task within an Experiment, combining a task definition with its execution environment.

    This class is primarily used internally by the Experiment class and is not typically instantiated directly by users.
    It encapsulates all the information needed to run a single task, including the task itself, the executor
    configuration, and metadata about the job's state and execution.

    Attributes:
        id (str): A unique identifier for the job within the experiment.
        task (Union[Partial, Script]): The task to be executed, either as a Partial (configured function) or a Script.
        executor (Executor): The executor configuration for running the task.
        handle (str): A unique identifier for the running job, set when the job is launched.
        launched (bool): Indicates whether the job has been launched.
        state (AppState): The current state of the job (e.g., UNSUBMITTED, RUNNING, SUCCEEDED, FAILED).
        plugins (Optional[list[ExperimentPlugin]]): Any plugins to be applied to this job.
        tail_logs (bool): Whether to tail the logs of this job during execution.

    The Job class is responsible for:
    - Serializing and deserializing job configurations
    - Launching the task on the specified executor
    - Monitoring the job's status
    - Retrieving logs
    - Cancelling the job if necessary
    - Cleaning up resources after job completion

    While users typically interact with jobs through the Experiment interface, understanding the Job class
    can be helpful for advanced usage scenarios or when developing custom plugins or executors.
    """

    id: str
    task: Union[Partial, Script]
    executor: Executor
    handle: str = ""
    launched: bool = False
    state: AppState = AppState.UNSUBMITTED
    plugins: Optional[list[ExperimentPlugin]] = None
    tail_logs: bool = False
    dependencies: list[str] = field(default_factory=list)

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

    def prepare(self):
        self._executable = package(
            self.id, self.task, executor=self.executor, serialize_to_file=True
        )

    def launch(
        self,
        wait: bool,
        runner: Runner,
        dryrun: bool = False,
        log_dryrun: bool = False,
        direct: bool = False,
    ):
        if not isinstance(self.task, (Partial, Config, Script)):
            raise TypeError(f"Need a configured Buildable or run.Script. Got {self.task}.")

        executor_str = get_executor_str(self.executor)
        assert hasattr(self, "_executable") and self._executable

        if direct:
            direct_run_fn(self.task, dryrun=dryrun)
            self.launched = True
            self.handle = f"{executor_str}://nemo_run/{self.id}_direct_run"
            self.state = AppState.SUCCEEDED
            return

        if dryrun:
            launch(
                executable=self._executable,
                executor_name=executor_str,
                executor=self.executor,
                dryrun=True,
                log_dryrun=log_dryrun,
                wait=wait,
                log=self.tail_logs,
                runner=runner,
            )
            return

        self.handle, status = launch(
            executable=self._executable,
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
    """
    A JobGroup represents a collection of related tasks within an Experiment that are managed together.

    This class is primarily used internally by the Experiment class and is not typically instantiated directly by users.
    It allows for the grouping of multiple tasks that share common characteristics or need to be executed
    in a coordinated manner, such as tasks that should run on the same node or share resources.

    Attributes:
        id (str): A unique identifier for the job group within the experiment.
        tasks (list[Union[Partial, Script]]): A list of tasks to be executed in this group.
        executors (Union[Executor, list[Executor]]): The executor(s) for running the tasks. Can be a single
                                                     executor shared by all tasks or a list of executors.
        handles (list[str]): Unique identifiers for the running jobs, set when the jobs are launched.
        launched (bool): Indicates whether the job group has been launched.
        states (list[AppState]): The current states of the jobs in the group.
        plugins (Optional[list[ExperimentPlugin]]): Any plugins to be applied to this job group.
        tail_logs (bool): Whether to tail the logs of the jobs in this group during execution.

    The JobGroup class is responsible for:
    - Managing the execution of multiple related tasks
    - Handling task dependencies within the group
    - Serializing and deserializing job group configurations
    - Launching tasks on specified executors
    - Monitoring the status of all tasks in the group
    - Retrieving logs for the entire group
    - Cancelling all jobs in the group if necessary
    - Cleaning up resources after all jobs in the group have completed

    JobGroups are particularly useful for:
    - Executing multiple tasks that need to share resources or run on the same node
    - Managing sets of tasks with internal dependencies
    - Optimizing resource allocation for related tasks

    While users typically interact with job groups through the Experiment interface, understanding the JobGroup class
    can be helpful for designing complex workflows or when developing custom plugins or executors that need to
    handle groups of related tasks.
    """

    SUPPORTED_EXECUTORS = [SlurmExecutor, DockerExecutor]

    id: str
    tasks: list[Union[Partial, Script]]
    executors: Union[Executor, list[Executor]]
    handles: list[str] = field(default_factory=list)
    launched: bool = False
    states: list[AppState] = field(default_factory=list)
    plugins: Optional[list[ExperimentPlugin]] = None
    tail_logs: bool = False
    dependencies: list[str] = field(default_factory=list)

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
        elif executor_type == DockerExecutor:
            self._merge = True
            self.executors = DockerExecutor.merge(
                cast(list[DockerExecutor], executors), num_tasks=len(self.tasks)
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

    @property
    def executor(self) -> Executor:
        return self.executors if isinstance(self.executors, Executor) else self.executors[0]

    def serialize(self) -> tuple[str, str]:
        cfg = self.to_config()
        tasks_cfg = cfg.tasks
        cfg.tasks = [None for _ in range(len(tasks_cfg))]
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

    def prepare(self):
        self._executables: list[tuple[AppDef, Executor]] = []
        for i, task in enumerate(self.tasks):
            executor = self.executors if self._merge else self.executors[i]  # type: ignore
            assert isinstance(executor, Executor)
            executable = package(
                f"{self.id}-{i}",
                task,
                executor=executor,
                serialize_to_file=True,
            )
            self._executables.append((executable, executor))

        if self._merge:
            executable = merge_executables(map(lambda x: x[0], self._executables), self.id)
            self._executables = [(executable, self._executables[0][1])]

    def launch(
        self,
        wait: bool,
        runner: Runner,
        dryrun: bool = False,
        log_dryrun: bool = False,
        direct: bool = False,
    ):
        for task in self.tasks:
            if not isinstance(task, (Partial, Config, Script)):
                raise TypeError(f"Need a configured Buildable or run.Script. Got {task}.")

        if direct:
            raise NotImplementedError("Direct launch is not supported yet for JobGroups.")

        assert hasattr(self, "_executables") and self._executables

        for executable, executor in self._executables:
            executor_str = get_executor_str(executor)

            if dryrun:
                launch(
                    executable=executable,
                    executor_name=executor_str,
                    executor=executor,
                    dryrun=True,
                    log_dryrun=log_dryrun,
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
        for i in range(len(self.tasks)):
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
