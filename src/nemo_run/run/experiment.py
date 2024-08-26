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

import contextvars
import copy
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Type, Union

import fiddle as fdl
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TimeElapsedColumn
from rich.progress import Task as RichTask
from rich.syntax import Syntax
from torchx.specs.api import AppState, is_terminal

import nemo_run as run
from nemo_run.config import NEMORUN_HOME, Config, Partial, Script, get_type_namespace
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.frontend.console.api import CONSOLE, configure_logging, deconfigure_logging
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.core.tunnel.client import SSHTunnel, Tunnel
from nemo_run.run.plugin import ExperimentPlugin
from nemo_run.run.task import ExperimentTask, ExperimentTaskGroup
from nemo_run.run.torchx_backend.runner import get_runner
from nemo_run.run.utils import TeeStdoutStderr

_current_experiment: contextvars.ContextVar["Experiment"] = contextvars.ContextVar(
    "nemo_current_experiment"
)

_SUPPORTED_EXECUTORS = (SlurmExecutor, LocalExecutor, SkypilotExecutor)


class Experiment:
    """
    A context manager to launch and manage multiple runs, all using pure Python.

    run.Experiment provides researchers with
    a simple and flexible way to create and manage their ML experiments.

    Building on the core blocks of nemo_run,
    the Experiment can be used as an umbrella under which a user can
    launch different configured functions on multiple remote clusters.

    The Experiment takes care of storing the run metadata,
    launching it on the specified cluster, and syncing the logs and artifacts.

    Additionally, the Experiment also provides management tools to easily inspect and reproduce past experiments.
    Some of the use-cases that it enables are listed below:

    1. Check the status and logs of a past experiment
    2. Reconstruct a past experiment and relaunch it after some changes
    3. Compare different runs of the same experiment.

    This API allows users to programmatically define their experiments.
    To get a glance of the flexibility provided, here are some use cases
    which can be supported by the Experiment in just a few lines of code.

    1. Launch a benchmarking run on different GPUs at the same time in parallel
    2. Launch a sequential data processing pipeline on a CPU heavy cluster
    3. Launch hyperparameter grid search runs on a single cluster in parallel
    4. Launch hyperparameter search runs distributed across all available clusters

    The design is heavily inspired from `XManager <https://github.com/google-deepmind/xmanager/blob/main/docs/xm_launch_api_principles.md>`_.

    Under the hood, the Experiment metadata is stored in the local filesystem
    inside a user specified directory controlled by NEMORUN_HOME env var.
    We will explore making the metadata more persistent in the future.

    .. note::
        `Experiment.add` and `Experiment.run` methods inside Experiment can currently only be used within its context manager.

    Examples
    --------
    .. code-block:: python

        # An experiment that runs a pre-configured training example
        # on multiple GPU specific clusters (A100 and H100 shown here) in parallel using torchrun
        # Assumes that example_to_run is pre-configured using run.Partial
        with run.Experiment("example-multiple-gpus", executor="h100_cluster") as exp:
            # Set up the run on H100
            # Setting up a single task is identical to setting up a single run outside the experiment
            h100_cluster: run.SlurmExecutor = exp.executor.clone()
            h100_cluster.nodes = 2

            # torchrun manages the processes on a single node
            h100_cluster.ntasks_per_node = 1
            h100_cluster.gpus_per_task = 8

            h100_cluster.packager.subpath = "subpath/to/your/code/repo"
            h100_cluster.launcher = "torchrun"

            exp.add(
                "example_h100",
                fn=example_to_run,
                tail_logs=True,
                executor=h100_cluster,
            )

            # Set up the run on A100
            a100_cluster: run.Config[SlurmExecutor] = h100_cluster.clone()
            a100_cluster.tunnel = run.Config(
                SSHTunnel,
                host=os.environ["A100_HOST"],
                user="your_user_in_cluster",
                identity="path_to_your_ssh_key"
            )

            exp.add(
                "example_a100",
                fn=example_to_run,
                tail_logs=True,
                executor=a100_cluster,
            )

            # Runs all the task in the experiment.
            # By default, all tasks will be run in parallel if all different executors support parallel execution.
            # You can set sequential=True to run the tasks sequentially.
            exp.run()

        # Upon exiting the context manager, the Experiment will automatically wait for all tasks to complete,
        # and optionally tail logs for tasks that have tail_logs=True.
        # A detach mode (if the executors support it) will be available soon.
        # Once all tasks have completed,
        # the Experiment will display a status table and clean up resources like ssh tunnels.

        # You can also manage the experiment at a later point in time
        exp = run.Experiment.from_title("example-multiple-gpus")
        exp.status()
        exp.logs(task_id="example_a100")

    """

    GOODBYE_MESSAGE_PYTHON = """
# The experiment was run with the following tasks: {tasks}
# You can inspect and reconstruct this experiment at a later point in time using:
experiment = run.Experiment.from_id("{exp_id}")
experiment.status() # Gets the overall status
experiment.logs("{tasks[0]}") # Gets the log for the provided task
experiment.cancel("{tasks[0]}") # Cancels the provided task if still running
"""

    GOODBYE_MESSAGE_BASH = """
# You can inspect this experiment at a later point in time using the CLI as well:
nemorun experiment status {exp_id}
nemorun experiment logs {exp_id} 0
nemorun experiment cancel {exp_id} 0
"""
    _PARALLEL_SUPPORTED_EXECUTORS = (SlurmExecutor, LocalExecutor, SkypilotExecutor)
    _DETACH_SUPPORTED_EXECUTORS = (SlurmExecutor, SkypilotExecutor)
    _DEPENDENCY_SUPPORTED_EXECUTORS = (SlurmExecutor,)
    _RUNNER_DEPENDNET_EXECUTORS = (LocalExecutor,)
    _CONFIG_FILE = "_CONFIG"
    _VERSION_FILE = "_VERSION"
    _TASK_FILE = "_TASKS"
    _DONE_FILE = "_DONE"
    _current_experiment_token: Optional[contextvars.Token]

    @classmethod
    def catalog(
        cls: Type["Experiment"],
        title: str = "",
    ) -> list[str]:
        """
        List all experiments inside NEMORUN_HOME, optionally with the provided title.
        """
        parent_dir = os.path.join(NEMORUN_HOME, "experiments", title)
        return _get_sorted_dirs(parent_dir)

    @classmethod
    def _from_config(cls: Type["Experiment"], exp_dir: str) -> "Experiment":
        id = os.path.basename(exp_dir)
        with open(os.path.join(exp_dir, cls._CONFIG_FILE), "r") as f:
            config = f.read()

        serializer = ZlibJSONSerializer()
        cfg: Config["Experiment"] = fdl.cast(Config, serializer.deserialize(config))
        if "id" not in cfg.__arguments__:
            cfg.id = id

        cfg._reconstruct = True

        exp: "Experiment" = fdl.build(cfg)
        exp.tasks = exp._load_tasks()

        return exp

    @classmethod
    def from_id(
        cls: Type["Experiment"],
        id: str,
    ) -> "Experiment":
        """
        Reconstruct an experiment with the specified id.
        """
        title, _, _ = id.rpartition("_")
        parent_dir = os.path.join(NEMORUN_HOME, "experiments", title)
        exp_dir = os.path.join(parent_dir, id)

        assert os.path.isdir(exp_dir), f"Experiment {id} not found."

        exp = cls._from_config(exp_dir)
        return exp

    @classmethod
    def from_title(
        cls: Type["Experiment"],
        title: str,
    ) -> "Experiment":
        """
        Reconstruct an experiment with the specified title.
        """
        parent_dir = os.path.join(NEMORUN_HOME, "experiments", title)
        exp_dir = _get_latest_dir(parent_dir)

        assert os.path.isdir(exp_dir), f"Experiment {id} not found."

        exp = cls._from_config(exp_dir)
        return exp

    def __init__(
        self,
        title: str,
        executor: Executor | None = None,  # type: ignore
        id: str | None = None,
        log_level: str = "INFO",
        _reconstruct: bool = False,
    ) -> None:
        """
        Initializes an experiment run by creating its metadata directory and saving the experiment config.

        Args:
            title: Title or name for the experiment
            executor: Any executor that subclasses run.Executor and is supported by NeMo-Run.
                This will be used as the default executor for tasks if an explicit one is not specified.
                Users can also clone this and make task specific executor changes.
            id (Optional): Unique id for the experiment run.
                If not specified, will be set automatically based on the current timestamp.
            log_level: Set log level for the experiment. Defaults to WARN.
            _reconstruct: Generally, the user does not need to specify this flag.
                This is only set to True when using run.Experiment.from_dir.
        """
        configure_logging(level=log_level)
        self._reconstruct = _reconstruct
        if _reconstruct:
            assert id, "Cannot reconstruct an experiment without id."

        self._title = title
        self._id = id or f"{title}_{int(time.time())}"

        self._exp_dir = os.path.join(NEMORUN_HOME, "experiments", title, self._id)

        self.log_level = log_level
        self._runner = get_runner()

        if not _reconstruct:
            os.makedirs(self._exp_dir, exist_ok=False)

            self.executor = executor if executor else LocalExecutor()
            self._save_config()
        else:
            assert isinstance(executor, Executor)
            self.executor = executor

        self.tasks: list[ExperimentTask | ExperimentTaskGroup] = []
        self.tunnels: dict[str, Tunnel] = {}
        self.console = CONSOLE
        self._launched = False
        self._live_progress = None
        self._current_experiment_token = None

    def _to_config(self) -> Config:
        return Config(
            self.__class__,
            title=self._title,
            id=self._id,
            executor=self.executor.to_config(),
            log_level=self.log_level,
        )

    def _save_config(self):
        with open(os.path.join(self._exp_dir, self.__class__._CONFIG_FILE), "w+") as f:
            f.write(ZlibJSONSerializer().serialize(self._to_config()))

        with open(os.path.join(self._exp_dir, self.__class__._VERSION_FILE), "w+") as f:
            f.write(f"{run.__version__}\n")

    def _save_tasks(self):
        serialized_tasks = list(map(lambda task: task.serialize(), self.tasks))
        with open(os.path.join(self._exp_dir, self.__class__._TASK_FILE), "w+") as f:
            json.dump(serialized_tasks, f)

        if "__main__" in sys.modules:
            main_module = sys.modules["__main__"]
            with open(os.path.join(self._exp_dir, "__main__.py"), "w+") as f:
                f.write(inspect.getsource(main_module))

    def _load_tasks(self) -> list[ExperimentTask | ExperimentTaskGroup]:
        maybe_load_external_main(self._exp_dir)
        with open(os.path.join(self._exp_dir, self._TASK_FILE)) as f:
            serialized_tasks = json.load(f)

        serializer = ZlibJSONSerializer()
        tasks = []
        for task_cfg, fn_or_script_cfg in serialized_tasks:
            task_cfg = serializer.deserialize(task_cfg)

            built_task: ExperimentTask | ExperimentTaskGroup = fdl.build(task_cfg)
            if isinstance(built_task, ExperimentTask):
                built_task.fn_or_script = fn_or_script_cfg  # type: ignore
            elif isinstance(built_task, ExperimentTaskGroup):
                built_task.fn_or_scripts = fn_or_script_cfg  # type: ignore
            else:
                raise ValueError(f"Unknown task type: {task_cfg.__fn_or_cls__}")

            tasks.append(built_task)

        return tasks

    def _add_single_task(
        self,
        fn_or_script: Union[Partial, Script],
        executor: Executor,
        name: str = "",
        plugins: Optional[list[ExperimentPlugin]] = None,
        tail_logs: bool = False,
    ):
        if isinstance(fn_or_script, Script):
            default_name = fn_or_script.get_name()
        else:
            default_name = get_type_namespace(fn_or_script.__fn_or_cls__)

        reuse_job_dir = True if name else False
        name = name or default_name
        if any(map(lambda task: task.id == name, self.tasks)):
            task_id = f"{name}_{len(self.tasks)}"
        else:
            task_id = name

        executor = executor.clone()
        executor.assign(
            self._id,
            self._exp_dir,
            task_id=task_id,
            task_dir=name if reuse_job_dir else task_id,
        )

        fn_or_script = (
            copy.deepcopy(fn_or_script)
            if isinstance(fn_or_script, Script)
            else fn_or_script.clone()
        )
        task = ExperimentTask(
            id=task_id,
            fn_or_script=fn_or_script,
            executor=executor,
            plugins=plugins,
            tail_logs=tail_logs,
        )
        plugins = plugins or []
        for plugin in plugins:
            plugin.assign(self._id)
            plugin.setup(fn_or_script, executor)

        self.tasks.append(task)

    def _add_task_group(
        self,
        fn_or_scripts: list[Partial | Script],
        executor: list[Executor] | Executor,
        name: str,
        plugins: Optional[list[ExperimentPlugin]] = None,
        tail_logs: bool = False,
    ):
        if any(map(lambda task: task.id == name, self.tasks)):
            name = f"{name}_{len(self.tasks)}"
        executors = executor if isinstance(executor, list) else [executor]
        cloned_executors = []
        for executor in executors:
            new_executor = executor.clone()
            cloned_executors.append(new_executor)
            new_executor.assign(self._id, self._exp_dir, name, task_dir=name)

        cloned_fn_or_scripts = []
        for fn_or_script in fn_or_scripts:
            cloned_fn_or_script = (
                copy.deepcopy(fn_or_script)
                if isinstance(fn_or_script, Script)
                else fn_or_script.clone()
            )
            cloned_fn_or_scripts.append(cloned_fn_or_script)

        task_group = ExperimentTaskGroup(
            id=name,
            fn_or_scripts=cloned_fn_or_scripts,
            executors=cloned_executors,
            plugins=plugins,
            tail_logs=tail_logs,
        )
        plugins = plugins or []
        for plugin in plugins:
            for i, _fn_or_script in enumerate(cloned_fn_or_scripts):
                _executor = task_group.executors if task_group._merge else task_group.executors[i]  # type: ignore
                assert isinstance(_executor, Executor)
                plugin.setup(_fn_or_script, _executor)

        self.tasks.append(task_group)

    def add(
        self,
        fn_or_script: Union[Partial, Script] | list[Union[Partial, Script]],
        executor: Executor | list[Executor] | None = None,
        name: str = "",
        plugins: Optional[list[ExperimentPlugin]] = None,
        tail_logs: bool = False,
    ):
        """
        Add a configured function along with its executor config to the experiment.
        """
        assert (
            _current_experiment.get(None) == self
        ), "Using Experiment without it's context manager is not permitted."

        executor = executor or self.executor
        if not isinstance(fn_or_script, list):
            assert executor and isinstance(executor, Executor)
            self._add_single_task(
                fn_or_script, executor, name, plugins=plugins, tail_logs=tail_logs
            )
        else:
            assert name, "name is required for task group."
            self._add_task_group(fn_or_script, executor, name, plugins=plugins, tail_logs=tail_logs)

        self._save_tasks()

    def dryrun(self):
        """
        Logs the raw scripts that will be executed for each task.
        """
        self.console.log(f"[bold magenta]Experiment {self._id} dryrun...")
        for task in self.tasks:
            if isinstance(task, ExperimentTask):
                self.console.log(f"[bold magenta]Task {task.id}\n")
            elif isinstance(task, ExperimentTaskGroup):
                self.console.log(f"[bold magenta]Task Group {task.id}\n")
            task.launch(wait=False, runner=self._runner, dryrun=True)

    def run(
        self,
        sequential: bool = False,
        detach: bool = False,
        tail_logs: bool = False,
        direct: bool = False,
    ):
        """
        Runs all the tasks in the experiment.

        By default, all tasks are run in parallel.

        If sequential=True, all tasks will be run one after the other.
        The order is based on the order in which they were added.

        Parallel mode only works if all exectuors in the experiment support it.
        Currently, all executors support parallel mode.

        In sequential mode, if all executor supports dependencies, then all tasks will be scheduled at once
        by specifying the correct dependencies to each task.
        Otherwise, the experiment.run call will block and each task that is scheduled will be executed sequentially.
        In this particular case, we cannot guarantee the state of the exeperiment if the process exits in the middle.

        Currently, only the slurm executor supports dependencies.

        Args:
            sequential: If True, runs all tasks sequentially in the order they were added. Defaults to False.
            detach: If True, detaches from the process after launching the tasks. Only supported for Slurm and Skypilot. Defaults to False.
            tail_logs: If True, tails logs from all tasks in the experiment. If False, relies on task specific setting. Defaults to False.
            direct: If True, runs all tasks in the experiment sequentially in the same process. Note that if direct=True, then sequential also will be True. Defaults to False.
        """
        assert (
            _current_experiment.get(None) == self
        ), "Using Experiment without it's context manager is not permitted."

        if self._launched:
            self.console.log("[bold magenta]Experiment already running...")
            return

        if self._reconstruct:
            self.console.log("[bold magenta]Experiment in inspection mode...")
            return

        if direct:
            self.console.log(
                "[bold magenta]Running the experiment with direct=True"
                "This will launch all tasks sequentially in the same process."
            )
            assert all(
                map(lambda task: isinstance(task, ExperimentTask), self.tasks)
            ), "Tasks in this experiment contain ExperimentTaskGroup which cannot be run directly for now."
            for task in self.tasks:
                assert isinstance(task, ExperimentTask)
                with TeeStdoutStderr(
                    os.path.join(task.executor.job_dir, f"log_{task.id}_direct_run.out")
                ):
                    task.launch(wait=True, direct=True, runner=self._runner)
                self._save_tasks()

            self._launched = any(map(lambda task: task.launched, self.tasks))
            self._direct = True
            return

        executors = set()
        for task in self.tasks:
            if isinstance(task, ExperimentTask):
                executors.add(task.executor.__class__)
            elif isinstance(task, ExperimentTaskGroup):
                if isinstance(task.executors, list):
                    for executor in task.executors:
                        executors.add(executor.__class__)
                else:
                    executors.add(task.executors.__class__)

        if detach and any(map(lambda x: x not in self._DETACH_SUPPORTED_EXECUTORS, executors)):
            self.console.log(
                "[bold red] Cannot detach from this experiment. Please keep it running until completion."
            )
            detach = False

        add_deps = False
        task_deps: list[list[ExperimentTask | ExperimentTaskGroup]] = [[]]
        if sequential:
            if all(map(lambda x: x in self._DEPENDENCY_SUPPORTED_EXECUTORS, executors)):
                wait = False
                add_deps = True
                if len(self.tasks) > 1:
                    self.console.log(
                        "[bold cyan]Tasks will be scheduled all at once but executed sequentially."
                    )
                    for i in range(1, len(self.tasks)):
                        task_deps.append([self.tasks[i - 1]])

                self.detach = detach
            else:
                wait = True
                if len(self.tasks) > 1:
                    self.console.log(
                        f"[bold cyan]Dependencies not supported for atleast one of {executors}."
                        "Tasks will be run one after the other, please keep the process alive."
                    )
                if detach:
                    self.console.log(
                        "[bold red] Cannot detach from this experiment. Please keep it running until completion."
                    )
        else:
            assert all(
                map(lambda x: x in self._PARALLEL_SUPPORTED_EXECUTORS, executors)
            ), f"Parallel mode not supported for atleast one of {executors}. Set sequential=True."
            wait = False
            self.detach = detach

        for i, task in enumerate(self.tasks):
            self.console.log(f"[bold cyan]Launching task {task.id} for experiment {self._title}")
            if tail_logs:
                task.tail_logs = True

            try:
                if add_deps:
                    deps = []
                    for dep in task_deps[i]:
                        handle = dep.handle if isinstance(dep, ExperimentTask) else dep.handles[0]
                        assert (
                            dep.launched and handle
                        ), f"Dependency {dep.id} for {task.id} not yet launched."
                        deps.append(handle)

                    task.executor.dependencies = deps  # type: ignore
                task.launch(wait=wait, runner=self._runner)
                if wait:
                    self._update_progress(task, task.state)

                self._save_tasks()
            except Exception as e:
                self.console.log(f"Error running task {task.id}: {e}")
                self.console.log(*traceback.format_exception(e))

                self._update_progress(
                    task,
                    AppState.FAILED,
                )

        self._launched = any(map(lambda task: task.launched, self.tasks))

    def status(self):
        """
        Prints a table specifying the status of all tasks.

        .. note::
            status is not supported for local executor
            and the status for a task using the local executor
            will be listed as UNKNOWN in most cases
        """
        _set_current_experiment = False
        if not self._current_experiment_token:
            _current_experiment.set(self)
            _set_current_experiment = True

        try:
            task_infos = []
            for i, task in enumerate(self.tasks):
                task_info = []
                task_info.append(f"[bold green]Task {i}[/bold green]: [bold orange1]{task.id}")
                task_info.append(
                    f"- [bold green]Status[/bold green]: {str(task.status(runner=self._runner))}"
                )
                task_executor = (
                    task.executor
                    if isinstance(task, ExperimentTask)
                    else (
                        task.executors
                        if isinstance(task.executors, Executor)
                        else task.executors[0]
                    )
                )
                task_info.append(f"- [bold green]Executor[/bold green]: {task_executor.info()}")

                try:
                    _, _, path_str = task.handle.partition("://")
                    path = path_str.split("/")
                    app_id = path[1]
                except Exception:
                    app_id = ""

                task_info.append(f"- [bold green]Job id[/bold green]: {app_id}")
                directory_info = [
                    "- [bold green]Local Directory[/bold green]: " + task_executor.job_dir,
                ]
                if isinstance(task_executor, SlurmExecutor) and isinstance(
                    task_executor.tunnel, SSHTunnel
                ):
                    directory_info.extend(
                        [
                            "- [bold green]Remote Directory[/bold green]: "
                            + os.path.join(
                                task_executor.tunnel.job_dir,
                                Path(task_executor.job_dir).name,
                            ),
                        ]
                    )
                task_info.extend(directory_info)
                task_infos.append(Group(*task_info))

            self.console.print()
            self.console.print(
                f"[bold green]Experiment Status for[/bold green] [bold orange1]{self._id}",
                new_line_start=True,
            )
            for task_info in task_infos:
                self.console.print(task_info, soft_wrap=True, new_line_start=True, highlight=False)
            self.console.print()
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

    def cancel(self, task_id: str):
        """
        Cancels an existing task if still running.
        """
        _set_current_experiment = False
        if not self._current_experiment_token:
            _current_experiment.set(self)
            _set_current_experiment = True

        self.console.log(f"[bold cyan]Cancelling {task_id} if still running")
        try:
            task = next(filter(lambda x: x.id == task_id, self.tasks))
            task.cancel(runner=self._runner)
        except StopIteration:
            self.console.log(f"[bold red]Task {task_id} not found")
        except Exception as e:
            self.console.log(f"[bold red]Failed to cancel {task_id}\nError: {e}\n")
            self.console.log(*traceback.format_exception(e))
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

    def logs(self, task_id: str, regex: str | None = None):
        """
        Prints the logs of the specified task_id, optionally filtered by regex.
        """
        _set_current_experiment = False
        if not self._current_experiment_token:
            _current_experiment.set(self)
            _set_current_experiment = True

        self.console.log(f"[bold cyan]Fetching logs for {task_id}")
        try:
            task = next(filter(lambda x: x.id == task_id, self.tasks))
            if isinstance(task, ExperimentTask) and task.handle.endswith("direct_run"):
                self.console.log("This task was run with direct=True.")
                self.console.log(
                    f"Logs may be present in task directory at:\n[bold]{task.executor.job_dir}."
                )
                return

            try:
                task.logs(runner=self._runner, regex=regex)
            except Exception as e:
                self.console.log(f"[bold red]Failed to get logs for {task_id}\nError: {e}\n")
                task_executor = (
                    task.executor
                    if isinstance(task, ExperimentTask)
                    else (
                        task.executors
                        if isinstance(task.executors, Executor)
                        else task.executors[0]
                    )
                )
                self.console.log(
                    f"Logs may be present in task directory at:\n[bold]{task_executor.job_dir}."
                )
        except StopIteration:
            self.console.log(f"[bold red]Task {task_id} not found")
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

    def reset(self):
        """
        Resets an experiment to make it ready for a relaunch.
        Only works if the current experiment run has already been launched.
        """
        if not self._reconstruct and not os.path.isfile(
            os.path.join(self._exp_dir, self._DONE_FILE)
        ):
            self.console.log(
                f"[bold magenta]Experiment {self._id} has not run yet, skipping reset..."
            )
            return

        old_id, old_exp_dir, old_launched = self._id, self._exp_dir, self._launched
        self._id = f"{self._title}_{int(time.time())}"
        self._exp_dir = os.path.join(NEMORUN_HOME, "experiments", self._title, self._id)
        os.makedirs(self._exp_dir, exist_ok=False)
        self._launched = False
        self._live_progress = None

        tasks = self.tasks
        self.tasks = []
        serializer = ZlibJSONSerializer()
        _set_current_experiment = False
        if not self._current_experiment_token:
            _current_experiment.set(self)
            _set_current_experiment = True

        try:
            for task in tasks:
                if isinstance(task, ExperimentTask):
                    if isinstance(task.fn_or_script, str):
                        fn_or_script = serializer.deserialize(task.fn_or_script)
                        if fn_or_script.__fn_or_cls__ == Script:
                            task.fn_or_script = fdl.build(fn_or_script)
                        else:
                            task.fn_or_script = fn_or_script  # type: ignore

                    self.add(
                        task.fn_or_script,
                        task.executor,
                        name=task.id,
                        tail_logs=task.tail_logs,
                    )
                else:
                    if isinstance(task.fn_or_scripts, str):
                        fn_or_scripts = serializer.deserialize(task.fn_or_scripts)
                        task.fn_or_scripts = [
                            fdl.build(fn_or_script)
                            if fn_or_script.__fn_or_cls__ == Script
                            else fn_or_script
                            for fn_or_script in fn_or_scripts
                        ]

                    self.add(
                        task.fn_or_scripts,
                        task.executors,
                        name=task.id,
                        tail_logs=task.tail_logs,
                    )
        except Exception as e:
            self.console.log(
                f"[bold magenta]Failed resetting Experiment {self._id} due to error: {e}"
            )
            # Double check exp dir is unchanged
            new_path = os.path.join(NEMORUN_HOME, "experiments", self._title, self._id)
            if self._exp_dir == new_path and new_path != old_exp_dir:
                shutil.rmtree(self._exp_dir)

            self._id = old_id
            self._exp_dir = old_exp_dir
            self._launched = old_launched
            self.tasks = self._load_tasks()
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

        self._reconstruct = False
        self._save_config()

    def _initialize_live_progress(self):
        if not self._live_progress:
            # Disable live progress if we are tailing logs for any task
            # as tty output consistency can not be guaranteed as of now
            if any(map(lambda task: task.tail_logs, self.tasks)):
                return

            self._progress = Progress(
                "{task.description}",
                SpinnerColumn(),
                BarColumn(bar_width=None),
                TimeElapsedColumn(),
            )
            self._exp_panel = Panel(
                self._progress,
                title=f"[b]{self._id}",
                padding=(1, 3),
            )
            self._task_progress: dict[str, TaskID] = {}
            self._live_progress = Live(self._exp_panel, console=self.console, refresh_per_second=10)
            self._live_progress.start(refresh=True)

    def _add_progress(self, task: ExperimentTask | ExperimentTaskGroup):
        if self._live_progress:
            self._task_progress[task.id] = self._progress.add_task(
                f"[bold green]{task.id}", total=None
            )

    def _update_progress(self, task: ExperimentTask | ExperimentTaskGroup, state: AppState):
        if self._live_progress:
            color = "[bold green]" if state == AppState.SUCCEEDED else "[bold red]"
            task_progress_id = self._task_progress[task.id]
            self._progress.stop_task(task_progress_id)
            self._progress.update(
                task_progress_id,
                description=f"{color}{task.id} {state}",
            )
            progress_task: RichTask = self._progress._tasks[task_progress_id]
            progress_task.finished_time = progress_task.elapsed
            progress_task.completed = progress_task.elapsed or 0.0
            progress_task.total = progress_task.elapsed

            self._progress.refresh()

    def _cleanup(self, tunnels: bool = True):
        if tunnels and hasattr(self, "tunnels"):
            for tunnel in self.tunnels.values():
                try:
                    tunnel.cleanup()
                except Exception:
                    ...

        self._runner.close()

        if (
            _current_experiment is not None
            and _current_experiment.get(None)
            and self._current_experiment_token
        ):
            _current_experiment.reset(self._current_experiment_token)
            self._current_experiment_token = None

    def __enter__(self) -> "Experiment":
        self._current_experiment_token = _current_experiment.set(self)
        self.console.rule(
            f"[bold magenta]Entering Experiment {self._title} with id: {self._id}",
        )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        try:
            if hasattr(self, "detach") and self.detach:
                self.console.rule(f"[bold magenta]Detaching from Experiment {self._id}.")
                self.console.log(
                    "Task specific cleanup won't be run.\n"
                    "Ephemeral logs and artifacts may be lost.",
                )

                if self._launched:
                    self.status()
                return

            if self._launched:
                if hasattr(self, "_direct") and self._direct:
                    self.console.rule(
                        f"[bold magenta]Direct run Experiment {self._id}",
                    )
                    self.status()
                    return

                self.console.rule(
                    f"[bold magenta]Waiting for Experiment {self._id} to finish",
                )
                self.status()

                def set_context(context: contextvars.Context):
                    for var, value in context.items():
                        var.set(value)

                context = contextvars.copy_context()
                with ThreadPoolExecutor(initializer=set_context, initargs=(context,)) as executor:
                    futures: dict[Future, ExperimentTask | ExperimentTaskGroup] = {}
                    for task in self.tasks:
                        if isinstance(task, ExperimentTask):
                            handle_exists = task.handle
                        else:
                            handle_exists = len(task.handles) > 0 and all(task.handles)

                        if task.launched and handle_exists and not is_terminal(task.state):
                            self._initialize_live_progress()
                            self._add_progress(task=task)
                            task_executor = (
                                task.executor
                                if isinstance(task, ExperimentTask)
                                else (
                                    task.executors[0]
                                    if isinstance(task.executors, list)
                                    else task.executors
                                )
                            )
                            future = executor.submit(
                                task.wait,
                                runner=self._runner
                                if isinstance(
                                    task_executor,
                                    self._RUNNER_DEPENDNET_EXECUTORS,
                                )
                                else get_runner(),
                            )
                            futures[future] = task

                    for future in as_completed(futures.keys()):
                        task = futures[future]
                        try:
                            future.result()
                            self._update_progress(task, task.state)
                        except Exception as e:
                            self.console.log(f"Exception while waiting for Task {task.id}: {e}")
                            self.console.log(*traceback.format_exception(e))
                            self._update_progress(task, AppState.UNKNOWN)
                        finally:
                            task.cleanup()
        finally:
            if self._live_progress:
                self._live_progress.stop()

            self._cleanup(tunnels=False)
            if self._launched:
                Path(os.path.join(self._exp_dir, self._DONE_FILE)).touch()
                self.console.print(
                    Syntax(
                        self.GOODBYE_MESSAGE_PYTHON.format(
                            exp_id=self._id,
                            tasks=list(map(lambda task: task.id, self.tasks)),
                        ),
                        "python",
                    )
                )
                self.console.print(
                    Syntax(
                        self.GOODBYE_MESSAGE_BASH.format(
                            exp_id=self._id,
                            tasks=list(map(lambda task: task.id, self.tasks)),
                        ),
                        "shell",
                    )
                )

    def __del__(self):
        try:
            deconfigure_logging()
            self._cleanup()
        except Exception:
            pass


def _get_latest_dir(path) -> str:
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    latest_dir = max(dirs, key=lambda d: os.path.getctime(os.path.join(path, d)))
    return os.path.join(path, latest_dir)


def _get_sorted_dirs(path: str) -> list[str]:
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirs = sorted(dirs, key=lambda d: os.path.getctime(os.path.join(path, d)))
    return list(dirs)


def maybe_load_external_main(exp_dir: str):
    main_file = Path(exp_dir) / "__main__.py"
    if main_file.exists():
        spec = importlib.util.spec_from_file_location("__external_main__", main_file)
        new_main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_main_module)
        existing_main = sys.modules["__main__"]

        for attr in dir(new_main_module):
            if not attr.startswith("__"):
                setattr(existing_main, attr, getattr(new_main_module, attr))
