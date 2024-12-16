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
import pprint
import shutil
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Type, Union

import fiddle as fdl
import networkx as nx
from fiddle._src import daglish, diffing
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TimeElapsedColumn
from rich.progress import Task as RichTask
from rich.syntax import Syntax
from torchx.specs.api import AppState

import nemo_run as run
from nemo_run.config import (
    NEMORUN_HOME,
    Config,
    ConfigurableMixin,
    Partial,
    Script,
    get_type_namespace,
)
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.frontend.console.api import CONSOLE, configure_logging, deconfigure_logging
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.core.tunnel.client import SSHTunnel, Tunnel
from nemo_run.core.tunnel.rsync import rsync
from nemo_run.run.job import Job, JobGroup
from nemo_run.run.plugin import ExperimentPlugin
from nemo_run.run.torchx_backend.runner import get_runner
from nemo_run.run.utils import TeeStdoutStderr

_current_experiment: contextvars.ContextVar["Experiment"] = contextvars.ContextVar(
    "nemo_current_experiment"
)


class Experiment(ConfigurableMixin):
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
nemo experiment status {exp_id}
nemo experiment logs {exp_id} 0
nemo experiment cancel {exp_id} 0
"""
    _PARALLEL_SUPPORTED_EXECUTORS = (SlurmExecutor, LocalExecutor, SkypilotExecutor, DockerExecutor)
    _DETACH_SUPPORTED_EXECUTORS = (SlurmExecutor, SkypilotExecutor)
    _DEPENDENCY_SUPPORTED_EXECUTORS = (SlurmExecutor,)
    _RUNNER_DEPENDENT_EXECUTORS = (LocalExecutor,)
    _CONFIG_FILE = "_CONFIG"
    _VERSION_FILE = "_VERSION"
    _TASK_FILE = "_TASKS"
    _DONE_FILE = "_DONE"
    _TUNNELS_FILE = "_TUNNELS"
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

        if not config:
            raise ValueError(f"Experiment {id} not found.")

        serializer = ZlibJSONSerializer()
        cfg: Config["Experiment"] = fdl.cast(Config, serializer.deserialize(config))
        if "id" not in cfg.__arguments__:
            cfg.id = id

        cfg._reconstruct = True

        exp: "Experiment" = fdl.build(cfg)
        exp._jobs = exp._load_jobs()
        try:
            exp.tunnels = exp._load_tunnels()
        except Exception as e:
            exp.console.log(
                f"Exception {e} loading tunnels for experiment {id}, will continue without loading tunnels."
            )

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
        jobs: list[Job | JobGroup] | None = None,
        base_dir: str | None = None,
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

        base_dir = base_dir or NEMORUN_HOME
        self._exp_dir = os.path.join(base_dir, "experiments", title, self._id)

        self.log_level = log_level
        self._runner = get_runner()

        if not _reconstruct:
            os.makedirs(self._exp_dir, exist_ok=False)

            self.executor = executor if executor else LocalExecutor()
            self._save_config()
        else:
            assert isinstance(executor, Executor)
            self.executor = executor

        self._jobs: list[Job | JobGroup] = jobs or []
        self.tunnels: dict[str, Tunnel] = {}
        self.console = CONSOLE
        self._launched = False
        self._live_progress = None
        self._current_experiment_token = None

    def to_config(self) -> Config:
        return Config(
            self.__class__,
            title=self._title,
            id=self._id,
            executor=self.executor.to_config(),
            log_level=self.log_level,
        )

    def _save_config(self):
        with open(os.path.join(self._exp_dir, self.__class__._CONFIG_FILE), "w+") as f:
            f.write(ZlibJSONSerializer().serialize(self.to_config()))

        with open(os.path.join(self._exp_dir, self.__class__._VERSION_FILE), "w+") as f:
            f.write(f"{run.__version__}\n")

    def _save_tunnels(self):
        serializer = ZlibJSONSerializer()
        serialized_tunnels = {
            k: serializer.serialize(v.to_config()) for k, v in self.tunnels.items()
        }
        with open(os.path.join(self._exp_dir, self.__class__._TUNNELS_FILE), "w+") as f:
            json.dump(serialized_tunnels, f)

    def _load_tunnels(self) -> dict[str, Tunnel]:
        with open(os.path.join(self._exp_dir, self.__class__._TUNNELS_FILE)) as f:
            serialized_tunnels = json.load(f)
        serializer = ZlibJSONSerializer()
        return {k: fdl.build(serializer.deserialize(v)) for k, v in serialized_tunnels.items()}

    def _save_jobs(self):
        serialized_jobs = list(map(lambda job: job.serialize(), self.jobs))
        with open(os.path.join(self._exp_dir, self.__class__._TASK_FILE), "w+") as f:
            json.dump(serialized_jobs, f)

        if "__main__" in sys.modules:
            main_module = sys.modules["__main__"]
            try:
                with open(os.path.join(self._exp_dir, "__main__.py"), "w+") as f:
                    f.write(inspect.getsource(main_module))
            except TypeError:
                ...

    def _load_jobs(self) -> list[Job | JobGroup]:
        with open(os.path.join(self._exp_dir, self._TASK_FILE)) as f:
            serialized_jobs = json.load(f)

        serializer = ZlibJSONSerializer()
        jobs = []
        for job_cfg, task_cfg in serialized_jobs:
            job_cfg = serializer.deserialize(job_cfg)

            job: Job | JobGroup = fdl.build(job_cfg)
            if isinstance(job, Job):
                job.task = task_cfg  # type: ignore
            elif isinstance(job, JobGroup):
                job.tasks = task_cfg  # type: ignore
            else:
                raise ValueError(f"Unknown task type: {task_cfg.__fn_or_cls__}")

            jobs.append(job)

        return jobs

    def _add_single_job(
        self,
        task: Union[Partial, Script],
        executor: Executor,
        name: str = "",
        plugins: Optional[list[ExperimentPlugin]] = None,
        tail_logs: bool = False,
        dependencies: Optional[list[str]] = None,
    ) -> str:
        if isinstance(task, Script):
            default_name = task.get_name()
        else:
            default_name = get_type_namespace(task.__fn_or_cls__)

        reuse_job_dir = True if name else False
        name = name or default_name
        if any(map(lambda job: job.id == name, self.jobs)):
            task_id = f"{name}_{len(self.jobs)}"
        else:
            task_id = name

        self._validate_task(task_info=task_id, task=task)

        executor = executor.clone()
        executor.assign(
            self._id,
            self._exp_dir,
            task_id=task_id,
            task_dir=name if reuse_job_dir else task_id,
        )

        cloned = copy.deepcopy(task) if isinstance(task, Script) else task.clone()
        job = Job(
            id=task_id,
            task=cloned,
            executor=executor,
            plugins=plugins,
            tail_logs=tail_logs,
            dependencies=dependencies or [],
        )
        plugins = plugins or []
        for plugin in plugins:
            plugin.assign(self._id)
            plugin.setup(cloned, executor)

        job.prepare()
        self._jobs.append(job)
        return job.id

    def _add_job_group(
        self,
        tasks: list[Partial | Script],
        executor: list[Executor] | Executor,
        name: str,
        plugins: Optional[list[ExperimentPlugin]] = None,
        tail_logs: bool = False,
        dependencies: Optional[list[str]] = None,
    ) -> str:
        if any(map(lambda task: task.id == name, self.jobs)):
            task_id = f"{name}_{len(self.jobs)}"
        else:
            task_id = name

        for i, _task in enumerate(tasks):
            self._validate_task(task_info=f"Job Group: {task_id}, job index: {i}", task=_task)

        executors = executor if isinstance(executor, list) else [executor]
        cloned_executors = []
        for executor in executors:
            new_executor = executor.clone()
            cloned_executors.append(new_executor)
            new_executor.assign(self._id, self._exp_dir, task_id, task_dir=name)

        cloned_tasks = []
        for task in tasks:
            cloned_task = copy.deepcopy(task) if isinstance(task, Script) else task.clone()
            cloned_tasks.append(cloned_task)

        job_group = JobGroup(
            id=task_id,
            tasks=cloned_tasks,
            executors=cloned_executors,
            plugins=plugins,
            tail_logs=tail_logs,
            dependencies=dependencies or [],
        )
        plugins = plugins or []
        for plugin in plugins:
            for i, task in enumerate(cloned_tasks):
                _executor = job_group.executors if job_group._merge else job_group.executors[i]  # type: ignore
                assert isinstance(_executor, Executor)
                plugin.setup(task, _executor)

        job_group.prepare()
        self._jobs.append(job_group)
        return job_group.id

    def _validate_task(self, task_info: str, task: Union[Partial, Script]) -> None:
        valid = True
        message = ""
        if isinstance(task, Partial):
            serializer = ZlibJSONSerializer()
            serialized = serializer.serialize(task)
            deserialized = serializer.deserialize(serialized)
            diff = diffing.build_diff(deserialized, task)
            diff = {
                daglish.path_str(d.target): (d.new_value if hasattr(d, "new_value") else None)  # type: ignore
                for d in diff.changes
            }
            if deserialized != task:
                valid = False
                message += f"""
Deserialized task does not match original task. The following paths in your task need to be wrapped in `run.Config` or `run.Partial`:

{pprint.PrettyPrinter(indent=4).pformat(diff)}

For more information about `run.Config` and `run.Partial`, please refer to https://github.com/NVIDIA/NeMo-Run/blob/main/docs/source/guides/configuration.md.
"""
        if not valid:
            raise RuntimeError(f"Failed to validate task {task_info}.\n{message}")

    def add(
        self,
        task: Union[Partial, Script] | list[Union[Partial, Script]],
        executor: Executor | list[Executor] | None = None,
        name: str = "",
        plugins: Optional[list[ExperimentPlugin]] = None,
        tail_logs: bool = False,
        dependencies: Optional[list[str]] = None,
    ) -> str:
        """
        Add a configured function along with its executor config to the experiment.
        """
        assert (
            _current_experiment.get(None) == self
        ), "Using Experiment without it's context manager is not permitted."

        job_ids = set([job.id for job in self.jobs])
        for dep in dependencies or []:
            assert dep in job_ids, f"Dependency {dep} not found."

        executor = executor or self.executor
        if not isinstance(task, list):
            assert executor and isinstance(executor, Executor)
            job_id = self._add_single_job(
                task,
                executor,
                name,
                plugins=plugins,
                tail_logs=tail_logs,
                dependencies=dependencies.copy() if dependencies else None,
            )
        else:
            assert name, "name is required for task group."
            job_id = self._add_job_group(
                task,
                executor,
                name,
                plugins=plugins,
                tail_logs=tail_logs,
                dependencies=dependencies.copy() if dependencies else None,
            )

        self._save_jobs()
        return job_id

    def dryrun(self, log: bool = True):
        """
        Logs the raw scripts that will be executed for each task.
        """
        if log:
            self.console.log(f"[bold magenta]Experiment {self._id} dryrun...")

        for job in self.jobs:
            if isinstance(job, Job):
                if log:
                    self.console.log(f"[bold magenta]Task {job.id}\n")
            elif isinstance(job, JobGroup):
                if log:
                    self.console.log(f"[bold magenta]Task Group {job.id}\n")
            job.launch(wait=False, runner=self._runner, dryrun=True, direct=False, log_dryrun=log)

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
                "[bold magenta]Running the experiment with direct=True. "
                "This will launch all jobs sequentially in the same process."
            )
            if not self.jobs:
                self.console.log("[bold red]No jobs to run in this experiment.")
                return

            assert all(
                map(lambda job: isinstance(job, Job), self.jobs)
            ), "Jobs in this experiment contain JobGroup which cannot be run directly for now."

            assert all(
                map(lambda job: not job.dependencies, self.jobs)
            ), "Jobs in this experiment contain dependencies which cannot be run directly for now."

            for job in self.jobs:
                assert isinstance(job, Job)
                with TeeStdoutStderr(
                    os.path.join(job.executor.job_dir, f"log_{job.id}_direct_run.out")
                ):
                    job.launch(wait=True, direct=True, runner=self._runner)
                self._save_jobs()

            self._launched = any(map(lambda job: job.launched, self.jobs))
            self._direct = True
            return

        executors = set()
        for job in self.jobs:
            if isinstance(job, Job):
                executors.add(job.executor.__class__)
            elif isinstance(job, JobGroup):
                if isinstance(job.executors, list):
                    for executor in job.executors:
                        executors.add(executor.__class__)
                else:
                    executors.add(job.executors.__class__)

        if detach and any(map(lambda x: x not in self._DETACH_SUPPORTED_EXECUTORS, executors)):
            self.console.log(
                "[bold red] Cannot detach from this experiment. Please keep it running until completion."
            )
            detach = False

        is_dag = any(map(lambda job: len(job.dependencies) > 0, self.jobs))
        assert not (
            is_dag and sequential
        ), "Jobs in this experiment have dependencies, they cannot be run sequentially. Set sequential=False."

        if sequential:
            for i in range(1, len(self.jobs)):
                self.jobs[i].dependencies.append(self.jobs[i - 1].id)

        self.dryrun(log=False)
        for tunnel in self.tunnels.values():
            if isinstance(tunnel, SSHTunnel):
                tunnel.connect()
                assert tunnel.session, f"SSH tunnel {tunnel.key} failed to connect."
                rsync(tunnel.session, self._exp_dir, os.path.dirname(tunnel.job_dir))

            symlink_cmds = []
            for packaging_job in tunnel.packaging_jobs.values():
                if packaging_job.symlink:
                    symlink_cmds.append(packaging_job.symlink_cmd())

            if symlink_cmds:
                tunnel.run(" && ".join(symlink_cmds))

        self._save_tunnels()

        return self._run_dag(detach=detach, tail_logs=tail_logs, executors=executors)

    def _run_dag(self, detach: bool, tail_logs: bool, executors: set[Executor]):
        job_map = {job.id: job for job in self._jobs}
        graph = nx.DiGraph()
        job_ids = set([job.id for job in self.jobs])
        for job in self.jobs:
            graph.add_node(job.id, job=job)
            for dep in job.dependencies:
                assert dep in job_ids, f"Dependency {dep} not found in job list {job_ids}."
                graph.add_edge(dep, job.id)

        assert nx.is_directed_acyclic_graph(graph), "Jobs have cyclic dependencies."
        order = [sorted(generation) for generation in nx.topological_generations(graph)]
        add_deps = False
        if len(order) > 1:
            if all(map(lambda x: x in self._DEPENDENCY_SUPPORTED_EXECUTORS, executors)):
                wait = False
                add_deps = True
                self.detach = detach
            else:
                wait = True
                if len(self.jobs) > 1:
                    self.console.log(
                        f"[bold cyan]Dependencies not supported for atleast one of {executors}."
                        "All jobs will be run one after the other based on their dependencies, please keep the process alive."
                    )
                if detach:
                    self.console.log(
                        "[bold red] Cannot detach from this experiment. Please keep it running until completion."
                    )
        else:
            # All jobs will be executed in parallel
            assert all(
                map(lambda x: x in self._PARALLEL_SUPPORTED_EXECUTORS, executors)
            ), f"Parallel mode not supported for atleast one of {executors}. Set sequential=True."
            wait = False
            self.detach = detach

        for level in order:
            for _, node in enumerate(level):
                job: Job | JobGroup = job_map[node]
                self.console.log(f"[bold cyan]Launching job {job.id} for experiment {self._title}")
                if tail_logs:
                    job.tail_logs = True

                try:
                    if add_deps:
                        deps = []
                        for dep_id in job.dependencies:
                            dep = job_map[dep_id]
                            handle = dep.handle
                            assert (
                                dep.launched and handle
                            ), f"Dependency {dep.id} for {job.id} not yet launched."
                            deps.append(handle)

                        job.executor.dependencies = deps  # type: ignore
                    job.launch(wait=False, runner=self._runner)

                    self._save_jobs()
                except Exception as e:
                    self.console.log(f"Error running job {job.id}: {e}")
                    raise e

            if wait:
                self._wait_for_jobs(jobs=[job_map[node] for node in level])

        self._launched = any(map(lambda job: job.launched, self.jobs))
        self._waited = wait

    def _wait_for_jobs(self, jobs: list[Job | JobGroup]):
        def set_context(context: contextvars.Context):
            for var, value in context.items():
                var.set(value)

        context = contextvars.copy_context()
        with ThreadPoolExecutor(initializer=set_context, initargs=(context,)) as executor:
            futures: dict[Future, Job | JobGroup] = {}
            for job in jobs:
                if isinstance(job, Job):
                    handle_exists = job.handle
                else:
                    handle_exists = len(job.handles) > 0 and all(job.handles)

                if job.launched and handle_exists:
                    self._initialize_live_progress()
                    self._add_progress(job=job)
                    future = executor.submit(
                        job.wait,
                        runner=self._runner
                        if isinstance(
                            job.executor,
                            self._RUNNER_DEPENDENT_EXECUTORS,
                        )
                        else get_runner(),
                    )
                    futures[future] = job

            for future in as_completed(futures.keys()):
                job = futures[future]
                try:
                    future.result()
                    self._update_progress(job, job.state)
                except Exception as e:
                    self.console.log(f"Exception while waiting for Job {job.id}: {e}")
                    self.console.log(*traceback.format_exception(e))
                    self._update_progress(job, AppState.UNKNOWN)
                finally:
                    job.cleanup()

    def status(self, return_dict: bool = False) -> Optional[dict[str, str]]:
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

        def _get_job_info_and_dict(
            idx: int, job: Job | JobGroup
        ) -> tuple[list[str], dict[str, str]]:
            job_info = []
            job_info.append(f"[bold green]Task {idx}[/bold green]: [bold orange1]{job.id}")
            job_info.append(
                f"- [bold green]Status[/bold green]: {str(job.status(runner=self._runner))}"
            )
            job_info.append(f"- [bold green]Executor[/bold green]: {job.executor.info()}")

            try:
                _, _, path_str = job.handle.partition("://")
                path = path_str.split("/")
                app_id = path[1]
            except Exception:
                app_id = ""

            job_info.append(f"- [bold green]Job id[/bold green]: {app_id}")
            directory_info = [
                "- [bold green]Local Directory[/bold green]: " + job.executor.job_dir,
            ]
            job_dict = {
                "name": job.id,
                "status": job.status(runner=self._runner),
                "executor": job.executor.info(),
                "job_id": app_id,
                "local_dir": job.executor.job_dir,
            }

            if isinstance(job.executor, SlurmExecutor) and isinstance(
                job.executor.tunnel, SSHTunnel
            ):
                directory_info.extend(
                    [
                        "- [bold green]Remote Directory[/bold green]: "
                        + os.path.join(
                            job.executor.tunnel.job_dir,
                            Path(job.executor.job_dir).name,
                        ),
                    ]
                )
                job_dict["remote_dir"] = os.path.join(
                    job.executor.tunnel.job_dir,
                    Path(job.executor.job_dir).name,
                )
            job_info.extend(directory_info)
            return job_info, job_dict

        try:
            result_dict = {}
            job_infos = []
            for i, job in enumerate(self.jobs):
                job_info, job_dict = _get_job_info_and_dict(i, job)
                job_infos.append(Group(*job_info))
                result_dict[job.id] = job_dict

            if return_dict:
                return result_dict

            self.console.print()
            self.console.print(
                f"[bold green]Experiment Status for[/bold green] [bold orange1]{self._id}",
                new_line_start=True,
            )
            for job_info in job_infos:
                self.console.print(job_info, soft_wrap=True, new_line_start=True, highlight=False)
            self.console.print()
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

    def cancel(self, job_id: str):
        """
        Cancels an existing job if still running.
        """
        _set_current_experiment = False
        if not self._current_experiment_token:
            _current_experiment.set(self)
            _set_current_experiment = True

        self.console.log(f"[bold cyan]Cancelling {job_id} if still running")
        try:
            job = next(filter(lambda x: x.id == job_id, self.jobs))
            job.cancel(runner=self._runner)
        except StopIteration:
            self.console.log(f"[bold red]Job {job_id} not found")
        except Exception as e:
            self.console.log(f"[bold red]Failed to cancel {job_id}\nError: {e}\n")
            self.console.log(*traceback.format_exception(e))
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

    def logs(self, job_id: str, regex: str | None = None):
        """
        Prints the logs of the specified job_id, optionally filtered by regex.
        """
        _set_current_experiment = False
        if not self._current_experiment_token:
            _current_experiment.set(self)
            _set_current_experiment = True

        self.console.log(f"[bold cyan]Fetching logs for {job_id}")
        try:
            job = next(filter(lambda x: x.id == job_id, self.jobs))
            if isinstance(job, Job) and job.handle.endswith("direct_run"):
                self.console.log("This job was run with direct=True.")
                self.console.log(
                    f"Logs may be present in task directory at:\n[bold]{job.executor.job_dir}."
                )
                return

            try:
                job.logs(runner=self._runner, regex=regex)
            except Exception as e:
                self.console.log(f"[bold red]Failed to get logs for {job_id}\nError: {e}\n")
                self.console.log(
                    f"Logs may be present in job directory at:\n[bold]{job.executor.job_dir}."
                )
        except StopIteration:
            self.console.log(f"[bold red]Job {job_id} not found")
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

    def reset(self) -> "Experiment":
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

        jobs = self._jobs
        self._jobs = []
        serializer = ZlibJSONSerializer()
        _set_current_experiment = False
        if not self._current_experiment_token:
            _current_experiment.set(self)
            _set_current_experiment = True

        if "__main__.py" in os.listdir(old_exp_dir):
            shutil.copy(os.path.join(old_exp_dir, "__main__.py"), self._exp_dir)

        try:
            if "__external_main__" not in sys.modules:
                maybe_load_external_main(self._exp_dir)

            for job in jobs:
                if isinstance(job, Job):
                    if isinstance(job.task, str):
                        _task = serializer.deserialize(job.task)
                        if _task.__fn_or_cls__ == Script:
                            job.task = fdl.build(_task)
                        else:
                            job.task = _task  # type: ignore

                    self.add(
                        job.task,
                        job.executor,
                        name=job.id,
                        tail_logs=job.tail_logs,
                    )
                else:
                    if isinstance(job.tasks, str):
                        tasks = serializer.deserialize(job.tasks)
                        job.tasks = [
                            fdl.build(task) if task.__fn_or_cls__ == Script else task
                            for task in tasks
                        ]

                    self.add(
                        job.tasks,
                        job.executors,
                        name=job.id,
                        tail_logs=job.tail_logs,
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
            self._jobs = self._load_jobs()
        finally:
            if _set_current_experiment and self._current_experiment_token:
                _current_experiment.reset(self._current_experiment_token)
                self._current_experiment_token = None

        self._reconstruct = False
        self._save_config()

        return self

    def _initialize_live_progress(self):
        if not self._live_progress:
            # Disable live progress if we are tailing logs for any task
            # as tty output consistency can not be guaranteed as of now
            if any(map(lambda job: job.tail_logs, self.jobs)):
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

    def _add_progress(self, job: Job | JobGroup):
        if self._live_progress:
            self._task_progress[job.id] = self._progress.add_task(
                f"[bold green]{job.id}", total=None
            )

    def _update_progress(self, job: Job | JobGroup, state: AppState):
        if self._live_progress:
            color = "[bold green]" if state == AppState.SUCCEEDED else "[bold red]"
            task_progress_id = self._task_progress[job.id]
            self._progress.stop_task(task_progress_id)
            self._progress.update(
                task_progress_id,
                description=f"{color}{job.id} {state}",
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

                if hasattr(self, "_waited") and self._waited:
                    self.console.rule(
                        f"[bold magenta]Done waiting for Experiment {self._id}",
                    )
                    self.status()
                    return

                self.console.rule(
                    f"[bold magenta]Waiting for Experiment {self._id} to finish",
                )
                self.status()

                self._wait_for_jobs(jobs=self.jobs)
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
                            tasks=list(map(lambda job: job.id, self.jobs)),
                        ),
                        "python",
                        theme=os.environ.get("NEMO_RUN_CODE_THEME", "monokai"),
                    )
                )
                self.console.print(
                    Syntax(
                        self.GOODBYE_MESSAGE_BASH.format(
                            exp_id=self._id,
                            tasks=list(map(lambda job: job.id, self.jobs)),
                        ),
                        "shell",
                        theme=os.environ.get("NEMO_RUN_CODE_THEME", "monokai"),
                    )
                )

    def _repr_svg_(self):
        return self.to_config()._repr_svg_()

    def __del__(self):
        try:
            deconfigure_logging()
            self._cleanup()
        except Exception:
            pass

    @property
    def jobs(self) -> list[Job | JobGroup]:
        return Jobs(self._jobs)

    @jobs.setter
    def jobs(self, jobs: list[Job | JobGroup]):
        self._jobs = jobs

    @property
    def tasks(self) -> list[Config]:
        serializer = ZlibJSONSerializer()

        for job in self._jobs:
            if isinstance(job, Job):
                if isinstance(job.task, str):
                    _task = serializer.deserialize(job.task)
                    if _task.__fn_or_cls__ == Script:
                        job.task = fdl.build(_task)
                    else:
                        job.task = _task  # type: ignore
            else:
                if isinstance(job.tasks, str):
                    tasks = serializer.deserialize(job.tasks)
                    job.tasks = [
                        fdl.build(task) if task.__fn_or_cls__ == Script else task for task in tasks
                    ]

        return Tasks((job.task if isinstance(job, Job) else job.tasks) for job in self._jobs)


class Tasks(list, ConfigurableMixin): ...


class Jobs(list, ConfigurableMixin): ...


def _get_latest_dir(path) -> str:
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    latest_dir = max(dirs, key=lambda d: os.path.getctime(os.path.join(path, d)))
    return os.path.join(path, latest_dir)


def _get_sorted_dirs(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirs = sorted(dirs, key=lambda d: os.path.getctime(os.path.join(path, d)))
    return list(dirs)


_LOADED_MAINS = set()


def maybe_load_external_main(exp_dir: str):
    main_file = Path(exp_dir) / "__main__.py"
    if main_file.exists() and main_file not in _LOADED_MAINS:
        _LOADED_MAINS.add(main_file)

        spec = importlib.util.spec_from_file_location("__external_main__", main_file)
        new_main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_main_module)

        if "__external_main__" not in sys.modules:
            sys.modules["__external_main__"] = new_main_module
        else:
            external = sys.modules["__external_main__"]
            for attr in dir(new_main_module):
                if not attr.startswith("__"):
                    setattr(external, attr, getattr(new_main_module, attr))

        existing_main = sys.modules["__main__"]
        for attr in dir(new_main_module):
            if not attr.startswith("__"):
                setattr(existing_main, attr, getattr(new_main_module, attr))
