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

import copy
import logging
import os
import shlex
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeAlias, Union

import invoke
from rich.console import Console
from rich.text import Text

from nemo_run.core.execution.base import (
    _RUNDIR_NAME,
    Executor,
    ExecutorMacros,
    FaultTolerance,
    Launcher,
    Torchrun,
)
from nemo_run.core.execution.utils import fill_template
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.core.tunnel.callback import Callback
from nemo_run.core.tunnel.client import LocalTunnel, SSHConfigFile, SSHTunnel, Tunnel
from nemo_run.core.tunnel.server import TunnelMetadata, server_dir
from nemo_run.devspace.base import DevSpace

logger = logging.getLogger(__name__)
noquote: TypeAlias = str


class JobPaths:
    """Creates paths related to the slurm job and its submission"""

    def __init__(
        self,
        folder: Union[Path, str],
        job_name: str,
    ) -> None:
        self._folder = Path(folder).expanduser().absolute()
        self._job_name = job_name

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def results_folder(self) -> Path:
        return self._folder / "results"

    @property
    def submission_file(self) -> Path:
        return Path(self.folder / f"{self._job_name}_submission.sh")

    @property
    def config_file(self) -> Path:
        return Path(self.folder / f"{self._job_name}_config.yaml")

    @property
    def stderr(self) -> Path:
        return Path(self.folder / f"sbatch_{self._job_name}_%j.err")

    @property
    def stdout(self) -> Path:
        return Path(self.folder / f"sbatch_{self._job_name}_%j.out")

    @property
    def srun_stderr(self) -> Path:
        return Path(self.folder / f"log-{self._job_name}_%j_${{SLURM_RESTART_COUNT:-0}}.err")

    @property
    def srun_stdout(self) -> Path:
        return Path(self.folder / f"log-{self._job_name}_%j_${{SLURM_RESTART_COUNT:-0}}.out")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.folder})"


@dataclass(kw_only=True)
class SlurmExecutor(Executor):
    """
    Dataclass to configure a Slurm Cluster.
    During execution, sbatch related parameters will automatically get parsed to their corresponding sbatch flags.

    .. note::
        We assume that the underlying Slurm cluster has `Pyxis <https://github.com/NVIDIA/pyxis>`_ enabled.
        The slurm executor will fail if the slurm cluster doesn't support pyxis.

    Example:

    .. code-block:: python

        def your_slurm_executor() -> run.SlurmExecutor:
            ssh_tunnel = SSHTunnel(
                host=os.environ["SLURM_HOST"],
                user=os.environ["SLURM_USER"],
                job_dir=os.environ["SLURM_JOBDIR"],
            )
            packager = GitArchivePackager()
            launcher = "torchrun"
            executor = SlurmExecutor(
                account=os.environ["SLURM_ACCT"],
                partition=os.environ["SLURM_PARTITION"],
                nodes=1,
                ntasks_per_node=1,
                tunnel=ssh_tunnel,
                container_image=os.environ["BASE_IMAGE"],
                time="00:30:00",
                packager=packager,
                launcher=launcher,
            )
            return executor

        ...

        your_executor = your_slurm_executor()

    """

    HEAD_NODE_IP_VAR = "head_node_ip"
    NPROC_PER_NODE_VAR = "SLURM_NTASKS_PER_NODE"
    NUM_NODES_VAR = "SLURM_NNODES"
    NODE_RANK_VAR = "SLURM_NODEID"
    HET_GROUP_HOST_VAR = "het_group_host"

    #: List of sbatch flags in snake case
    SBATCH_FLAGS = [
        "account",
        "acctg_freq",
        "array",
        "batch",
        "clusters",
        "constraint",
        "container",
        "container_id",
        "core_spec",
        "cpus_per_gpu",
        "cpus_per_task",
        "comment",
        "debug",
        "delay_boot",
        "dependency",
        "distribution",
        "error",
        "exclude",
        "exclusive",
        "export",
        "get_user_env",
        "gid",
        "gpu_bind",
        "gpu_freq",
        "gpus",
        "gpus_per_node",
        "gpus_per_socket",
        "gpus_per_task",
        "gres",
        "gres_flags",
        "help",
        "hold",
        "ignore_pbs",
        "input",
        "job_name",
        "kill_on_invalid_dep",
        "licenses",
        "mail_type",
        "mail_user",
        "mcs_label",
        "mem",
        "mem_bind",
        "mem_per_cpu",
        "mem_per_gpu",
        "mincpus",
        "network",
        "nice",
        "no_kill",
        "no_requeue",
        "nodefile",
        "nodelist",
        "nodes",
        "ntasks",
        "ntasks_per_core",
        "ntasks_per_gpu",
        "ntasks_per_node",
        "ntasks_per_socket",
        "open_mode",
        "output",
        "overcommit",
        "oversubscribe",
        "parsable",
        "partition",
        "power",
        "prefer",
        "priority",
        "profile",
        "propagate",
        "qos",
        "quiet",
        "reboot",
        "requeue",
        "reservation",
        "signal",
        "sockets_per_node",
        "spread_job",
        "switches",
        "test_only",
        "thread_spec",
        "threads_per_core",
        "time",
        "time_min",
        "tmp",
        "tres_bind",
        "tres_per_task",
        "uid",
        "usage",
        "verbose",
        "version",
        "wait",
        "wait_all_nodes",
        "wckey",
        "wrap",
    ]

    SRUN_ARGS = [
        "account",
        "partition",
        "job-name",
        "time",
        "nodes",
        "ntasks",
        "ntasks-per-node",
        "cpus-per-task",
        "gpus-per-node",
        "gpus-per-task",
        "qos",
        "mem",
        "mem-per-gpu",
        "mem-per-cpu",
        "comment",
        "constraint",
        "exclude",
        "gres",
        "exclusive",
        "array",
        "additional-parameters",
        "container-image",
        "container-mounts",
        "container-workdir",
    ]

    ALLOC_ARGS = [
        "account",
        "partition",
        "job-name",
        "time",
        "nodes",
        "ntasks-per-node",
        "qos",
        "mem",
        "mem-per-gpu",
        "mem-per-cpu",
    ]

    @dataclass(kw_only=True)
    class ResourceRequest:
        packager: GitArchivePackager
        nodes: int
        ntasks_per_node: int
        container_image: Optional[str] = None
        gpus_per_node: Optional[int] = None
        gpus_per_task: Optional[int] = None
        container_mounts: list[str] = field(default_factory=list)
        env_vars: dict[str, str] = field(default_factory=dict)
        srun_args: Optional[list[str]] = None

    account: str
    partition: Optional[str] = None
    job_name_prefix: Optional[str] = None
    time: str = "00:10:00"
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: Optional[int] = None
    cpus_per_gpu: Optional[int] = None
    gpus_per_node: Optional[int] = None
    gpus_per_task: Optional[int] = None
    qos: Optional[str] = None
    mem: Optional[str] = None
    mem_per_gpu: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    comment: Optional[str] = None
    constraint: Optional[str] = None
    exclude: Optional[str] = None
    gres: Optional[str] = None
    signal: Optional[str] = None
    exclusive: Optional[Union[bool, str]] = None
    array: Optional[str] = None
    open_mode: str = "append"
    container_image: Optional[str] = None
    container_mounts: list[str] = field(default_factory=list)
    additional_parameters: Optional[dict[str, Any]] = None
    srun_args: Optional[list[str]] = None
    heterogeneous: bool = False
    memory_measure: bool = False
    job_paths_cls: Type[JobPaths] = JobPaths
    tunnel: Union[SSHTunnel, LocalTunnel] = field(default_factory=lambda: LocalTunnel(job_dir=""))
    packager: GitArchivePackager = field(default_factory=lambda: GitArchivePackager())  # type: ignore
    #: List of TorchX app handles that will be parsed and passed to --dependency flag in sbatch.
    dependencies: list[str] = field(default_factory=list)
    #: Optional parameter to explicitly specify nproc_per_node for torchrun like components if the slurm cluster doesn't support granular resource allocation.
    torchrun_nproc_per_node: Optional[int] = None
    wait_time_for_group_job: int = 30
    monitor_group_job: bool = True
    monitor_group_job_wait_time: int = 60

    #: Set by the executor; cannot be initialized
    job_name: str = field(init=False, default="nemo-job")
    stderr_to_stdout: bool = field(init=False, default=True)
    resource_group: list[ResourceRequest] = field(init=False, default_factory=list)
    run_as_group: bool = field(init=False, default=False)

    @classmethod
    def merge(
        cls: Type["SlurmExecutor"], executors: list["SlurmExecutor"], num_tasks: int
    ) -> "SlurmExecutor":
        assert len(executors) in [1, num_tasks]
        if len(executors) == 1 and not executors[0].heterogeneous:
            executors[0].run_as_group = True
            return executors[0]

        if len(executors) == 1:
            executors = executors * num_tasks

        main_executor = executors[0]
        main_executor.run_as_group = True
        main_executor.resource_group = [
            cls.ResourceRequest(
                packager=copy.deepcopy(main_executor.packager),
                nodes=main_executor.nodes,
                ntasks_per_node=main_executor.ntasks_per_node,
                container_image=copy.deepcopy(main_executor.container_image),
                container_mounts=copy.deepcopy(main_executor.container_mounts),
                env_vars=copy.deepcopy(main_executor.env_vars),
                gpus_per_node=main_executor.gpus_per_node,
                gpus_per_task=main_executor.gpus_per_task,
                srun_args=main_executor.srun_args,
            )
        ]

        for executor in executors[1:]:
            main_executor.resource_group.append(
                cls.ResourceRequest(
                    packager=copy.deepcopy(executor.packager),
                    nodes=executor.nodes,
                    ntasks_per_node=executor.ntasks_per_node,
                    container_image=copy.deepcopy(executor.container_image),
                    container_mounts=copy.deepcopy(executor.container_mounts),
                    env_vars=copy.deepcopy(executor.env_vars),
                    gpus_per_node=executor.gpus_per_node,
                    gpus_per_task=executor.gpus_per_task,
                    srun_args=executor.srun_args,
                )
            )

        main_executor.env_vars = {}
        return main_executor

    def __post_init__(self):
        # TODO: Remove this
        assert isinstance(
            self.packager, GitArchivePackager
        ), "Only GitArchivePackager is currently supported for SlurmExecutor."

        if self.wait_time_for_group_job < 0:
            self.wait_time_for_group_job = 0

    def info(self) -> str:
        return f"{self.__class__.__qualname__} on {self.tunnel._key}"

    def alloc(self, job_name="interactive"):
        self.job_name = f"{self.job_name_prefix}{job_name}"
        args = [
            f"--{arg}={getattr(self, arg.replace('-', '_'))}"
            for arg in self.ALLOC_ARGS
            if getattr(self, arg.replace("-", "_"), None) is not None
        ]

        self.slurm.run(
            f"salloc {' '.join(args)} && cd {self.job_dir}",
            hide=False,
            echo=True,
            pty=True,
        )

    def srun(
        self,
        cmd: str,
        job_name="interactive",
        flags=None,
        env_vars: Optional[Dict[str, str]] = None,
        arg_dict=None,
        **kwargs,
    ):
        self.job_name = f"{self.job_name_prefix}{job_name}"
        _arg_dict = {
            arg: getattr(self, arg.replace("-", "_"))
            for arg in self.SRUN_ARGS
            if getattr(self, arg.replace("-", "_"), None) is not None
        }
        _arg_dict["container-mounts"] = ",".join(self.container_mounts)
        if env_vars:
            _arg_dict["container-env"] = ",".join(list(env_vars.keys()))
        if arg_dict:
            _arg_dict.update(arg_dict)

        add_quotes = ["container-image", "container-mounts", "container-workdir"]
        if env_vars:
            add_quotes.append("container-env")
        args = []
        for arg, value in _arg_dict.items():
            if arg in add_quotes:
                args.append(f"--{arg}={shlex.quote(value)}")
            else:
                args.append(f"--{arg}={value}")

        if flags:
            args.extend(flags)

        srun = f"srun {' '.join(args)} {cmd}"
        if env_vars:
            srun = (
                " ".join([f"{key}={shlex.quote(val)}" for key, val in env_vars.items()])
                + " "
                + srun
            )

        return self.slurm.run(srun, **kwargs)

    def bash(self, job_name="interactive"):
        self.srun("bash", job_name=job_name)

    def launch_devspace(
        self,
        space: DevSpace,
        job_name="interactive",
        env_vars: Optional[Dict[str, str]] = None,
        add_workspace_to_pythonpath: bool = True,
    ):
        cfg_zlib = ZlibJSONSerializer().serialize(space.__io__)

        _container_dir = f"/workspaces/{space.name}"

        mounts = self.container_mounts
        mounts.append(f"{self.job_dir}:{_container_dir}")

        if add_workspace_to_pythonpath:
            mounts.append(f"{self.job_dir}:/workspaces/.main")

        arg_dict = {}
        arg_dict["container-workdir"] = _container_dir
        arg_dict["container-mounts"] = ",".join(mounts)

        if self.local_is_slurm:
            srun_kwargs = dict(hide=False, echo=True, pty=True)
        else:
            srun_kwargs = dict(warn=True, hide=False, echo=False, asynchronous=True)

        _env_vars = env_vars or {}
        _env_vars["NEMO_DEVSPACE"] = space.name

        srun = self.srun(
            f"nemorun devspace sshserver {cfg_zlib}",
            job_name=job_name,
            env_vars=_env_vars,
            flags=["--no-container-remap-root"],
            arg_dict=arg_dict,
            **srun_kwargs,
        )

        if not self.local_is_slurm:
            return SlurmTunnelCallback(self, space=space, srun=srun)

    def connect_devspace(self, space, tunnel_dir=None):
        return SlurmTunnelCallback(self, space=space, tunnel_dir=tunnel_dir)

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        self.job_name = task_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        self.experiment_id = exp_id

        os.makedirs(self.job_dir, exist_ok=True)
        self.tunnel._set_job_dir(self.experiment_id)

    def get_launcher_prefix(self) -> Optional[list[str]]:
        launcher = self.get_launcher()
        if launcher.nsys_profile:
            return launcher.get_nsys_prefix(profile_dir=f"/{_RUNDIR_NAME}")

    def package_configs(self, *cfgs: tuple[str, str]) -> list[str]:
        filenames = []
        basepath = os.path.join(self.job_dir, "configs")
        for name, cfg in cfgs:
            filename = os.path.join(basepath, name)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(cfg)

            filenames.append(
                os.path.join(
                    "/",
                    _RUNDIR_NAME,
                    "configs",
                    name,
                )
            )

        return filenames

    def package(self, packager: Packager, job_name: str):
        if job_name in self.tunnel.packaging_jobs:
            logger.info(
                f"Packaging for job {job_name} in tunnel {self.tunnel} already done. Skipping subsequent packagings.\n"
                "This may cause issues if you have multiple tasks with the same name but different packagers, as only the first packager will be used."
            )
            return

        assert self.experiment_id, "Executor not assigned to an experiment."
        output = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
        )
        path = output.stdout.splitlines()[0].decode()
        base_path = Path(path).absolute()
        local_pkg = packager.package(base_path, self.job_dir, job_name)
        remote_code_extraction_path = os.path.join(self.tunnel.job_dir, job_name, "code")
        self.tunnel.run(f"mkdir -p {remote_code_extraction_path}")

        if self.get_launcher().nsys_profile:
            remote_nsys_extraction_path = os.path.join(
                self.tunnel.job_dir, job_name, self.get_launcher().nsys_folder
            )
            self.tunnel.run(f"mkdir -p {remote_nsys_extraction_path}")

        dst_pkg = os.path.join(self.tunnel.job_dir, job_name, os.path.basename(local_pkg))
        self.tunnel.put(local_path=local_pkg, remote_path=dst_pkg)
        self.tunnel.run(f"tar -xvzf {dst_pkg} -C {remote_code_extraction_path}")

        local_configs_path = os.path.join(self.job_dir, "configs")
        remote_config_extraction_path = os.path.join(self.tunnel.job_dir, job_name, "configs")
        if os.path.isdir(local_configs_path):
            self.tunnel.run(f"mkdir -p {remote_config_extraction_path}")
            for file in os.listdir(local_configs_path):
                self.tunnel.put(
                    os.path.join(local_configs_path, file),
                    os.path.join(remote_config_extraction_path, file),
                )

        local_main_path = os.path.join(self.job_dir, "__main__.py")
        if os.path.exists(local_main_path):
            remote_main_path = os.path.join(self.tunnel.job_dir, job_name, "__main__.py")
            self.tunnel.put(local_main_path, remote_main_path)

        self.tunnel.packaging_jobs.add(job_name)

    def parse_deps(self) -> list[str]:
        """
        Helper function to parse a list of TorchX app handles
        and return a list of Slurm Job IDs to use as dependencies.
        """
        deps = []
        for dep in self.dependencies:
            # Parse torchx app handle to get slurm job id
            _, _, path_str = dep.partition("://")

            # path is of the form ["", "app_id", "master", "0"]
            path = path_str.split("/")
            job_id = path[1]
            deps.append(job_id)

        return deps

    def nnodes(self) -> int:
        return self.nodes if isinstance(self.nodes, int) else self.nodes[0]

    def nproc_per_node(self) -> int:
        if self.torchrun_nproc_per_node:
            return self.torchrun_nproc_per_node

        if self.gpus_per_node and self.ntasks_per_node == 1:
            return self.gpus_per_node

        if self.gpus_per_task:
            return self.gpus_per_task

        return (
            self.ntasks_per_node
            if isinstance(self.ntasks_per_node, int)
            else self.ntasks_per_node[0]
        )

    def macro_values(self) -> Optional[ExecutorMacros]:
        return ExecutorMacros(
            head_node_ip_var=self.HEAD_NODE_IP_VAR,
            nproc_per_node_var=self.NPROC_PER_NODE_VAR,
            num_nodes_var=self.NUM_NODES_VAR,
            node_rank_var=self.NODE_RANK_VAR,
            het_group_host_var=self.HET_GROUP_HOST_VAR,
        )

    def _setup_launcher(self):
        super()._setup_launcher()
        launcher = self.launcher
        if launcher and isinstance(launcher, (FaultTolerance, Torchrun)):
            self.torchrun_nproc_per_node = self.torchrun_nproc_per_node or self.ntasks_per_node
            self.ntasks_per_node = 1
            CONSOLE.log(
                f"Detected {launcher.__class__.__name__} launcher, setting ntasks_per_node=1 and torchrun_nproc_per_node={self.torchrun_nproc_per_node}"
            )

        if launcher and isinstance(launcher, FaultTolerance):
            base_dir = os.path.join(self.tunnel.job_dir, Path(self.job_dir).name)
            launcher.cfg_path = os.path.join(base_dir, f"{self.job_name}_ft_cfg.yml")
            launcher.finished_flag_file = os.path.join(
                "/", _RUNDIR_NAME, f"{self.job_name}_finished_flag"
            )
            launcher.job_results_file = os.path.join(base_dir, f"{self.job_name}_job_results")

    @property
    def local(self) -> LocalTunnel:
        if not hasattr(self, "_local"):
            self._local = LocalTunnel(job_dir=self.tunnel.job_dir)
        return self._local

    @property
    def slurm(self) -> Tunnel:
        if self.local_is_slurm:
            return self.local

        self.tunnel.connect()
        return self.tunnel

    @property
    def local_is_slurm(self) -> bool:
        try:
            self.local.run("which srun", hide=True)

            return True
        except invoke.exceptions.UnexpectedExit:
            return False


def _as_sbatch_flag(key: str, value: Any) -> str:
    """Convert key value pairs to `#SBATCH --{key}={value}` flags"""
    key = key.replace("_", "-")
    if value is True:
        return f"#SBATCH --{key}"

    value = shlex.quote(str(value))
    return f"#SBATCH --{key}={value}"


@dataclass(kw_only=True)
class SlurmBatchRequest:
    cmd: list[str]
    jobs: list[str]
    command_groups: list[list[str]]
    slurm_config: SlurmExecutor
    max_retries: int
    setup: Optional[list[str]] = None
    extra_env: dict[str, str]
    launcher: Optional[Launcher] = None

    def materialize(self) -> str:
        """Creates the content of an sbatch file with provided parameters

        Parameters
        ----------
        See slurm sbatch documentation for most parameters:
        https://slurm.schedmd.com/sbatch.html

        Below are the parameters that differ from slurm documentation:

        command_groups:
            each command group will be assigned one srun
        folder: str/Path
            folder where print logs and error logs will be written
        setup: list
            a list of command to run in sbatch before running srun
        additional_parameters: dict
            Forces any parameter to a given value in sbatch. This can be useful
            to add parameters which are not currently available in nemo_launcher.
            Eg: {"mail-user": "blublu@nvidia.com", "mail-type": "BEGIN"}
        srun_args: List[str]
            Add each argument in the list to the srun call

        Raises
        ------
        ValueError
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values
        """
        args = asdict(self.slurm_config)  # noqa: F821
        parameters = {
            k: v for k, v in args.items() if v is not None and k in SlurmExecutor.SBATCH_FLAGS
        }

        # rename and reformat parameters

        if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
            warnings.warn(  # noqa: F821
                '"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")'
            )
        # add necessary parameters
        original_job_name: str = self.jobs[0]  # type: ignore
        if self.slurm_config.job_name_prefix is None:
            job_name = f"{self.slurm_config.account}-{self.slurm_config.account.split('_')[-1]}.{original_job_name}"
        else:
            job_name = f"{self.slurm_config.job_name_prefix}{original_job_name}"
        parameters["job_name"] = job_name
        slurm_job_dir = (
            self.slurm_config.tunnel.job_dir
            if self.slurm_config.tunnel
            else self.slurm_config.job_dir
        )
        job_directory_name = Path(self.slurm_config.job_dir).name
        paths = self.slurm_config.job_paths_cls(
            folder=os.path.join(slurm_job_dir, job_directory_name), job_name=job_name
        )
        stdout = str(paths.stdout)
        stderr = str(paths.stderr)

        if self.slurm_config.array is not None:
            stdout = stdout.replace("%j", "%A_%a")
            stderr = stderr.replace("%j", "%A_%a")
        parameters["output"] = stdout.replace("%t", "0")

        if not self.slurm_config.stderr_to_stdout:
            parameters["error"] = stderr.replace("%t", "0")

        # if NEMO_LAUNCHER_CI:  # Override output file for slurm
        #     parameters["output"] = parameters["error"] = str(paths.folder / "slurm_%j.out")
        #     stdout = stderr = parameters["output"]

        if self.slurm_config.additional_parameters is not None:
            parameters.update(self.slurm_config.additional_parameters)

        # now create
        sbatch_cmd = " ".join([shlex.quote(arg) for arg in self.cmd])

        sbatch_flags = []
        if self.slurm_config.heterogeneous:
            assert len(self.jobs) == len(self.slurm_config.resource_group)
            for i in range(len(self.slurm_config.resource_group)):
                resource_req = self.slurm_config.resource_group[i]
                het_parameters = parameters.copy()
                het_parameters["output"] = parameters["output"].replace(
                    original_job_name, self.jobs[i]
                )
                if "error" in parameters:
                    het_parameters["error"] = parameters["error"].replace(
                        original_job_name, self.jobs[i]
                    )
                het_parameters.update(
                    {
                        "job_name": f"{self.slurm_config.account}-{self.slurm_config.account.split('_')[-1]}.{self.jobs[i]}",
                        "nodes": resource_req.nodes,
                        "ntasks_per_node": resource_req.ntasks_per_node,
                        "gpus_per_node": resource_req.gpus_per_node,
                        "gpus_per_task": resource_req.gpus_per_task,
                    }
                )
                for k in sorted(parameters):
                    sbatch_flags.append(_as_sbatch_flag(k, het_parameters[k]))
                if i != len(self.slurm_config.resource_group) - 1:
                    sbatch_flags.append("#SBATCH hetjob")
        else:
            for k in sorted(parameters):
                sbatch_flags.append(_as_sbatch_flag(k, parameters[k]))

        if self.slurm_config.dependencies:
            slurm_deps = self.slurm_config.parse_deps()
            sbatch_flags.append(_as_sbatch_flag("dependency", f"afterok:{':'.join(slurm_deps)}"))

        env_vars = []
        full_env_vars = self.slurm_config.env_vars | self.extra_env
        for key, value in full_env_vars.items():
            env_vars.append(f"export {key.upper()}={value}")

        # commandline (this will run the function and args specified in the file provided as argument)
        # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern
        stderr_flags = [] if self.slurm_config.stderr_to_stdout else ["--error", stderr]

        srun_commands = []
        group_env_vars = []
        srun_stdout = noquote(paths.srun_stdout)
        stderr_flags = (
            [] if self.slurm_config.stderr_to_stdout else ["--error", noquote(paths.srun_stderr)]
        )
        memory_measure = None
        if self.slurm_config.memory_measure:
            memory_measure = srun_stdout

        def get_container_flags(
            base_mounts: list[str], src_job_dir: str, container_image: Optional[str]
        ) -> list[str]:
            _container_flags = ["--container-image", container_image] if container_image else []

            new_mounts = copy.deepcopy(base_mounts)
            new_mounts.append(f"{src_job_dir}:/{_RUNDIR_NAME}")
            _mount_arg = ",".join(new_mounts)
            _container_flags += ["--container-mounts", _mount_arg]
            _container_flags += [
                "--container-workdir",
                f"/{_RUNDIR_NAME}/code",
            ]

            return _container_flags

        for group_ind, command_group in enumerate(self.command_groups):
            if self.slurm_config.heterogeneous:
                resource_req = self.slurm_config.resource_group[group_ind]

                current_env_vars = []
                for key, value in resource_req.env_vars.items():
                    current_env_vars.append(f"export {key.upper()}={value}")

                group_env_vars.append(current_env_vars)

                het_group = f"--het-group={group_ind}"
                het_stdout = srun_stdout.replace(original_job_name, self.jobs[group_ind])
                het_stderr = stderr_flags.copy()
                if het_stderr:
                    het_stderr[-1] = het_stderr[-1].replace(original_job_name, self.jobs[group_ind])

                _group_srun_args = ["--wait=60", "--kill-on-bad-exit=1"]
                _group_srun_args.extend(resource_req.srun_args or [])
                srun_cmd = " ".join(
                    list(
                        map(
                            lambda arg: arg if isinstance(arg, noquote) else shlex.quote(arg),
                            [
                                "srun",
                                het_group,
                                "--output",
                                het_stdout,
                                *het_stderr,
                                *get_container_flags(
                                    base_mounts=resource_req.container_mounts,
                                    src_job_dir=os.path.join(slurm_job_dir, job_directory_name),
                                    container_image=resource_req.container_image,
                                ),
                                *_group_srun_args,
                            ],
                        )
                    )
                )

                command = ";\n  ".join(command_group)

                srun_command = f"{srun_cmd} {command} & pids[{group_ind}]=$!"
                if group_ind != len(self.slurm_config.resource_group) - 1:
                    srun_command += f"\n\nsleep {self.slurm_config.wait_time_for_group_job}\n"
                srun_commands.append(srun_command)
            else:
                cmd_stdout = srun_stdout.replace(original_job_name, self.jobs[group_ind])
                cmd_stderr = stderr_flags.copy()
                if cmd_stderr:
                    cmd_stderr[-1] = cmd_stderr[-1].replace(original_job_name, self.jobs[group_ind])

                if self.slurm_config.run_as_group and len(self.slurm_config.resource_group) == len(
                    self.command_groups
                ):
                    resource_req = self.slurm_config.resource_group[group_ind]
                    _container_flags = get_container_flags(
                        base_mounts=resource_req.container_mounts,
                        src_job_dir=os.path.join(
                            slurm_job_dir,
                            job_directory_name,
                        ),
                        container_image=resource_req.container_image,
                    )
                    _srun_args = ["--wait=60", "--kill-on-bad-exit=1"]
                    _srun_args.extend(resource_req.srun_args or [])
                else:
                    _container_flags = get_container_flags(
                        base_mounts=self.slurm_config.container_mounts,
                        src_job_dir=os.path.join(
                            slurm_job_dir,
                            job_directory_name,
                        ),
                        container_image=self.slurm_config.container_image,
                    )
                    _srun_args = ["--wait=60", "--kill-on-bad-exit=1"]
                    _srun_args.extend(self.slurm_config.srun_args or [])

                srun_cmd = " ".join(
                    list(
                        map(
                            lambda arg: arg if isinstance(arg, noquote) else shlex.quote(arg),
                            [
                                "srun",
                                "--output",
                                cmd_stdout,
                                *cmd_stderr,
                                *_container_flags,
                                *_srun_args,
                            ],
                        )
                    )
                )
                command = " ".join(command_group)

                if self.slurm_config.run_as_group:
                    srun_command = f"{srun_cmd} {command} & pids[{group_ind}]=$!"
                    if group_ind != len(self.command_groups) - 1:
                        srun_command += f"\n\nsleep {self.slurm_config.wait_time_for_group_job}\n"
                else:
                    srun_command = f"{srun_cmd} {command}"

                srun_commands.append(srun_command)

        vars_to_fill = {
            "sbatch_command": sbatch_cmd,
            "sbatch_flags": sbatch_flags,
            "max_retries": self.max_retries,
            "env_vars": env_vars,
            "head_node_ip_var": SlurmExecutor.HEAD_NODE_IP_VAR,
            "setup_lines": self.setup,
            "memory_measure": memory_measure,
            "srun_commands": srun_commands,
            "group_env_vars": group_env_vars,
            "heterogeneous": self.slurm_config.heterogeneous,
            "run_as_group": self.slurm_config.run_as_group,
            "monitor_group_job": self.slurm_config.run_as_group
            and self.slurm_config.monitor_group_job,
            "monitor_group_job_wait_time": self.slurm_config.monitor_group_job_wait_time,
            "het_group_host_var": SlurmExecutor.HET_GROUP_HOST_VAR,
            "ft_enabled": self.launcher and isinstance(self.launcher, FaultTolerance),
        }

        if self.launcher and isinstance(self.launcher, FaultTolerance):
            assert (
                self.launcher.cfg_path
                and self.launcher.finished_flag_file
                and self.launcher.job_results_file
            )
            vars_to_fill["fault_tol_cfg_path"] = self.launcher.cfg_path
            vars_to_fill["fault_tol_finished_flag_file"] = self.launcher.finished_flag_file
            vars_to_fill["fault_tol_job_results_file"] = self.launcher.job_results_file

        sbatch_script = fill_template("slurm.sh.j2", vars_to_fill)
        return sbatch_script

    def __repr__(self) -> str:
        return f"""{' '.join(self.cmd + ['$SBATCH_SCRIPT'])}

#----------------
# SBATCH_SCRIPT
#----------------
{self.materialize()}"""


class SlurmTunnelCallback(Callback):
    def __init__(self, executor: SlurmExecutor, space: DevSpace, srun=None, tunnel_dir=None):
        self.executor = executor
        self.srun = srun
        self.space = space
        self.ssh_config = SSHConfigFile()
        self.console = Console()
        self.editor_started = False
        self.tunnel_dir = tunnel_dir

    def on_start(self):
        if self.srun is not None:
            self.srun_status = self.console.status(
                Text("srun: ", style="bold green"), spinner="dots"
            )
            self.srun_status.start()
            self.srun_is_done = False
        else:
            self.srun_is_done = True

    def on_interval(self):
        from nemo_run.devspace.editor import launch_editor

        if not self.srun_is_done:
            status = self.srun.runner.stderr[-1] if self.srun.runner.stderr else None
            stdout = self.srun.runner.stdout

            if stdout:
                for line in stdout:
                    if (
                        "To connect to the tunnel, run the following command on your local machine:"
                        in line
                    ):
                        if not self.srun_is_done:
                            self.srun_is_done = True
                            self.srun_status.stop()
                            self.console.log(":white_check_mark: Server is launched")
                            self.console.log("[bold green]Devspace is active...")

            if not self.srun_is_done and status:
                self.srun_status.update(Text(status, style="bold green"))
        elif not self.editor_started:
            _tunnel_dir = self.tunnel_dir or server_dir(self.executor.job_dir, self.space.name)
            metadata = TunnelMetadata.restore(_tunnel_dir, tunnel=self.tunnel)
            self.forward_port_context = self.tunnel.session.forward_local(
                int(metadata.port), remote_host=metadata.hostname
            )
            self.forward_port_context.__enter__()

            self.ssh_config.add_entry(
                metadata.user, "localhost", int(metadata.port), self.tunnel_name
            )
            self.ssh_entry_added = True

            with self.console.status("Setting up port forwarding", spinner="dots"):
                time.sleep(3)

            self.console.print(
                f"[bold green]:white_check_mark: Port forwarding established. "
                f"Connect via SSH with: ssh tunnel.{self.tunnel_name}"
            )
            launch_editor(self.tunnel_name, f"/workspaces/{metadata.workspace_name}")
            self.editor_started = True

    def on_stop(self):
        # if hasattr(self, "forward_port_context"):
        #     self.forward_port_context.__exit__()
        if hasattr(self, "ssh_entry_added"):
            self.ssh_config.remove_entry(self.tunnel_name)

    @property
    def tunnel_name(self) -> str:
        workspace_name = self.space.name

        return ".".join([workspace_name, self.space.name])
