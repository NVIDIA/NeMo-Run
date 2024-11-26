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

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Union

from invoke.context import Context

from nemo_run.config import RUNDIR_NAME
from nemo_run.core.execution.base import (
    Executor,
    ExecutorMacros,
    FaultTolerance,
    Torchrun,
)
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

_SKYPILOT_AVAILABLE: bool = False
try:
    import sky
    import sky.task as skyt
    from sky import backends, status_lib

    _SKYPILOT_AVAILABLE = True
except ImportError:
    ...

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SkypilotExecutor(Executor):
    """
    Dataclass to configure a Skypilot Executor.

    Some familiarity with `Skypilot <https://skypilot.readthedocs.io/en/latest/docs/index.html>`_ is necessary.
    In order to use this executor, you need to install NeMo Run
    with either `skypilot` (for only Kubernetes) or `skypilot-all` (for all clouds) optional features.

    Example:

    .. code-block:: python

        skypilot = SkypilotExecutor(
            gpus="A10G",
            gpus_per_node=devices,
            container_image="nvcr.io/nvidia/nemo:dev",
            cloud="kubernetes",
            cluster_name="nemo_tester",
            file_mounts={
                "nemo_run.whl": "nemo_run.whl"
            },
            setup=\"\"\"
        conda deactivate
        nvidia-smi
        ls -al ./
        pip install nemo_run.whl
        cd /opt/NeMo && git pull origin main && pip install .
            \"\"\",
        )

    """

    HEAD_NODE_IP_VAR = "head_node_ip"
    NPROC_PER_NODE_VAR = "SKYPILOT_NUM_GPUS_PER_NODE"
    NUM_NODES_VAR = "num_nodes"
    NODE_RANK_VAR = "SKYPILOT_NODE_RANK"
    HET_GROUP_HOST_VAR = "het_group_host"

    container_image: Optional[str] = None
    cloud: Optional[Union[str, list[str]]] = None
    region: Optional[Union[str, list[str]]] = None
    zone: Optional[Union[str, list[str]]] = None
    gpus: Optional[Union[str, list[str]]] = None
    gpus_per_node: Optional[int] = None
    cpus: Optional[Union[int | float, list[int | float]]] = None
    memory: Optional[Union[int | float, list[int | float]]] = None
    instance_type: Optional[Union[str, list[str]]] = None
    num_nodes: int = 1
    use_spot: Optional[Union[bool, list[bool]]] = None
    disk_size: Optional[Union[int, list[int]]] = None
    disk_tier: Optional[Union[str, list[str]]] = None
    ports: Optional[tuple[str]] = None
    file_mounts: Optional[dict[str, str]] = None
    cluster_name: Optional[str] = None
    setup: Optional[str] = None
    autodown: bool = False
    idle_minutes_to_autostop: Optional[int] = None
    torchrun_nproc_per_node: Optional[int] = None
    packager: Packager = field(default_factory=lambda: GitArchivePackager())  # type: ignore  # noqa: F821

    def __post_init__(self):
        assert (
            _SKYPILOT_AVAILABLE
        ), "Skypilot is not installed. Please install it using `pip install nemo_run[skypilot]"
        assert isinstance(
            self.packager, GitArchivePackager
        ), "Only GitArchivePackager is currently supported for SkypilotExecutor."

    @classmethod
    def parse_app(cls: Type["SkypilotExecutor"], app_id: str) -> tuple[str, str, int]:
        app = app_id.split("___")
        _, cluster, task, job_id = app[0], app[1], app[2], app[3]
        assert cluster and task and job_id, f"Invalid app id for Skypilot: {app_id}"
        return cluster, task, int(job_id)

    def to_resources(self) -> Union[set["sky.Resources"], set["sky.Resources"]]:
        from sky.resources import Resources

        resources_cfg = {}
        accelerators = None
        if self.gpus:
            if not self.gpus_per_node:
                self.gpus_per_node = 1
            else:
                assert isinstance(self.gpus_per_node, int)

            gpus = [self.gpus] if isinstance(self.gpus, str) else self.gpus

            accelerators = {}
            for gpu in gpus:
                accelerators[gpu] = self.gpus_per_node

            resources_cfg["accelerators"] = accelerators

        if self.container_image:
            resources_cfg["image_id"] = self.container_image

        any_of = []

        def parse_attr(attr: str):
            if getattr(self, attr, None) is not None:
                value = getattr(self, attr)
                if isinstance(value, list):
                    for i, val in enumerate(value):
                        if len(any_of) < i + 1:
                            any_of.append({})

                        if val.lower() == "none":
                            any_of[i][attr] = val
                        else:
                            any_of[i][attr] = val
                else:
                    if value.lower() == "none":
                        resources_cfg[attr] = None
                    else:
                        resources_cfg[attr] = value

        # any_of = False
        attrs = [
            "cloud",
            "region",
            "zone",
            "cpus",
            "memory",
            "instance_type",
            "use_spot",
            "image_id",
            "disk_size",
            "disk_tier",
            "ports",
        ]

        for attr in attrs:
            parse_attr(attr)

        resources_cfg["any_of"] = any_of
        resources = Resources.from_yaml_config(resources_cfg)

        return resources  # type: ignore

    @classmethod
    def status(
        cls: Type["SkypilotExecutor"], app_id: str
    ) -> tuple[Optional["status_lib.ClusterStatus"], Optional[dict]]:
        import sky.core as sky_core
        import sky.exceptions as sky_exceptions
        from sky import status_lib

        cluster, _, job_id = cls.parse_app(app_id)
        try:
            cluster_details = sky_core.status(cluster)[0]
            cluster_status: status_lib.ClusterStatus = cluster_details["status"]
        except Exception:
            return None, None

        try:
            job_queue = sky_core.queue(cluster, all_users=True)
            job_details = next(filter(lambda job: job["job_id"] == job_id, job_queue))
        except sky_exceptions.ClusterNotUpError:
            return cluster_status, None

        return cluster_status, job_details

    @classmethod
    def cancel(cls: Type["SkypilotExecutor"], app_id: str):
        from sky.core import cancel

        cluster_name, _, job_id = cls.parse_app(app_id=app_id)
        _, job_details = cls.status(app_id=app_id)
        if not job_details:
            return

        cancel(cluster_name=cluster_name, job_ids=[job_id])

    @classmethod
    def logs(cls: Type["SkypilotExecutor"], app_id: str, fallback_path: Optional[str]):
        import sky.core as sky_core
        from sky.skylet import job_lib

        cluster, _, job_id = cls.parse_app(app_id)
        _, job_details = cls.status(app_id)

        is_terminal = False
        if job_details and job_lib.JobStatus.is_terminal(job_details["status"]):
            is_terminal = True
        elif not job_details:
            is_terminal = True
        if fallback_path and is_terminal:
            log_path = os.path.expanduser(os.path.join(fallback_path, "run.log"))
            if os.path.isfile(log_path):
                with open(os.path.expanduser(os.path.join(fallback_path, "run.log"))) as f:
                    for line in f:
                        print(line, end="", flush=True)

                return

        sky_core.tail_logs(cluster, job_id)

    @property
    def workdir(self) -> str:
        return os.path.join(f"{self.job_dir}", "workdir")

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
                    RUNDIR_NAME,
                    "configs",
                    name,
                )
            )

        return filenames

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

    def package(self, packager: Packager, job_name: str):
        assert self.experiment_id, "Executor not assigned to an experiment."
        if isinstance(packager, GitArchivePackager):
            output = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                stdout=subprocess.PIPE,
            )
            path = output.stdout.splitlines()[0].decode()
            base_path = Path(path).absolute()
        else:
            base_path = Path(os.getcwd()).absolute()

        local_pkg = packager.package(base_path, self.job_dir, job_name)
        local_code_extraction_path = os.path.join(self.job_dir, "code")
        ctx = Context()
        ctx.run(f"mkdir -p {local_code_extraction_path}")

        if self.get_launcher().nsys_profile:
            remote_nsys_extraction_path = os.path.join(
                self.job_dir, job_name, self.get_launcher().nsys_folder
            )
            ctx.run(f"mkdir -p {remote_nsys_extraction_path}")
        if local_pkg:
            ctx.run(
                f"tar -xvzf {local_pkg} -C {local_code_extraction_path} --ignore-zeros", hide=True
            )

    def nnodes(self) -> int:
        return self.num_nodes

    def nproc_per_node(self) -> int:
        if self.torchrun_nproc_per_node:
            return self.torchrun_nproc_per_node

        return self.gpus_per_node or 1

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
        # Dynamic rendezvous has an error in Skypilot Kubernetes currently
        if (
            launcher
            and isinstance(launcher, (Torchrun, FaultTolerance))
            and self.cloud == "kubernetes"
        ):
            launcher.rdzv_backend = "static"
            launcher.rdzv_port = 49500

    def to_task(
        self,
        name: str,
        cmd: Optional[list[str]] = None,
        env_vars: Optional[dict[str, str]] = None,
    ) -> "skyt.Task":
        from sky.task import Task

        run_cmd = None
        if cmd:
            run_cmd = f"""
conda deactivate

num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
echo "num_nodes=$num_nodes"

head_node_ip=`echo "$SKYPILOT_NODE_IPS" | head -n1`
echo "head_node_ip=$head_node_ip"

cd /nemo_run/code

{" ".join(cmd)}
"""
        task = Task(
            name=name,
            setup=self.setup if self.setup else "",
            run=run_cmd,
            envs=self.env_vars,
            num_nodes=self.num_nodes,
        )
        file_mounts = self.file_mounts or {}
        file_mounts["/nemo_run"] = self.job_dir
        task.set_file_mounts(self.file_mounts)
        task.set_resources(self.to_resources())

        if env_vars:
            task.update_envs(env_vars)

        return task

    def launch(
        self,
        task: "skyt.Task",
        cluster_name: Optional[str] = None,
        num_nodes: Optional[int] = None,
        detach_run: bool = True,
        dryrun: bool = False,
    ) -> tuple[Optional[int], Optional["backends.ResourceHandle"]]:
        from sky import backends
        from sky.execution import launch
        from sky.utils import common_utils

        task_yml = os.path.join(self.job_dir, "skypilot_task.yml")
        with open(task_yml, "w+") as f:
            f.write(common_utils.dump_yaml_str(task.to_yaml_config()))

        backend = backends.CloudVmRayBackend()
        if num_nodes:
            task.num_nodes = num_nodes

        cluster_name = cluster_name or self.cluster_name or self.experiment_id
        job_id, handle = launch(
            task,
            dryrun=dryrun,
            stream_logs=False,
            cluster_name=cluster_name,
            detach_setup=False,
            detach_run=detach_run,
            backend=backend,
            idle_minutes_to_autostop=self.idle_minutes_to_autostop,
            down=self.autodown,
            # retry_until_up=retry_until_up,
            no_setup=True if (self.cluster_name and not self.setup) else False,
            # clone_disk_from=clone_disk_from,
        )

        return job_id, handle

    def cleanup(self, handle: str):
        import sky.core as sky_core

        _, _, path_str = handle.partition("://")
        path = path_str.split("/")
        app_id = path[1]

        cluster, _, job_id = self.parse_app(app_id)
        sky_core.download_logs(cluster, job_ids=[job_id])
