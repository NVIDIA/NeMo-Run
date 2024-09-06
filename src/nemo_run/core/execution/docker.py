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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type

import fiddle._src.experimental.dataclasses as fdl_dc
from invoke.context import Context

from nemo_run.config import RUNDIR_NAME
from nemo_run.core.execution.base import Executor
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager
from nemo_run.core.serialization.yaml import YamlSerializer

if TYPE_CHECKING:
    from docker import DockerClient
    from docker.models.containers import Container

logger = logging.getLogger(__name__)

NETWORK = "nemo_run"

LABEL_EXPERIMENT_ID: str = "nemo-run/experiment-id"
LABEL_NAME: str = "nemo-run/name"
LABEL_ID: str = "nemo-run/id"


def ensure_network(client: Optional["DockerClient"] = None) -> None:
    """
    This creates the torchx docker network. Multi-process safe.
    """
    import filelock
    from docker.errors import APIError

    if client is None:
        import docker

        client = docker.from_env()

    lock_path = os.path.join(tempfile.gettempdir(), "nemorun_docker_network_lock")

    # Docker networks.create check_duplicate has a race condition so we need
    # to do client side locking to ensure only one network is created.
    with filelock.FileLock(lock_path, timeout=10):
        try:
            client.networks.create(name=NETWORK, driver="bridge", check_duplicate=True)
        except APIError as e:
            if "already exists" not in str(e):
                raise


@dataclass(kw_only=True)
class DockerExecutor(Executor):
    """
    Dataclass to configure a docker based executor.

    Example:

    .. code-block:: python

        run.DockerExecutor()

    """

    container_image: str
    #: Used by components like torchrun to deduce the number of tasks to launch.
    ntasks_per_node: int = 1
    runtime: Optional[str] = None
    #: Number of gpus to use, -1 means use all gpus
    num_gpus: Optional[int] = None
    shm_size: Optional[str] = None
    #: Specify --ulimit with a soft and hard limit in the format <type>=<soft limit>[:<hard limit>]
    ulimits: list[str] = field(default_factory=list)
    ipc_mode: Optional[str] = None
    #: Refer to https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.ContainerCollection.run for healthcheck format.
    healthcheck: dict[str, Any] = field(default_factory=dict)
    privileged: bool = False
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)
    packager: GitArchivePackager = field(default_factory=lambda: GitArchivePackager())  # type: ignore  # noqa: F821

    job_name: str = field(init=False, default="nemo-job")

    run_as_group: bool = field(init=False, default=False)
    resource_group: list["DockerExecutor"] = field(init=False, default_factory=list)

    @classmethod
    def merge(
        cls: Type["DockerExecutor"], executors: list["DockerExecutor"], num_tasks: int
    ) -> "DockerExecutor":
        assert len(executors) in [1, num_tasks]
        if len(executors) == 1:
            executors = [executors[0]] + [executors[0].clone() for _ in range(1, num_tasks)]

        main_executor = executors[0]
        main_executor.run_as_group = True
        main_executor.resource_group = executors
        return main_executor

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        self.job_name = task_id
        self.experiment_id = exp_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        os.makedirs(self.job_dir, exist_ok=True)

    def nnodes(self) -> int:
        return 1

    def nproc_per_node(self) -> int:
        return self.ntasks_per_node

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

    def package(self, packager: Packager, job_name: str):
        assert self.experiment_id, "Executor not assigned to an experiment."
        output = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
        )
        path = output.stdout.splitlines()[0].decode()
        base_path = Path(path).absolute()
        local_pkg = packager.package(base_path, self.job_dir, job_name)
        local_code_extraction_path = os.path.join(self.job_dir, "code")
        ctx = Context()
        ctx.run(f"mkdir -p {local_code_extraction_path}")

        if self.get_launcher().nsys_profile:
            remote_nsys_extraction_path = os.path.join(
                self.job_dir, job_name, self.get_launcher().nsys_folder
            )
            ctx.run(f"mkdir -p {remote_nsys_extraction_path}")

        ctx.run(f"tar -xvzf {local_pkg} -C {local_code_extraction_path}", hide=True)


@dataclass(kw_only=True)
class DockerContainer:
    name: str
    command: list[str]
    executor: DockerExecutor
    extra_env: dict[str, str]

    def run(self, client: "DockerClient", id: str) -> "Container":
        from docker.types import DeviceRequest, Ulimit

        container_kwargs = {}

        if self.executor.runtime:
            container_kwargs["runtime"] = self.executor.runtime

        ulimits = None
        if self.executor.ulimits:
            ulimits = []
            for ulimit in self.executor.ulimits:
                vals = ulimit.split(":")
                name = vals[0]
                soft = vals[1]
                hard = vals[2] if len(vals) > 2 else None
                ulimits.append(Ulimit(name=name, soft=int(soft), hard=int(hard) if hard else None))

        container_kwargs = {
            "runtime": self.executor.runtime,
            "volumes": self.executor.volumes,
            "shm_size": self.executor.shm_size,
            "ulimits": ulimits,
            "ipc_mode": self.executor.ipc_mode,
            "healthcheck": self.executor.healthcheck,
            "privileged": self.executor.privileged,
        }

        if self.executor.num_gpus:
            container_kwargs["device_requests"] = [
                DeviceRequest(
                    count=self.executor.num_gpus,
                    capabilities=[["compute", "utility"]],
                )
            ]

        if self.executor.retries:
            container_kwargs["restart_policy"] = {
                "Name": "on-failure",
                "MaximumRetryCount": self.executor.retries,
            }

        container_kwargs.update(self.executor.additional_kwargs)
        assert self.executor.experiment_id
        tee_cmd = f" 2>&1 | tee -a /{RUNDIR_NAME}/log_{self.name}.out"
        command = " ".join(self.command)
        command = f'bash -c "{command}{tee_cmd}"'

        return client.containers.run(
            self.executor.container_image,
            command,
            detach=True,
            remove=True,
            name=self.name,
            hostname=self.name,
            network=NETWORK,
            working_dir=f"/{RUNDIR_NAME}/code",
            labels={
                LABEL_EXPERIMENT_ID: self.executor.experiment_id,
                LABEL_NAME: self.name,
                LABEL_ID: id,
            },
            environment=self.executor.env_vars | self.extra_env,
            **container_kwargs,
        )

    def get_container(self, client: "DockerClient", id: str) -> Optional["Container"]:
        containers = client.containers.list(
            all=True,
            filters={
                "label": [
                    f"{LABEL_ID}={id}",
                    f"{LABEL_NAME}={self.name}",
                ]
            },
        )
        return containers[0] if len(containers) >= 1 else None


@dataclass(kw_only=True)
class DockerJobRequest:
    id: str
    executor: DockerExecutor
    containers: list[DockerContainer]

    def to_config(self):
        return fdl_dc.convert_dataclasses_to_configs(self, allow_post_init=True)

    def __str__(self) -> str:
        return YamlSerializer().serialize(
            self.to_config(),
        )

    def __repr__(self) -> str:
        return str(self)

    def run(self, client: "DockerClient") -> list["Container"]:
        ensure_network(client)

        container_details = []
        for container in self.containers:
            container_details.append(container.run(client=client, id=self.id))

        return container_details

    def get_containers(self, client: "DockerClient") -> list["Container"]:
        return client.containers.list(all=True, filters={"label": f"{LABEL_ID}={self.id}"})
