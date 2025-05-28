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

from dataclasses import dataclass
from typing import Optional, Type

from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.frontend.console.api import configure_logging
from nemo_run.run.ray.slurm import SlurmRayCluster

# Import guard for Kubernetes dependencies
try:
    from nemo_run.core.execution.kuberay import KubeRayExecutor
    from nemo_run.run.ray.kuberay import KubeRayCluster

    _KUBERAY_AVAILABLE = True
except ImportError:
    KubeRayExecutor = None
    KubeRayCluster = None
    _KUBERAY_AVAILABLE = False


@dataclass(kw_only=True)
class RayCluster:
    name: str
    executor: Executor
    log_level: str = "INFO"

    def __post_init__(self):
        configure_logging(level=self.log_level)
        backend_map: dict[Type[Executor], Type] = {
            SlurmExecutor: SlurmRayCluster,
        }

        if _KUBERAY_AVAILABLE and KubeRayExecutor is not None and KubeRayCluster is not None:
            backend_map[KubeRayExecutor] = KubeRayCluster

        if self.executor.__class__ not in backend_map:
            raise ValueError(f"Unsupported executor: {self.executor.__class__}")

        backend_cls = backend_map[self.executor.__class__]
        self.backend = backend_cls(name=self.name, executor=self.executor)  # type: ignore[arg-type]

        self._port_forward_map = {}

    def start(
        self,
        wait_until_ready: bool = True,
        timeout: int = 1000,
        dryrun: bool = False,
        pre_ray_start_commands: Optional[list[str]] = None,
    ):
        assert isinstance(self.executor, self.backend.EXECUTOR_CLS)
        self.backend.create(
            pre_ray_start_commands=pre_ray_start_commands,
            dryrun=dryrun,
        )
        if wait_until_ready and not dryrun:
            self.backend.wait_until_running(timeout=timeout)

    def status(self, display: bool = True):
        return self.backend.status(display=display)  # type: ignore[attr-defined]

    def port_forward(self, port: int = 8265, target_port: int = 8265, wait: bool = False):
        assert isinstance(self.executor, self.backend.EXECUTOR_CLS)
        if self._port_forward_map.get(port) is not None:
            self._port_forward_map[port].stop_forwarding()

        self._port_forward_map[port] = self.backend.port_forward(
            port=port,
            target_port=target_port,
            wait=wait,
        )

    def stop(self):
        assert isinstance(self.executor, self.backend.EXECUTOR_CLS)
        for port_forward in self._port_forward_map.values():
            port_forward.stop_forwarding()

        self.backend.delete(wait=True)
