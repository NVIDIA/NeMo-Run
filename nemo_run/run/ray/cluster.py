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

from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.kuberay import KubeRayExecutor
from nemo_run.run.ray.kuberay import KubeRayCluster


@dataclass(kw_only=True)
class RayCluster:
    BACKEND_MAP = {
        KubeRayExecutor: KubeRayCluster,
    }

    name: str
    executor: Executor

    def __post_init__(self):
        if self.executor.__class__ not in self.BACKEND_MAP:
            raise ValueError(f"Unsupported executor: {self.executor}")

        self.backend = self.BACKEND_MAP[self.executor.__class__]()

        self._port_forward_map = {}

    def start(self, wait_until_ready: bool = True, timeout: int = 1000):
        assert isinstance(self.executor, self.backend.EXECUTOR_CLS)
        self.backend.create_ray_cluster(name=self.name, executor=self.executor)
        if wait_until_ready:
            self.backend.wait_until_ray_cluster_running(
                name=self.name, executor=self.executor, timeout=timeout
            )

    def port_forward(self, port: int = 8265, target_port: int = 8265, wait: bool = False):
        assert isinstance(self.executor, self.backend.EXECUTOR_CLS)
        if self._port_forward_map.get(port) is not None:
            self._port_forward_map[port].stop_forwarding()

        self._port_forward_map[port] = self.backend.port_forward(
            name=self.name,
            k8s_namespace=self.executor.namespace,
            port=port,
            target_port=target_port,
            wait=wait,
        )

    def stop(self):
        assert isinstance(self.executor, self.backend.EXECUTOR_CLS)
        for port_forward in self._port_forward_map.values():
            port_forward.stop_forwarding()

        self.backend.delete_ray_cluster(name=self.name, executor=self.executor, wait=True)
