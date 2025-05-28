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
from typing import Any, Optional, Type

from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.frontend.console.api import configure_logging
from nemo_run.run.ray.slurm import SlurmRayJob

# Import guard for Kubernetes dependencies
try:
    from nemo_run.core.execution.kuberay import KubeRayExecutor
    from nemo_run.run.ray.kuberay import KubeRayJob

    _KUBERAY_AVAILABLE = True
except ImportError:
    KubeRayExecutor = None
    KubeRayJob = None
    _KUBERAY_AVAILABLE = False


@dataclass(kw_only=True)
class RayJob:
    """Backend-agnostic convenience wrapper around Ray *jobs*."""

    name: str
    executor: Executor
    pre_ray_start_commands: Optional[list[str]] = None
    log_level: str = "INFO"

    def __post_init__(self) -> None:  # noqa: D401 – simple implementation
        configure_logging(level=self.log_level)
        backend_map: dict[Type[Executor], Type[Any]] = {
            SlurmExecutor: SlurmRayJob,
        }

        if _KUBERAY_AVAILABLE and KubeRayExecutor is not None and KubeRayJob is not None:
            backend_map[KubeRayExecutor] = KubeRayJob

        if self.executor.__class__ not in backend_map:
            raise ValueError(f"Unsupported executor: {self.executor.__class__}")

        backend_cls = backend_map[self.executor.__class__]
        self.backend = backend_cls(name=self.name, executor=self.executor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        command: str,
        workdir: str,
        runtime_env_yaml: Optional[str] | None = None,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
    ) -> Any:
        """Submit a Ray job and return a live helper (backend specific).

        The *pre_ray_start_commands* provided at construction time are forwarded
        to the backend implementation so callers can inject arbitrary shell
        commands that run inside the Ray *head* container right before the
        cluster starts.
        """
        self.backend.start(  # type: ignore[attr-defined]
            command=command,
            workdir=workdir,
            runtime_env_yaml=runtime_env_yaml,
            pre_ray_start_commands=pre_ray_start_commands,
            dryrun=dryrun,
        )

    def stop(self) -> None:
        self.backend.stop()  # type: ignore[attr-defined]

    def status(self, display: bool = True):
        return self.backend.status(display=display)  # type: ignore[attr-defined]

    def logs(self, *, follow: bool = False, lines: int = 100, timeout: int = 100):
        self.backend.logs(follow=follow, lines=lines, timeout=timeout)  # type: ignore[attr-defined]
