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

from typing import Type

from torchx.specs import AppDef, AppDryRunInfo

from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.core.execution.slurm import SlurmExecutor

EXECUTOR_MAPPING: dict[Type[Executor], str] = {
    SlurmExecutor: "slurm_tunnel",
    SkypilotExecutor: "skypilot",
    LocalExecutor: "local_persistent",
    DockerExecutor: "docker_persistent",
}

REVERSE_EXECUTOR_MAPPING: dict[str, Type[Executor]] = {
    "slurm_tunnel": SlurmExecutor,
    "skypilot": SkypilotExecutor,
    "local_persistent": LocalExecutor,
    "docker_persistent": DockerExecutor,
}


def get_executor_str(executor: Executor) -> str:
    """
    Maps the executor config class to the underlying executor identifier.
    """
    return EXECUTOR_MAPPING[executor.__class__]


class SchedulerMixin:
    def submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo:
        dryrun_info = self._submit_dryrun(app, cfg)  # type: ignore
        for role in app.roles:
            dryrun_info = role.pre_proc(self.backend, dryrun_info)  # type: ignore
        dryrun_info._app = app
        return dryrun_info
