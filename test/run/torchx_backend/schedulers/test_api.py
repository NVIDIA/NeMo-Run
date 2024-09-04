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

import pytest
from torchx.specs import AppDef, AppDryRunInfo, Role

from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.run.torchx_backend.schedulers.api import (
    SchedulerMixin,
    get_executor_str,
)


class MockExecutor(Executor):
    def __init__(self, executor_str: str):
        self.executor_str = executor_str

    def __str__(self):
        return self.executor_str


# class MockRole:
#     def __init__(self, pre_proc_fn: Any = None):
#         self.pre_proc = pre_proc_fn


# class MockAppDef(AppDef):
#     def __init__(self, roles: list[MockRole]):
#         self.roles = roles


class MockSchedulerMixin(SchedulerMixin):
    backend: str = ""

    def _submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo:
        return AppDryRunInfo(request="", fmt=repr)


@pytest.fixture
def mock_app_def():
    return AppDef(name="mock", roles=[Role(name="mock_role", image="")])


@pytest.mark.parametrize(
    "executor, expected_str",
    [
        (SlurmExecutor(account=""), "slurm_tunnel"),
        (SkypilotExecutor(job_dir=""), "skypilot"),
        (LocalExecutor(), "local_persistent"),
        (DockerExecutor(container_image=""), "docker_persistent"),
    ],
)
def test_get_executor_str(executor, expected_str):
    assert get_executor_str(executor) == expected_str


def test_submit_dryrun(mock_app_def):
    scheduler_mixin = MockSchedulerMixin()
    executor = MockExecutor("test_executor")
    dryrun_info = scheduler_mixin.submit_dryrun(mock_app_def, executor)
    assert dryrun_info._app == mock_app_def


def test_submit_dryrun_with_role_preproc(mock_app_def):
    def mock_preproc(backend, dryrun_info):
        return dryrun_info

    scheduler_mixin = MockSchedulerMixin()
    executor = MockExecutor("test_executor")
    dryrun_info = scheduler_mixin.submit_dryrun(mock_app_def, executor)
    assert dryrun_info._app == mock_app_def
