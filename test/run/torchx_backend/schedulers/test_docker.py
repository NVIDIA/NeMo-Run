# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, Role

from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.run.torchx_backend.schedulers.docker import (
    PersistentDockerScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="ubuntu:latest")])


@pytest.fixture
def docker_executor():
    return DockerExecutor(container_image="ubuntu:latest", job_dir=tempfile.mkdtemp())


@pytest.fixture
def docker_scheduler():
    with mock.patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"Docker version 20.10.0, build abcdef\n"
        scheduler = create_scheduler(session_name="test_session")
        yield scheduler


def test_create_scheduler():
    with mock.patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"Docker version 20.10.0, build abcdef\n"
        scheduler = create_scheduler(session_name="test_session")
        assert isinstance(scheduler, PersistentDockerScheduler)
        assert scheduler.session_name == "test_session"


def test_submit_dryrun(docker_scheduler, mock_app_def, docker_executor):
    with mock.patch.object(DockerExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = docker_scheduler._submit_dryrun(mock_app_def, docker_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_check_docker_version_success():
    with mock.patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = b"Docker version 20.10.0, build abcdef\n"

        scheduler = create_scheduler(session_name="test_session")
        assert isinstance(scheduler, PersistentDockerScheduler)


def test_docker_scheduler_methods(docker_scheduler):
    # Test that basic methods exist
    assert hasattr(docker_scheduler, "_submit_dryrun")
    assert hasattr(docker_scheduler, "schedule")
    assert hasattr(docker_scheduler, "describe")
    assert hasattr(docker_scheduler, "log_iter")
    assert hasattr(docker_scheduler, "close")
