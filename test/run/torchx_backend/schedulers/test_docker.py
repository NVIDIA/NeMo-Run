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
    DockerContainer,
    DockerJobRequest,
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


def test_schedule(docker_scheduler, mock_app_def, docker_executor):
    with (
        mock.patch.object(DockerExecutor, "package") as mock_package,
        mock.patch.object(DockerContainer, "run") as mock_run,
    ):
        mock_package.return_value = None
        mock_run.return_value = ("test_container_id", "RUNNING")

        # Set job_name on executor
        docker_executor.job_name = "test_job"

        dryrun_info = docker_scheduler._submit_dryrun(mock_app_def, docker_executor)
        docker_scheduler.schedule(dryrun_info)

        mock_package.assert_called_once()
        mock_run.assert_called_once()


def test_describe(docker_scheduler, docker_executor):
    with (
        mock.patch.object(DockerJobRequest, "load") as mock_load,
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
    ):
        mock_load.return_value = DockerJobRequest(
            id="test_session___test_role___test_container_id",
            executor=docker_executor,
            containers=[
                DockerContainer(
                    name="test_role",
                    command=["test"],
                    executor=docker_executor,
                    extra_env={},
                )
            ],
        )
        mock_get_container.return_value = None

        response = docker_scheduler.describe("test_session___test_role___test_container_id")
        assert response is not None
        assert response.app_id == "test_session___test_role___test_container_id"
        assert "SUCCEEDED" in str(response.state)
        assert len(response.roles) == 1


def test_save_and_get_job_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        from nemo_run.config import set_nemorun_home

        set_nemorun_home(temp_dir)

        from nemo_run.run.torchx_backend.schedulers.docker import DockerJobRequest

        executor = DockerExecutor(
            container_image="test:latest",
            job_dir=temp_dir,
        )

        req = DockerJobRequest(
            id="test_app_id",
            executor=executor,
            containers=[
                DockerContainer(
                    name="test_role",
                    command=["test"],
                    executor=executor,
                    extra_env={},
                )
            ],
        )
        req.save()

        loaded_req = DockerJobRequest.load("test_app_id")
        assert loaded_req is not None
        assert loaded_req.id == "test_app_id"
        assert isinstance(loaded_req.executor, DockerExecutor)


def test_run_opts(docker_scheduler):
    opts = docker_scheduler._run_opts()
    assert "copy_env" in str(opts)
    assert "env" in str(opts)
    assert "privileged" in str(opts)


def test_log_iter(docker_scheduler, docker_executor):
    with (
        mock.patch.object(DockerJobRequest, "load") as mock_load,
        mock.patch.object(DockerContainer, "get_container") as mock_get_container,
    ):
        mock_load.return_value = DockerJobRequest(
            id="test_session___test_role___test_container_id",
            executor=docker_executor,
            containers=[
                DockerContainer(
                    name="test_role",
                    command=["test"],
                    executor=docker_executor,
                    extra_env={},
                )
            ],
        )
        container_mock = mock.Mock()
        container_mock.logs = mock.Mock(return_value=["log1", "log2"])
        mock_get_container.return_value = container_mock

        logs = list(
            docker_scheduler.log_iter("test_session___test_role___test_container_id", "test_role")
        )
        assert logs == ["log1", "log2"]
        assert mock_get_container.call_count == 1
        assert container_mock.logs.call_count == 1


def test_close(docker_scheduler):
    with mock.patch.object(DockerContainer, "delete") as mock_delete:
        docker_scheduler._scheduled_reqs = []  # No requests to clean up
        docker_scheduler.close()
        mock_delete.assert_not_called()  # No cleanup needed since no requests
