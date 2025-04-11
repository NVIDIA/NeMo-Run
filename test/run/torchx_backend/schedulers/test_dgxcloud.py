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

from nemo_run.core.execution.dgxcloud import DGXCloudExecutor
from nemo_run.run.torchx_backend.schedulers.dgxcloud import (
    DGXCloudScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(
        name="test_app", roles=[Role(name="test_role", image="nvcr.io/nvidia/nemo:latest")]
    )


@pytest.fixture
def dgx_cloud_executor():
    return DGXCloudExecutor(
        base_url="https://dgx.example.com",
        app_id="test_app_id",
        app_secret="test_secret",
        project_name="test_project",
        container_image="nvcr.io/nvidia/test:latest",
        pvc_nemo_run_dir="/workspace/nemo_run",
        job_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def dgx_cloud_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, DGXCloudScheduler)
    assert scheduler.session_name == "test_session"


def test_submit_dryrun(dgx_cloud_scheduler, mock_app_def, dgx_cloud_executor):
    # Mock any external calls that might be made
    with mock.patch.object(DGXCloudExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = dgx_cloud_scheduler._submit_dryrun(mock_app_def, dgx_cloud_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_dgx_cloud_scheduler_methods(dgx_cloud_scheduler):
    # Test that basic methods exist
    assert hasattr(dgx_cloud_scheduler, "_submit_dryrun")
    assert hasattr(dgx_cloud_scheduler, "schedule")
    assert hasattr(dgx_cloud_scheduler, "describe")
    assert hasattr(dgx_cloud_scheduler, "_cancel_existing")
    assert hasattr(dgx_cloud_scheduler, "_validate")


def test_schedule(dgx_cloud_scheduler, mock_app_def, dgx_cloud_executor):
    with (
        mock.patch.object(DGXCloudExecutor, "package") as mock_package,
        mock.patch.object(DGXCloudExecutor, "launch") as mock_launch,
    ):
        mock_package.return_value = None
        mock_launch.return_value = ("test_job_id", "RUNNING")

        # Set job_name and experiment_id on executor
        dgx_cloud_executor.job_name = "test_job"
        dgx_cloud_executor.experiment_id = "test_experiment"

        dryrun_info = dgx_cloud_scheduler._submit_dryrun(mock_app_def, dgx_cloud_executor)
        app_id = dgx_cloud_scheduler.schedule(dryrun_info)

        assert app_id == "test_experiment___test_role___test_job_id"
        mock_package.assert_called_once()
        mock_launch.assert_called_once()


def test_describe(dgx_cloud_scheduler, dgx_cloud_executor):
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.dgxcloud._get_job_dirs"
        ) as mock_get_job_dirs,
        mock.patch.object(DGXCloudExecutor, "status") as mock_status,
    ):
        mock_get_job_dirs.return_value = {
            "test_experiment___test_role___test_job_id": {
                "job_status": "RUNNING",
                "executor": dgx_cloud_executor,
            }
        }
        mock_status.return_value = "RUNNING"

        response = dgx_cloud_scheduler.describe("test_experiment___test_role___test_job_id")
        assert response is not None
        assert response.app_id == "test_experiment___test_role___test_job_id"
        assert len(response.roles) == 1
        assert response.roles[0].name == "test_role"


def test_cancel_existing(dgx_cloud_scheduler, dgx_cloud_executor):
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.dgxcloud._get_job_dirs"
        ) as mock_get_job_dirs,
        mock.patch.object(DGXCloudExecutor, "cancel") as mock_cancel,
    ):
        mock_get_job_dirs.return_value = {
            "test_experiment___test_role___test_job_id": {
                "job_status": "RUNNING",
                "executor": dgx_cloud_executor,
            }
        }

        dgx_cloud_scheduler._cancel_existing("test_experiment___test_role___test_job_id")
        mock_cancel.assert_called_once_with("test_job_id")


def test_save_and_get_job_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        from nemo_run.config import set_nemorun_home

        set_nemorun_home(temp_dir)

        from nemo_run.run.torchx_backend.schedulers.dgxcloud import _get_job_dirs, _save_job_dir

        executor = DGXCloudExecutor(
            base_url="https://test.com",
            app_id="test_id",
            app_secret="test_secret",
            project_name="test_project",
            container_image="test:latest",
            job_dir=temp_dir,
            pvc_nemo_run_dir="/workspace/nemo_run",
        )

        _save_job_dir("test_app_id", "RUNNING", executor)
        job_dirs = _get_job_dirs()

        assert "test_app_id" in job_dirs
        assert isinstance(job_dirs["test_app_id"]["executor"], DGXCloudExecutor)
