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

from nemo_run.core.execution.runai import RunAIExecutor
from nemo_run.run.torchx_backend.schedulers.runai import (
    RunAICloudScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(
        name="test_app", roles=[Role(name="test_role", image="nvcr.io/nvidia/nemo:latest")]
    )


@pytest.fixture
def runai_executor():
    return RunAIExecutor(
        base_url="https://runai.example.com",
        app_id="test_app_id",
        app_secret="test_secret",
        project_name="test_project",
        container_image="nvcr.io/nvidia/test:latest",
        pvc_nemo_run_dir="/workspace/nemo_run",
        job_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def runai_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, RunAICloudScheduler)
    assert scheduler.session_name == "test_session"


def test_submit_dryrun(runai_scheduler, mock_app_def, runai_executor):
    # Mock any external calls that might be made
    with mock.patch.object(RunAIExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = runai_scheduler._submit_dryrun(mock_app_def, runai_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_runai_scheduler_methods(runai_scheduler):
    # Test that basic methods exist
    assert hasattr(runai_scheduler, "_submit_dryrun")
    assert hasattr(runai_scheduler, "schedule")
    assert hasattr(runai_scheduler, "describe")
    assert hasattr(runai_scheduler, "_cancel_existing")
    assert hasattr(runai_scheduler, "_validate")


def test_schedule(runai_scheduler, mock_app_def, runai_executor):
    with (
        mock.patch.object(RunAIExecutor, "package") as mock_package,
        mock.patch.object(RunAIExecutor, "launch") as mock_launch,
    ):
        mock_package.return_value = None
        mock_launch.return_value = ("test_job_id", "RUNNING")

        # Set job_name and experiment_id on executor
        runai_executor.job_name = "test_job"
        runai_executor.experiment_id = "test_experiment"

        dryrun_info = runai_scheduler._submit_dryrun(mock_app_def, runai_executor)
        app_id = runai_scheduler.schedule(dryrun_info)

        assert app_id == "test_experiment___test_role___test_job_id"
        mock_package.assert_called_once()
        mock_launch.assert_called_once()


def test_describe(runai_scheduler, runai_executor):
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.runai._get_job_dirs"
        ) as mock_get_job_dirs,
        mock.patch.object(RunAIExecutor, "status") as mock_status,
    ):
        mock_get_job_dirs.return_value = {
            "test_experiment___test_role___test_job_id": {
                "job_status": "RUNNING",
                "executor": runai_executor,
            }
        }
        mock_status.return_value = "RUNNING"

        response = runai_scheduler.describe("test_experiment___test_role___test_job_id")
        assert response is not None
        assert response.app_id == "test_experiment___test_role___test_job_id"
        assert len(response.roles) == 1
        assert response.roles[0].name == "test_role"


def test_cancel_existing(runai_scheduler, runai_executor):
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.runai._get_job_dirs"
        ) as mock_get_job_dirs,
        mock.patch.object(RunAIExecutor, "cancel") as mock_cancel,
    ):
        mock_get_job_dirs.return_value = {
            "test_experiment___test_role___test_job_id": {
                "job_status": "RUNNING",
                "executor": runai_executor,
            }
        }

        runai_scheduler._cancel_existing("test_experiment___test_role___test_job_id")
        mock_cancel.assert_called_once_with("test_job_id")


def test_save_and_get_job_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        from nemo_run.config import set_nemorun_home

        set_nemorun_home(temp_dir)

        from nemo_run.run.torchx_backend.schedulers.runai import _get_job_dirs, _save_job_dir

        executor = RunAIExecutor(
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
        assert isinstance(job_dirs["test_app_id"]["executor"], RunAIExecutor)
