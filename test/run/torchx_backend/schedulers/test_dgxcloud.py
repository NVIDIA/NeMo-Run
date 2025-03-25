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
