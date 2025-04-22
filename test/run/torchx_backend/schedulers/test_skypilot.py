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

from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.run.torchx_backend.schedulers.skypilot import (
    SkypilotScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def skypilot_executor():
    return SkypilotExecutor(
        job_dir=tempfile.mkdtemp(),
        gpus="V100",
        gpus_per_node=1,
        cloud="aws",
    )


@pytest.fixture
def skypilot_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, SkypilotScheduler)
    assert scheduler.session_name == "test_session"


def test_skypilot_scheduler_methods(skypilot_scheduler):
    # Test that basic methods exist
    assert hasattr(skypilot_scheduler, "_submit_dryrun")
    assert hasattr(skypilot_scheduler, "schedule")
    assert hasattr(skypilot_scheduler, "describe")
    assert hasattr(skypilot_scheduler, "_validate")


def test_submit_dryrun(skypilot_scheduler, mock_app_def, skypilot_executor):
    with mock.patch.object(SkypilotExecutor, "package") as mock_package:
        mock_package.return_value = None

        dryrun_info = skypilot_scheduler._submit_dryrun(mock_app_def, skypilot_executor)
        assert isinstance(dryrun_info, AppDryRunInfo)
        assert dryrun_info.request is not None


def test_schedule(skypilot_scheduler, mock_app_def, skypilot_executor):
    class MockHandle:
        def get_cluster_name(self):
            return "test_cluster_name"

    with (
        mock.patch.object(SkypilotExecutor, "package") as mock_package,
        mock.patch.object(SkypilotExecutor, "launch") as mock_launch,
    ):
        mock_package.return_value = None
        mock_launch.return_value = (123, MockHandle())

        # Set job_name and experiment_id on executor
        skypilot_executor.job_name = "test_job"
        skypilot_executor.experiment_id = "test_session"

        dryrun_info = skypilot_scheduler._submit_dryrun(mock_app_def, skypilot_executor)
        app_id = skypilot_scheduler.schedule(dryrun_info)

        assert app_id == "test_session___test_cluster_name___test_role___123"
        mock_package.assert_called_once()
        mock_launch.assert_called_once()


def test_cancel_existing(skypilot_scheduler, skypilot_executor):
    with (
        mock.patch.object(SkypilotExecutor, "parse_app") as mock_parse_app,
        mock.patch.object(SkypilotExecutor, "cancel") as mock_cancel,
    ):
        mock_parse_app.return_value = ("test_cluster_name", "test_role", 123)

        skypilot_scheduler._cancel_existing("test_session___test_cluster_name___test_role___123")
        mock_cancel.assert_called_once_with(
            app_id="test_session___test_cluster_name___test_role___123"
        )


def test_validate(skypilot_scheduler, mock_app_def):
    # Test that validation doesn't raise any errors
    skypilot_scheduler._validate(mock_app_def, "skypilot")
