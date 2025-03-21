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
from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse
from torchx.specs import AppDef, AppState, Role

from nemo_run.core.execution.local import LocalExecutor
from nemo_run.run.torchx_backend.schedulers.local import (
    PersistentLocalScheduler,
    _get_job_dirs,
    _save_job_dir,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def local_executor():
    return LocalExecutor(job_dir=tempfile.mkdtemp())


@pytest.fixture
def local_scheduler():
    return create_scheduler(session_name="test_session", cache_size=10)


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session", cache_size=10)
    assert isinstance(scheduler, PersistentLocalScheduler)
    assert scheduler.session_name == "test_session"
    assert scheduler._cache_size == 10


def test_submit_dryrun(local_scheduler, mock_app_def, local_executor):
    dryrun_info = local_scheduler._submit_dryrun(mock_app_def, local_executor)
    assert isinstance(dryrun_info, AppDryRunInfo)
    assert dryrun_info.request is not None
    # AppDryRunInfo has changed and no longer has a fmt attribute
    # assert callable(dryrun_info.fmt)


@mock.patch("nemo_run.run.torchx_backend.schedulers.local._save_job_dir")
def test_schedule(mock_save, local_scheduler, mock_app_def, local_executor):
    dryrun_info = local_scheduler._submit_dryrun(mock_app_def, local_executor)

    with mock.patch(
        "torchx.schedulers.local_scheduler.LocalScheduler.schedule"
    ) as mock_super_schedule:
        mock_super_schedule.return_value = "test_app_id"
        app_id = local_scheduler.schedule(dryrun_info)

        assert app_id == "test_app_id"
        mock_super_schedule.assert_called_once_with(dryrun_info=dryrun_info)
        mock_save.assert_called_once()


@mock.patch("nemo_run.run.torchx_backend.schedulers.local._save_job_dir")
def test_describe_existing_app(mock_save, local_scheduler):
    app_id = "test_app_id"
    expected_response = DescribeAppResponse()
    expected_response.app_id = app_id

    with mock.patch(
        "torchx.schedulers.local_scheduler.LocalScheduler.describe"
    ) as mock_super_describe:
        mock_super_describe.return_value = expected_response
        response = local_scheduler.describe(app_id)

        assert response == expected_response
        mock_super_describe.assert_called_once_with(app_id=app_id)
        mock_save.assert_called_once()


@mock.patch("nemo_run.run.torchx_backend.schedulers.local._get_job_dirs")
def test_describe_from_saved_apps(mock_get_job_dirs, local_scheduler):
    app_id = "test_app_id"

    # First simulate the app not in current apps
    with mock.patch(
        "torchx.schedulers.local_scheduler.LocalScheduler.describe"
    ) as mock_super_describe:
        mock_super_describe.return_value = None

        from torchx.schedulers.local_scheduler import _LocalAppDef

        mock_app_def = _LocalAppDef(id=app_id, log_dir="/tmp/test")
        mock_app_def.role_replicas = {"test_role": []}
        mock_app_def.set_state(AppState.SUCCEEDED)

        mock_get_job_dirs.return_value = {app_id: mock_app_def}

        response = local_scheduler.describe(app_id)

        assert response is not None
        assert response.app_id == app_id
        assert len(response.roles) == 1
        assert response.roles[0].name == "test_role"
        assert response.state == AppState.SUCCEEDED
        assert response.ui_url == "file:///tmp/test"


def test_log_iter_warns_on_since_until(local_scheduler):
    with mock.patch("warnings.warn") as mock_warn:
        with mock.patch.object(local_scheduler, "_apps", {"test_app_id": mock.MagicMock()}):
            with mock.patch("os.path.isfile", return_value=True):
                with mock.patch("nemo_run.run.torchx_backend.schedulers.local.LogIterator"):
                    # Call with since parameter
                    list(
                        local_scheduler.log_iter("test_app_id", "test_role", since=mock.MagicMock())
                    )
                    mock_warn.assert_called_once()

                    mock_warn.reset_mock()

                    # Call with until parameter
                    list(
                        local_scheduler.log_iter("test_app_id", "test_role", until=mock.MagicMock())
                    )
                    mock_warn.assert_called_once()


def test_save_and_get_job_dirs():
    from torchx.schedulers.local_scheduler import _LocalAppDef

    # Create a test app
    app_id = "test_app_id"
    app_def = _LocalAppDef(id=app_id, log_dir="/tmp/test")
    app_def.role_replicas = {"test_role": []}
    app_def.set_state(AppState.SUCCEEDED)

    test_apps = {app_id: app_def}

    # Create a temporary file to mock LOCAL_JOB_DIRS
    with tempfile.NamedTemporaryFile() as temp_file:
        with mock.patch(
            "nemo_run.run.torchx_backend.schedulers.local.LOCAL_JOB_DIRS", temp_file.name
        ):
            # Test _save_job_dir
            _save_job_dir(test_apps)

            # Test _get_job_dirs
            loaded_apps = _get_job_dirs()

            assert app_id in loaded_apps
            assert loaded_apps[app_id].id == app_id
            assert loaded_apps[app_id].log_dir == "/tmp/test"
            assert "test_role" in loaded_apps[app_id].role_replicas
            assert loaded_apps[app_id].state == AppState.SUCCEEDED
