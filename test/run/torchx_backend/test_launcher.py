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

import contextvars
import sys
import threading
from unittest.mock import MagicMock, Mock, patch

import pytest
from torchx.schedulers.api import Stream
from torchx.specs import AppDef, AppStatus

from nemo_run.core.execution.base import Executor
from nemo_run.exceptions import UnknownStatusError
from nemo_run.run.logs import get_logs
from nemo_run.run.torchx_backend.launcher import ContextThread, launch, wait_and_exit


@pytest.fixture
def mock_runner():
    return Mock()


@pytest.fixture
def mock_executor():
    return Mock(spec=Executor)


@pytest.fixture
def mock_executable():
    return Mock(spec=AppDef)


@pytest.fixture
def setup_and_teardown():
    test_var = contextvars.ContextVar("test_var")
    token = test_var.set("original")
    yield test_var
    test_var.reset(token)


def test_launch_dryrun(mock_runner, mock_executor, mock_executable):
    dryrun_info = "Dryrun Info"
    mock_runner.dryrun.return_value = dryrun_info

    result = launch(
        mock_executable, "test_executor", mock_executor, dryrun=True, runner=mock_runner
    )

    mock_runner.dryrun.assert_called_once_with(
        mock_executable, "test_executor", cfg=mock_executor, parent_run_id=None
    )
    assert result == (None, None)


def test_launch_non_dryrun(mock_runner, mock_executor, mock_executable):
    mock_app_handle = "dummy://nemo_run/my-test-run"
    mock_runner.run.return_value = mock_app_handle

    result = launch(
        mock_executable,
        "test_executor",
        mock_executor,
        dryrun=False,
        runner=mock_runner,
    )

    mock_runner.run.assert_called_once_with(
        mock_executable, "test_executor", cfg=mock_executor, parent_run_id=None
    )
    assert result[0] == mock_app_handle


def test_launch_wait(mock_runner, mock_executor, mock_executable):
    mock_app_handle = "dummy://nemo_run/my-test-run"
    mock_runner.run.return_value = mock_app_handle
    mock_runner.status.return_value = MagicMock(spec=AppStatus, state="RUNNING")
    mock_runner.wait.return_value = MagicMock(spec=AppStatus, state="SUCCEEDED")

    result = launch(
        mock_executable,
        "test_executor",
        mock_executor,
        dryrun=False,
        wait=True,
        runner=mock_runner,
    )

    mock_runner.wait.assert_called_once_with(mock_app_handle, wait_interval=2)
    assert result[1].state == "SUCCEEDED"


def test_wait_and_exit_success(mock_runner):
    mock_app_handle = "dummy://nemo_run/my-test-run"
    mock_runner.wait.return_value = MagicMock(spec=AppStatus, state="SUCCEEDED")

    result = wait_and_exit(app_handle=mock_app_handle, log=False, runner=mock_runner)

    mock_runner.wait.assert_called()
    assert result.state == "SUCCEEDED"


def test_wait_and_exit_timeout(mock_runner):
    mock_app_handle = "dummy://nemo_run/my-test-run"
    mock_runner.wait.return_value = None

    with pytest.raises(UnknownStatusError):
        wait_and_exit(app_handle=mock_app_handle, log=False, runner=mock_runner, timeout=0)


@patch("nemo_run.run.torchx_backend.launcher.ContextThread")
def test_wait_and_exit_log_thread_started(mock_thread, mock_runner):
    mock_app_handle = "dummy://nemo_run/my-test-run"
    mock_runner.wait.return_value = MagicMock(spec=AppStatus, state="SUCCEEDED")

    wait_and_exit(app_handle=mock_app_handle, log=True, runner=mock_runner)

    mock_thread.assert_called_once_with(
        target=get_logs,
        kwargs={
            "file": sys.stdout,
            "runner": mock_runner,
            "identifier": mock_app_handle,
            "regex": None,
            "should_tail": True,
            "streams": Stream.COMBINED,
        },
    )


@patch("nemo_run.run.torchx_backend.launcher.ContextThread")
def test_wait_and_exit_log_thread_not_started(mock_thread, mock_runner):
    mock_app_handle = "dummy://nemo_run/my-test-run"
    mock_runner.wait.return_value = MagicMock(spec=AppStatus, state="SUCCEEDED")
    wait_and_exit(app_handle=mock_app_handle, log=False, runner=mock_runner)
    mock_thread.assert_not_called()


def test_context_thread_init(setup_and_teardown):
    def test_function():
        assert setup_and_teardown.get() == "original"

        thread = ContextThread(target=test_function)
        assert isinstance(thread, threading.Thread)
        assert hasattr(thread, "ctx")
        assert isinstance(thread.ctx, contextvars.Context)


@patch("threading.Thread.run")
def test_context_thread_run(mocked_run, setup_and_teardown):
    def test_function():
        assert setup_and_teardown.get() == "test"

        setup_and_teardown.set("test")
        thread = ContextThread(target=test_function)
        thread.start()
        mocked_run.assert_called_once()
