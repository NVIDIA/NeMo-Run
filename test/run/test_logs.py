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

import io
import queue
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from torchx.specs.api import AppState, AppStatus, Role

from nemo_run.core.execution.base import Executor
from nemo_run.run import logs
from nemo_run.run.torchx_backend.runner import Runner
from nemo_run.run.torchx_backend.schedulers.api import (
    REVERSE_EXECUTOR_MAPPING,
)


class MockExecutorNoLogs(Executor):
    def __init__(self, executor_str: str):
        self.executor_str = executor_str

    def __str__(self):
        return self.executor_str


class MockExecutor(Executor):
    def __init__(self, executor_str: str):
        self.executor_str = executor_str

    def __str__(self):
        return self.executor_str

    def logs(self, app_id: str, fallback_path: Optional[str]): ...


@pytest.fixture
def mock_runner() -> Runner:
    return MagicMock(spec=Runner)


@pytest.fixture
def mock_status() -> AppStatus:
    return MagicMock(spec=AppStatus, state="COMPLETED")


@pytest.fixture
def mock_app() -> MagicMock:
    mock = MagicMock()
    mock.roles = [Role(name="master", image="")]
    return mock


def test_print_log_lines_with_log_supported_executor(mock_runner: Runner, mock_status: AppStatus):
    executor_cls = MockExecutor
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = MockExecutor
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = MagicMock(
        spec=AppState
    )  # assuming app_handle and role_name are correctly passed
    que = queue.Queue()
    with patch.object(executor_cls, "logs", return_value=None) as mock_logs:
        logs.print_log_lines(
            io.StringIO(),
            mock_runner,
            "dummy_backend://nemo_run/12345",
            "main",
            0,
            "",
            False,
            que,
            None,
            None,
        )
        mock_logs.assert_called_once_with("12345", fallback_path=None)

        logs.print_log_lines(
            io.StringIO(),
            mock_runner,
            "dummy_backend://nemo_run/12345",
            "main",
            0,
            "",
            False,
            que,
            None,
            log_path="test_path",
        )
        mock_logs.assert_called_with("12345", fallback_path="test_path")


def test_print_log_lines_with_unsupported_executor(
    mock_runner: Runner, mock_status: AppStatus, capsys
):
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = MagicMock(
        spec=AppState
    )  # assuming app_handle and role_name are correctly passed
    mock_runner.log_lines.return_value = ["test_line"]
    que = queue.Queue()
    logs.print_log_lines(
        sys.stderr,
        mock_runner,
        "dummy_backend://nemo_run/12345",
        "main",
        0,
        "",
        False,
        que,
        None,
        None,
    )
    captured = capsys.readouterr()
    assert "main/0 test_line" in captured.err


def test_print_log_lines_with_exception(mock_runner, mock_status):
    que = queue.Queue()
    with patch("nemo_run.run.logs.parse_app_handle", side_effect=Exception("Parse Error")):
        with pytest.raises(Exception):
            logs.print_log_lines(
                io.StringIO(),
                mock_runner,
                "example://app_id",
                "master",
                0,
                "",
                False,
                que,
                None,
                None,
            )
    assert not que.empty()
    exception = que.get()
    assert isinstance(exception, Exception)
    assert "Parse Error" in str(exception)


def test_get_logs_without_running_app(mock_runner: Runner, capsys):
    mock_runner.status.return_value = None
    with pytest.raises(SystemExit):
        logs.get_logs(
            sys.stderr,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            mock_runner,
            wait_timeout=0,
        )
    captured = capsys.readouterr()
    assert "Waiting for app state response before fetching logs..." in captured.out


def test_get_logs_with_invalid_role(mock_runner: Runner, mock_app: MagicMock, capsys):
    mock_runner.describe.return_value = mock_app
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls
    mock_runner.status.return_value = MagicMock(spec=AppStatus, state="RUNNING")
    with patch("nemo_run.run.logs.find_role_replicas", return_value=[]):
        with pytest.raises(SystemExit):
            logs.get_logs(
                sys.stderr,
                "dummy_backend://nemo_run/12345",
                None,
                False,
                mock_runner,
                wait_timeout=0,
            )
    captured = capsys.readouterr()
    assert "No role [None] found for app" in captured.out


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_get_logs_exception_handling(mock_runner, mock_status, mock_app):
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = mock_app
    mock_app.roles = [
        Role("main", image=""),
        Role("worker", image=""),
    ]
    with patch("nemo_run.run.logs.parse_app_handle", side_effect=Exception("Log Error")):
        with pytest.raises(Exception):
            logs.get_logs(
                sys.stdout,
                "dummy_backend://nemo_run/12345",
                None,
                False,
                mock_runner,
                wait_timeout=0,
            )


def test_get_logs_calls_print_log_lines(mock_runner, mock_status, mock_app):
    mock_runner.status.return_value = mock_status
    mock_runner.describe.return_value = mock_app
    mock_runner.log_lines.return_value = ["test_line"]
    executor_cls = MockExecutorNoLogs
    REVERSE_EXECUTOR_MAPPING["dummy_backend"] = executor_cls

    mock_app.roles = [
        Role("main", image=""),
        Role("worker", image=""),
    ]
    with patch("nemo_run.run.logs.print_log_lines") as mock_print_log_lines:
        logs.get_logs(
            sys.stderr,
            "dummy_backend://nemo_run/12345",
            None,
            False,
            mock_runner,
            wait_timeout=0,
        )
        roles_and_replicas = [
            ("main", 0),
            ("worker", 1),
        ]
        assert mock_print_log_lines.call_count == len(roles_and_replicas)
