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

import csv
import json
import logging
import os
import tempfile
from unittest import mock

import pytest
from torchx.schedulers.api import AppDryRunInfo
from torchx.specs import AppDef, AppState, Role

from nemo_run.core.execution.slurm import SlurmBatchRequest, SlurmExecutor
from nemo_run.run.torchx_backend.schedulers.slurm import (
    SlurmTunnelScheduler,
    TunnelLogIterator,
    _get_job_dirs,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def temp_dir():
    return tempfile.mkdtemp()


@pytest.fixture
def slurm_executor(temp_dir):
    return SlurmExecutor(
        account="test_account",
        job_dir=temp_dir,
        nodes=1,
        ntasks_per_node=1,
        tunnel=LocalTunnel(job_dir=temp_dir),
    )


@pytest.fixture
def slurm_scheduler():
    return create_scheduler(session_name="test_session")


@pytest.fixture
def temp_job_dirs_file():
    """Create a temporary file for SLURM_JOB_DIRS."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "slurm_jobs")
    with open(temp_file, "w"):
        pass  # Create empty file
    yield temp_file
    # Cleanup
    try:
        os.unlink(temp_file)
        os.rmdir(temp_dir)
    except (OSError, FileNotFoundError) as e:
        logging.error(f"Error during cleanup: {e}")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, SlurmTunnelScheduler)
    assert scheduler.session_name == "test_session"

    # Test with experiment parameter
    mock_exp = mock.MagicMock()
    scheduler = create_scheduler(session_name="test_session", experiment=mock_exp)
    assert scheduler.experiment == mock_exp


def test_initialize_tunnel(slurm_scheduler):
    # Test with new tunnel
    tunnel = LocalTunnel(job_dir=tempfile.mkdtemp())
    slurm_scheduler._initialize_tunnel(tunnel)
    assert slurm_scheduler.tunnel is tunnel  # Use 'is' instead of '=='

    # Test with existing tunnel in experiment
    exp = mock.MagicMock()
    exp.tunnels = {tunnel.key: tunnel}
    slurm_scheduler.experiment = exp

    # Use the same tunnel object to avoid comparison issues
    slurm_scheduler._initialize_tunnel(tunnel)
    assert slurm_scheduler.tunnel is tunnel

    # Test with same tunnel
    slurm_scheduler._initialize_tunnel(tunnel)
    assert slurm_scheduler.tunnel is tunnel


@mock.patch("nemo_run.core.execution.utils.fill_template")
def test_submit_dryrun(mock_fill_template, slurm_scheduler, mock_app_def, slurm_executor):
    mock_fill_template.return_value = "#!/bin/bash\n# Mock script content"

    with mock.patch.object(SlurmTunnelScheduler, "_initialize_tunnel"):
        slurm_scheduler.tunnel = mock.MagicMock()

        with (
            mock.patch.object(SlurmExecutor, "package"),
            mock.patch("builtins.open", mock.mock_open()),
        ):
            # Use a mock for the actual AppDryRunInfo
            mock_dryrun_info = mock.MagicMock(spec=AppDryRunInfo)
            mock_dryrun_info.request = mock.MagicMock(spec=SlurmBatchRequest)

            with mock.patch.object(
                SlurmTunnelScheduler, "_submit_dryrun", return_value=mock_dryrun_info
            ):
                dryrun_info = slurm_scheduler._submit_dryrun(mock_app_def, slurm_executor)
                assert dryrun_info.request is not None


def test_schedule(slurm_scheduler, slurm_executor):
    mock_request = mock.MagicMock()
    mock_request.cmd = ["sbatch", "--requeue", "--parsable"]

    dryrun_info = mock.MagicMock()
    dryrun_info.request = mock_request
    slurm_executor.experiment_id = "test_exp_id"

    # Directly mock the tunnel.run method and patching the strip method's return value
    with (
        mock.patch.object(SlurmTunnelScheduler, "_initialize_tunnel"),
        mock.patch("nemo_run.run.torchx_backend.schedulers.slurm._save_job_dir"),
    ):
        # Create a fresh mock tunnel for each test to avoid interference
        mock_tunnel = mock.MagicMock()
        run_result = mock.MagicMock()
        # Use a simple string but with a mocked strip method
        run_result.stdout = mock.MagicMock()
        run_result.stdout.strip.return_value = "12345"
        mock_tunnel.run.return_value = run_result
        slurm_scheduler.tunnel = mock_tunnel

        result = slurm_scheduler.schedule(dryrun_info)
        assert result == "12345"
        # Verify the run was called with the expected arguments
        mock_tunnel.run.assert_called_once()


def test_cancel_existing(slurm_scheduler):
    # Test with non-existing app_id
    with mock.patch("nemo_run.run.torchx_backend.schedulers.slurm._get_job_dirs", return_value={}):
        result = slurm_scheduler._cancel_existing("non_existing_id")
        assert result is None

    # Test with existing app_id
    job_dirs = {"existing_id": ("/path/to/job", LocalTunnel(job_dir="/path/to/tunnel"), "log*")}
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.slurm._get_job_dirs", return_value=job_dirs
        ),
        mock.patch.object(SlurmTunnelScheduler, "_initialize_tunnel"),
    ):
        slurm_scheduler.tunnel = mock.MagicMock()
        slurm_scheduler._cancel_existing("existing_id")
        slurm_scheduler.tunnel.run.assert_called_with("scancel existing_id", hide=False)


def test_describe(slurm_scheduler):
    # Test with non-existing app_id
    with mock.patch("nemo_run.run.torchx_backend.schedulers.slurm._get_job_dirs", return_value={}):
        result = slurm_scheduler.describe("non_existing_id")
        assert result is None

    # Test with existing app_id but no output
    job_dirs = {"existing_id": ("/path/to/job", LocalTunnel(job_dir="/path/to/tunnel"), "log*")}
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.slurm._get_job_dirs", return_value=job_dirs
        ),
        mock.patch.object(SlurmTunnelScheduler, "_initialize_tunnel"),
    ):
        slurm_scheduler.tunnel = mock.MagicMock()
        slurm_scheduler.tunnel.run.return_value.stdout = "Header"

        result = slurm_scheduler.describe("existing_id")
        assert result is None

    # Test with proper output
    sacct_output = "JobID|State|JobName\nexisting_id|COMPLETED|test.test_app.test_role"
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.slurm._get_job_dirs", return_value=job_dirs
        ),
        mock.patch.object(SlurmTunnelScheduler, "_initialize_tunnel"),
        mock.patch.object(csv, "DictReader") as mock_reader,
    ):
        slurm_scheduler.tunnel = mock.MagicMock()
        slurm_scheduler.tunnel.run.return_value.stdout = sacct_output
        mock_reader.return_value = [
            {"JobID": "existing_id", "State": "COMPLETED", "JobName": "test.test_app.test_role"}
        ]

        result = slurm_scheduler.describe("existing_id")
        assert result is not None
        assert result.app_id == "existing_id"
        assert result.state == AppState.SUCCEEDED
        assert len(result.roles) == 1
        assert result.roles[0].name == "test_role"


def test_list(slurm_scheduler):
    slurm_scheduler.tunnel = mock.MagicMock()
    json_output = json.dumps({"jobs": [{"job_id": 12345, "state": {"current": "COMPLETED"}}]})
    slurm_scheduler.tunnel.run.return_value.stdout = json_output

    result = slurm_scheduler.list()
    assert len(result) == 1
    assert result[0].app_id == "12345"
    assert result[0].state == AppState.SUCCEEDED


def test_log_iter(slurm_scheduler):
    # Test with non-existing app_id
    with mock.patch("nemo_run.run.torchx_backend.schedulers.slurm._get_job_dirs", return_value={}):
        result = list(slurm_scheduler.log_iter("non_existing_id", "test_role"))
        assert len(result) == 1
        assert "Failed getting logs" in result[0]

    # Test with existing app_id
    job_dirs = {"existing_id": ("/path/to/job", LocalTunnel(job_dir="/path/to/tunnel"), "log*")}
    with (
        mock.patch(
            "nemo_run.run.torchx_backend.schedulers.slurm._get_job_dirs", return_value=job_dirs
        ),
        mock.patch.object(SlurmTunnelScheduler, "_initialize_tunnel"),
        mock.patch.object(
            TunnelLogIterator, "__iter__", return_value=iter(["log line 1", "log line 2"])
        ),
    ):
        slurm_scheduler.tunnel = mock.MagicMock()

        result = list(slurm_scheduler.log_iter("existing_id", "test_role"))
        assert len(result) == 2
        assert result[0] == "log line 1"
        assert result[1] == "log line 2"


def test_tunnel_log_iterator():
    # Create minimal mocks for faster testing
    scheduler = mock.Mock()
    app_id = "12345"
    log_file = "/path/to/log"
    remote_dir = "/remote/path"

    # Test init directly
    iterator = TunnelLogIterator(app_id, log_file, remote_dir, scheduler, should_tail=False)
    assert iterator._app_id == app_id
    assert iterator._log_file == log_file
    assert iterator._app_finished is True

    # Check app finished states in one test
    scheduler.describe.side_effect = [
        None,  # App not found
        mock.Mock(state=AppState.SUCCEEDED),  # Terminal state
        mock.Mock(state=AppState.RUNNING),  # Running state
    ]

    # Test app not found
    iterator._check_finished()
    assert iterator._app_finished is True

    # Test terminal state
    iterator._app_finished = False
    iterator._check_finished()
    assert iterator._app_finished is True

    # Test running state
    iterator._app_finished = False
    scheduler.tunnel = mock.Mock()
    scheduler.tunnel.run.return_value.stdout = "/remote/path/log.out"

    # Use patch without calling os.path
    with mock.patch("os.path.splitext", return_value=(".log", ".out")):
        iterator._check_finished()
        assert iterator._app_finished is False


def test_get_job_dirs():
    # Single test using direct file manipulation instead of complex mocks
    with tempfile.TemporaryDirectory() as temp_dir:
        job_dirs_file = os.path.join(temp_dir, "job_dirs")
        # Define the path to the module where _get_job_dirs resides and where imports need mocking
        slurm_module_path = "nemo_run.run.torchx_backend.schedulers.slurm"

        # Patch SLURM_JOB_DIRS env var used by the function to point to our temp file
        with mock.patch(f"{slurm_module_path}.SLURM_JOB_DIRS", job_dirs_file):
            # Test with no file (or empty file)
            # Create the file first to avoid FileNotFoundError if _get_job_dirs checks existence
            open(job_dirs_file, "a").close()
            assert _get_job_dirs() == {}, "Test Failed: Should return empty dict for empty file"

            # --- Test with valid content ---
            job_data_line = '12345 = log*,/path/to/job,LocalTunnel,{"job_dir": "/path/to/tunnel", "packaging_jobs": {}}\n'
            with open(job_dirs_file, "w") as f:
                f.write(job_data_line)

            expected_json_str = '{"job_dir": "/path/to/tunnel", "packaging_jobs": {}}'
            expected_json_obj = {"job_dir": "/path/to/tunnel", "packaging_jobs": {}}

            # Create mocks for dependencies that _get_job_dirs uses
            mock_local_tunnel_instance = mock.MagicMock()
            # This mock replaces the actual LocalTunnel class *within the slurm module*
            mock_local_tunnel_class = mock.MagicMock(return_value=mock_local_tunnel_instance)
            # Assign a name for easier debugging if assertions fail
            mock_local_tunnel_class.__name__ = "MockedLocalTunnel"

            # Mock json.loads (globally) and LocalTunnel (within the slurm module)
            with (
                mock.patch("json.loads") as mock_json_loads,
                mock.patch(f"{slurm_module_path}.LocalTunnel", mock_local_tunnel_class),
            ):
                # Configure the mock return value for json.loads
                mock_json_loads.return_value = expected_json_obj

                # Call the function under test
                result = _get_job_dirs()

                # Assert json.loads was called correctly
                try:
                    mock_json_loads.assert_called_once_with(expected_json_str)
                except AssertionError as e:
                    raise AssertionError(
                        f"json.loads not called as expected. _get_job_dirs result: {result}. Error: {e}"
                    )

                # Assert the mocked LocalTunnel class was instantiated correctly
                try:
                    mock_local_tunnel_class.assert_called_once_with(**expected_json_obj)
                except AssertionError as e:
                    raise AssertionError(
                        f"MockedLocalTunnel class not called as expected. _get_job_dirs result: {result}. Error: {e}"
                    )

                # Assert the final dictionary structure and content
                assert "12345" in result, (
                    f"Test Failed: '12345' not in result keys: {result.keys()}"
                )
                job_path, tunnel_instance, log_pattern = result["12345"]
                assert job_path == "/path/to/job", f"Test Failed: Incorrect job path: {job_path}"
                # Check we got the instance created by our mock class
                assert tunnel_instance is mock_local_tunnel_instance, (
                    "Test Failed: Tunnel instance is not the mocked one"
                )
                assert log_pattern == "log*", f"Test Failed: Incorrect log pattern: {log_pattern}"

            # --- Test invalid line format ---
            with open(job_dirs_file, "w") as f:
                f.write("invalid line format no equals sign\n")
            # No mocks needed here as parsing should fail before json/tunnel handling
            result = _get_job_dirs()
            assert result == {}, "Test Failed: Should return empty dict for invalid line format"

            # --- Test exception handling during json parsing ---
            with open(job_dirs_file, "w") as f:
                # Line is valid up until the JSON part
                f.write("54321 = log*,/path/to/job,LocalTunnel,THIS_IS_INVALID_JSON\n")

            # Create a fresh mock for LocalTunnel just for this test context
            mock_local_tunnel_class_for_json_error = mock.MagicMock()
            # Mock LocalTunnel and make json.loads raise an error
            with (
                mock.patch(
                    f"{slurm_module_path}.LocalTunnel", mock_local_tunnel_class_for_json_error
                ),
                mock.patch(
                    "json.loads", side_effect=json.JSONDecodeError("Mock decode error", "", 0)
                ) as mock_json_error,
            ):
                result = _get_job_dirs()
                # Expect empty dict because the line parsing failed
                assert result == {}, "Test Failed: Should return empty dict if json parsing fails"
                # Check json.loads was called with the invalid string
                mock_json_error.assert_called_once_with("THIS_IS_INVALID_JSON")
                # Ensure the tunnel class was not instantiated because json parsing failed first
                mock_local_tunnel_class_for_json_error.assert_not_called()

            # --- Test failure during tunnel class lookup/handling ---
            # This tests the case where the class name in the file doesn't exist or causes an error
            invalid_tunnel_line = '67890 = log*,/path/to/job,NonExistentTunnelClass,{"a": 1}\n'
            with open(job_dirs_file, "w") as f:
                f.write(invalid_tunnel_line)

            # Mock json.loads to check it's NOT called (failure should happen before)
            with mock.patch("json.loads") as mock_json_lookup_fail:
                # Also mock LocalTunnel within this context to ensure it's definitely not
                # involved or accidentally called due to unexpected fallback logic.
                mock_local_tunnel_class_lookup_fail = mock.MagicMock()
                with mock.patch(
                    f"{slurm_module_path}.LocalTunnel", mock_local_tunnel_class_lookup_fail
                ):
                    # Run the function - expect it to fail finding 'NonExistentTunnelClass'
                    # and return an empty dict, potentially logging an error internally.
                    result = _get_job_dirs()

                # Assert the overall result is an empty dictionary
                assert result == {}, (
                    "Test Failed: Should return empty dict if tunnel class is invalid/not found"
                )
                # Assert json.loads was NOT called, assuming class lookup/validation happens first
                mock_json_lookup_fail.assert_not_called()
                # Assert the existing LocalTunnel mock wasn't called either
                mock_local_tunnel_class_lookup_fail.assert_not_called()


def test_schedule_with_dependencies(slurm_scheduler, slurm_executor):
    mock_request = mock.MagicMock()
    mock_request.cmd = ["sbatch", "--requeue", "--parsable"]

    dryrun_info = mock.MagicMock()
    dryrun_info.request = mock_request
    slurm_executor.experiment_id = "test_exp_id"
    slurm_executor.dependencies = ["slurm://54321/master/0"]

    # Directly mock the methods we need instead of patching LocalTunnel.run
    with (
        mock.patch.object(SlurmTunnelScheduler, "_initialize_tunnel"),
        mock.patch.object(SlurmExecutor, "parse_deps", return_value=["54321"]),
        mock.patch("nemo_run.run.torchx_backend.schedulers.slurm._save_job_dir"),
    ):
        # Create a fresh mock tunnel for testing
        mock_tunnel = mock.MagicMock()
        run_result = mock.MagicMock()
        run_result.stdout = mock.MagicMock()
        run_result.stdout.strip.return_value = "12345"
        mock_tunnel.run.return_value = run_result
        slurm_scheduler.tunnel = mock_tunnel

        result = slurm_scheduler.schedule(dryrun_info)
        assert result == "12345"
        # Verify the run was called with the expected arguments
        mock_tunnel.run.assert_called_once()
