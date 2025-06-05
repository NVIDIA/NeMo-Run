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

import json
import os
import threading
from unittest.mock import Mock, mock_open, patch

import pytest

from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.tunnel.client import SSHTunnel
from nemo_run.run.ray.slurm import (
    SlurmRayCluster,
    SlurmRayJob,
    cancel_slurm_job,
    get_last_job_id,
)

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "core", "execution", "artifacts"
)


class TestSlurmRayCluster:
    @pytest.fixture
    def mock_tunnel(self):
        """Create a mocked SSH tunnel."""
        tunnel = Mock(spec=SSHTunnel)
        tunnel.job_dir = "/tmp/test_jobs"
        tunnel.key = "test-host"
        tunnel.connect.return_value = None
        tunnel.run.return_value = Mock(stdout="", return_code=0)
        tunnel.put.return_value = None
        return tunnel

    @pytest.fixture
    def basic_executor(self, mock_tunnel):
        """Create a basic SlurmExecutor with mocked tunnel."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
        )
        executor.tunnel = mock_tunnel
        return executor

    @pytest.fixture
    def cluster(self, basic_executor):
        """Create a SlurmRayCluster instance."""
        return SlurmRayCluster(name="test-cluster", executor=basic_executor)

    def test_cluster_initialization(self, cluster):
        """Test cluster initialization."""
        assert cluster.name == "test-cluster"
        assert cluster.cluster_map == {}
        assert isinstance(cluster.executor, SlurmExecutor)

    def test_get_ray_cluster_info_success(self, cluster, mock_tunnel):
        """Test successful retrieval of Ray cluster info."""
        cluster_info = {"head_ip": "192.168.1.100", "dashboard_port": "8265", "port": "6379"}
        mock_tunnel.run.return_value = Mock(return_code=0, stdout=json.dumps(cluster_info))

        result = cluster._get_ray_cluster_info()

        assert result == cluster_info
        mock_tunnel.run.assert_called_once()

    def test_get_ray_cluster_info_file_not_found(self, cluster, mock_tunnel):
        """Test when cluster info file doesn't exist."""
        mock_tunnel.run.return_value = Mock(return_code=1, stdout="")

        result = cluster._get_ray_cluster_info()

        assert result == {}

    def test_get_ray_cluster_info_invalid_json(self, cluster, mock_tunnel):
        """Test when cluster info file contains invalid JSON."""
        mock_tunnel.run.return_value = Mock(return_code=0, stdout="invalid json content")

        result = cluster._get_ray_cluster_info()

        assert result == {}

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_status_job_running(self, mock_get_last_job_id, cluster, mock_tunnel):
        """Test status when job is running."""
        # Mock squeue to return job ID
        mock_tunnel.run.side_effect = [
            Mock(stdout="12345", return_code=0),  # squeue -n job_name
            Mock(stdout="RUNNING", return_code=0),  # squeue -j job_id
        ]

        # Mock ray cluster info
        with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
            mock_cluster_info.return_value = {"head_ip": "192.168.1.100"}

            status = cluster.status()

            assert status["state"] == "RUNNING"
            assert status["job_id"] == "12345"
            assert status["ray_ready"] is True

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_status_job_not_found(self, mock_get_last_job_id, cluster, mock_tunnel):
        """Test status when job is not found."""
        mock_tunnel.run.return_value = Mock(stdout="", return_code=0)  # squeue returns empty
        mock_get_last_job_id.return_value = None

        status = cluster.status()

        assert status["state"] == "NOT_FOUND"
        assert status["job_id"] is None
        assert status["ray_ready"] is False

    def test_status_with_display(self, cluster, mock_tunnel, caplog):
        """Test status with display flag."""
        mock_tunnel.run.side_effect = [
            Mock(stdout="12345", return_code=0),  # squeue -n job_name
            Mock(stdout="RUNNING", return_code=0),  # squeue -j job_id
        ]

        with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
            mock_cluster_info.return_value = {"head_ip": "192.168.1.100"}

            with caplog.at_level("INFO"):  # Capture INFO level logs
                cluster.status(display=True)

            # Check that status info was logged
            assert "Ray Cluster Status (Slurm)" in caplog.text
            assert "Host:" in caplog.text
            assert "Name:" in caplog.text

    @patch("tempfile.NamedTemporaryFile")
    def test_create_success(self, mock_tempfile, cluster, mock_tunnel):
        """Test successful cluster creation."""
        # Mock status to show no existing cluster
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": None}

            # Mock tempfile with proper fileno
            mock_file = Mock()
            mock_file.name = "/tmp/mock_script"
            mock_file.fileno.return_value = 1  # Return valid file descriptor
            mock_tempfile.return_value.__enter__.return_value = mock_file

            # Mock sbatch response
            mock_tunnel.run.side_effect = [
                Mock(stdout="", return_code=0),  # mkdir -p command
                Mock(stdout="12345", return_code=0),  # sbatch command
            ]

            with patch("os.fsync"):  # Mock fsync to avoid file descriptor issues
                job_id = cluster.create()

            assert job_id == "12345"
            assert cluster.cluster_map["test-cluster"] == "12345"
            mock_tunnel.put.assert_called()

    def test_create_cluster_already_exists(self, cluster):
        """Test creation when cluster already exists."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "state": "RUNNING"}

            job_id = cluster.create()

            assert job_id is None

    def test_create_dryrun(self, cluster, capsys):
        """Test dry run mode."""
        job_id = cluster.create(dryrun=True)

        captured = capsys.readouterr()
        assert "#SBATCH" in captured.out
        assert job_id is None

    @patch("time.time")
    @patch("time.sleep")
    def test_wait_until_running_success(self, mock_sleep, mock_time, cluster):
        """Test successful wait until running."""
        mock_time.return_value = 0  # Fixed time

        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"ray_ready": True}

            result = cluster.wait_until_running(timeout=10)

            assert result is True

    @patch("nemo_run.run.ray.slurm.cancel_slurm_job")
    def test_delete_success(self, mock_cancel, cluster):
        """Test successful cluster deletion."""
        mock_cancel.return_value = True

        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "state": "RUNNING"}

            result = cluster.delete()

            assert result is True
            mock_cancel.assert_called_once()

    def test_delete_already_completed(self, cluster):
        """Test deletion of already completed job."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "state": "COMPLETED"}

            result = cluster.delete()

            assert result is True

    @patch("socket.socket")
    @patch("subprocess.Popen")
    @patch("queue.Queue")
    def test_port_forward_success(self, mock_queue_class, mock_popen, mock_socket, cluster):
        """Test successful port forwarding setup."""
        # Mock cluster status
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "ray_ready": True}

            # Mock cluster info
            with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
                mock_cluster_info.return_value = {"head_ip": "192.168.1.100"}

                # Mock socket binding
                mock_sock = Mock()
                mock_socket.return_value = mock_sock

                # Mock subprocess
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                # Mock queue
                mock_queue = Mock()
                mock_queue.get.return_value = ("success", None)
                mock_queue_class.return_value = mock_queue

                thread = cluster.port_forward(port=8080, target_port=8265)

                assert isinstance(thread, threading.Thread)

    def test_port_forward_cluster_not_found(self, cluster):
        """Test port forwarding when cluster doesn't exist."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": None}

            with pytest.raises(RuntimeError, match="Could not find Ray cluster"):
                cluster.port_forward()

    def test_port_forward_ray_not_ready(self, cluster):
        """Test port forwarding when Ray is not ready."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "ray_ready": False}

            with pytest.raises(RuntimeError, match="Ray cluster .* is not running or not ready"):
                cluster.port_forward()

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_status_job_not_in_squeue_use_sacct(self, mock_get_last_job_id, cluster, mock_tunnel):
        """Test status when job not in squeue but found via sacct."""
        mock_tunnel.run.side_effect = [
            Mock(stdout="", return_code=0),  # squeue -n job_name (empty)
            Mock(stdout="", return_code=1),  # squeue -j job_id (not found)
            Mock(stdout="COMPLETED\n", return_code=0),  # sacct command
        ]
        mock_get_last_job_id.return_value = 12345

        status = cluster.status()

        assert status["state"] == "COMPLETED"
        assert status["job_id"] == "12345"
        assert status["ray_ready"] is True  # COMPLETED jobs are considered ray_ready

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_status_job_not_in_squeue_no_sacct_output(
        self, mock_get_last_job_id, cluster, mock_tunnel
    ):
        """Test status when job not in squeue and sacct returns no output."""
        mock_tunnel.run.side_effect = [
            Mock(stdout="", return_code=0),  # squeue -n job_name (empty)
            Mock(stdout="", return_code=1),  # squeue -j job_id (not found)
            Mock(stdout="", return_code=0),  # sacct command (no output)
        ]
        mock_get_last_job_id.return_value = 12345

        status = cluster.status()

        assert status["state"] == "UNKNOWN"
        assert status["job_id"] == "12345"
        assert status["ray_ready"] is False

    def test_status_job_found_by_name_stores_in_cluster_map(self, cluster, mock_tunnel):
        """Test that job_id found by name is stored in cluster_map."""
        mock_tunnel.run.side_effect = [
            Mock(stdout="67890", return_code=0),  # squeue -n job_name
            Mock(stdout="RUNNING", return_code=0),  # squeue -j job_id
        ]

        with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
            mock_cluster_info.return_value = {}

            status = cluster.status()

            assert status["job_id"] == "67890"
            assert cluster.cluster_map["test-cluster"] == "67890"

    def test_create_job_exists_with_warning_state(self, cluster, mock_tunnel):
        """Test creation when job exists in non-standard state."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "state": "SUSPENDED"}

            with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                mock_file = Mock()
                mock_file.name = "/tmp/mock_script"
                mock_file.fileno.return_value = 1
                mock_tempfile.return_value.__enter__.return_value = mock_file

                mock_tunnel.run.side_effect = [
                    Mock(stdout="", return_code=0),  # mkdir -p command
                    Mock(stdout="54321", return_code=0),  # sbatch command
                ]

                with patch("os.fsync"):
                    job_id = cluster.create()

                assert job_id == "54321"

    @patch("time.time")
    @patch("time.sleep")
    def test_wait_until_running_job_failed(self, mock_sleep, mock_time, cluster):
        """Test wait_until_running when job fails."""
        mock_time.return_value = 0

        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"ray_ready": False, "state": "FAILED"}

            result = cluster.wait_until_running(timeout=10)

            assert result is False

    def test_delete_job_already_terminal_state(self, cluster):
        """Test delete when job is already in terminal state."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "state": "CANCELLED"}

            result = cluster.delete()

            assert result is True
            # Should not be in cluster_map anymore
            assert "test-cluster" not in cluster.cluster_map

    def test_port_forward_no_head_ip_in_cluster_info(self, cluster):
        """Test port forwarding when cluster info lacks head_ip."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "ray_ready": True}

            with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
                mock_cluster_info.return_value = {"some_other_key": "value"}

                with pytest.raises(RuntimeError, match="does not contain head_ip"):
                    cluster.port_forward()

    def test_port_forward_empty_cluster_info(self, cluster):
        """Test port forwarding when cluster info is empty."""
        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "ray_ready": True}

            with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
                mock_cluster_info.return_value = {}

                with pytest.raises(RuntimeError, match="Could not find Ray cluster info"):
                    cluster.port_forward()

    @patch("socket.socket")
    def test_port_forward_port_already_in_use(self, mock_socket_class, cluster):
        """Test port forwarding when local port is already in use."""
        mock_sock = Mock()
        mock_sock.bind.side_effect = OSError("Port already in use")
        mock_socket_class.return_value = mock_sock

        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "ray_ready": True}

            with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
                mock_cluster_info.return_value = {"head_ip": "192.168.1.100"}

                # Set up a mock thread to avoid actually starting port forwarding
                with patch("threading.Thread") as mock_thread_class:
                    mock_thread = Mock()
                    mock_thread_class.return_value = mock_thread

                    with patch("queue.Queue") as mock_queue_class:
                        mock_queue = Mock()
                        mock_queue.get.side_effect = RuntimeError(
                            "Port 8265 is already in use locally"
                        )
                        mock_queue_class.return_value = mock_queue

                        with pytest.raises(RuntimeError, match="Port .* is already in use locally"):
                            cluster.port_forward()

    @patch("subprocess.Popen")
    @patch("socket.socket")
    @patch("queue.Queue")
    def test_port_forward_ssh_process_fails(
        self, mock_queue_class, mock_socket, mock_popen, cluster
    ):
        """Test port forwarding when SSH process fails to start."""
        # Mock successful socket binding
        mock_sock = Mock()
        mock_socket.return_value = mock_sock

        # Mock process that fails
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process failed
        mock_process.pid = 12345
        mock_process.communicate.return_value = (b"stdout", b"Connection failed")
        mock_popen.return_value = mock_process

        # Mock queue
        mock_queue = Mock()
        mock_queue.get.return_value = ("success", None)  # Initial success, then process fails
        mock_queue_class.return_value = mock_queue

        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "ray_ready": True}

            with patch.object(cluster, "_get_ray_cluster_info") as mock_cluster_info:
                mock_cluster_info.return_value = {"head_ip": "192.168.1.100"}

                # Mock the executor tunnel attributes
                cluster.executor.tunnel.user = "testuser"
                cluster.executor.tunnel.host = "testhost"
                cluster.executor.tunnel.identity = "/path/to/key"

                with patch("threading.Thread"):
                    with patch("threading.Event"):
                        thread = cluster.port_forward()
                        assert thread is not None

    @patch("time.time")
    @patch("time.sleep")
    def test_wait_until_running_long_timeout(self, mock_sleep, mock_time, cluster):
        """Test wait_until_running with continuous checking."""
        # Mock multiple status checks before success
        mock_time.side_effect = [0, 100, 200, 300, 400, 500, 550]  # Simulate time progression

        with patch.object(cluster, "status") as mock_status:
            # First few calls return not ready, then ready
            mock_status.side_effect = [
                {"ray_ready": False, "state": "PENDING"},
                {"ray_ready": False, "state": "RUNNING"},
                {"ray_ready": True, "state": "RUNNING"},
            ]

            result = cluster.wait_until_running(timeout=600, delay_between_attempts=100)

            assert result is True
            assert mock_status.call_count == 3

    def test_delete_cluster_not_in_map(self, cluster):
        """Test delete when cluster is not in cluster_map."""
        # Ensure cluster name is not in map
        cluster.cluster_map.clear()

        with patch.object(cluster, "status") as mock_status:
            mock_status.return_value = {"job_id": "12345", "state": "RUNNING"}

            with patch("nemo_run.run.ray.slurm.cancel_slurm_job") as mock_cancel:
                mock_cancel.return_value = True

                result = cluster.delete()

                assert result is True
                mock_cancel.assert_called_once()

    def test_cluster_status_with_existing_cluster_map(self, cluster, mock_tunnel):
        """Test status when job_id is already in cluster_map."""
        # Pre-populate cluster_map
        cluster.cluster_map["test-cluster"] = "99999"

        # Mock squeue to return empty (job not found by name)
        mock_tunnel.run.side_effect = [
            Mock(stdout="", return_code=0),  # squeue -n job_name (empty)
            Mock(stdout="COMPLETED", return_code=0),  # squeue -j job_id from cluster_map
        ]

        status = cluster.status()

        assert status["job_id"] == "99999"
        assert status["state"] == "COMPLETED"


class TestSlurmRayJob:
    @pytest.fixture
    def mock_tunnel(self):
        """Create a mocked SSH tunnel."""
        tunnel = Mock(spec=SSHTunnel)
        tunnel.job_dir = "/tmp/test_jobs"
        tunnel.key = "test-host"
        tunnel.connect.return_value = None
        tunnel.run.return_value = Mock(stdout="", return_code=0)
        return tunnel

    @pytest.fixture
    def basic_executor(self, mock_tunnel):
        """Create a basic SlurmExecutor."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
        )
        executor.tunnel = mock_tunnel
        return executor

    @pytest.fixture
    def job(self, basic_executor):
        """Create a SlurmRayJob instance."""
        return SlurmRayJob(name="test-job", executor=basic_executor)

    def test_job_initialization(self, job):
        """Test job initialization."""
        assert job.name == "test-job"
        assert job.executor.account == "test_account"
        assert job.cluster_dir == "/tmp/test_jobs/test-job"
        assert job.job_id is None

    def test_logs_path(self, job):
        """Test logs path construction."""
        expected_path = "/tmp/test_jobs/test-job/logs/ray-job.log"
        assert job._logs_path() == expected_path

    @patch("nemo_run.run.ray.slurm.cancel_slurm_job")
    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_stop_success(self, mock_get_last_job_id, mock_cancel, job):
        """Test successful job stopping."""
        mock_get_last_job_id.return_value = 12345
        mock_cancel.return_value = True

        result = job.stop()

        assert result is True
        mock_cancel.assert_called_once_with(
            job.executor, "test-job", 12345, wait=False, timeout=60, poll_interval=5
        )

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_stop_no_job_id(self, mock_get_last_job_id, job):
        """Test stopping when no job ID exists."""
        mock_get_last_job_id.return_value = None

        with pytest.raises(RuntimeError, match="Ray job .* has no job_id"):
            job.stop()

    @patch("time.time")
    @patch("time.sleep")
    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_logs_follow(self, mock_get_last_job_id, mock_sleep, mock_time, job, mock_tunnel):
        """Test following logs."""
        mock_get_last_job_id.return_value = 12345
        mock_time.side_effect = [0, 1, 2, 3]  # More values for multiple time calls

        # Mock file exists check
        mock_tunnel.run.side_effect = [
            Mock(return_code=0),  # test -f log_path
            Mock(return_code=0),  # tail command
        ]

        job.logs(follow=True, lines=50)

        # Verify tail command was called
        assert mock_tunnel.run.call_count == 2

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_logs_file_not_found(self, mock_get_last_job_id, job, mock_tunnel):
        """Test logs when log file doesn't exist."""
        mock_get_last_job_id.return_value = 12345

        # Mock file doesn't exist
        mock_tunnel.run.return_value = Mock(return_code=1)

        with patch("time.time") as mock_time:
            mock_time.side_effect = [0, 0.5, 1, 1.5, 2]  # Multiple time calls for the loop

            job.logs(timeout=0.1)  # Short timeout

            # Should not raise, just warn and return

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_status(self, mock_get_last_job_id, job, caplog):
        """Test job status."""
        mock_get_last_job_id.return_value = 12345

        with patch("nemo_run.run.ray.slurm.SlurmRayCluster") as mock_cluster_class:
            mock_cluster = Mock()
            mock_cluster.status.return_value = {"state": "RUNNING", "ray_ready": True}
            mock_cluster.cluster_map = {}  # Add the missing attribute
            mock_cluster_class.return_value = mock_cluster

            with caplog.at_level("INFO"):  # Capture INFO level logs
                status = job.status()

            assert status["state"] == "RUNNING"
            assert status["ray_ready"] is True
            assert "Ray Job Status (Slurm)" in caplog.text

    @patch("tempfile.mkdtemp")
    @patch("nemo_run.core.tunnel.rsync.rsync")
    def test_start_with_workdir(self, mock_rsync, mock_mkdtemp, job, mock_tunnel):
        """Test starting job with local workdir."""
        mock_mkdtemp.return_value = "/tmp/mock_temp"
        # Add session attribute to mock tunnel with proper connect_kwargs and host attributes
        mock_tunnel.session = Mock()
        mock_tunnel.session.connect_kwargs = {"key_filename": ["/path/to/key"]}
        mock_tunnel.session.user = "testuser"
        mock_tunnel.session.host = "testhost.example.com"
        mock_tunnel.session.port = 22

        with patch("nemo_run.run.ray.slurm.SlurmRayCluster") as mock_cluster_class:
            mock_cluster = Mock()
            mock_cluster.create.return_value = "12345"
            mock_cluster_class.return_value = mock_cluster

            # Mock the status call that happens at the end
            with patch.object(job, "status") as mock_status:
                mock_status.return_value = {"state": "RUNNING"}

                job.start(command="python train.py", workdir="/local/code", dryrun=False)

            # Verify main functionality
            mock_cluster.create.assert_called_once()
            assert job.job_id == "12345"

    def test_start_dryrun(self, job):
        """Test starting job in dryrun mode."""
        with patch("nemo_run.run.ray.slurm.SlurmRayCluster") as mock_cluster_class:
            mock_cluster = Mock()
            mock_cluster.create.return_value = None
            mock_cluster_class.return_value = mock_cluster

            # Mock the status call that happens at the end - return empty to avoid JSON issues
            with patch.object(job, "status") as mock_status:
                mock_status.return_value = {"state": "NOT_FOUND"}

                job.start(command="python train.py", workdir="/workspace", dryrun=True)

            # Verify cluster.create was called with dryrun=True
            mock_cluster.create.assert_called_once()
            call_args = mock_cluster.create.call_args
            assert call_args.kwargs["dryrun"] is True

    @patch("tempfile.mkdtemp")
    @patch("subprocess.run")
    @patch("os.makedirs")
    @patch("os.getcwd")
    @patch("nemo_run.core.tunnel.rsync.rsync")
    def test_start_with_packager(
        self,
        mock_rsync,
        mock_getcwd,
        mock_makedirs,
        mock_subprocess,
        mock_mkdtemp,
        job,
        mock_tunnel,
    ):
        """Test starting job with packager functionality."""
        mock_mkdtemp.return_value = "/tmp/mock_temp"
        mock_getcwd.return_value = "/repo/root"
        mock_tunnel.session = Mock()
        mock_tunnel.session.connect_kwargs = {"key_filename": ["/path/to/key"]}
        mock_tunnel.session.user = "testuser"
        mock_tunnel.session.host = "testhost.example.com"
        mock_tunnel.session.port = 22

        # Set up a mock packager
        mock_packager = Mock()
        mock_packager.package.return_value = "/tmp/mock_temp/test-job.tar.gz"
        job.executor.packager = mock_packager

        # Mock git rev-parse for GitArchivePackager
        mock_subprocess.return_value = Mock(stdout=[b"/repo/root"], returncode=0)

        with patch("nemo_run.run.ray.slurm.SlurmRayCluster") as mock_cluster_class:
            mock_cluster = Mock()
            mock_cluster.create.return_value = "12345"
            mock_cluster_class.return_value = mock_cluster

            with patch.object(job, "status") as mock_status:
                mock_status.return_value = {"state": "RUNNING"}

                job.start(command="python train.py", workdir=None, dryrun=False)

            # Verify packager was used
            mock_packager.package.assert_called_once()
            mock_cluster.create.assert_called_once()
            assert job.job_id == "12345"

    @patch("tempfile.mkdtemp")
    @patch("subprocess.run")
    @patch("os.makedirs")
    @patch("os.getcwd")
    def test_start_with_packager_local_tunnel(
        self, mock_getcwd, mock_makedirs, mock_subprocess, mock_mkdtemp, job
    ):
        """Test starting job with packager and local tunnel."""
        mock_mkdtemp.return_value = "/tmp/mock_temp"
        mock_getcwd.return_value = "/repo/root"

        # Use local tunnel (not SSH)
        job.executor.tunnel = Mock()
        job.executor.tunnel.job_dir = "/local/jobs"

        # Set up a mock packager
        mock_packager = Mock()
        mock_packager.package.return_value = "/tmp/mock_temp/test-job.tar.gz"
        job.executor.packager = mock_packager

        # Mock git command to return a path
        mock_subprocess.return_value = Mock(stdout=[b"/repo/root"], returncode=0)

        with patch("nemo_run.run.ray.slurm.SlurmRayCluster") as mock_cluster_class:
            mock_cluster = Mock()
            mock_cluster.create.return_value = "12345"
            mock_cluster_class.return_value = mock_cluster

            with patch.object(job, "status") as mock_status:
                mock_status.return_value = {"state": "RUNNING"}

                job.start(command="python train.py", workdir=None, dryrun=False)

            # Verify packager was used but no rsync for local tunnel
            mock_packager.package.assert_called_once()
            assert job.job_id == "12345"

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_logs_with_timeout_file_never_appears(self, mock_get_last_job_id, job, mock_tunnel):
        """Test logs when file never appears within timeout."""
        mock_get_last_job_id.return_value = 12345

        # Mock file check to always return not found
        mock_tunnel.run.return_value = Mock(return_code=1)

        with patch("time.time") as mock_time:
            with patch("time.sleep"):
                # Provide enough time values for the loop and logging
                mock_time.side_effect = [0, 50, 105, 110, 115]

                # Should not raise, just warn and return
                job.logs(timeout=100)

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_logs_follow_keyboard_interrupt(self, mock_get_last_job_id, job, mock_tunnel):
        """Test logs follow mode when user interrupts with Ctrl+C."""
        mock_get_last_job_id.return_value = 12345

        # Mock file exists
        mock_tunnel.run.side_effect = [
            Mock(return_code=0),  # test -f log_path
        ]

        # Mock the run command for tail to raise KeyboardInterrupt
        def run_side_effect(cmd, **kwargs):
            if "tail" in cmd:
                raise KeyboardInterrupt()
            return Mock(return_code=0)

        mock_tunnel.run.side_effect = run_side_effect

        with patch("time.time") as mock_time:
            mock_time.return_value = 0

            # Should not raise, just return gracefully
            job.logs(follow=True)

    def test_status_display_false(self, job):
        """Test job status with display=False."""
        with patch("nemo_run.run.ray.slurm.get_last_job_id") as mock_get_last_job_id:
            mock_get_last_job_id.return_value = 12345

            with patch("nemo_run.run.ray.slurm.SlurmRayCluster") as mock_cluster_class:
                mock_cluster = Mock()
                mock_cluster.status.return_value = {"state": "RUNNING", "ray_ready": True}
                mock_cluster.cluster_map = {}
                mock_cluster_class.return_value = mock_cluster

                # No logging should occur with display=False
                status = job.status(display=False)

                assert status["state"] == "RUNNING"
                assert status["ray_ready"] is True

    def test_start_no_workdir_no_packager(self, job):
        """Test starting job with no workdir and no packager."""
        # Clear the packager
        job.executor.packager = None

        with pytest.raises(AssertionError, match="workdir could not be determined"):
            job.start(command="python train.py", workdir=None)

    def test_start_assertion_error_handling(self, job):
        """Test starting job handles assertion errors properly."""
        # Clear the packager to trigger assertion
        job.executor.packager = None

        with pytest.raises(AssertionError):
            job.start(command="python train.py", workdir=None)

    @patch("nemo_run.run.ray.slurm.get_last_job_id")
    def test_status_with_none_job_id(self, mock_get_last_job_id, job):
        """Test job status when get_last_job_id returns None."""
        mock_get_last_job_id.return_value = None

        with patch("nemo_run.run.ray.slurm.SlurmRayCluster") as mock_cluster_class:
            mock_cluster = Mock()
            mock_cluster.status.return_value = {"state": "NOT_FOUND", "ray_ready": False}
            mock_cluster.cluster_map = {}
            mock_cluster_class.return_value = mock_cluster

            status = job.status()

            assert status["state"] == "NOT_FOUND"
            assert status["ray_ready"] is False


class TestUtilityFunctions:
    @pytest.fixture
    def mock_tunnel(self):
        """Create a mocked SSH tunnel."""
        tunnel = Mock(spec=SSHTunnel)
        tunnel.run.return_value = Mock(stdout="", return_code=0)
        return tunnel

    @pytest.fixture
    def basic_executor(self, mock_tunnel):
        """Create a basic SlurmExecutor."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
        )
        executor.tunnel = mock_tunnel
        return executor

    def test_cancel_slurm_job_success(self, basic_executor, mock_tunnel):
        """Test successful job cancellation."""
        mock_tunnel.run.return_value = Mock(return_code=0)

        result = cancel_slurm_job(basic_executor, "test-job", 12345)

        assert result is True
        mock_tunnel.connect.assert_called_once()
        mock_tunnel.run.assert_called_with("scancel 12345")

    def test_get_last_job_id_ssh_tunnel(self, basic_executor, mock_tunnel):
        """Test getting last job ID with SSH tunnel."""
        job_ids = ["12345", "12346", "12347"]
        mock_tunnel.run.return_value = Mock(return_code=0, stdout=json.dumps(job_ids))

        result = get_last_job_id("/tmp/test_cluster", basic_executor)

        assert result == 12347
        mock_tunnel.run.assert_called_with("cat /tmp/test_cluster/job_ids.json", warn=True)

    def test_get_last_job_id_file_not_found(self, basic_executor, mock_tunnel):
        """Test getting last job ID when file doesn't exist."""
        mock_tunnel.run.return_value = Mock(return_code=1)

        result = get_last_job_id("/tmp/test_cluster", basic_executor)

        assert result is None

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_last_job_id_local_tunnel(self, mock_file, mock_exists, basic_executor):
        """Test getting last job ID with local tunnel."""
        # Change tunnel to non-SSH
        basic_executor.tunnel = Mock()
        mock_exists.return_value = True

        job_ids = ["12345", "12346", "12347"]
        mock_file.return_value.read.return_value = json.dumps(job_ids)

        result = get_last_job_id("/tmp/test_cluster", basic_executor)

        assert result == 12347
        mock_file.assert_called_with("/tmp/test_cluster/job_ids.json", "r")

    @patch("os.path.exists")
    def test_get_last_job_id_local_file_not_found(self, mock_exists, basic_executor):
        """Test getting last job ID when local file doesn't exist."""
        basic_executor.tunnel = Mock()  # Non-SSH tunnel
        mock_exists.return_value = False

        result = get_last_job_id("/tmp/test_cluster", basic_executor)

        assert result is None

    def test_cancel_slurm_job_exception(self, basic_executor, mock_tunnel):
        """Test job cancellation when scancel command raises exception."""
        mock_tunnel.run.side_effect = Exception("SSH connection failed")

        result = cancel_slurm_job(basic_executor, "test-job", 12345)

        assert result is False
        mock_tunnel.connect.assert_called_once()

    def test_get_last_job_id_ssh_invalid_json(self, basic_executor, mock_tunnel):
        """Test getting last job ID when SSH returns invalid JSON."""
        mock_tunnel.run.return_value = Mock(return_code=0, stdout="invalid json")

        with pytest.raises(json.JSONDecodeError):
            get_last_job_id("/tmp/test_cluster", basic_executor)

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_last_job_id_local_invalid_json(self, mock_file, mock_exists, basic_executor):
        """Test getting last job ID when local file contains invalid JSON."""
        basic_executor.tunnel = Mock()  # Non-SSH tunnel
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "invalid json"

        with pytest.raises(json.JSONDecodeError):
            get_last_job_id("/tmp/test_cluster", basic_executor)
