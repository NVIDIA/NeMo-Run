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

import copy
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from nemo_run.core.execution.launcher import SlurmTemplate, Torchrun
from nemo_run.core.execution.slurm import (
    SlurmExecutor,
    SlurmJobDetails,
    SlurmTunnelCallback,
    get_packaging_job_key,
)
from nemo_run.core.tunnel.client import LocalTunnel
from nemo_run.devspace.base import DevSpace


class TestSlurmJobDetails:
    def test_job_details_properties(self):
        """Test SlurmJobDetails property methods."""
        details = SlurmJobDetails(job_name="test_job", folder="/path/to/job")

        # Test property methods
        assert str(details.stderr) == "/path/to/job/sbatch_test_job_%j.err"
        assert str(details.stdout) == "/path/to/job/sbatch_test_job_%j.out"
        assert (
            str(details.srun_stderr) == "/path/to/job/log-test_job_%j_${SLURM_RESTART_COUNT:-0}.err"
        )
        assert (
            str(details.srun_stdout) == "/path/to/job/log-test_job_%j_${SLURM_RESTART_COUNT:-0}.out"
        )
        assert details.ls_term == "/path/to/job/log*"

        # Test repr method
        assert repr(details) == "SlurmJobDetails(/path/to/job)"


class TestGetPackagingJobKey:
    def test_packaging_job_key(self):
        """Test the get_packaging_job_key function."""
        key = get_packaging_job_key("exp_123", "job_456")
        assert key == "exp_123:job_456"


class TestSlurmExecutorExtended:
    @pytest.fixture
    def mock_context(self):
        with patch("invoke.context.Context") as mock_ctx:
            mock_context = MagicMock()
            mock_ctx.return_value = mock_context
            yield mock_context

    @pytest.fixture
    def mock_subprocess(self):
        with patch("subprocess.run") as mock_run:
            mock_process = MagicMock()
            mock_process.stdout = b"/path/to/repo\n"
            mock_run.return_value = mock_process
            yield mock_run

    def test_post_init(self):
        """Test the __post_init__ method with negative wait time."""
        executor = SlurmExecutor(account="test", wait_time_for_group_job=-10)
        assert executor.wait_time_for_group_job == 0

    def test_info(self):
        """Test the info method."""
        executor = SlurmExecutor(account="test", tunnel=LocalTunnel(job_dir="/test"))

        # Use a more flexible assertion since the exact output can vary
        info = executor.info()
        assert "SlurmExecutor on" in info

    def test_nnodes_and_nproc_per_node(self):
        """Test the nnodes and nproc_per_node methods."""
        executor = SlurmExecutor(account="test", nodes=2, ntasks_per_node=4)
        assert executor.nnodes() == 2
        assert executor.nproc_per_node() == 4

        # Test with torchrun_nproc_per_node
        executor = SlurmExecutor(
            account="test", nodes=2, ntasks_per_node=4, torchrun_nproc_per_node=8
        )
        assert executor.nproc_per_node() == 8

        # Test with gpus_per_node and ntasks_per_node=1
        executor = SlurmExecutor(account="test", nodes=2, ntasks_per_node=1, gpus_per_node=8)
        assert executor.nproc_per_node() == 8

        # Test with gpus_per_task
        executor = SlurmExecutor(account="test", nodes=2, ntasks_per_node=4, gpus_per_task=2)
        assert executor.nproc_per_node() == 2

    def test_macro_values(self):
        """Test the macro_values method."""
        executor = SlurmExecutor(account="test")
        macros = executor.macro_values()
        assert macros.head_node_ip_var == "head_node_ip"
        assert macros.nproc_per_node_var == "SLURM_NTASKS_PER_NODE"
        assert macros.num_nodes_var == "SLURM_NNODES"
        assert macros.node_rank_var == "SLURM_NODEID"
        assert macros.het_group_host_var == "het_group_host"

    def test_setup_launcher_with_torchrun(self):
        """Test the _setup_launcher method with Torchrun launcher."""
        executor = SlurmExecutor(account="test", ntasks_per_node=8)
        executor.launcher = Torchrun()
        executor._setup_launcher()
        assert executor.ntasks_per_node == 1
        assert executor.torchrun_nproc_per_node == 8

    def test_local_is_slurm_true(self):
        """Test the local_is_slurm property when srun is available."""
        executor = SlurmExecutor(account="test")

        with patch.object(executor.local, "run") as mock_run:
            # Simulate successful srun detection
            mock_run.return_value = MagicMock()
            assert executor.local_is_slurm is True

    def test_local_is_slurm_false(self):
        """Test the local_is_slurm property when srun is not available."""
        executor = SlurmExecutor(account="test")

        with patch.object(executor.local, "run") as mock_run:
            # Simulate failed srun detection
            import invoke.exceptions

            mock_run.side_effect = invoke.exceptions.UnexpectedExit(MagicMock())
            assert executor.local_is_slurm is False

    def test_assign(self):
        """Test the assign method with mock executor."""
        # Create executor with a mock tunnel
        tunnel = MagicMock(spec=LocalTunnel)
        executor = SlurmExecutor(account="test", tunnel=tunnel)

        # Initial job_name
        initial_job_name = executor.job_name

        # Call assign
        executor.assign("exp_id", "/path/to/exp", "task_id", "task_dir")

        # Check updated values
        assert executor.job_name == "task_id"
        assert executor.experiment_dir == "/path/to/exp"
        assert executor.job_dir == "/path/to/exp/task_dir"
        assert executor.experiment_id == "exp_id"
        assert initial_job_name != executor.job_name

    def test_get_launcher_prefix(self):
        """Test the get_launcher_prefix method with nsys_profile."""
        executor = SlurmExecutor(account="test")

        # Test with launcher that has nsys_profile
        launcher_mock = MagicMock()
        launcher_mock.nsys_profile = True
        launcher_mock.get_nsys_prefix.return_value = ["nsys", "profile"]
        launcher_mock.nsys_gpu_metrics = False

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            assert executor.get_launcher_prefix() == ["nsys", "profile"]

    def test_get_launcher_prefix_with_gpu_metrics(self):
        """Test the get_launcher_prefix method with nsys_profile when gpu metrics is enabled."""
        executor = SlurmExecutor(account="test")

        # Test with launcher that has nsys_profile
        launcher_mock = MagicMock()
        launcher_mock.nsys_profile = True
        launcher_mock.get_nsys_prefix.return_value = ["nsys", "profile"]
        launcher_mock.nsys_gpu_metrics = True

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            assert executor.get_launcher_prefix() == ["nsys", "profile", "$GPU_METRICS_FLAG"]

    def test_get_nsys_entrypoint(self):
        """Test the get_nsys_entrypoint method with nsys_profile."""
        executor = SlurmExecutor(account="test")

        # Test with launcher that has nsys_profile
        launcher_mock = MagicMock()
        launcher_mock.nsys_gpu_metrics = True

        with patch.object(executor, "get_launcher", return_value=launcher_mock):
            assert executor.get_nsys_entrypoint() == (
                'bash -c \'GPU_METRICS_FLAG=""; if [ "$SLURM_PROCID" -eq 0 ]; then GPU_METRICS_FLAG="--gpu-metrics-devices=all"; fi; nsys',
                "'",
            )

    def test_supports_launcher_transform(self):
        """Test the supports_launcher_transform method."""
        executor = SlurmExecutor(account="test")

        # Test with SlurmTemplate launcher
        with patch.object(
            executor, "get_launcher", return_value=SlurmTemplate(template_inline="content")
        ):
            assert executor.supports_launcher_transform() is True

        # Test with non-SlurmTemplate launcher
        with patch.object(executor, "get_launcher", return_value=Torchrun()):
            assert executor.supports_launcher_transform() is False

    def test_bash(self):
        """Test the bash method."""
        executor = SlurmExecutor(account="test")

        with patch.object(executor, "srun") as mock_srun:
            executor.bash(job_name="test_job")

            mock_srun.assert_called_once_with("bash", job_name="test_job")

    @patch("nemo_run.core.execution.slurm.ZlibJSONSerializer")
    def test_launch_devspace(self, mock_serializer_cls):
        """Test the launch_devspace method."""
        # Set up mocks
        mock_serializer = MagicMock()
        mock_serializer.serialize.return_value = "serialized_space_config"
        mock_serializer_cls.return_value = mock_serializer

        # Create executor and mock space
        executor = SlurmExecutor(
            account="test",
            job_dir="/path/to/job",
            container_mounts=["/path1:/path1"],
        )
        mock_space = MagicMock(spec=DevSpace)
        mock_space.name = "test_space"
        mock_space.__io__ = {"config": "value"}

        # Mock the local_is_slurm property and srun method
        with patch(
            "nemo_run.core.execution.slurm.SlurmExecutor.local_is_slurm", new_callable=PropertyMock
        ) as mock_local_is_slurm:
            with patch.object(executor, "srun") as mock_srun:
                # Case 1: local_is_slurm = True
                mock_local_is_slurm.return_value = True
                mock_srun.return_value = None

                executor.launch_devspace(mock_space, job_name="test_job")

                # Check that srun was called
                mock_srun.assert_called_once()

    def test_connect_devspace(self):
        """Test the connect_devspace method."""
        executor = SlurmExecutor(account="test")
        mock_space = MagicMock(spec=DevSpace)

        with patch("nemo_run.core.execution.slurm.SlurmTunnelCallback") as mock_callback_cls:
            mock_callback = MagicMock()
            mock_callback_cls.return_value = mock_callback

            # Call connect_devspace
            callback = executor.connect_devspace(mock_space, tunnel_dir="/path/to/tunnel")

            # Verify SlurmTunnelCallback was created correctly
            mock_callback_cls.assert_called_once_with(
                executor, space=mock_space, tunnel_dir="/path/to/tunnel"
            )
            assert callback == mock_callback


class TestSlurmTunnelCallback:
    @pytest.fixture
    def mock_space(self):
        space = MagicMock(spec=DevSpace)
        space.name = "test_space"
        return space

    @pytest.fixture
    def mock_executor(self):
        executor = MagicMock(spec=SlurmExecutor)
        executor.job_dir = "/path/to/job"
        return executor

    @pytest.fixture
    def mock_srun(self):
        srun = MagicMock()
        srun.runner = MagicMock()
        srun.runner.stderr = ["Starting server..."]
        srun.runner.stdout = []
        return srun

    def test_init(self, mock_executor, mock_space, mock_srun):
        """Test SlurmTunnelCallback initialization."""
        callback = SlurmTunnelCallback(mock_executor, mock_space, mock_srun)

        assert callback.executor == mock_executor
        assert callback.srun == mock_srun
        assert callback.space == mock_space
        assert callback.editor_started is False
        assert callback.tunnel_name == "test_space.test_space"

    def test_on_start_with_srun(self, mock_executor, mock_space, mock_srun):
        """Test on_start method with srun."""
        with patch("nemo_run.core.execution.slurm.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            callback = SlurmTunnelCallback(mock_executor, mock_space, mock_srun)
            callback.on_start()

            assert callback.srun_is_done is False
            mock_console.status.assert_called_once()
            mock_console.status().start.assert_called_once()

    def test_on_start_without_srun(self, mock_executor, mock_space):
        """Test on_start method without srun."""
        callback = SlurmTunnelCallback(mock_executor, mock_space)
        callback.on_start()

        assert callback.srun_is_done is True

    def test_on_interval_srun_processing(self, mock_executor, mock_space, mock_srun):
        """Test on_interval method for srun status processing."""
        # Set up mocks
        callback = SlurmTunnelCallback(mock_executor, mock_space, mock_srun)
        callback.srun_is_done = False
        callback.editor_started = False

        # Mock console
        with patch("nemo_run.core.execution.slurm.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console
            callback.console = mock_console
            callback.srun_status = MagicMock()

            # Case 1: No connection message yet
            callback.on_interval()
            assert callback.srun_is_done is False
            callback.srun_status.update.assert_called_once()

            # Case 2: Connection message appears
            mock_srun.runner.stdout = [
                "Starting...",
                "To connect to the tunnel, run the following command on your local machine:",
            ]
            callback.on_interval()

            assert callback.srun_is_done is True
            callback.srun_status.stop.assert_called_once()
            mock_console.log.assert_called()

    def test_on_stop(self, mock_executor, mock_space):
        """Test on_stop method."""
        callback = SlurmTunnelCallback(mock_executor, mock_space)

        # Add ssh_entry_added attribute
        callback.ssh_entry_added = True
        callback.ssh_config = MagicMock()

        callback.on_stop()

        callback.ssh_config.remove_entry.assert_called_once_with(callback.tunnel_name)


class TestSlurmExecutor:
    def test_merge_single_executor(self):
        executor = SlurmExecutor(account="account", heterogeneous=True)
        merged_executor = SlurmExecutor.merge([executor], num_tasks=3)
        assert len(merged_executor.resource_group) == 3
        assert merged_executor.run_as_group

    def test_merge_multiple_executor(self):
        executor = SlurmExecutor(account="account", heterogeneous=True)
        executor_2 = SlurmExecutor(
            account="account_2", nodes=2, ntasks_per_node=4, container_image="abcd"
        )
        merged_executor = SlurmExecutor.merge([executor, executor_2], num_tasks=2)
        assert len(merged_executor.resource_group) == 2
        assert merged_executor.resource_group[1].container_image == "abcd"
        assert merged_executor.resource_group[1].nodes == 2
        assert merged_executor.resource_group[1].ntasks_per_node == 4
        assert merged_executor.run_as_group

    def test_merge_single_executor_non_heterogeneous(self):
        executor = SlurmExecutor(account="account", heterogeneous=False)
        expected = copy.deepcopy(executor)
        expected.run_as_group = True
        merged_executor = SlurmExecutor.merge([executor], num_tasks=3)
        assert merged_executor == expected
        assert merged_executor.run_as_group

    def test_merge_mismatch(self):
        with pytest.raises(AssertionError):
            SlurmExecutor.merge(
                [SlurmExecutor(account="account1"), SlurmExecutor(account="account2")],
                num_tasks=3,
            )
