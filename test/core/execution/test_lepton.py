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

import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest
from leptonai.api.v1.types.common import LeptonVisibility, Metadata
from leptonai.api.v1.types.deployment import (
    LeptonContainer,
    LeptonResourceAffinity,
    Mount,
)
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec

from nemo_run.core.execution.lepton import LeptonExecutor, LeptonJobState
from nemo_run.core.packaging.git import GitArchivePackager


class MockLeptonJob:
    def __init__(self, state, ready=0, active=0):
        self.status = MagicMock()
        self.status.state = state
        self.status.ready = ready
        self.status.active = active


class TestLeptonExecutor:
    def test_init(self):
        executor = LeptonExecutor(
            resource_shape="gpu.8xh100-80gb",
            node_group="my-node-group",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=2,
            gpus_per_node=8,
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.resource_shape == "gpu.8xh100-80gb"
        assert executor.node_group == "my-node-group"
        assert executor.container_image == "nvcr.io/nvidia/test:latest"
        assert executor.nodes == 2
        assert executor.gpus_per_node == 8
        assert executor.nemo_run_dir == "/workspace/nemo_run"
        assert executor.mounts == [{"path": "/workspace", "mount_path": "/workspace"}]

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_stop_job(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=2, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.stop_job("job123")

        mock_job_api.update.assert_called_once_with("job123", spec={"spec": {"stopped": True}})

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_stop_job_not_running(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(
                LeptonJobState.Completed,
            )
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.stop_job("job123")

        mock_job_api.update.assert_not_called()

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_stop_job_not_found(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(return_value=None)

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.stop_job("job123")

        mock_job_api.update.assert_not_called()

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open, read_data=b"mock tarball")
    def test_copy_directory_data_command_success(self, mock_file, mock_subprocess):
        local_dir_path = "/mock/local/dir"
        dest_path = "/mock/destination/path"

        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )
        response = executor.copy_directory_data_command(local_dir_path, dest_path)

        # The response is in the format ["sh", "-c", "<command>"]
        # The actual command is in the final index of the response
        command = response[-1]
        mock_subprocess.assert_called_once()
        assert mock_file.call_count == 1

        assert "rm -rf /mock/destination/path && mkdir -p /mock/destination/path && echo" in command
        assert (
            "base64 -d > /mock/destination/path/archive.tar.gz && tar -xzf /mock/destination/path/archive.tar.gz -C /mock/destination/path && rm /mock/destination/path/archive.tar.gz"
            in command
        )

    @patch("tempfile.TemporaryDirectory")
    def test_copy_directory_data_command_fails(self, mock_tempdir):
        local_dir_path = "/mock/local/dir"
        dest_path = "/mock/destination/path"

        mock_tempdir.side_effect = OSError("Temporary directory creation failed")

        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )
        with pytest.raises(OSError, match="Temporary directory creation failed"):
            executor.copy_directory_data_command(local_dir_path, dest_path)

    @patch.object(LeptonExecutor, "copy_directory_data_command")
    @patch("nemo_run.core.execution.lepton.datetime")
    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_move_data_success(self, mock_APIClient, mock_datetime, mock_copy):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api
        mock_copy.return_value = ["sh", "-c", "echo 'hello world'"]
        mock_APIClient.return_value = mock_instance
        mock_client = mock_APIClient.return_value
        mock_nodegroup = MagicMock()
        mock_datetime_now = MagicMock()
        mock_datetime.now.return_value = mock_datetime_now
        mock_datetime_now.timestamp.return_value = 1
        mock_client.nodegroup = mock_nodegroup
        mock_nodegroup.list_all.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(name="123456", id_="my-node-id"))
        ]
        mock_nodegroup.list_nodes.return_value = [
            SimpleNamespace(metadata=SimpleNamespace(id_="10-10-10-10"))
        ]
        mock_job_api.get.return_value = SimpleNamespace(status=SimpleNamespace(state="Completed"))

        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            node_group="123456",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        executor.move_data()

        expected_cmd = ["sh", "-c", "echo 'hello world'"]
        expected_spec = LeptonJobUserSpec(
            resource_shape="cpu.small",
            affinity=LeptonResourceAffinity(
                allowed_dedicated_node_groups=["my-node-id"],
                allowed_nodes_in_node_group=["10-10-10-10"],
            ),
            container=LeptonContainer(
                image="busybox:1.37.0",
                command=expected_cmd,
            ),
            completions=1,
            parallelism=1,
            mounts=[Mount(path="/workspace", mount_path="/workspace")],
        )

        custom_name = "data-mover-1"
        expected_job = LeptonJob(
            metadata=Metadata(
                id=custom_name,
                name=custom_name,
                visibility=LeptonVisibility("private"),
            ),
            spec=expected_spec,
        )

        mock_copy.assert_called_once_with(executor.job_dir, executor.lepton_job_dir)
        mock_job_api.create.assert_called_once_with(expected_job)
        mock_job_api.delete.assert_called_once_with(mock_job_api.create.return_value.metadata.id_)

    def test_node_group_id(self):
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_all=MagicMock(
                    return_value=[
                        SimpleNamespace(metadata=SimpleNamespace(name="123456")),
                        SimpleNamespace(metadata=SimpleNamespace(name="abcdef")),
                    ]
                )
            )
        )

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        node_group_id = executor._node_group_id(mock_client)

        assert node_group_id == SimpleNamespace(metadata=SimpleNamespace(name="123456"))

    def test_node_group_id_no_groups(self):
        mock_client = MagicMock(nodegroup=MagicMock(list_all=MagicMock(return_value=[])))

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        with pytest.raises(RuntimeError):
            executor._node_group_id(mock_client)

    def test_node_group_id_unmatched_node_id(self):
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_all=MagicMock(
                    return_value=[
                        SimpleNamespace(metadata=SimpleNamespace(name="123456")),
                        SimpleNamespace(metadata=SimpleNamespace(name="abcdef")),
                    ]
                )
            )
        )

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="zzzzz",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        with pytest.raises(RuntimeError):
            executor._node_group_id(mock_client)

    def test_valid_node_id(self):
        mock_client = MagicMock(
            nodegroup=MagicMock(
                list_nodes=MagicMock(
                    return_value=[
                        SimpleNamespace(metadata=SimpleNamespace(id_="10-10-10-10")),
                        SimpleNamespace(metadata=SimpleNamespace(id_="20-20-20-20")),
                    ]
                )
            )
        )

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        node_ids = executor._valid_node_ids(None, mock_client)

        assert node_ids == set(["10-10-10-10", "20-20-20-20"])

    def test_valid_node_id_no_ids(self):
        mock_client = MagicMock(nodegroup=MagicMock(list_nodes=MagicMock(return_value=[])))

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        node_ids = executor._valid_node_ids(None, mock_client)

        assert node_ids == set([])

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_create_lepton_job(self, mock_APIClient_class):
        mock_client = mock_APIClient_class.return_value
        mock_client.job.create.return_value = LeptonJob(metadata=Metadata(id="my-lepton-job"))
        node_group = SimpleNamespace(metadata=SimpleNamespace(id_="123456"))

        mock_client.nodegroup.list_all.return_value = []
        valid_node_ids = ["node-id-1", "node-id-2"]

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            node_group="123456",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )
        executor._valid_node_ids = MagicMock(return_value=valid_node_ids)
        executor._node_group_id = MagicMock(return_value=node_group)

        executor.create_lepton_job("my-lepton-job")

        mock_client.job.create.assert_called_once()

    def test_nnodes(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            nodes=3,
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nnodes() == 3

    def test_nnodes_default(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nnodes() == 1

    def test_nproc_per_node_with_gpus(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            gpus_per_node=4,
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nproc_per_node() == 4

    def test_nproc_per_node_with_nprocs(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            gpus_per_node=0,
            nprocs_per_node=3,
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nproc_per_node() == 3

    def test_nproc_per_node_default(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor.nproc_per_node() == 1

    def test_valid_storage_mounts(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace"}],
        )

        assert executor._validate_mounts() is None

    def test_valid_storage_mounts_with_mount_from(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[
                {"path": "/workspace", "mount_path": "/workspace", "from": "local-storage:nfs"}
            ],
        )

        assert executor._validate_mounts() is None

    def test_missing_storage_mount_options(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace"}],
        )

        with pytest.raises(RuntimeError):
            executor._validate_mounts()

    def test_missing_storage_mount_options_mount_path(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"mount_path": "/workspace"}],
        )

        with pytest.raises(RuntimeError):
            executor._validate_mounts()

    def test_valid_storage_mounts_with_random_args(self):
        executor = LeptonExecutor(
            container_image="nvcr.io/nvidia/test:latest",
            nemo_run_dir="/workspace/nemo_run",
            mounts=[{"path": "/workspace", "mount_path": "/workspace", "random": True}],
        )

        assert executor._validate_mounts() is None

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_running_and_ready(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=2, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Running

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_running_and_not_all_ready(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=1, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Starting

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_starting(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Starting, ready=0, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Starting

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_unknown(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(
                LeptonJobState.Unknown,
            )
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Unknown

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_status_no_job(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(return_value=None)

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        status = executor.status("job123")
        assert status == LeptonJobState.Unknown

        mock_job_api.get.assert_called_once_with("job123")

    @patch("nemo_run.core.execution.lepton.APIClient")
    def test_cancel_job(self, mock_APIClient):
        mock_instance = MagicMock()
        mock_job_api = MagicMock()
        mock_instance.job = mock_job_api

        mock_job_api.get = MagicMock(
            return_value=MockLeptonJob(LeptonJobState.Running, ready=2, active=2)
        )

        mock_APIClient.return_value = mock_instance

        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        executor.cancel("job123")

        mock_job_api.delete.assert_called_once_with("job123")

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_package_configs(self, mock_file, mock_makedirs):
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        configs = [("config1.yaml", "key: value"), ("subdir/config2.yaml", "another: config")]

        filenames = executor.package_configs(*configs)

        assert len(filenames) == 2
        assert filenames[0] == "/nemo_run/configs/config1.yaml"
        assert filenames[1] == "/nemo_run/configs/subdir/config2.yaml"
        mock_makedirs.assert_called()
        assert mock_file.call_count == 2

    @patch("invoke.context.Context.run")
    @patch("subprocess.run")
    def test_package_git_packager(self, mock_subprocess_run, mock_context_run):
        # Mock subprocess.run which is used to get the git repo path
        mock_process = MagicMock()
        mock_process.stdout = b"/path/to/repo\n"
        mock_subprocess_run.return_value = mock_process

        # Mock the Context.run to avoid actually running commands
        mock_context_run.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = LeptonExecutor(
                container_image="test-image",
                nemo_run_dir="/test/path",
                mounts=[{"path": "/test", "mount_path": "/test"}],
            )
            executor.experiment_id = "test_exp"
            executor.job_dir = tmp_dir

            packager = GitArchivePackager()
            # Mock the package method to avoid real git operations
            with patch.object(packager, "package", return_value="/mocked/package.tar.gz"):
                executor.package(packager, "test_job")

                # Check that the right methods were called
                mock_subprocess_run.assert_called_once_with(
                    ["git", "rev-parse", "--show-toplevel"],
                    check=True,
                    stdout=subprocess.PIPE,
                )
                assert mock_context_run.called

    def test_macro_values(self):
        executor = LeptonExecutor(
            container_image="test-image",
            nemo_run_dir="/test/path",
            mounts=[{"path": "/test", "mount_path": "/test"}],
        )

        result = executor.macro_values()

        assert result is None
