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

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.execution.dgxcloud import DGXCloudExecutor, DGXCloudState
from nemo_run.core.packaging.git import GitArchivePackager


class TestDGXCloudExecutor:
    def test_init(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=2,
            gpus_per_node=8,
            pvcs=[{"path": "/workspace", "claimName": "test-claim"}],
        )

        assert executor.base_url == "https://dgxapi.example.com"
        assert executor.app_id == "test_app_id"
        assert executor.app_secret == "test_app_secret"
        assert executor.project_name == "test_project"
        assert executor.container_image == "nvcr.io/nvidia/test:latest"
        assert executor.nodes == 2
        assert executor.gpus_per_node == 8
        assert executor.pvcs == [{"path": "/workspace", "claimName": "test-claim"}]
        assert executor.distributed_framework == "PyTorch"

    @patch("requests.post")
    def test_get_auth_token_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = '{"accessToken": "test_token"}'
        mock_post.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        token = executor.get_auth_token()

        assert token == "test_token"
        mock_post.assert_called_once_with(
            "https://dgxapi.example.com/token",
            json={
                "grantType": "app_token",
                "appId": "test_app_id",
                "appSecret": "test_app_secret",
            },
            headers=executor._default_headers(),
        )

    @patch("requests.post")
    def test_get_auth_token_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = '{"error": "Invalid credentials"}'
        mock_post.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        token = executor.get_auth_token()

        assert token is None

    @patch("requests.get")
    def test_get_project_and_cluster_id_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = '{"projects": [{"name": "other_project", "id": "proj1", "clusterId": "clust1"}, {"name": "test_project", "id": "proj2", "clusterId": "clust2"}]}'
        mock_get.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        project_id, cluster_id = executor.get_project_and_cluster_id("test_token")

        assert project_id == "proj2"
        assert cluster_id == "clust2"
        mock_get.assert_called_once_with(
            "https://dgxapi.example.com/org-unit/projects",
            headers=executor._default_headers(token="test_token"),
        )

    @patch("requests.get")
    def test_get_project_and_cluster_id_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = (
            '{"projects": [{"name": "other_project", "id": "proj1", "clusterId": "clust1"}]}'
        )
        mock_get.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        project_id, cluster_id = executor.get_project_and_cluster_id("test_token")

        assert project_id is None
        assert cluster_id is None

    @patch("requests.post")
    def test_create_distributed_job(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "submitted"}'
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                nodes=2,
                gpus_per_node=8,
                pvcs=[{"path": tmp_dir, "claimName": "test-claim"}],
            )
            executor.job_dir = tmp_dir
            executor.env_vars = {"TEST_VAR": "test_value"}

            response = executor.create_distributed_job(
                token="test_token",
                project_id="proj_id",
                cluster_id="cluster_id",
                name="test_job",
                cmd=["python", "train.py"],
            )

            assert response == mock_response
            assert os.path.exists(os.path.join(tmp_dir, "launch_script.sh"))

            # Check if the API call is made correctly
            mock_post.assert_called_once()
            # The URL is the first argument to post
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["name"] == "test_job"
            assert kwargs["json"]["projectId"] == "proj_id"
            assert kwargs["json"]["clusterId"] == "cluster_id"
            assert kwargs["json"]["spec"]["image"] == "nvcr.io/nvidia/test:latest"
            assert kwargs["json"]["spec"]["numWorkers"] == 2
            assert kwargs["json"]["spec"]["compute"]["gpuDevicesRequest"] == 8
            assert kwargs["json"]["spec"]["environmentVariables"] == [
                {"name": "TEST_VAR", "value": "test_value"}
            ]
            assert kwargs["headers"] == executor._default_headers(token="test_token")

    @patch.object(DGXCloudExecutor, "get_auth_token")
    @patch.object(DGXCloudExecutor, "get_project_and_cluster_id")
    @patch.object(DGXCloudExecutor, "create_distributed_job")
    def test_launch_success(self, mock_create_job, mock_get_ids, mock_get_token):
        mock_get_token.return_value = "test_token"
        mock_get_ids.return_value = ("proj_id", "cluster_id")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"workloadId": "job123", "actualPhase": "Pending"}
        mock_create_job.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        job_id, status = executor.launch("test_job", ["python", "train.py"])

        assert job_id == "job123"
        assert status == "Pending"
        mock_get_token.assert_called_once()
        mock_get_ids.assert_called_once_with("test_token")
        mock_create_job.assert_called_once_with(
            "test_token", "proj_id", "cluster_id", "test-job", ["python", "train.py"]
        )

    @patch.object(DGXCloudExecutor, "get_auth_token")
    def test_launch_no_token(self, mock_get_token):
        mock_get_token.return_value = None

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        with pytest.raises(RuntimeError, match="Failed to get auth token"):
            executor.launch("test_job", ["python", "train.py"])

    @patch.object(DGXCloudExecutor, "get_auth_token")
    @patch.object(DGXCloudExecutor, "get_project_and_cluster_id")
    def test_launch_no_project_id(self, mock_get_ids, mock_get_token):
        mock_get_token.return_value = "test_token"
        mock_get_ids.return_value = (None, None)

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        with pytest.raises(RuntimeError, match="Unable to determine project/cluster IDs"):
            executor.launch("test_job", ["python", "train.py"])

    @patch.object(DGXCloudExecutor, "get_auth_token")
    @patch.object(DGXCloudExecutor, "get_project_and_cluster_id")
    @patch.object(DGXCloudExecutor, "create_distributed_job")
    def test_launch_job_creation_failed(self, mock_create_job, mock_get_ids, mock_get_token):
        mock_get_token.return_value = "test_token"
        mock_get_ids.return_value = ("proj_id", "cluster_id")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_create_job.return_value = mock_response

        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        with pytest.raises(RuntimeError, match="Failed to create job"):
            executor.launch("test_job", ["python", "train.py"])

    def test_nnodes(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            nodes=3,
        )

        assert executor.nnodes() == 3

    def test_nproc_per_node_with_gpus(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            gpus_per_node=4,
        )

        assert executor.nproc_per_node() == 4

    def test_nproc_per_node_with_nprocs(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            gpus_per_node=0,
            nprocs_per_node=3,
        )

        assert executor.nproc_per_node() == 3

    def test_nproc_per_node_default(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
            gpus_per_node=0,
            nprocs_per_node=0,
        )

        assert executor.nproc_per_node() == 1

    @patch("requests.get")
    def test_status(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"actualPhase": "Running"}
        mock_get.return_value = mock_response

        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="test_token"):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
            )

            status = executor.status("job123")

            assert status == DGXCloudState.RUNNING
            mock_get.assert_called_once_with(
                "https://dgxapi.example.com/workloads/distributed/job123",
                headers=executor._default_headers(token="test_token"),
            )

    @patch("requests.get")
    def test_status_no_token(self, mock_get):
        with patch.object(DGXCloudExecutor, "get_auth_token", return_value=None):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
            )

            status = executor.status("job123")

            assert status is None
            mock_get.assert_not_called()

    @patch("requests.get")
    def test_status_error_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="test_token"):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
            )

            status = executor.status("job123")

            assert status == DGXCloudState.UNKNOWN

    @patch("requests.get")
    def test_cancel(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.object(DGXCloudExecutor, "get_auth_token", return_value="test_token"):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
            )

            executor.cancel("job123")

            mock_get.assert_called_once_with(
                "https://dgxapi.example.com/workloads/distributed/job123/suspend",
                headers=executor._default_headers(token="test_token"),
            )

    @patch("requests.get")
    def test_cancel_no_token(self, mock_get):
        with patch.object(DGXCloudExecutor, "get_auth_token", return_value=None):
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
            )

            executor.cancel("job123")

            mock_get.assert_not_called()

    def test_logs(self):
        with patch("logging.Logger.warning") as mock_warning:
            DGXCloudExecutor.logs("app123", "/path/to/fallback")
            mock_warning.assert_called_once()
            assert "Logs not available" in mock_warning.call_args[0][0]

    def test_assign(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvcs=[{"path": tmp_dir, "claimName": "test-claim"}],
            )

            task_dir = "test_task"
            executor.assign(
                exp_id="test_exp",
                exp_dir=tmp_dir,
                task_id="test_task",
                task_dir=task_dir,
            )

            assert executor.job_name == "test_task"
            assert executor.experiment_dir == tmp_dir
            assert executor.job_dir == os.path.join(tmp_dir, task_dir)
            assert executor.experiment_id == "test_exp"

    def test_assign_no_pvc(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvcs=[{"path": "/other/path", "claimName": "test-claim"}],
            )

            with pytest.raises(AssertionError, match="Need to specify atleast one PVC"):
                executor.assign(
                    exp_id="test_exp",
                    exp_dir=tmp_dir,
                    task_id="test_task",
                    task_dir="test_task",
                )

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
            executor = DGXCloudExecutor(
                base_url="https://dgxapi.example.com",
                app_id="test_app_id",
                app_secret="test_app_secret",
                project_name="test_project",
                container_image="nvcr.io/nvidia/test:latest",
                pvcs=[{"path": tmp_dir, "claimName": "test-claim"}],
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
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        result = executor.macro_values()

        assert result is None

    def test_default_headers_without_token(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        headers = executor._default_headers()

        # Check that the headers include Content-Type but don't require an exact match on all fields
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    def test_default_headers_with_token(self):
        executor = DGXCloudExecutor(
            base_url="https://dgxapi.example.com",
            app_id="test_app_id",
            app_secret="test_app_secret",
            project_name="test_project",
            container_image="nvcr.io/nvidia/test:latest",
        )

        headers = executor._default_headers(token="test_token")

        # Check that the headers include Authorization but don't require an exact match on all fields
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"
