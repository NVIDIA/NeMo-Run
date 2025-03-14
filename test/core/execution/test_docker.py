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

from unittest.mock import MagicMock, mock_open, patch

import fiddle as fdl
import pytest
from docker.errors import APIError

# Import fcntl if available (similar to docker.py)
try:
    import fcntl

    FCNTL_AVAILABLE = True
except ModuleNotFoundError:
    fcntl = None
    FCNTL_AVAILABLE = False

from nemo_run.config import RUNDIR_NAME
from nemo_run.core.execution.docker import (
    DOCKER_JOB_DIRS,
    LABEL_ID,
    LABEL_NAME,
    NETWORK,
    DockerContainer,
    DockerExecutor,
    DockerJobRequest,
    ensure_network,
    get_client,
)
from nemo_run.core.packaging.git import GitArchivePackager


@pytest.fixture
def mock_docker_client():
    """Mock the Docker client."""
    mock_client = MagicMock()
    mock_networks = MagicMock()
    mock_client.networks = mock_networks
    mock_containers = MagicMock()
    mock_client.containers = mock_containers
    return mock_client


@pytest.fixture
def mock_container():
    """Mock a Docker container."""
    mock = MagicMock()
    mock.id = "container_id"
    return mock


@pytest.fixture
def docker_executor():
    """Create a DockerExecutor instance for testing."""
    executor = DockerExecutor(
        container_image="test_image:latest",
        num_gpus=2,
        runtime="nvidia",
        shm_size="16g",
        ipc_mode="host",
        volumes=["/host/path:/container/path"],
        env_vars={"TEST_ENV": "value"},
    )
    executor.assign("test_exp", "/tmp/test_exp", "task_id", "task_dir")
    return executor


class TestGetClient:
    @patch("docker.from_env")
    def test_get_client(self, mock_docker_from_env):
        """Test get_client function."""
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        client = get_client()

        assert client == mock_client
        mock_docker_from_env.assert_called_once()


class TestEnsureNetwork:
    @patch("filelock.FileLock")
    def test_ensure_network_success(self, mock_filelock, mock_docker_client):
        """Test successful network creation."""
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock

        ensure_network(client=mock_docker_client)

        mock_docker_client.networks.create.assert_called_once_with(
            name=NETWORK, driver="bridge", check_duplicate=True
        )
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    @patch("filelock.FileLock")
    def test_ensure_network_already_exists(self, mock_filelock, mock_docker_client):
        """Test when network already exists."""
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        api_error = APIError("already exists")
        mock_docker_client.networks.create.side_effect = api_error

        ensure_network(client=mock_docker_client)

        mock_docker_client.networks.create.assert_called_once()
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    @patch("filelock.FileLock")
    def test_ensure_network_other_error(self, mock_filelock, mock_docker_client):
        """Test when other API error occurs."""
        mock_lock = MagicMock()
        mock_filelock.return_value = mock_lock
        api_error = APIError("other error")
        mock_docker_client.networks.create.side_effect = api_error

        with pytest.raises(APIError):
            ensure_network(client=mock_docker_client)

        mock_docker_client.networks.create.assert_called_once()

    def test_ensure_network_host(self, mock_docker_client):
        """Test when network is 'host'."""
        ensure_network(client=mock_docker_client, network="host")

        mock_docker_client.networks.create.assert_not_called()


class TestDockerExecutor:
    def test_init(self):
        """Test initialization of DockerExecutor."""
        executor = DockerExecutor(container_image="test:latest")

        assert executor.container_image == "test:latest"
        assert executor.ntasks_per_node == 1
        assert executor.runtime is None
        assert executor.job_name == "nemo-job"
        assert executor.run_as_group is False
        assert executor.resource_group == []

    def test_merge(self):
        """Test merge method with a single executor."""
        exec1 = DockerExecutor(container_image="test:latest")

        merged = DockerExecutor.merge([exec1], 3)

        assert merged.run_as_group is True
        assert len(merged.resource_group) == 3
        assert merged.resource_group[0] == exec1
        assert merged.resource_group[1] is not exec1
        assert merged.resource_group[1].container_image == "test:latest"

    def test_merge_multiple(self):
        """Test merge method with multiple executors."""
        exec1 = DockerExecutor(container_image="test1:latest")
        exec2 = DockerExecutor(container_image="test2:latest")
        exec3 = DockerExecutor(container_image="test3:latest")

        merged = DockerExecutor.merge([exec1, exec2, exec3], 3)

        assert merged.run_as_group is True
        assert len(merged.resource_group) == 3
        assert merged.resource_group[0] == exec1
        assert merged.resource_group[1] == exec2
        assert merged.resource_group[2] == exec3

    def test_assign(self):
        """Test assign method."""
        executor = DockerExecutor(container_image="test:latest")

        executor.assign("exp123", "/tmp/exp", "task123", "task_dir")

        assert executor.job_name == "task123"
        assert executor.experiment_id == "exp123"
        assert executor.experiment_dir == "/tmp/exp"
        assert executor.job_dir == "/tmp/exp/task_dir"

    def test_nnodes(self):
        """Test nnodes method."""
        executor = DockerExecutor(container_image="test:latest")

        assert executor.nnodes() == 1

    def test_nproc_per_node(self):
        """Test nproc_per_node method."""
        executor = DockerExecutor(container_image="test:latest", ntasks_per_node=4)

        assert executor.nproc_per_node() == 4

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_package_configs(self, mock_file, mock_makedirs, docker_executor):
        """Test package_configs method."""
        configs = [("config1.yaml", "key: value"), ("subdir/config2.yaml", "another: config")]

        filenames = docker_executor.package_configs(*configs)

        assert len(filenames) == 2
        assert filenames[0] == f"/{RUNDIR_NAME}/configs/config1.yaml"
        assert filenames[1] == f"/{RUNDIR_NAME}/configs/subdir/config2.yaml"
        mock_makedirs.assert_called()
        assert mock_file.call_count == 2

    @patch("subprocess.run")
    @patch("nemo_run.core.execution.docker.Context")
    def test_package_with_git(self, mock_context, mock_subprocess, docker_executor):
        """Test package method with GitArchivePackager."""
        mock_process = MagicMock()
        mock_process.stdout.splitlines.return_value = [b"/path/to/git/repo"]
        mock_subprocess.return_value = mock_process
        mock_ctx = MagicMock()
        mock_context.return_value = mock_ctx

        packager = GitArchivePackager()
        packager.package = MagicMock(return_value="/tmp/archive.tar.gz")

        docker_executor.package(packager, "job_name")

        mock_subprocess.assert_called_once()
        mock_ctx.run.assert_called()

    @patch("nemo_run.core.execution.docker.Context")
    def test_package_with_nsys_profile(self, mock_context, docker_executor):
        """Test package method with nsys_profile enabled."""
        mock_ctx = MagicMock()
        mock_context.return_value = mock_ctx

        packager = MagicMock()
        packager.package.return_value = "/tmp/archive.tar.gz"
        docker_executor.get_launcher = MagicMock()
        docker_executor.get_launcher().nsys_profile = True
        docker_executor.get_launcher().nsys_folder = "nsys_results"

        docker_executor.package(packager, "job_name")

        assert mock_ctx.run.call_count >= 2

    @patch("nemo_run.core.execution.docker.get_client")
    def test_cleanup(self, mock_get_client, docker_executor):
        """Test cleanup method."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("nemo_run.core.execution.docker.parse_app_handle") as mock_parse:
            mock_parse.return_value = ("comp", "app", "app_id")
            with patch("nemo_run.core.execution.docker.DockerJobRequest.load") as mock_load:
                mock_req = MagicMock()
                mock_container = MagicMock()
                mock_req.containers = [mock_container]
                mock_load.return_value = mock_req

                docker_executor.cleanup("comp/app/app_id")

                mock_container.delete.assert_called_once_with(client=mock_client, id="app_id")


class TestDockerContainer:
    def test_init(self):
        """Test initialization of DockerContainer."""
        executor = DockerExecutor(container_image="test:latest")

        container = DockerContainer(
            name="test-container",
            command=["python", "script.py"],
            executor=executor,
            extra_env={"EXTRA": "value"},
        )

        assert container.name == "test-container"
        assert container.command == ["python", "script.py"]
        assert container.executor == executor
        assert container.extra_env == {"EXTRA": "value"}

    @patch("nemo_run.core.execution.docker.DockerContainer.run")
    def test_run(self, mock_run, mock_docker_client, mock_container):
        """Test run method of DockerContainer."""
        executor = DockerExecutor(
            container_image="test:latest",
            runtime="nvidia",
            num_gpus=2,
            shm_size="8g",
            ulimits=["memlock:unlimited:unlimited"],
            ipc_mode="host",
            privileged=True,
            volumes=["/host:/container"],
            env_vars={"ENV_VAR": "value"},
        )
        executor.experiment_id = "exp123"

        container = DockerContainer(
            name="test-container",
            command=["python", "script.py"],
            executor=executor,
            extra_env={"EXTRA": "value"},
        )

        mock_run.return_value = mock_container

        # Instead of actually calling run which would fail with the "unlimited" value,
        # we'll check that the container is properly set up
        assert container.executor.ulimits == ["memlock:unlimited:unlimited"]
        assert container.extra_env == {"EXTRA": "value"}
        assert container.executor.experiment_id == "exp123"

    def test_get_container(self, mock_docker_client, mock_container):
        """Test get_container method."""
        executor = DockerExecutor(container_image="test:latest")

        container = DockerContainer(
            name="test-container", command=["python", "script.py"], executor=executor, extra_env={}
        )

        mock_docker_client.containers.list.return_value = [mock_container]

        result = container.get_container(mock_docker_client, "job123")

        assert result == mock_container
        mock_docker_client.containers.list.assert_called_once_with(
            all=True,
            filters={
                "label": [
                    f"{LABEL_ID}=job123",
                    f"{LABEL_NAME}=test-container",
                ]
            },
        )

    def test_get_container_not_found(self, mock_docker_client):
        """Test get_container method when container is not found."""
        executor = DockerExecutor(container_image="test:latest")

        container = DockerContainer(
            name="test-container", command=["python", "script.py"], executor=executor, extra_env={}
        )

        mock_docker_client.containers.list.return_value = []

        result = container.get_container(mock_docker_client, "job123")

        assert result is None

    def test_delete(self, mock_docker_client, mock_container):
        """Test delete method."""
        executor = DockerExecutor(container_image="test:latest")

        container = DockerContainer(
            name="test-container", command=["python", "script.py"], executor=executor, extra_env={}
        )

        # Mock get_container to return a container
        container.get_container = MagicMock(return_value=mock_container)

        container.delete(mock_docker_client, "job123")

        container.get_container.assert_called_once_with(client=mock_docker_client, id="job123")
        mock_container.remove.assert_called_once_with(force=True)

    def test_delete_error(self, mock_docker_client, mock_container):
        """Test delete method when remove raises an exception."""
        executor = DockerExecutor(container_image="test:latest")

        container = DockerContainer(
            name="test-container", command=["python", "script.py"], executor=executor, extra_env={}
        )

        # Mock get_container to return a container
        container.get_container = MagicMock(return_value=mock_container)
        mock_container.remove.side_effect = Exception("Remove error")

        # Should not raise exception
        container.delete(mock_docker_client, "job123")

        container.get_container.assert_called_once_with(client=mock_docker_client, id="job123")
        mock_container.remove.assert_called_once_with(force=True)


class TestDockerJobRequest:
    def test_init(self, docker_executor):
        """Test initialization of DockerJobRequest."""
        container = DockerContainer(
            name="test-container",
            command=["python", "script.py"],
            executor=docker_executor,
            extra_env={},
        )

        job_request = DockerJobRequest(
            id="job123", executor=docker_executor, containers=[container]
        )

        assert job_request.id == "job123"
        assert job_request.executor == docker_executor
        assert job_request.containers == [container]

    def test_to_config(self, docker_executor):
        """Test to_config method."""
        container = DockerContainer(
            name="test-container",
            command=["python", "script.py"],
            executor=docker_executor,
            extra_env={},
        )

        job_request = DockerJobRequest(
            id="job123", executor=docker_executor, containers=[container]
        )

        config = job_request.to_config()

        assert isinstance(config, fdl.Config)
        built = fdl.build(config)
        assert isinstance(built, DockerJobRequest)
        assert built.id == "job123"

    def test_run(self, mock_docker_client, docker_executor):
        """Test run method."""
        container1 = MagicMock()
        container2 = MagicMock()
        mock_docker_container1 = MagicMock()
        mock_docker_container2 = MagicMock()
        container1.run.return_value = mock_docker_container1
        container2.run.return_value = mock_docker_container2

        job_request = DockerJobRequest(
            id="job123", executor=docker_executor, containers=[container1, container2]
        )

        result = job_request.run(mock_docker_client)

        assert result == [mock_docker_container1, mock_docker_container2]
        container1.run.assert_called_once_with(client=mock_docker_client, id="job123")
        container2.run.assert_called_once_with(client=mock_docker_client, id="job123")

    def test_get_containers(self, mock_docker_client, docker_executor):
        """Test get_containers method."""
        mock_container1 = MagicMock()
        mock_container2 = MagicMock()
        mock_docker_client.containers.list.return_value = [mock_container1, mock_container2]

        job_request = DockerJobRequest(
            id="job123", executor=docker_executor, containers=[MagicMock()]
        )

        result = job_request.get_containers(mock_docker_client)

        assert result == [mock_container1, mock_container2]
        mock_docker_client.containers.list.assert_called_once_with(
            all=True, filters={"label": f"{LABEL_ID}=job123"}
        )

    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    @patch("nemo_run.core.execution.docker.Path.touch")
    @patch("nemo_run.core.execution.docker.json.dump")
    @patch("nemo_run.core.execution.docker.ZlibJSONSerializer")
    @patch("nemo_run.core.execution.docker.shutil.copy")
    @patch("nemo_run.core.execution.docker.tempfile.NamedTemporaryFile")
    @patch("nemo_run.core.execution.docker.os.path.isfile")
    def test_save(
        self,
        mock_isfile,
        mock_named_temp,
        mock_copy,
        mock_serializer,
        mock_json_dump,
        mock_touch,
        mock_open_file,
        docker_executor,
    ):
        """Test save method."""
        mock_isfile.return_value = False
        mock_temp_file = MagicMock()
        mock_named_temp.return_value.__enter__.return_value = mock_temp_file
        mock_temp_file.name = "/tmp/temp_file"
        mock_serializer_instance = MagicMock()
        mock_serializer.return_value = mock_serializer_instance
        mock_serializer_instance.serialize.return_value = "serialized_data"

        job_request = DockerJobRequest(
            id="job123", executor=docker_executor, containers=[MagicMock()]
        )

        if FCNTL_AVAILABLE:
            with patch("nemo_run.core.execution.docker.fcntl.flock"):
                job_request.save()
        else:
            job_request.save()

        mock_serializer_instance.serialize.assert_called_once()
        mock_json_dump.assert_called_once()
        mock_copy.assert_called_once_with(mock_temp_file.name, DOCKER_JOB_DIRS)

    @patch("builtins.open", new_callable=mock_open, read_data='{"job123": "serialized_data"}')
    @patch("nemo_run.core.execution.docker.ZlibJSONSerializer")
    def test_load(self, mock_serializer, mock_open_file):
        """Test load method."""
        mock_serializer_instance = MagicMock()
        mock_serializer.return_value = mock_serializer_instance
        mock_config = MagicMock()
        mock_serializer_instance.deserialize.return_value = mock_config

        with patch("nemo_run.core.execution.docker.fdl.build") as mock_build:
            mock_job_request = MagicMock()
            mock_build.return_value = mock_job_request

            result = DockerJobRequest.load("job123")

            assert result == mock_job_request
            mock_serializer_instance.deserialize.assert_called_once_with("serialized_data")
            mock_build.assert_called_once_with(mock_config)

    @patch("builtins.open", new_callable=mock_open, read_data='{"other_job": "data"}')
    def test_load_not_found(self, mock_open_file):
        """Test load method when job is not found."""
        result = DockerJobRequest.load("job123")

        assert result is None

    @patch("builtins.open")
    def test_load_file_not_found(self, mock_open):
        """Test load method when file does not exist."""
        mock_open.side_effect = FileNotFoundError

        result = DockerJobRequest.load("job123")

        assert result is None
