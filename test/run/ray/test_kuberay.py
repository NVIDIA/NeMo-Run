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

import os
import queue
import subprocess
import threading
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml
from kubernetes.client.rest import ApiException

from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup
from nemo_run.run.ray.kuberay import KubeRayCluster, KubeRayJob, get_user

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "core", "execution", "artifacts"
)


class TestKubeRayCluster:
    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(
            namespace="test-namespace",
            ray_version="2.43.0",
            head_cpu="2",
            head_memory="4Gi",
            worker_groups=[
                KubeRayWorkerGroup(
                    group_name="workers",
                    replicas=2,
                    cpu_requests="4",
                    memory_requests="8Gi",
                    gpus_per_worker=1,
                )
            ],
        )

    @pytest.fixture
    def advanced_executor(self):
        """Create an advanced KubeRayExecutor with volumes and custom settings."""
        return KubeRayExecutor(
            namespace="production",
            ray_version="2.43.0",
            image="custom/ray:latest",
            head_cpu="4",
            head_memory="16Gi",
            ray_head_start_params={"dashboard-host": "0.0.0.0", "log-level": "debug"},
            ray_worker_start_params={"num-cpus": "8"},
            worker_groups=[
                KubeRayWorkerGroup(
                    group_name="gpu-workers",
                    replicas=4,
                    min_replicas=2,
                    max_replicas=8,
                    cpu_requests="8",
                    memory_requests="32Gi",
                    gpus_per_worker=2,
                ),
                KubeRayWorkerGroup(
                    group_name="cpu-workers",
                    replicas=2,
                    cpu_requests="16",
                    memory_requests="64Gi",
                ),
            ],
            volumes=[
                {"name": "data-volume", "persistentVolumeClaim": {"claimName": "data-pvc"}},
                {"name": "model-volume", "persistentVolumeClaim": {"claimName": "model-pvc"}},
            ],
            volume_mounts=[
                {"name": "data-volume", "mountPath": "/data"},
                {"name": "model-volume", "mountPath": "/models"},
            ],
            labels={"team": "ml", "project": "training"},
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1", "NCCL_DEBUG": "INFO"},
            lifecycle_kwargs={
                "postStart": {"exec": {"command": ["/bin/sh", "-c", "echo 'Starting Ray'"]}}
            },
        )

    @pytest.fixture
    def cluster_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayCluster with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayCluster(name="test-cluster", executor=basic_executor)

    @pytest.fixture
    def cluster_with_advanced_executor(self, advanced_executor, mock_k8s_clients):
        """Create a KubeRayCluster with advanced executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayCluster(name="advanced-cluster", executor=advanced_executor)

    def test_cluster_initialization(self, cluster_with_basic_executor):
        """Test cluster initialization."""
        cluster = cluster_with_basic_executor
        assert cluster.name == "test-cluster"
        assert cluster.cluster_name == "testuser-test-cluster-raycluster"
        assert cluster.user == "testuser"
        assert cluster.executor.namespace == "test-namespace"

    def test_get_cluster_not_found(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test getting a non-existent cluster."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object.side_effect = ApiException(status=404)

        result = cluster_with_basic_executor._get()
        assert result is None

    def test_get_cluster_success(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test successfully getting a cluster."""
        mock_api, _ = mock_k8s_clients
        expected_resource = {
            "metadata": {"name": "testuser-test-cluster-raycluster", "namespace": "test-namespace"},
            "status": {"state": "ready", "head": {"serviceIP": "10.0.0.1"}},
        }
        mock_api.get_namespaced_custom_object.return_value = expected_resource

        result = cluster_with_basic_executor._get()
        assert result == expected_resource

    def test_status_with_timeout(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test status method with timeout."""
        mock_api, _ = mock_k8s_clients
        # First call returns no status, second call returns status
        mock_api.get_namespaced_custom_object_status.side_effect = [
            {"metadata": {"name": "test"}, "status": None},
            {
                "metadata": {"name": "test"},
                "status": {"state": "ready", "head": {"serviceIP": "10.0.0.1"}},
            },
        ]

        with patch("time.sleep"):
            status = cluster_with_basic_executor.status(timeout=10, delay_between_attempts=1)

        assert status == {"state": "ready", "head": {"serviceIP": "10.0.0.1"}}

    def test_wait_until_running_success(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test waiting for cluster to be running."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock status responses
        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {"serviceIP": "10.0.0.1"}}

            # Mock pod listing
            mock_pod = Mock()
            mock_pod.metadata.labels = {"ray.io/node-type": "head"}
            mock_pod.metadata.name = "test-cluster-head"
            mock_pod.status.phase = "Running"
            mock_pod.status.conditions = [Mock(type="Ready", status="True")]

            mock_pods = Mock()
            mock_pods.items = [mock_pod]
            mock_core_api.list_namespaced_pod.return_value = mock_pods

            result = cluster_with_basic_executor.wait_until_running(timeout=10)
            assert result is True

    def test_wait_until_running_timeout(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test wait_until_running timeout."""
        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {}}  # No serviceIP

            with patch("time.sleep"):
                result = cluster_with_basic_executor.wait_until_running(
                    timeout=1, delay_between_attempts=0.5
                )
            assert result is False

    def test_create_basic_cluster(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test creating a basic Ray cluster."""
        mock_api, _ = mock_k8s_clients

        # Create expected body from artifact
        os.path.join(ARTIFACTS_DIR, "expected_kuberay_cluster_basic.yaml")

        # Mock the API response
        mock_api.create_namespaced_custom_object.return_value = {
            "metadata": {"name": "test-cluster"}
        }

        cluster_with_basic_executor.create()

        # Verify API was called
        assert mock_api.create_namespaced_custom_object.called
        call_args = mock_api.create_namespaced_custom_object.call_args

        # Verify basic structure
        body = call_args.kwargs["body"]
        assert body["apiVersion"] == "ray.io/v1alpha1"
        assert body["kind"] == "RayCluster"
        assert body["metadata"]["name"] == "testuser-test-cluster-raycluster"
        assert body["metadata"]["namespace"] == "test-namespace"
        assert body["spec"]["rayVersion"] == "2.43.0"

    def test_create_advanced_cluster(self, cluster_with_advanced_executor, mock_k8s_clients):
        """Test creating an advanced Ray cluster with volumes and custom settings."""
        mock_api, _ = mock_k8s_clients

        # Mock the API response
        mock_api.create_namespaced_custom_object.return_value = {
            "metadata": {"name": "advanced-cluster"}
        }

        # Add pre-ray-start commands
        pre_commands = ["export PYTHONPATH=/app", "pip install -r requirements.txt"]
        cluster_with_advanced_executor.create(pre_ray_start_commands=pre_commands)

        # Verify API was called
        assert mock_api.create_namespaced_custom_object.called
        call_args = mock_api.create_namespaced_custom_object.call_args

        # Verify advanced structure
        body = call_args.kwargs["body"]
        assert body["metadata"]["namespace"] == "production"
        assert body["metadata"]["labels"] == {"team": "ml", "project": "training"}

        # Check head spec
        head_spec = body["spec"]["headGroupSpec"]
        assert head_spec["rayStartParams"]["dashboard-host"] == "0.0.0.0"
        assert head_spec["rayStartParams"]["log-level"] == "debug"

        head_container = head_spec["template"]["spec"]["containers"][0]
        assert head_container["image"] == "custom/ray:latest"
        assert head_container["resources"]["requests"]["cpu"] == "4"
        assert head_container["resources"]["requests"]["memory"] == "16Gi"

        # Check lifecycle hooks
        assert "postStart" in head_container["lifecycle"]
        assert head_container["lifecycle"]["postStart"]["exec"]["command"][2] == "\n".join(
            pre_commands
        )

        # Check volumes
        assert len(head_spec["template"]["spec"]["volumes"]) == 2
        assert head_spec["template"]["spec"]["volumes"][0]["name"] == "data-volume"

        # Check worker groups
        worker_specs = body["spec"]["workerGroupSpecs"]
        assert len(worker_specs) == 2

        # GPU workers
        gpu_workers = worker_specs[0]
        assert gpu_workers["groupName"] == "gpu-workers"
        assert gpu_workers["replicas"] == 4
        assert gpu_workers["minReplicas"] == 2
        assert gpu_workers["maxReplicas"] == 8
        assert (
            gpu_workers["template"]["spec"]["containers"][0]["resources"]["requests"][
                "nvidia.com/gpu"
            ]
            == 2
        )

    def test_create_cluster_already_exists(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test creating a cluster that already exists."""
        mock_api, _ = mock_k8s_clients
        mock_api.create_namespaced_custom_object.side_effect = ApiException(
            status=409, reason="AlreadyExists"
        )

        result = cluster_with_basic_executor.create()
        assert result is None

    def test_create_cluster_dryrun(self, cluster_with_basic_executor, mock_k8s_clients, capsys):
        """Test creating a cluster in dryrun mode."""
        result = cluster_with_basic_executor.create(dryrun=True)

        # Check that YAML was printed
        captured = capsys.readouterr()
        assert "apiVersion: ray.io/v1alpha1" in captured.out
        assert "kind: RayCluster" in captured.out

        # Verify structure is returned
        assert result["apiVersion"] == "ray.io/v1alpha1"
        assert result["kind"] == "RayCluster"

    def test_delete_cluster_success(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test deleting a cluster successfully."""
        mock_api, mock_core_api = mock_k8s_clients

        result = cluster_with_basic_executor.delete()
        assert result is True

        mock_api.delete_namespaced_custom_object.assert_called_once()

    def test_delete_cluster_with_wait(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test deleting a cluster and waiting for completion."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock cluster deletion
        mock_api.delete_namespaced_custom_object.return_value = None

        # Mock get calls - first exists, then doesn't
        mock_api.get_namespaced_custom_object.side_effect = [
            {"metadata": {"name": "test"}},
            ApiException(status=404),
        ]

        # Mock pod listing - first has pods, then empty
        mock_pod = Mock()
        mock_pod.metadata.name = "test-pod"  # Set the name attribute properly

        mock_pods_with_items = Mock()
        mock_pods_with_items.items = [mock_pod]

        mock_pods_empty = Mock()
        mock_pods_empty.items = []

        mock_core_api.list_namespaced_pod.side_effect = [
            mock_pods_with_items,
            mock_pods_empty,
        ]

        with patch("time.sleep"):
            result = cluster_with_basic_executor.delete(wait=True, timeout=10, poll_interval=1)

        assert result is True

    def test_delete_cluster_not_found(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test deleting a non-existent cluster."""
        mock_api, _ = mock_k8s_clients
        mock_api.delete_namespaced_custom_object.side_effect = ApiException(status=404)

        result = cluster_with_basic_executor.delete()
        assert result is None

    def test_patch_cluster(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test patching a cluster."""
        mock_api, _ = mock_k8s_clients

        patch_body = {"spec": {"workerGroupSpecs": [{"replicas": 5}]}}
        result = cluster_with_basic_executor.patch(patch_body)

        assert result is True
        mock_api.patch_namespaced_custom_object.assert_called_once()

    def test_port_forward_success(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test successful port forwarding."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock cluster exists
        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = {"metadata": {"namespace": "test-namespace"}}

            # Mock service exists
            mock_core_api.read_namespaced_service.return_value = Mock()

            # Mock subprocess
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.poll.return_value = None  # Process is running
                mock_process.returncode = None
                mock_popen.return_value = mock_process

                # Mock queue for thread communication
                with patch("queue.Queue") as mock_queue_class:
                    mock_queue = Mock()
                    mock_queue.get.return_value = ("success", None)
                    mock_queue_class.return_value = mock_queue

                    thread = cluster_with_basic_executor.port_forward(port=8080, target_port=8265)

                    assert isinstance(thread, threading.Thread)
                    assert thread.daemon is True

    def test_port_forward_cluster_not_found(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test port forwarding when cluster doesn't exist."""
        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = None

            with pytest.raises(RuntimeError, match="Could not find Ray cluster"):
                cluster_with_basic_executor.port_forward(port=8080, target_port=8265)


class TestKubeRayJob:
    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(
            namespace="test-namespace",
            volumes=[
                {"name": "workspace", "persistentVolumeClaim": {"claimName": "workspace-pvc"}}
            ],
            volume_mounts=[{"name": "workspace", "mountPath": "/workspace"}],
        )

    @pytest.fixture
    def job_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayJob with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayJob(name="test-job", executor=basic_executor)

    def test_job_initialization(self, job_with_basic_executor):
        """Test job initialization."""
        job = job_with_basic_executor
        assert job.name == "test-job"
        assert job.job_name == "testuser-test-job-rayjob"
        assert job.user == "testuser"

    def test_stop_job(self, job_with_basic_executor, mock_k8s_clients):
        """Test stopping a job."""
        mock_api, _ = mock_k8s_clients

        job_with_basic_executor.stop()

        mock_api.delete_namespaced_custom_object.assert_called_once_with(
            group="ray.io",
            version="v1",
            plural="rayjobs",
            name="testuser-test-job-rayjob",
            namespace="test-namespace",
        )

    def test_stop_job_not_found(self, job_with_basic_executor, mock_k8s_clients):
        """Test stopping a non-existent job."""
        mock_api, _ = mock_k8s_clients
        mock_api.delete_namespaced_custom_object.side_effect = ApiException(status=404)

        # Should not raise, just log warning
        job_with_basic_executor.stop()

    def test_logs_follow(self, job_with_basic_executor):
        """Test following logs."""
        with patch("subprocess.run") as mock_run:
            job_with_basic_executor.logs(follow=True, lines=100, timeout=30)

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "kubectl" in args
            assert "logs" in args
            assert "-f" in args
            assert f"job-name={job_with_basic_executor.job_name}" in " ".join(args)

    def test_logs_no_follow(self, job_with_basic_executor, capsys):
        """Test showing logs without following."""
        with patch("subprocess.check_output") as mock_output:
            mock_output.return_value = "Log line 1\nLog line 2\n"

            job_with_basic_executor.logs(follow=False, lines=50)

            captured = capsys.readouterr()
            assert "Log line 1" in captured.out
            assert "Log line 2" in captured.out

    def test_status(self, job_with_basic_executor, mock_k8s_clients):
        """Test getting job status."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object.return_value = {
            "status": {
                "jobStatus": "RUNNING",
                "jobDeploymentStatus": "Running",
            }
        }

        status = job_with_basic_executor.status(display=False)

        assert status["jobStatus"] == "RUNNING"
        assert status["jobDeploymentStatus"] == "Running"

    def test_start_basic_job(self, job_with_basic_executor, mock_k8s_clients):
        """Test starting a basic Ray job."""
        mock_api, _ = mock_k8s_clients

        job_with_basic_executor.start(
            command="python train.py",
            workdir=None,
        )

        # Verify RayJob creation
        mock_api.create_namespaced_custom_object.assert_called_once()
        call_args = mock_api.create_namespaced_custom_object.call_args

        body = call_args.kwargs["body"]
        assert body["apiVersion"] == "ray.io/v1"
        assert body["kind"] == "RayJob"
        assert body["metadata"]["name"] == "testuser-test-job-rayjob"
        assert body["spec"]["entrypoint"] == "python train.py"
        assert body["spec"]["shutdownAfterJobFinishes"] is True

    def test_start_job_with_workdir(self, job_with_basic_executor, mock_k8s_clients):
        """Test starting a job with workdir sync."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock pod creation and status
        mock_core_api.create_namespaced_pod.return_value = None

        # Mock watch for pod status
        with patch("nemo_run.core.execution.kuberay.watch.Watch") as mock_watch_class:
            mock_watch = Mock()
            mock_event = {"object": Mock(status=Mock(phase="Running"))}
            mock_watch.stream.return_value = [mock_event]
            mock_watch_class.return_value = mock_watch

            # Mock subprocess calls for rsync - sync_workdir_via_pod uses subprocess.check_call
            with patch("nemo_run.core.execution.kuberay.subprocess.check_call") as mock_check_call:
                mock_check_call.return_value = None

                job_with_basic_executor.start(
                    command="python train.py",
                    workdir="/local/path",
                )

                # Verify data mover pod was created
                mock_core_api.create_namespaced_pod.assert_called_once()

                # Verify rsync was called (subprocess.check_call is called twice - mkdir and rsync)
                assert mock_check_call.call_count >= 2

    def test_start_job_with_runtime_env(self, job_with_basic_executor, mock_k8s_clients):
        """Test starting a job with runtime environment YAML."""
        mock_api, _ = mock_k8s_clients

        runtime_env = """
pip:
  - numpy
  - pandas
env_vars:
  MY_VAR: value
"""

        with patch("os.path.isfile", return_value=False):  # Treat as string, not file
            job_with_basic_executor.start(
                command="python train.py",
                runtime_env_yaml=runtime_env,
            )

        call_args = mock_api.create_namespaced_custom_object.call_args
        body = call_args.kwargs["body"]
        assert body["spec"]["runtimeEnvYAML"] == runtime_env

    def test_start_job_dryrun(self, job_with_basic_executor, mock_k8s_clients, capsys):
        """Test starting a job in dryrun mode."""
        result = job_with_basic_executor.start(
            command="python train.py",
            dryrun=True,
        )

        # Check that YAML was printed
        captured = capsys.readouterr()
        assert "apiVersion: ray.io/v1" in captured.out
        assert "kind: RayJob" in captured.out

        # Verify structure is returned
        assert result["apiVersion"] == "ray.io/v1"
        assert result["kind"] == "RayJob"

    def test_follow_logs_until_completion(self, job_with_basic_executor):
        """Test following logs until job completion."""
        # Mock status progression
        status_sequence = [
            {"jobDeploymentStatus": "Pending"},
            {"jobDeploymentStatus": "Running"},
            {"jobDeploymentStatus": "Running"},
            {"jobDeploymentStatus": "Complete"},
        ]

        with patch.object(job_with_basic_executor, "status") as mock_status:
            mock_status.side_effect = status_sequence

            with patch.object(job_with_basic_executor, "logs") as mock_logs:
                with patch.object(job_with_basic_executor, "stop") as mock_stop:
                    with patch("time.sleep"):
                        job_with_basic_executor.follow_logs_until_completion(
                            poll_interval=1,
                            delete_on_finish=True,
                        )

                    # Verify logs were called with follow=True
                    mock_logs.assert_called_once_with(follow=True)

                    # Verify job was deleted
                    mock_stop.assert_called_once()


class TestUtilityFunctions:
    def test_get_user_normal(self):
        """Test getting username normally."""
        with patch("getpass.getuser", return_value="TestUser"):
            user = get_user()
            assert user == "testuser"  # Should be lowercased

    def test_get_user_with_special_chars(self):
        """Test getting username with special characters."""
        with patch("getpass.getuser", return_value="test.user@domain"):
            user = get_user()
            assert user == "test-user-domain"  # Special chars replaced with hyphens

    def test_get_user_fallback_to_env(self):
        """Test falling back to environment variables."""
        with patch("getpass.getuser", side_effect=Exception("No user")):
            with patch.dict(os.environ, {"USER": "env_user"}):
                user = get_user()
                assert user == "env-user"

    def test_get_user_unknown(self):
        """Test unknown user fallback."""
        with patch("getpass.getuser", side_effect=Exception("No user")):
            with patch.dict(os.environ, {}, clear=True):
                user = get_user()
                assert user == "unknown"


class TestKubeRayArtifacts:
    """Test that generated Kubernetes resources match expected artifacts."""

    def test_basic_cluster_artifact(self):
        """Test basic cluster generation matches artifact."""
        artifact_path = os.path.join(ARTIFACTS_DIR, "expected_kuberay_cluster_basic.yaml")

        # Create the artifact if it doesn't exist
        executor = KubeRayExecutor(
            namespace="default",
            ray_version="2.43.0",
            head_cpu="1",
            head_memory="2Gi",
            worker_groups=[
                KubeRayWorkerGroup(
                    group_name="workers",
                    replicas=2,
                    cpu_requests="2",
                    memory_requests="4Gi",
                )
            ],
        )

        body = executor.get_cluster_body("test-cluster")

        # Save artifact for reference
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "w") as f:
            yaml.dump(body, f, default_flow_style=False, sort_keys=False)

        # Verify structure
        assert body["apiVersion"] == "ray.io/v1alpha1"
        assert body["kind"] == "RayCluster"
        assert body["spec"]["rayVersion"] == "2.43.0"
        assert len(body["spec"]["workerGroupSpecs"]) == 1

    def test_advanced_cluster_artifact(self):
        """Test advanced cluster generation with GPUs and volumes."""
        artifact_path = os.path.join(ARTIFACTS_DIR, "expected_kuberay_cluster_advanced.yaml")

        executor = KubeRayExecutor(
            namespace="ml-team",
            ray_version="2.43.0",
            image="custom/ray:gpu",
            head_cpu="4",
            head_memory="16Gi",
            ray_head_start_params={"dashboard-host": "0.0.0.0", "num-cpus": "4"},
            worker_groups=[
                KubeRayWorkerGroup(
                    group_name="gpu-workers",
                    replicas=4,
                    min_replicas=2,
                    max_replicas=8,
                    cpu_requests="8",
                    memory_requests="32Gi",
                    gpus_per_worker=2,
                )
            ],
            volumes=[
                {"name": "data", "persistentVolumeClaim": {"claimName": "data-pvc"}},
            ],
            volume_mounts=[
                {"name": "data", "mountPath": "/data"},
            ],
            labels={"team": "ml", "env": "prod"},
            env_vars={"NCCL_DEBUG": "INFO"},
        )

        body = executor.get_cluster_body("ml-training-cluster")

        # Save artifact
        with open(artifact_path, "w") as f:
            yaml.dump(body, f, default_flow_style=False, sort_keys=False)

        # Verify GPU resources
        worker_spec = body["spec"]["workerGroupSpecs"][0]
        resources = worker_spec["template"]["spec"]["containers"][0]["resources"]
        assert resources["requests"]["nvidia.com/gpu"] == 2
        assert resources["limits"]["nvidia.com/gpu"] == 2

    def test_rayjob_artifact(self):
        """Test RayJob generation matches expected structure."""
        artifact_path = os.path.join(ARTIFACTS_DIR, "expected_kuberay_job_basic.yaml")

        # This would be generated by the start() method
        job_body = {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {
                "name": "test-job",
                "namespace": "default",
            },
            "spec": {
                "entrypoint": "python train.py",
                "shutdownAfterJobFinishes": True,
                "rayClusterSpec": {},  # Would contain full cluster spec
                "runtimeEnvYAML": None,
            },
        }

        # Save artifact
        with open(artifact_path, "w") as f:
            yaml.dump(job_body, f, default_flow_style=False, sort_keys=False)

        assert job_body["apiVersion"] == "ray.io/v1"
        assert job_body["kind"] == "RayJob"


class TestKubeRayExecutorUtilityFunctions:
    """Test utility functions from nemo_run.core.execution.kuberay module."""

    @pytest.fixture
    def basic_cluster_dict(self):
        """Create a basic cluster dictionary for testing."""
        from nemo_run.core.execution.kuberay import (
            populate_meta,
            populate_ray_head,
            populate_worker_group,
        )

        cluster = {}
        cluster = populate_meta(cluster, "test-cluster", "default", {}, "2.43.0")
        cluster = populate_ray_head(
            cluster,
            ray_image="rayproject/ray:2.43.0",
            service_type="ClusterIP",
            cpu_requests="1",
            memory_requests="2Gi",
            cpu_limits="1",
            memory_limits="2Gi",
            ray_start_params={"dashboard-host": "0.0.0.0"},
            head_ports=[],
            env_vars={},
            volume_mounts=[],
            volumes=[],
            spec_kwargs={},
            lifecycle_kwargs={},
            container_kwargs={},
        )
        cluster = populate_worker_group(
            cluster,
            group_name="workers",
            ray_image="rayproject/ray:2.43.0",
            gpus_per_worker=None,
            cpu_requests="2",
            memory_requests="4Gi",
            cpu_limits="2",
            memory_limits="4Gi",
            replicas=2,
            min_replicas=2,
            max_replicas=2,
            ray_start_params={},
            volume_mounts=[],
            volumes=[],
            labels={},
            annotations={},
            spec_kwargs={},
            lifecycle_kwargs={},
            container_kwargs={},
            env_vars={},
        )
        return cluster

    def test_is_valid_name_valid(self):
        """Test valid name validation."""
        from nemo_run.core.execution.kuberay import is_valid_name

        assert is_valid_name("test-cluster") is True
        assert is_valid_name("cluster.test") is True
        assert is_valid_name("cluster123") is True

    def test_is_valid_name_invalid(self):
        """Test invalid name validation."""
        from nemo_run.core.execution.kuberay import is_valid_name

        assert is_valid_name("") is False
        assert is_valid_name("Test-Cluster") is False  # uppercase
        assert is_valid_name("test_cluster") is False  # underscore
        assert is_valid_name("test cluster") is False  # space
        assert is_valid_name("a" * 64) is False  # too long

    def test_is_valid_label_valid(self):
        """Test valid label validation."""
        from nemo_run.core.execution.kuberay import is_valid_label

        assert is_valid_label("test-label") is True
        assert is_valid_label("label.test") is True
        assert is_valid_label("label_test") is True

    def test_is_valid_label_invalid(self):
        """Test invalid label validation."""
        from nemo_run.core.execution.kuberay import is_valid_label

        assert is_valid_label("") is False
        assert is_valid_label("Test-Label") is False  # uppercase
        assert is_valid_label("test label") is False  # space
        assert is_valid_label("a" * 64) is False  # too long

    def test_populate_meta_invalid_name(self):
        """Test populate_meta with invalid cluster name."""
        from nemo_run.core.execution.kuberay import populate_meta

        with pytest.raises(AssertionError, match="Invalid cluster name"):
            populate_meta({}, "Invalid_Name", "default", {}, "2.43.0")

    def test_populate_ray_head_missing_spec(self):
        """Test populate_ray_head with missing spec in cluster."""
        from nemo_run.core.execution.kuberay import populate_ray_head

        cluster = {}  # No spec
        result = populate_ray_head(
            cluster,
            ray_image="rayproject/ray:2.43.0",
            service_type="ClusterIP",
            cpu_requests="1",
            memory_requests="2Gi",
            cpu_limits="1",
            memory_limits="2Gi",
            ray_start_params={},
            head_ports=[],
            env_vars={},
            volume_mounts=[],
            volumes=[],
            spec_kwargs={},
            lifecycle_kwargs={},
            container_kwargs={},
        )
        # Should return cluster unchanged due to missing spec
        assert result == cluster

    def test_populate_ray_head_without_dashboard_host(self):
        """Test populate_ray_head automatically adds dashboard-host."""
        from nemo_run.core.execution.kuberay import populate_meta, populate_ray_head

        cluster = populate_meta({}, "test-cluster", "default", {}, "2.43.0")
        ray_start_params = {}

        populate_ray_head(
            cluster,
            ray_image="rayproject/ray:2.43.0",
            service_type="ClusterIP",
            cpu_requests="1",
            memory_requests="2Gi",
            cpu_limits="1",
            memory_limits="2Gi",
            ray_start_params=ray_start_params,
            head_ports=[],
            env_vars={},
            volume_mounts=[],
            volumes=[],
            spec_kwargs={},
            lifecycle_kwargs={},
            container_kwargs={},
        )

        assert ray_start_params["dashboard-host"] == "0.0.0.0"

    def test_update_worker_group_replicas_success(self, basic_cluster_dict):
        """Test successfully updating worker group replicas."""
        from nemo_run.core.execution.kuberay import update_worker_group_replicas

        cluster, success = update_worker_group_replicas(
            basic_cluster_dict, "workers", max_replicas=5, min_replicas=1, replicas=3
        )

        assert success is True
        worker_group = cluster["spec"]["workerGroupSpecs"][0]
        assert worker_group["maxReplicas"] == 5
        assert worker_group["minReplicas"] == 1
        assert worker_group["replicas"] == 3

    def test_update_worker_group_replicas_not_found(self, basic_cluster_dict):
        """Test updating non-existent worker group replicas."""
        from nemo_run.core.execution.kuberay import update_worker_group_replicas

        cluster, success = update_worker_group_replicas(
            basic_cluster_dict, "nonexistent", max_replicas=5, min_replicas=1, replicas=3
        )

        assert success is False

    def test_update_worker_group_resources_success(self, basic_cluster_dict):
        """Test successfully updating worker group resources."""
        from nemo_run.core.execution.kuberay import update_worker_group_resources

        cluster, success = update_worker_group_resources(
            basic_cluster_dict,
            "workers",
            cpu_requests="4",
            memory_requests="8Gi",
            cpu_limits="4",
            memory_limits="8Gi",
        )

        assert success is True
        container = cluster["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["cpu"] == "4"
        assert container["resources"]["requests"]["memory"] == "8Gi"
        assert container["resources"]["limits"]["cpu"] == "4"
        assert container["resources"]["limits"]["memory"] == "8Gi"

    def test_update_worker_group_resources_all_containers(self, basic_cluster_dict):
        """Test updating resources for all containers in a worker group."""
        from nemo_run.core.execution.kuberay import update_worker_group_resources

        # Add another container to test "all_containers" functionality
        basic_cluster_dict["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"].append(
            {
                "name": "sidecar",
                "image": "nginx",
                "resources": {"requests": {"cpu": "100m"}, "limits": {"cpu": "100m"}},
            }
        )

        cluster, success = update_worker_group_resources(
            basic_cluster_dict,
            "workers",
            cpu_requests="4",
            memory_requests="8Gi",
            cpu_limits="4",
            memory_limits="8Gi",
            container_name="all_containers",
        )

        assert success is True
        containers = cluster["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"]
        for container in containers:
            assert container["resources"]["requests"]["cpu"] == "4"
            assert container["resources"]["requests"]["memory"] == "8Gi"

    def test_update_worker_group_resources_specific_container(self, basic_cluster_dict):
        """Test updating resources for a specific container."""
        from nemo_run.core.execution.kuberay import update_worker_group_resources

        # Add another container
        basic_cluster_dict["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"].append(
            {
                "name": "sidecar",
                "image": "nginx",
                "resources": {"requests": {"cpu": "100m"}, "limits": {"cpu": "100m"}},
            }
        )

        cluster, success = update_worker_group_resources(
            basic_cluster_dict,
            "workers",
            cpu_requests="4",
            memory_requests="8Gi",
            cpu_limits="4",
            memory_limits="8Gi",
            container_name="sidecar",
        )

        assert success is True
        sidecar = cluster["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"][1]
        assert sidecar["resources"]["requests"]["cpu"] == "4"

    def test_update_worker_group_resources_no_containers(self):
        """Test updating resources when worker group has no containers."""
        from nemo_run.core.execution.kuberay import update_worker_group_resources

        cluster = {
            "spec": {
                "workerGroupSpecs": [
                    {
                        "groupName": "workers",
                        "template": {"spec": {"containers": []}},
                    }
                ]
            }
        }

        cluster, success = update_worker_group_resources(
            cluster,
            "workers",
            cpu_requests="4",
            memory_requests="8Gi",
            cpu_limits="4",
            memory_limits="8Gi",
        )

        assert success is False

    def test_duplicate_worker_group_success(self, basic_cluster_dict):
        """Test successfully duplicating a worker group."""
        from nemo_run.core.execution.kuberay import duplicate_worker_group

        cluster, success = duplicate_worker_group(basic_cluster_dict, "workers", "new-workers")

        assert success is True
        assert len(cluster["spec"]["workerGroupSpecs"]) == 2
        assert cluster["spec"]["workerGroupSpecs"][1]["groupName"] == "new-workers"

    def test_duplicate_worker_group_not_found(self, basic_cluster_dict):
        """Test duplicating non-existent worker group."""
        from nemo_run.core.execution.kuberay import duplicate_worker_group

        cluster, success = duplicate_worker_group(basic_cluster_dict, "nonexistent", "new-workers")

        assert success is False

    def test_delete_worker_group_success(self, basic_cluster_dict):
        """Test successfully deleting a worker group."""
        from nemo_run.core.execution.kuberay import delete_worker_group

        # Add another worker group first
        basic_cluster_dict["spec"]["workerGroupSpecs"].append(
            {
                "groupName": "gpu-workers",
                "replicas": 1,
            }
        )

        cluster, success = delete_worker_group(basic_cluster_dict, "gpu-workers")

        assert success is True
        assert len(cluster["spec"]["workerGroupSpecs"]) == 1
        assert cluster["spec"]["workerGroupSpecs"][0]["groupName"] == "workers"

    def test_delete_worker_group_not_found(self, basic_cluster_dict):
        """Test deleting non-existent worker group."""
        from nemo_run.core.execution.kuberay import delete_worker_group

        cluster, success = delete_worker_group(basic_cluster_dict, "nonexistent")

        assert success is False

    def test_worker_group_with_labels_and_annotations(self):
        """Test creating worker group with labels and annotations."""
        from nemo_run.core.execution.kuberay import populate_meta, populate_worker_group

        cluster = populate_meta({}, "test-cluster", "default", {}, "2.43.0")
        cluster["spec"]["workerGroupSpecs"] = []

        labels = {"team": "ml", "env": "prod"}
        annotations = {"prometheus.io/scrape": "true"}

        cluster = populate_worker_group(
            cluster,
            group_name="workers",
            ray_image="rayproject/ray:2.43.0",
            gpus_per_worker=None,
            cpu_requests="2",
            memory_requests="4Gi",
            cpu_limits="2",
            memory_limits="4Gi",
            replicas=2,
            min_replicas=2,
            max_replicas=2,
            ray_start_params={},
            volume_mounts=[],
            volumes=[],
            labels=labels,
            annotations=annotations,
            spec_kwargs={},
            lifecycle_kwargs={},
            container_kwargs={},
            env_vars={},
        )

        worker_group = cluster["spec"]["workerGroupSpecs"][0]
        assert worker_group["metadata"]["labels"] == labels
        assert worker_group["metadata"]["annotations"] == annotations

    def test_kuberay_executor_post_init_with_custom_image(self):
        """Test KubeRayExecutor post_init with custom image."""
        executor = KubeRayExecutor(
            image="custom/ray:latest",
            reuse_volumes_in_worker_groups=False,
        )

        assert executor.image == "custom/ray:latest"

    def test_kuberay_executor_post_init_volume_reuse(self):
        """Test KubeRayExecutor post_init with volume reuse."""
        volumes = [{"name": "data", "emptyDir": {}}]
        volume_mounts = [{"name": "data", "mountPath": "/data"}]

        worker_group = KubeRayWorkerGroup(group_name="workers")

        KubeRayExecutor(
            volumes=volumes,
            volume_mounts=volume_mounts,
            worker_groups=[worker_group],
            reuse_volumes_in_worker_groups=True,
        )

        assert worker_group.volumes == volumes
        assert worker_group.volume_mounts == volume_mounts

    def test_kuberay_worker_group_post_init(self):
        """Test KubeRayWorkerGroup post_init sets min/max replicas."""
        worker_group = KubeRayWorkerGroup(
            group_name="workers",
            replicas=3,
        )

        assert worker_group.min_replicas == 3
        assert worker_group.max_replicas == 3

    def test_kuberay_worker_group_post_init_with_custom_replicas(self):
        """Test KubeRayWorkerGroup post_init with custom min/max replicas."""
        worker_group = KubeRayWorkerGroup(
            group_name="workers",
            replicas=3,
            min_replicas=1,
            max_replicas=5,
        )

        assert worker_group.min_replicas == 1
        assert worker_group.max_replicas == 5


class TestSyncWorkdirViaPod:
    """Test sync_workdir_via_pod function and related error paths."""

    @pytest.fixture
    def mock_core_v1_api(self):
        """Mock CoreV1Api for testing."""
        return Mock()

    def test_sync_workdir_via_pod_success(self, mock_core_v1_api):
        """Test successful workdir sync."""
        from nemo_run.core.execution.kuberay import sync_workdir_via_pod

        # Mock successful pod creation and watch
        mock_core_v1_api.create_namespaced_pod.return_value = None

        with patch("nemo_run.core.execution.kuberay.watch.Watch") as mock_watch_class:
            mock_watch = Mock()
            mock_event = {"object": Mock(status=Mock(phase="Running"))}
            mock_watch.stream.return_value = [mock_event]
            mock_watch_class.return_value = mock_watch

            with patch("nemo_run.core.execution.kuberay.subprocess.check_call") as mock_check_call:
                with patch("os.path.isfile", return_value=False):
                    with patch("os.path.abspath", return_value="/abs/path"):
                        sync_workdir_via_pod(
                            pod_name="test-pod",
                            namespace="default",
                            user_workspace_path="/workspace",
                            workdir="/local/path",
                            core_v1_api=mock_core_v1_api,
                            volumes=[],
                            volume_mounts=[],
                        )

                # Verify pod was created and commands were called
                mock_core_v1_api.create_namespaced_pod.assert_called_once()
                assert mock_check_call.call_count >= 2  # mkdir and rsync

    def test_sync_workdir_via_pod_already_exists(self, mock_core_v1_api):
        """Test workdir sync when pod already exists."""
        from nemo_run.core.execution.kuberay import sync_workdir_via_pod

        # Mock pod already exists
        mock_core_v1_api.create_namespaced_pod.side_effect = ApiException(status=409)

        with patch("nemo_run.core.execution.kuberay.watch.Watch") as mock_watch_class:
            mock_watch = Mock()
            mock_event = {"object": Mock(status=Mock(phase="Running"))}
            mock_watch.stream.return_value = [mock_event]
            mock_watch_class.return_value = mock_watch

            with patch("nemo_run.core.execution.kuberay.subprocess.check_call"):
                with patch("os.path.isfile", return_value=False):
                    with patch("os.path.abspath", return_value="/abs/path"):
                        sync_workdir_via_pod(
                            pod_name="test-pod",
                            namespace="default",
                            user_workspace_path="/workspace",
                            workdir="/local/path",
                            core_v1_api=mock_core_v1_api,
                            volumes=[],
                            volume_mounts=[],
                        )

    def test_sync_workdir_via_pod_creation_error(self, mock_core_v1_api):
        """Test workdir sync with pod creation error."""
        from nemo_run.core.execution.kuberay import sync_workdir_via_pod

        # Mock pod creation failure with non-409 error
        mock_core_v1_api.create_namespaced_pod.side_effect = ApiException(status=500)

        with pytest.raises(ApiException):
            sync_workdir_via_pod(
                pod_name="test-pod",
                namespace="default",
                user_workspace_path="/workspace",
                workdir="/local/path",
                core_v1_api=mock_core_v1_api,
                volumes=[],
                volume_mounts=[],
            )

    def test_sync_workdir_via_pod_timeout(self, mock_core_v1_api):
        """Test workdir sync with pod timeout."""
        from nemo_run.core.execution.kuberay import sync_workdir_via_pod

        mock_core_v1_api.create_namespaced_pod.return_value = None

        with patch("nemo_run.core.execution.kuberay.watch.Watch") as mock_watch_class:
            mock_watch = Mock()
            # Pod never reaches Running state
            mock_watch.stream.return_value = []
            mock_watch_class.return_value = mock_watch

            with pytest.raises(RuntimeError, match="Data-mover pod did not reach Running state"):
                sync_workdir_via_pod(
                    pod_name="test-pod",
                    namespace="default",
                    user_workspace_path="/workspace",
                    workdir="/local/path",
                    core_v1_api=mock_core_v1_api,
                    volumes=[],
                    volume_mounts=[],
                )

    def test_sync_workdir_via_pod_with_gitignore(self, mock_core_v1_api):
        """Test workdir sync respects .gitignore file."""
        from nemo_run.core.execution.kuberay import sync_workdir_via_pod

        mock_core_v1_api.create_namespaced_pod.return_value = None

        with patch("nemo_run.core.execution.kuberay.watch.Watch") as mock_watch_class:
            mock_watch = Mock()
            mock_event = {"object": Mock(status=Mock(phase="Running"))}
            mock_watch.stream.return_value = [mock_event]
            mock_watch_class.return_value = mock_watch

            with patch("nemo_run.core.execution.kuberay.subprocess.check_call") as mock_check_call:
                with patch("os.path.isfile", return_value=True):  # .gitignore exists
                    with patch("os.path.abspath", return_value="/abs/path"):
                        sync_workdir_via_pod(
                            pod_name="test-pod",
                            namespace="default",
                            user_workspace_path="/workspace",
                            workdir="/local/path",
                            core_v1_api=mock_core_v1_api,
                            volumes=[],
                            volume_mounts=[],
                        )

                # Check that rsync command includes .gitignore filter
                rsync_call = mock_check_call.call_args_list[1]  # Second call is rsync
                rsync_cmd = rsync_call[0][0]
                assert "--filter=:- .gitignore" in rsync_cmd

    def test_sync_workdir_via_pod_with_cleanup(self, mock_core_v1_api):
        """Test workdir sync with cleanup."""
        from nemo_run.core.execution.kuberay import sync_workdir_via_pod

        mock_core_v1_api.create_namespaced_pod.return_value = None
        mock_core_v1_api.delete_namespaced_pod.return_value = None
        mock_core_v1_api.read_namespaced_pod.side_effect = ApiException(status=404)

        with patch("nemo_run.core.execution.kuberay.watch.Watch") as mock_watch_class:
            mock_watch = Mock()
            mock_event = {"object": Mock(status=Mock(phase="Running"))}
            mock_watch.stream.return_value = [mock_event]
            mock_watch_class.return_value = mock_watch

            with patch("nemo_run.core.execution.kuberay.subprocess.check_call"):
                with patch("os.path.isfile", return_value=False):
                    with patch("os.path.abspath", return_value="/abs/path"):
                        with patch("time.sleep"):  # Speed up test
                            sync_workdir_via_pod(
                                pod_name="test-pod",
                                namespace="default",
                                user_workspace_path="/workspace",
                                workdir="/local/path",
                                core_v1_api=mock_core_v1_api,
                                volumes=[],
                                volume_mounts=[],
                                cleanup=True,
                            )

                # Verify cleanup was called
                mock_core_v1_api.delete_namespaced_pod.assert_called_once()


class TestKubeRayJobAdditionalPaths:
    """Test additional code paths in KubeRayJob for increased coverage."""

    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(
            namespace="test-namespace",
            volumes=[
                {"name": "workspace", "persistentVolumeClaim": {"claimName": "workspace-pvc"}}
            ],
            volume_mounts=[{"name": "workspace", "mountPath": "/workspace"}],
        )

    @pytest.fixture
    def job_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayJob with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayJob(name="test-job", executor=basic_executor)

    def test_logs_kubectl_not_found(self, job_with_basic_executor):
        """Test logs when kubectl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            job_with_basic_executor.logs(follow=False)  # Should not raise

    def test_logs_kubectl_error(self, job_with_basic_executor):
        """Test logs when kubectl returns error."""
        with patch(
            "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "kubectl")
        ):
            job_with_basic_executor.logs(follow=False)  # Should not raise

    def test_logs_timeout(self, job_with_basic_executor):
        """Test logs with timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("kubectl", 30)):
            job_with_basic_executor.logs(follow=True, timeout=30)  # Should not raise

    def test_status_api_error(self, job_with_basic_executor, mock_k8s_clients):
        """Test status with API error."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object.side_effect = ApiException(status=500)

        status = job_with_basic_executor.status(display=False)
        assert status["jobStatus"] == "ERROR"
        assert status["jobDeploymentStatus"] == "ERROR"

    def test_start_job_already_exists(self, job_with_basic_executor, mock_k8s_clients):
        """Test starting job that already exists."""
        mock_api, _ = mock_k8s_clients
        mock_api.create_namespaced_custom_object.side_effect = ApiException(
            status=409, reason="AlreadyExists"
        )

        with pytest.raises(RuntimeError, match="already exists"):
            job_with_basic_executor.start(command="python train.py")

    def test_start_job_api_error(self, job_with_basic_executor, mock_k8s_clients):
        """Test starting job with API error."""
        mock_api, _ = mock_k8s_clients
        mock_api.create_namespaced_custom_object.side_effect = ApiException(status=500)

        with pytest.raises(RuntimeError, match="Error creating RayJob"):
            job_with_basic_executor.start(command="python train.py")

    def test_start_job_workdir_without_volumes(self):
        """Test starting job with workdir but no volumes."""
        executor = KubeRayExecutor(namespace="test")

        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
                with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi"):
                    with patch("nemo_run.run.ray.kuberay.client.CoreV1Api"):
                        job = KubeRayJob(name="test-job", executor=executor)

                        with pytest.raises(
                            ValueError, match="workdir.*specified but executor has no volumes"
                        ):
                            job.start(command="python train.py", workdir="/local/path")

    def test_start_job_with_runtime_env_file(self, job_with_basic_executor, mock_k8s_clients):
        """Test starting job with runtime environment from file."""
        mock_api, _ = mock_k8s_clients

        runtime_env_content = "pip:\n  - numpy\n"

        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open", mock_open(read_data=runtime_env_content)):
                job_with_basic_executor.start(
                    command="python train.py",
                    runtime_env_yaml="runtime_env.yaml",
                )

        call_args = mock_api.create_namespaced_custom_object.call_args
        body = call_args.kwargs["body"]
        assert body["spec"]["runtimeEnvYAML"] == runtime_env_content

    def test_follow_logs_early_termination(self, job_with_basic_executor):
        """Test follow_logs_until_completion when job terminates early."""
        # Mock status progression
        status_sequence = [
            {"jobDeploymentStatus": "Pending"},
            {"jobDeploymentStatus": "Failed"},  # Job fails before reaching Running
        ]

        with patch.object(job_with_basic_executor, "status") as mock_status:
            mock_status.side_effect = status_sequence

            with patch.object(job_with_basic_executor, "stop") as mock_stop:
                with patch("time.sleep"):
                    job_with_basic_executor.follow_logs_until_completion(
                        poll_interval=1,
                        delete_on_finish=True,
                    )

                # Job should be stopped due to early termination
                mock_stop.assert_called_once()


class TestKubeRayClusterAdditionalPaths:
    """Test additional code paths in KubeRayCluster for increased coverage."""

    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(namespace="test-namespace")

    @pytest.fixture
    def cluster_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayCluster with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayCluster(name="test-cluster", executor=basic_executor)

    def test_get_cluster_api_error(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test getting cluster with non-404 API error."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object.side_effect = ApiException(status=500)

        result = cluster_with_basic_executor._get()
        assert result is None

    def test_status_api_error_404(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test status with 404 API error."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object_status.side_effect = ApiException(status=404)

        status = cluster_with_basic_executor.status(timeout=1)
        assert status is None

    def test_status_api_error_non_404(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test status with non-404 API error."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object_status.side_effect = ApiException(status=500)

        status = cluster_with_basic_executor.status(timeout=1)
        assert status is None

    def test_wait_until_running_head_pod_not_ready(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test wait_until_running when head pod is not ready."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {"serviceIP": "10.0.0.1"}}

            # Mock pod that's Running but not Ready
            mock_pod = Mock()
            mock_pod.metadata.labels = {"ray.io/node-type": "head"}
            mock_pod.metadata.name = "test-cluster-head"
            mock_pod.status.phase = "Running"
            mock_pod.status.conditions = [Mock(type="Ready", status="False")]

            mock_pods = Mock()
            mock_pods.items = [mock_pod]
            mock_core_api.list_namespaced_pod.return_value = mock_pods

            with patch("time.sleep"):
                result = cluster_with_basic_executor.wait_until_running(
                    timeout=1, delay_between_attempts=0.5
                )
            assert result is False

    def test_wait_until_running_api_error(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test wait_until_running with API error during pod listing."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {"serviceIP": "10.0.0.1"}}
            mock_core_api.list_namespaced_pod.side_effect = ApiException(status=500)

            with patch("time.sleep"):
                result = cluster_with_basic_executor.wait_until_running(
                    timeout=1, delay_between_attempts=0.5
                )
            assert result is False

    def test_wait_until_running_head_pod_heuristic(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test wait_until_running with head pod identified by name heuristic."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {"serviceIP": "10.0.0.1"}}

            # Mock pod without ray.io/node-type label but with -head in name
            mock_pod = Mock()
            mock_pod.metadata.labels = {}  # No node-type label
            mock_pod.metadata.name = "testuser-test-cluster-raycluster-head-abc123"
            mock_pod.status.phase = "Running"
            mock_pod.status.conditions = [Mock(type="Ready", status="True")]

            mock_pods = Mock()
            mock_pods.items = [mock_pod]
            mock_core_api.list_namespaced_pod.return_value = mock_pods

            result = cluster_with_basic_executor.wait_until_running(timeout=10)
            assert result is True

    def test_patch_cluster_error(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test patching cluster with API error."""
        mock_api, _ = mock_k8s_clients
        mock_api.patch_namespaced_custom_object.side_effect = ApiException(status=500)

        patch_body = {"spec": {"workerGroupSpecs": [{"replicas": 5}]}}
        result = cluster_with_basic_executor.patch(patch_body)

        assert result is False

    def test_port_forward_service_not_found(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test port forwarding when service doesn't exist."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = {"metadata": {"namespace": "test-namespace"}}
            mock_core_api.read_namespaced_service.side_effect = ApiException(status=404)

            with pytest.raises(RuntimeError, match="Could not find Ray head service"):
                cluster_with_basic_executor.port_forward(port=8080, target_port=8265)

    def test_port_forward_service_error(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test port forwarding with service API error."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = {"metadata": {"namespace": "test-namespace"}}
            mock_core_api.read_namespaced_service.side_effect = ApiException(status=500)

            with pytest.raises(RuntimeError, match="Error getting Ray head service"):
                cluster_with_basic_executor.port_forward(port=8080, target_port=8265)

    def test_port_forward_timeout_establishment(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test port forwarding timeout during establishment."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = {"metadata": {"namespace": "test-namespace"}}
            mock_core_api.read_namespaced_service.return_value = Mock()

            with patch("queue.Queue") as mock_queue_class:
                mock_queue = Mock()
                mock_queue.get.side_effect = queue.Empty()  # Timeout
                mock_queue_class.return_value = mock_queue

                with pytest.raises(TimeoutError, match="Timed out waiting for port forwarding"):
                    cluster_with_basic_executor.port_forward(port=8080, target_port=8265)

    def test_display_banner(self, cluster_with_basic_executor):
        """Test display banner method."""
        status_dict = {"state": "ready", "head": {"serviceIP": "10.0.0.1"}}

        # Just test that it doesn't crash - it's a logging method
        cluster_with_basic_executor._display_banner("test-cluster", status_dict)


class TestKubeRayPortForwardingEdgeCases:
    """Test complex port forwarding scenarios for increased coverage."""

    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(namespace="test-namespace")

    @pytest.fixture
    def cluster_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayCluster with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayCluster(name="test-cluster", executor=basic_executor)

    def test_port_forward_process_failure_retry(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test port forwarding with process failure and retry logic."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = {"metadata": {"namespace": "test-namespace"}}
            mock_core_api.read_namespaced_service.return_value = Mock()

            # Mock failed process followed by timeout
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.poll.return_value = 1  # Process failed
                mock_process.stderr.read.return_value = "Connection refused"
                mock_popen.return_value = mock_process

                with patch("queue.Queue") as mock_queue_class:
                    mock_queue = Mock()
                    mock_queue.get.side_effect = queue.Empty()  # Timeout waiting for establishment
                    mock_queue_class.return_value = mock_queue

                    with patch("time.sleep"):  # Speed up retries
                        with pytest.raises(
                            TimeoutError, match="Timed out waiting for port forwarding"
                        ):
                            cluster_with_basic_executor.port_forward(port=8080, target_port=8265)

    def test_delete_with_wait_error_during_final_check(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test delete with wait=True that has error during final state check."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock successful deletion call
        mock_api.delete_namespaced_custom_object.return_value = None

        # Mock that CR still exists during the wait period, then gets deleted
        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            # During wait loop: cluster still exists, then gets deleted
            # During final check: exception (but it's caught and logged)
            mock_get.side_effect = [
                {"metadata": {"name": "test"}},  # Wait loop - exists
                {"metadata": {"name": "test"}},  # Wait loop - still exists (force timeout)
                ApiException(status=500, reason="Network error"),  # Final check - error (caught)
            ]

            with patch("time.time") as mock_time:
                with patch("time.sleep"):
                    # Mock time to force timeout - need enough values
                    mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Force timeout

                    # Mock empty pods
                    mock_core_api.list_namespaced_pod.return_value = Mock(items=[])

                    result = cluster_with_basic_executor.delete(
                        wait=True, timeout=5, poll_interval=1
                    )

                    assert result is False  # Should timeout

    def test_delete_with_wait_api_exception_during_wait(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test delete with wait=True that has API exception during wait loop."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock successful deletion call
        mock_api.delete_namespaced_custom_object.return_value = None

        # Mock exception during cluster status check after first successful call
        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            # First call succeeds, then ApiException during wait, then final check
            mock_get.side_effect = [
                {"metadata": {"name": "test"}},  # First call succeeds
                ApiException(status=500, reason="Network error"),  # Wait loop error (caught)
                ApiException(status=500, reason="Network error"),  # Final check error (also caught)
            ]

            with patch("time.time") as mock_time:
                with patch("time.sleep"):
                    # Mock time to force timeout - need enough values
                    mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Force timeout

                    # Mock empty pods for final check
                    mock_core_api.list_namespaced_pod.return_value = Mock(items=[])

                    result = cluster_with_basic_executor.delete(
                        wait=True, timeout=5, poll_interval=1
                    )

                    assert result is False  # Should timeout


class TestKubeRayClusterDeleteEdgeCases:
    """Test complex delete scenarios for increased coverage."""

    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(namespace="test-namespace")

    @pytest.fixture
    def cluster_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayCluster with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayCluster(name="test-cluster", executor=basic_executor)

    def test_delete_with_wait_timeout_and_final_check(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test delete with wait=True that times out and performs final state check."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock successful deletion call
        mock_api.delete_namespaced_custom_object.return_value = None

        # Mock that CR still exists during the wait period
        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = {"metadata": {"name": "test"}}  # Still exists

            # Mock pods still exist
            mock_pod = Mock()
            mock_pod.metadata.name = "test-pod-1"
            mock_pods = Mock()
            mock_pods.items = [
                mock_pod,
                mock_pod,
                mock_pod,
                mock_pod,
                mock_pod,
                mock_pod,
            ]  # 6 pods for truncation test
            mock_core_api.list_namespaced_pod.return_value = mock_pods

            with patch("time.time") as mock_time:
                with patch("time.sleep"):
                    # Mock time to force timeout - provide enough values
                    mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

                    result = cluster_with_basic_executor.delete(
                        wait=True, timeout=5, poll_interval=1
                    )

                    assert result is False  # Should timeout

    def test_delete_with_wait_error_during_final_check(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test delete with wait=True that has error during final state check."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock successful deletion call
        mock_api.delete_namespaced_custom_object.return_value = None

        # Mock that CR still exists during the wait period
        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            # During wait: cluster exists, then gets deleted, during final check: exception (caught)
            mock_get.side_effect = [
                {"metadata": {"name": "test"}},  # Wait loop
                {"metadata": {"name": "test"}},  # Wait loop - still exists (force timeout)
                ApiException(status=500, reason="Network error"),  # Final check (caught and logged)
            ]

            with patch("time.time") as mock_time:
                with patch("time.sleep"):
                    # Mock time to force timeout - provide enough values
                    mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                    # Mock empty pods for final check
                    mock_core_api.list_namespaced_pod.return_value = Mock(items=[])

                    result = cluster_with_basic_executor.delete(
                        wait=True, timeout=5, poll_interval=1
                    )

                    assert result is False  # Should timeout

    def test_delete_with_wait_api_exception_during_wait(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test delete with wait=True that has API exception during wait loop."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock successful deletion call
        mock_api.delete_namespaced_custom_object.return_value = None

        # Mock exception during cluster status check
        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            # ApiException during wait, then final check also fails
            mock_get.side_effect = [
                ApiException(status=500, reason="Network error"),  # Wait loop error (caught)
                ApiException(status=500, reason="Network error"),  # Final check error (also caught)
            ]

            with patch("time.time") as mock_time:
                with patch("time.sleep"):
                    # Mock time to force timeout - provide enough values
                    mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                    # Mock empty pods for final check
                    mock_core_api.list_namespaced_pod.return_value = Mock(items=[])

                    result = cluster_with_basic_executor.delete(
                        wait=True, timeout=5, poll_interval=1
                    )

                    assert result is False  # Should timeout

    def test_delete_with_wait_pod_check_error(self, cluster_with_basic_executor, mock_k8s_clients):
        """Test delete with wait=True that has error during pod checking."""
        mock_api, mock_core_api = mock_k8s_clients

        # Mock successful deletion call
        mock_api.delete_namespaced_custom_object.return_value = None

        with patch.object(cluster_with_basic_executor, "_get") as mock_get:
            mock_get.return_value = None  # CR is deleted

            # Mock pod checking error
            mock_core_api.list_namespaced_pod.side_effect = ApiException(status=500)

            with patch("time.time") as mock_time:
                with patch("time.sleep"):
                    # Mock time to force timeout - provide enough values
                    mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

                    result = cluster_with_basic_executor.delete(
                        wait=True, timeout=5, poll_interval=1
                    )

                    assert result is False  # Should timeout


class TestKubeRayJobStatusEdgeCases:
    """Test KubeRayJob status edge cases for increased coverage."""

    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(
            namespace="test-namespace",
            volumes=[
                {"name": "workspace", "persistentVolumeClaim": {"claimName": "workspace-pvc"}}
            ],
            volume_mounts=[{"name": "workspace", "mountPath": "/workspace"}],
        )

    @pytest.fixture
    def job_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayJob with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayJob(name="test-job", executor=basic_executor)

    def test_status_with_non_dict_resource(self, job_with_basic_executor, mock_k8s_clients):
        """Test status when API returns non-dict resource."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object.return_value = "invalid-response"  # Non-dict

        status = job_with_basic_executor.status(display=False)
        assert status["jobStatus"] == "UNKNOWN"
        assert status["jobDeploymentStatus"] == "UNKNOWN"

    def test_status_with_display_banner(self, job_with_basic_executor, mock_k8s_clients):
        """Test status with display=True to trigger banner display."""
        mock_api, _ = mock_k8s_clients
        mock_api.get_namespaced_custom_object.return_value = {
            "status": {
                "jobStatus": "RUNNING",
                "jobDeploymentStatus": "Running",
            }
        }

        # Just test that banner display doesn't crash
        status = job_with_basic_executor.status(display=True)
        assert status["jobStatus"] == "RUNNING"

    def test_follow_logs_cleanup_exception(self, job_with_basic_executor):
        """Test follow_logs_until_completion with cleanup exception."""
        status_sequence = [
            {"jobDeploymentStatus": "Running"},
            {"jobDeploymentStatus": "Complete"},
        ]

        with patch.object(job_with_basic_executor, "status") as mock_status:
            mock_status.side_effect = status_sequence

            with patch.object(job_with_basic_executor, "logs"):
                with patch.object(job_with_basic_executor, "stop") as mock_stop:
                    mock_stop.side_effect = Exception("Cleanup error")  # Exception during cleanup

                    with patch("time.sleep"):
                        # Should not raise despite cleanup exception
                        job_with_basic_executor.follow_logs_until_completion(
                            poll_interval=1,
                            delete_on_finish=True,
                        )


class TestKubeRayWaitUntilRunningEdgeCases:
    """Test additional edge cases in wait_until_running method."""

    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    @pytest.fixture
    def basic_executor(self):
        """Create a basic KubeRayExecutor."""
        return KubeRayExecutor(namespace="test-namespace")

    @pytest.fixture
    def cluster_with_basic_executor(self, basic_executor, mock_k8s_clients):
        """Create a KubeRayCluster with basic executor."""
        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            return KubeRayCluster(name="test-cluster", executor=basic_executor)

    def test_wait_until_running_no_pod_conditions(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test wait_until_running when head pod has no conditions."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {"serviceIP": "10.0.0.1"}}

            # Mock pod that's Running with no conditions
            mock_pod = Mock()
            mock_pod.metadata.labels = {"ray.io/node-type": "head"}
            mock_pod.metadata.name = "test-cluster-head"
            mock_pod.status.phase = "Running"
            mock_pod.status.conditions = None  # No conditions

            mock_pods = Mock()
            mock_pods.items = [mock_pod]
            mock_core_api.list_namespaced_pod.return_value = mock_pods

            result = cluster_with_basic_executor.wait_until_running(timeout=10)
            assert result is True  # Should return True based on phase only

    def test_wait_until_running_no_head_pod_found(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test wait_until_running when no head pod is found."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {"serviceIP": "10.0.0.1"}}

            # Mock pod that's not a head pod
            mock_pod = Mock()
            mock_pod.metadata.labels = {"ray.io/node-type": "worker"}
            mock_pod.metadata.name = "test-cluster-worker-1"
            mock_pod.status.phase = "Running"

            mock_pods = Mock()
            mock_pods.items = [mock_pod]
            mock_core_api.list_namespaced_pod.return_value = mock_pods

            with patch("time.sleep"):
                result = cluster_with_basic_executor.wait_until_running(
                    timeout=1, delay_between_attempts=0.5
                )
            assert result is False  # No head pod found

    def test_wait_until_running_pod_not_running(
        self, cluster_with_basic_executor, mock_k8s_clients
    ):
        """Test wait_until_running when head pod is not running."""
        mock_api, mock_core_api = mock_k8s_clients

        with patch.object(cluster_with_basic_executor, "status") as mock_status:
            mock_status.return_value = {"head": {"serviceIP": "10.0.0.1"}}

            # Mock pod that's Pending
            mock_pod = Mock()
            mock_pod.metadata.labels = {"ray.io/node-type": "head"}
            mock_pod.metadata.name = "test-cluster-head"
            mock_pod.status.phase = "Pending"

            mock_pods = Mock()
            mock_pods.items = [mock_pod]
            mock_core_api.list_namespaced_pod.return_value = mock_pods

            with patch("time.sleep"):
                result = cluster_with_basic_executor.wait_until_running(
                    timeout=1, delay_between_attempts=0.5
                )
            assert result is False


class TestKubeRayExecutorLifecycleEdgeCases:
    """Test KubeRayExecutor lifecycle edge cases."""

    @pytest.fixture
    def mock_k8s_clients(self):
        """Mock Kubernetes API clients."""
        with patch("nemo_run.run.ray.kuberay.config.load_kube_config"):
            with patch("nemo_run.run.ray.kuberay.client.CustomObjectsApi") as mock_api:
                with patch("nemo_run.run.ray.kuberay.client.CoreV1Api") as mock_core_api:
                    yield mock_api.return_value, mock_core_api.return_value

    def test_cluster_create_without_lifecycle_kwargs(self, mock_k8s_clients):
        """Test cluster creation when executor doesn't have lifecycle_kwargs."""
        mock_api, _ = mock_k8s_clients

        # Create executor without lifecycle_kwargs attribute
        executor = KubeRayExecutor(namespace="test-namespace")
        # Manually remove the attribute to test the missing attribute path
        if hasattr(executor, "lifecycle_kwargs"):
            delattr(executor, "lifecycle_kwargs")

        with patch("nemo_run.run.ray.kuberay.get_user", return_value="testuser"):
            cluster = KubeRayCluster(name="test-cluster", executor=executor)

            mock_api.create_namespaced_custom_object.return_value = {
                "metadata": {"name": "test-cluster"}
            }

            cluster.create(pre_ray_start_commands=["echo test"])

            # Should create lifecycle_kwargs and succeed
            assert hasattr(executor, "lifecycle_kwargs")
            assert mock_api.create_namespaced_custom_object.called
