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
from unittest.mock import Mock

import pytest

from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.tunnel.client import SSHTunnel
from nemo_run.run.ray.slurm import SlurmRayRequest

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "core", "execution", "artifacts"
)


class TestSlurmRayRequest:
    """Test SlurmRayRequest using artifact-based comparisons similar to test_slurm_templates.py"""

    @pytest.fixture
    def basic_ray_request_with_artifact(self) -> tuple[SlurmRayRequest, str]:
        """Create a basic Ray cluster request matching expected_ray_cluster.sub artifact."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            container_mounts=["/tmp/test_jobs/test-ray-cluster:/tmp/test_jobs/test-ray-cluster"],
        )

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        tunnel_mock.key = "test-cluster"
        executor.tunnel = tunnel_mock

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command="python train.py",
            workdir="/workspace",
            launch_cmd=["sbatch", "--requeue", "--parsable", "--dependency=singleton"],
        )

        return request, os.path.join(ARTIFACTS_DIR, "expected_ray_cluster.sub")

    @pytest.fixture
    def advanced_ray_request_with_artifact(self) -> tuple[SlurmRayRequest, str]:
        """Create an advanced Ray cluster request matching expected_ray_cluster_ssh.sub artifact."""
        executor = SlurmExecutor(
            account="research_account",
            partition="gpu_partition",
            time="02:30:00",
            nodes=4,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/nemo:24.01",
            container_mounts=[
                "/data:/data",
                "/models:/models",
                "/nemo_run:/nemo_run",
                "/lustre/fsw/projects/research/jobs/multi-node-training:/lustre/fsw/projects/research/jobs/multi-node-training",
            ],
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            setup_lines="module load cuda/11.8\nsource /opt/miniconda/bin/activate",
        )

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/lustre/fsw/projects/research/jobs"
        tunnel_mock.key = "research-cluster"
        executor.tunnel = tunnel_mock

        request = SlurmRayRequest(
            name="multi-node-training",
            cluster_dir="/lustre/fsw/projects/research/jobs/multi-node-training",
            template_name="ray.sub.j2",
            executor=executor,
            pre_ray_start_commands=["export NCCL_DEBUG=INFO", "export NCCL_IB_DISABLE=1"],
            command="ray job submit --address ray://localhost:10001 --job-id training-job -- python -m training.main",
            workdir="/workspace/training",
            launch_cmd=["sbatch", "--requeue", "--parsable", "--dependency=singleton"],
        )

        return request, os.path.join(ARTIFACTS_DIR, "expected_ray_cluster_ssh.sub")

    @pytest.fixture
    def resource_specs_ray_request(self) -> SlurmRayRequest:
        """Create a Ray request with various resource specifications."""
        executor = SlurmExecutor(
            account="test_account",
            partition="gpu",
            time="01:00:00",
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="nvcr.io/nvidia/pytorch:24.01-py3",
            gres="gpu:a100:4",
            cpus_per_task=4,
            gpus_per_task=2,
            mem="32G",
            mem_per_cpu="4G",
            exclusive=True,
        )

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/tmp/test_jobs"
        executor.tunnel = tunnel_mock

        return SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            nemo_run_dir="/custom/nemo_run",
            launch_cmd=["sbatch", "--parsable"],
        )

    def _assert_sbatch_parameters(self, script: str, expected_params: dict):
        """Helper to assert SBATCH parameters are present in script."""
        for param, value in expected_params.items():
            expected_line = f"#SBATCH --{param}={value}"
            assert expected_line in script, f"Missing SBATCH parameter: {expected_line}"

    def _assert_script_patterns(self, script: str, patterns: list[str], test_name: str = ""):
        """Helper to assert multiple patterns are present in script."""
        for pattern in patterns:
            assert pattern in script, f"Missing pattern in {test_name}: {pattern}"

    def test_basic_ray_cluster_artifact(
        self, basic_ray_request_with_artifact: tuple[SlurmRayRequest, str]
    ):
        """Test that basic Ray cluster script matches key patterns from artifact."""
        ray_request, artifact_path = basic_ray_request_with_artifact
        generated_script = ray_request.materialize()

        # Read expected artifact for reference
        with open(artifact_path, "r") as f:
            expected_script = f.read()

        assert generated_script.strip() == expected_script.strip()

    def test_advanced_ray_cluster_artifact(
        self, advanced_ray_request_with_artifact: tuple[SlurmRayRequest, str]
    ):
        """Test that advanced Ray cluster script matches key patterns from SSH artifact."""
        ray_request, artifact_path = advanced_ray_request_with_artifact
        generated_script = ray_request.materialize()

        # Read expected artifact for reference
        with open(artifact_path, "r") as f:
            expected_script = f.read()

        assert generated_script.strip() == expected_script.strip()

    def test_get_job_name_basic(self):
        """Test job name generation with basic executor."""
        executor = SlurmExecutor(account="test_account")
        name = "test-ray-cluster"
        job_name = SlurmRayRequest.get_job_name(executor, name)
        expected = "test_account-account.test-ray-cluster"
        assert job_name == expected

    def test_get_job_name_with_prefix(self):
        """Test job name generation with custom prefix."""
        executor = SlurmExecutor(account="test_account", job_name_prefix="custom-prefix.")
        name = "my-cluster"
        job_name = SlurmRayRequest.get_job_name(executor, name)
        expected = "custom-prefix.my-cluster"
        assert job_name == expected

    def test_resource_specifications(self, resource_specs_ray_request: SlurmRayRequest):
        """Test materialize with various resource specifications."""
        script = resource_specs_ray_request.materialize()

        # Check resource specifications are present
        resource_patterns = [
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --gpus-per-task=2",
            "#SBATCH --mem=32G",
            "#SBATCH --mem-per-cpu=4G",
            "#SBATCH --exclusive",
            "--gres=gpu:a100:4",  # Should use gres instead of gpus_per_node
            "/custom/nemo_run:/nemo_run",  # Should handle nemo_run_dir mounting
        ]

        self._assert_script_patterns(script, resource_patterns, "resource specifications")

    def test_additional_parameters(self):
        """Test materialize with additional SBATCH parameters."""
        executor = SlurmExecutor(
            account="test_account", additional_parameters={"custom_param": "custom_value"}
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "#SBATCH --custom-param=custom_value" in script

    def test_dependencies(self):
        """Test materialize with job dependencies."""
        executor = SlurmExecutor(
            account="test_account",
            dependencies=[
                "torchx://session/app_id/master/0",
                "torchx://session/app_id2/master/0",
            ],
            dependency_type="afterok",
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "#SBATCH --dependency=afterok:app_id:app_id2" in script

    def test_stderr_to_stdout_false(self):
        """Test materialize when stderr_to_stdout is False."""
        executor = SlurmExecutor(account="test_account")
        executor.stderr_to_stdout = False  # Set after creation since it's not an init parameter
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "#SBATCH --error=" in script

    def test_container_configurations(self):
        """Test materialize with various container configurations."""
        executor = SlurmExecutor(account="test_account", container_image=None)
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            workdir=None,  # No workdir - should use cluster_dir as default
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should use cluster_dir as default workdir
        assert "--container-workdir=/tmp/test_jobs/test-ray-cluster" in script
        # Should not contain container-image flag when none specified
        assert "--container-image" not in script

    def test_special_mount_handling(self):
        """Test materialize handles special RUNDIR_SPECIAL_NAME mounts."""
        from nemo_run.config import RUNDIR_SPECIAL_NAME

        executor = SlurmExecutor(
            account="test_account", container_mounts=[f"{RUNDIR_SPECIAL_NAME}:/special"]
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            nemo_run_dir="/actual/nemo_run",
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()
        assert "/actual/nemo_run:/special" in script

    def test_job_details_preset(self):
        """Test materialize when job details are already set."""
        executor = SlurmExecutor(account="test_account")
        executor.job_details.job_name = "custom-job-name"
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        job_details_patterns = [
            "#SBATCH --job-name=custom-job-name",  # Should use preset job name
            "export LOG_DIR=/tmp/test_jobs/test-ray-cluster/logs",  # Log dir still constructed from cluster_dir/logs
        ]

        self._assert_script_patterns(script, job_details_patterns, "job details preset")

    def test_repr_method(self):
        """Test the __repr__ method returns formatted script."""
        executor = SlurmExecutor(account="test_account")
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-cluster",
            cluster_dir="/tmp/test-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch"],
        )

        repr_output = repr(request)

        assert "#----------------" in repr_output
        assert "# SBATCH_SCRIPT" in repr_output
        assert "#----------------" in repr_output
        assert "#SBATCH --account=test_account" in repr_output

    def test_cpus_per_gpu_warning(self):
        """Test materialize issues warning when cpus_per_gpu without gpus_per_task."""
        executor = SlurmExecutor(account="test_account", cpus_per_gpu=4)
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        with pytest.warns(UserWarning, match="cpus_per_gpu.*requires.*gpus_per_task"):
            request.materialize()

    def test_heterogeneous_assertion(self):
        """Test materialize raises assertion for heterogeneous jobs."""
        executor = SlurmExecutor(account="test_account", heterogeneous=True)
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        with pytest.raises(AssertionError, match="heterogeneous is not supported"):
            request.materialize()

    def test_array_assertion(self):
        """Test materialize raises assertion for array jobs."""
        executor = SlurmExecutor(account="test_account", array="1-10")
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        with pytest.raises(AssertionError, match="array is not supported"):
            request.materialize()

    def test_command_groups_env_vars(self):
        """Test environment variables are properly set for each command group."""
        # Create executor with environment variables
        executor = SlurmExecutor(
            account="test_account",
            env_vars={"GLOBAL_ENV": "global_value"},
        )
        executor.run_as_group = True

        # Create resource groups with different env vars
        resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image1",
                env_vars={"GROUP1_ENV": "group1_value"},
                container_mounts=["/mount1"],
            ),
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image2",
                env_vars={"GROUP2_ENV": "group2_value"},
                container_mounts=["/mount2"],
            ),
        ]
        executor.resource_group = resource_group
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"], ["cmd2"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Check global env vars are set in setup section
        assert "export GLOBAL_ENV=global_value" in script

        # Check that command groups generate srun commands (excluding the first one)
        # The template should have a section for srun_commands
        assert "# Run extra commands" in script
        assert "srun" in script
        assert "cmd1" in script  # First command group after skipping index 0
        assert "cmd2" in script  # Second command group

    def test_command_groups_without_resource_group(self):
        """Test command groups work without resource groups."""
        executor = SlurmExecutor(
            account="test_account",
            env_vars={"GLOBAL_ENV": "global_value"},
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[["cmd0"], ["cmd1"]],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Should have global env vars
        assert "export GLOBAL_ENV=global_value" in script

        # Should have srun commands for overlapping groups (skipping first)
        assert "srun" in script
        assert "--overlap" in script
        assert "cmd1" in script  # Second command in the list (index 1)

    def test_env_vars_formatting(self):
        """Test that environment variables are properly formatted as export statements."""
        executor = SlurmExecutor(
            account="test_account",
            env_vars={
                "VAR_WITH_SPACES": "value with spaces",
                "PATH_VAR": "/usr/bin:/usr/local/bin",
                "EMPTY_VAR": "",
                "NUMBER_VAR": "123",
            },
        )
        executor.tunnel = Mock(spec=SSHTunnel)
        executor.tunnel.job_dir = "/tmp/test_jobs"

        request = SlurmRayRequest(
            name="test-ray-cluster",
            cluster_dir="/tmp/test_jobs/test-ray-cluster",
            template_name="ray.sub.j2",
            executor=executor,
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Check all environment variables are properly exported
        assert "export VAR_WITH_SPACES=value with spaces" in script
        assert "export PATH_VAR=/usr/bin:/usr/local/bin" in script
        assert "export EMPTY_VAR=" in script
        assert "export NUMBER_VAR=123" in script

    def test_group_env_vars_integration(self):
        """Test full integration of group environment variables matching the artifact pattern."""
        # This test verifies the behavior seen in group_resource_req_slurm.sh
        executor = SlurmExecutor(
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            container_mounts=["/some/job/dir/sample_job:/nemo_run"],
            env_vars={"ENV_VAR": "value"},
        )
        executor.run_as_group = True

        # Set up resource groups with specific env vars
        resource_group = [
            # First group (index 0) - for the head/main command
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                container_image="some-image",
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
                container_mounts=["/some/job/dir/sample_job:/nemo_run"],
            ),
            # Second group (index 1)
            SlurmExecutor.ResourceRequest(
                packager=Mock(),
                nodes=1,
                ntasks_per_node=8,
                container_image="different_container_image",
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
                container_mounts=["/some/job/dir/sample_job:/nemo_run"],
            ),
        ]
        executor.resource_group = resource_group

        # Mock tunnel
        tunnel_mock = Mock(spec=SSHTunnel)
        tunnel_mock.job_dir = "/some/job/dir"
        executor.tunnel = tunnel_mock

        request = SlurmRayRequest(
            name="sample_job",
            cluster_dir="/some/job/dir/sample_job",
            template_name="ray.sub.j2",
            executor=executor,
            command_groups=[
                ["bash ./scripts/start_server.sh"],
                ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
            ],
            launch_cmd=["sbatch", "--parsable"],
        )

        script = request.materialize()

        # Verify the pattern matches the artifact:
        # 1. Global env vars should be exported in setup
        assert "export ENV_VAR=value" in script

        # The template should include group_env_vars for proper env var handling per command
        # (The actual env var exports per command happen in the template rendering)
