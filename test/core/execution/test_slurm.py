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
import os
import re
from pathlib import Path

import pytest

from nemo_run.config import Script
from nemo_run.core.execution.base import ExecutorMacros, FaultTolerance
from nemo_run.core.execution.slurm import SlurmBatchRequest, SlurmExecutor, SlurmJobDetails
from nemo_run.core.packaging.git import GitArchivePackager
from nemo_run.core.tunnel.client import LocalTunnel, SSHTunnel
from nemo_run.run.torchx_backend.packaging import package

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "artifacts")


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


class TestSlurmBatchRequest:
    def apply_macros(self, executor: SlurmExecutor):
        values = executor.macro_values()

        if values:
            executor.env_vars = {
                key: values.substitute(arg) for key, arg in executor.env_vars.items()
            }
            for resource_req in executor.resource_group:
                resource_req.env_vars = {
                    key: values.substitute(arg) for key, arg in resource_req.env_vars.items()
                }

    @pytest.fixture
    def dummy_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["cmd1", "cmd2"]
        command_groups = [["cmd3", "cmd4"]]
        slurm_config = SlurmExecutor(
            account="account",
            job_dir="/root/sample_job",
            tunnel=LocalTunnel(job_dir="/root"),
        )
        slurm_config.job_name = "sample_job"
        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                cmd=cmd,
                jobs=["sample_job"],
                command_groups=command_groups,
                slurm_config=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "dummy_slurm.sh"),
        )

    @pytest.fixture
    def ft_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["cmd1", "cmd2"]
        slurm_config = SlurmExecutor(
            account="account",
            job_dir="/root/sample_job",
            tunnel=LocalTunnel(job_dir="/root/"),
        )
        slurm_config.job_name = "sample_job"
        slurm_config.launcher = FaultTolerance(
            workload_check_interval=10, rank_heartbeat_timeout=10
        )
        role = package(
            name="test_ft",
            fn_or_script=Script("test_ft.sh"),
            executor=slurm_config,
        ).roles[0]
        srun_cmd = [role.entrypoint] + role.args
        command_groups = [[" ".join(srun_cmd)]]
        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                cmd=cmd,
                jobs=["sample_job"],
                command_groups=command_groups,
                slurm_config=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
                launcher=slurm_config.get_launcher(),
            ),
            os.path.join(ARTIFACTS_DIR, "ft_slurm.sh"),
        )

    @pytest.fixture
    def group_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["sbatch", "--parsable"]
        command_groups = [
            ["bash ./scripts/start_server.sh"],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        slurm_config = SlurmExecutor(
            packager=GitArchivePackager(),
            experiment_id="some_experiment_12345",
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            heterogeneous=False,
            memory_measure=False,
            job_dir="/set/by/lib/sample_job",
            tunnel=SSHTunnel(
                job_dir="/some/job/dir",
                host="slurm-login-host",
                user="your-user",
            ),
            wait_time_for_group_job=10,
        )

        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                slurm_config=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "group_slurm.sh"),
        )

    @pytest.fixture
    def group_no_monitor_slurm_request_with_artifact(
        self, group_slurm_request_with_artifact
    ) -> tuple[SlurmBatchRequest, str]:
        req, _ = group_slurm_request_with_artifact
        req.slurm_config.monitor_group_job = False
        return (
            req,
            os.path.join(ARTIFACTS_DIR, "group_slurm_no_monitor.sh"),
        )

    @pytest.fixture
    def group_resource_req_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["sbatch", "--parsable"]
        command_groups = [
            ["bash ./scripts/start_server.sh"],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        executor_1 = SlurmExecutor(
            packager=GitArchivePackager(),
            experiment_id="some_experiment_12345",
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            heterogeneous=False,
            memory_measure=False,
            job_dir="/set/by/lib/sample_job",
            tunnel=SSHTunnel(
                job_dir="/some/job/dir",
                host="slurm-login-host",
                user="your-user",
            ),
            wait_time_for_group_job=10,
            env_vars={"CUSTOM_ENV_1": "some_value_1"},
        )
        executor_2 = executor_1.clone()
        executor_2.container_image = "different_container_image"
        executor_2.srun_args = ["--mpi=pmix"]

        executor = SlurmExecutor.merge([executor_1, executor_2], num_tasks=2)

        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                slurm_config=executor,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "group_resource_req_slurm.sh"),
        )

    @pytest.fixture
    def het_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["sbatch", "--parsable"]
        command_groups = [
            ["bash ./scripts/start_server.sh"],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        slurm_config = SlurmExecutor(
            packager=GitArchivePackager(),
            experiment_id="some_experiment_12345",
            account="your_account",
            partition="your_partition",
            time="00:30:00",
            nodes=1,
            ntasks_per_node=8,
            gpus_per_node=8,
            container_image="some-image",
            heterogeneous=True,
            memory_measure=False,
            job_dir="/set/by/lib/experiment/sample_job",
            tunnel=SSHTunnel(
                job_dir="/some/job/dir/experiment",
                host="slurm-login-host",
                user="your-user",
            ),
        )

        slurm_config.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=GitArchivePackager(),
                nodes=1,
                ntasks_per_node=8,
                container_image="image_1",
                gpus_per_node=8,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
            ),
            SlurmExecutor.ResourceRequest(
                packager=GitArchivePackager(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image_2",
                gpus_per_node=0,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={
                    "CUSTOM_ENV_2": "some_value_2",
                    "HOST_1": ExecutorMacros.group_host(0),
                },
            ),
        ]
        slurm_config.run_as_group = True

        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                slurm_config=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
            ),
            os.path.join(ARTIFACTS_DIR, "het_slurm.sh"),
        )

    @pytest.fixture
    def ft_het_slurm_request_with_artifact(
        self,
    ) -> tuple[SlurmBatchRequest, str]:
        cmd = ["cmd1", "cmd2"]
        slurm_config = SlurmExecutor(
            account="account",
            job_dir="/root/experiment/sample_job",
            tunnel=LocalTunnel(job_dir="/root/experiment"),
            heterogeneous=True,
        )
        slurm_config.job_name = "sample_job"
        slurm_config.launcher = FaultTolerance(
            workload_check_interval=10, rank_heartbeat_timeout=10
        )
        slurm_config.resource_group = [
            SlurmExecutor.ResourceRequest(
                packager=slurm_config.packager,
                nodes=1,
                ntasks_per_node=8,
                container_image="image_1",
                gpus_per_node=8,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={"CUSTOM_ENV_1": "some_value_1"},
            ),
            SlurmExecutor.ResourceRequest(
                packager=GitArchivePackager(),
                nodes=1,
                ntasks_per_node=1,
                container_image="image_2",
                gpus_per_node=0,
                gpus_per_task=None,
                container_mounts=[],
                env_vars={
                    "CUSTOM_ENV_2": "some_value_2",
                    "HOST_1": ExecutorMacros.group_host(0),
                },
            ),
        ]
        slurm_config.run_as_group = True
        role = package(
            name="test_ft",
            fn_or_script=Script("test_ft.sh"),
            executor=slurm_config,
        ).roles[0]
        srun_cmd = [role.entrypoint] + role.args
        command_groups = [
            [" ".join(srun_cmd)],
            ["bash ./scripts/echo.sh server_host=$het_group_host_0"],
        ]
        max_retries = 3
        extra_env = {"ENV_VAR": "value"}
        return (
            SlurmBatchRequest(
                cmd=cmd,
                jobs=["sample_job-0", "sample_job-1"],
                command_groups=command_groups,
                slurm_config=slurm_config,
                max_retries=max_retries,
                extra_env=extra_env,
                launcher=slurm_config.get_launcher(),
            ),
            os.path.join(ARTIFACTS_DIR, "ft_het_slurm.sh"),
        )

    def test_dummy_batch_request_materialize(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, artifact = dummy_slurm_request_with_artifact
        sbatch_script = dummy_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_dummy_batch_request_inline_materialize(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.command_groups = [["bash", "-c", "\"echo 'Hello World Mock Test'\""]]
        sbatch_script = dummy_slurm_request.materialize()
        assert "bash -c \"echo 'Hello World Mock Test'\"" in sbatch_script

        dummy_slurm_request.command_groups = [["bash", "-c", '"echo \\"Hello World Mock Test\\""']]
        sbatch_script = dummy_slurm_request.materialize()
        assert 'bash -c "echo \\"Hello World Mock Test\\""' in sbatch_script

    def test_dummy_batch_request_start(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        sbatch_script = dummy_slurm_request.materialize()
        assert sbatch_script[:11] == "#!/bin/bash"

    def test_dummy_batch_request_dependencies(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.dependencies = [
            "slurm_tunnel://nemo_run/depend1",
            "slurm_tunnel://nemo_run/depend2",
        ]
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --dependency=afterok:depend1:depend2" in sbatch_script

        dummy_slurm_request.slurm_config.dependency_type = "afterany"
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --dependency=afterany:depend1:depend2" in sbatch_script

    def test_dummy_batch_request_memory_measure(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.dependencies = [
            "slurm_tunnel://nemo_run/depend1",
            "slurm_tunnel://nemo_run/depend2",
        ]
        dummy_slurm_request.slurm_config.memory_measure = True
        sbatch_script = dummy_slurm_request.materialize()
        assert (
            "srun --ntasks=1 --ntasks-per-node=1 --output /root/sample_job/log-account-account.sample_job_%j_${SLURM_RESTART_COUNT:-0}.out --wait=60 --kill-on-bad-exit=1 --overlap nvidia-smi"
            in sbatch_script
        )

    def test_dummy_batch_request_custom_job_details_w_defaults(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        class CustomJobDetails(SlurmJobDetails):
            @property
            def stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / "sbatch_job.out"

            @property
            def srun_stdout(self) -> Path:
                assert self.folder
                return Path(self.folder) / "log_job.out"

        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.job_details = CustomJobDetails()
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --job-name=account-account.sample_job" in sbatch_script
        assert "--output /root/sample_job/log_job.out" in sbatch_script
        assert "#SBATCH --output=/root/sample_job/sbatch_job.out" in sbatch_script

    def test_dummy_batch_request_custom_job_details(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        class CustomJobDetails(SlurmJobDetails):
            @property
            def stdout(self) -> Path:
                assert self.folder
                return Path(self.folder / "sbatch_job.out")

            @property
            def srun_stdout(self) -> Path:
                assert self.folder
                return Path(self.folder / "log_job.out")

        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.job_details = CustomJobDetails(
            job_name="custom_sample_job", folder=Path("/custom_folder")
        )
        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --job-name=custom_sample_job" in sbatch_script
        assert "--output /custom_folder/log_job.out" in sbatch_script
        assert "#SBATCH --output=/custom_folder/sbatch_job.out" in sbatch_script

    def test_dummy_batch_request_nsys(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.get_launcher().nsys_profile = True
        launcher_prefix = dummy_slurm_request.slurm_config.get_launcher_prefix()
        assert launcher_prefix == [
            "profile",
            "-s",
            "none",
            "-t",
            "nvtx,cuda",
            "-o",
            "/nemo_run/nsys_profile/profile_%p",
            "--force-overwrite",
            "true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
        ]

    def test_dummy_batch_request_warn(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.cpus_per_gpu = 10
        dummy_slurm_request.slurm_config.gpus_per_task = None

        with pytest.warns(match='"cpus_per_gpu" requires to set "gpus_per_task"'):
            dummy_slurm_request.materialize()

    def test_dummy_batch_request_array(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.array = "0-10"

        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --array=0-10" in sbatch_script
        assert (
            "#SBATCH --output=/root/sample_job/sbatch_account-account.sample_job_%A_%a.out"
            in sbatch_script
        )

    def test_dummy_batch_additonal_params(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.additional_parameters = {"abc": "def"}

        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --abc=def" in sbatch_script

    def test_dummy_batch_job_name_prefix(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, _ = dummy_slurm_request_with_artifact
        dummy_slurm_request.slurm_config.job_name_prefix = "my-custom-prefix:"

        sbatch_script = dummy_slurm_request.materialize()
        assert "#SBATCH --job-name=my-custom-prefix:sample_job" in sbatch_script

    def test_dummy_batch_repr(
        self,
        dummy_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        dummy_slurm_request, artifact = dummy_slurm_request_with_artifact

        expected = Path(artifact).read_text()
        sbatch_repr = str(dummy_slurm_request)
        assert expected.strip() in sbatch_repr

    def test_het_batch_request_materialize(
        self,
        het_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        het_slurm_request, artifact = het_slurm_request_with_artifact
        executor = het_slurm_request.slurm_config
        self.apply_macros(executor)
        sbatch_script = het_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_het_batch_request_dependencies(
        self,
        het_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        het_slurm_request, _ = het_slurm_request_with_artifact
        het_slurm_request.slurm_config.dependencies = [
            "slurm_tunnel://nemo_run/depend1",
            "slurm_tunnel://nemo_run/depend2",
        ]
        sbatch_script = het_slurm_request.materialize()
        assert "#SBATCH --dependency=afterok:depend1:depend2" in sbatch_script

    def test_group_batch_request_materialize(
        self,
        group_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        group_slurm_request, artifact = group_slurm_request_with_artifact
        executor = group_slurm_request.slurm_config
        group_slurm_request.slurm_config = SlurmExecutor.merge([executor], num_tasks=2)
        self.apply_macros(executor)
        sbatch_script = group_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_group_no_monitor_batch_request_materialize(
        self,
        group_no_monitor_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        group_slurm_request, artifact = group_no_monitor_slurm_request_with_artifact
        executor = group_slurm_request.slurm_config
        group_slurm_request.slurm_config = SlurmExecutor.merge([executor], num_tasks=2)
        self.apply_macros(executor)
        sbatch_script = group_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_group_resource_req_batch_request_materialize(
        self,
        group_resource_req_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        group_slurm_request, artifact = group_resource_req_slurm_request_with_artifact
        executor = group_slurm_request.slurm_config
        group_slurm_request.slurm_config = SlurmExecutor.merge([executor], num_tasks=2)
        self.apply_macros(executor)
        sbatch_script = group_slurm_request.materialize()
        expected = Path(artifact).read_text()
        assert sbatch_script.strip() == expected.strip()

    def test_group_resource_req_request_custom_job_details(
        self,
        group_resource_req_slurm_request_with_artifact: tuple[SlurmBatchRequest, str],
    ):
        class CustomJobDetails(SlurmJobDetails):
            @property
            def stdout(self) -> Path:
                assert self.folder
                return Path(self.folder / "sbatch_job.out")

            @property
            def srun_stdout(self) -> Path:
                assert self.folder
                return Path(self.folder / f"log_{self.job_name}.out")

        group_resource_req_slurm_request, _ = group_resource_req_slurm_request_with_artifact
        group_resource_req_slurm_request.slurm_config.job_details = CustomJobDetails(
            job_name="custom_sample_job", folder=Path("/custom_folder")
        )
        group_resource_req_slurm_request.slurm_config.resource_group[0].job_details = copy.deepcopy(
            group_resource_req_slurm_request.slurm_config.job_details
        )
        group_resource_req_slurm_request.slurm_config.resource_group[
            1
        ].job_details = CustomJobDetails(
            job_name="custom_sample_job_2", folder=Path("/custom_folder_2")
        )

        sbatch_script = group_resource_req_slurm_request.materialize()
        assert "#SBATCH --job-name=custom_sample_job" in sbatch_script
        assert "srun --output /custom_folder/log_custom_sample_job.out" in sbatch_script
        assert "srun --output /custom_folder_2/log_custom_sample_job_2.out" in sbatch_script
        assert "#SBATCH --output=/custom_folder/sbatch_job.out" in sbatch_script

    def test_ft_slurm_request_materialize(
        self, ft_slurm_request_with_artifact: tuple[SlurmBatchRequest, str]
    ):
        ft_slurm_request, artifact = ft_slurm_request_with_artifact
        sbatch_script = ft_slurm_request.materialize()
        expected = Path(artifact).read_text()
        sbatch_script = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", sbatch_script)
        expected = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", expected)
        assert sbatch_script.strip() == expected.strip()

    def test_ft_het_slurm_request_materialize(
        self, ft_het_slurm_request_with_artifact: tuple[SlurmBatchRequest, str]
    ):
        ft_het_slurm_request, artifact = ft_het_slurm_request_with_artifact
        executor = ft_het_slurm_request.slurm_config
        self.apply_macros(executor)
        sbatch_script = ft_het_slurm_request.materialize()
        expected = Path(artifact).read_text()
        sbatch_script = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", sbatch_script)
        expected = re.sub(r"--rdzv-id \d+", "--rdzv-id 1", expected)
        assert sbatch_script.strip() == expected.strip()
