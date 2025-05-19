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
import json
import logging
import os
import queue
import shlex
import socket
import subprocess
import tempfile
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TypeAlias, Union

from nemo_run.config import RUNDIR_NAME, RUNDIR_SPECIAL_NAME
from nemo_run.core.execution.slurm import SlurmExecutor, _as_sbatch_flag
from nemo_run.core.execution.utils import fill_template
from nemo_run.core.packaging.git import GitArchivePackager
from nemo_run.core.tunnel.client import SSHTunnel
from nemo_run.core.tunnel.rsync import rsync

noquote: TypeAlias = str

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SlurmRayRequest:
    name: str
    cluster_dir: str
    template_name: str
    template_dir: Optional[str] = None
    executor: SlurmExecutor
    pre_ray_start_commands: Optional[list[str]] = None
    command: Optional[str] = None
    workdir: Optional[str] = None
    nemo_run_dir: Optional[str] = None
    launch_cmd: list[str]

    @staticmethod
    def get_job_name(executor: SlurmExecutor, name: str) -> str:
        job_name_prefix = (
            executor.job_name_prefix
            if executor.job_name_prefix
            else f"{executor.account}-{executor.account.split('_')[-1]}."
        )
        return f"{job_name_prefix}{name}"

    def materialize(self) -> str:
        args = asdict(self.executor)  # noqa: F821
        parameters = {
            k: v for k, v in args.items() if v is not None and k in SlurmExecutor.SBATCH_FLAGS
        }

        # rename and reformat parameters

        if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
            warnings.warn(  # noqa: F821
                '"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")'
            )
        # add necessary parameters
        job_name = SlurmRayRequest.get_job_name(self.executor, self.name)
        slurm_job_dir = self.cluster_dir
        job_details = self.executor.job_details

        if not job_details.job_name:
            job_details.job_name = job_name

        if not job_details.folder:
            job_details.folder = os.path.join(slurm_job_dir, "logs")

        parameters["job_name"] = job_details.job_name

        stdout = str(job_details.stdout)
        stderr = str(job_details.stderr)

        assert self.executor.array is None, "array is not supported for ray clusters"
        parameters["output"] = stdout.replace("%t", "0")

        if not self.executor.stderr_to_stdout:
            parameters["error"] = stderr.replace("%t", "0")

        if self.executor.additional_parameters is not None:
            parameters.update(self.executor.additional_parameters)

        sbatch_flags = []
        assert not self.executor.heterogeneous, "heterogeneous is not supported for ray clusters"
        for k in sorted(parameters):
            sbatch_flags.append(_as_sbatch_flag(k, parameters[k]))

        if self.executor.dependencies:
            slurm_deps = self.executor.parse_deps()
            sbatch_flags.append(
                _as_sbatch_flag(
                    "dependency", f"{self.executor.dependency_type}:{':'.join(slurm_deps)}"
                )
            )

        env_vars = []
        for key, value in self.executor.env_vars.items():
            env_vars.append(f"export {key.upper()}={value}")

        def get_srun_flags(mounts: list[str], container_image: Optional[str]) -> str:
            _srun_flags = [f"--container-image={container_image}"] if container_image else []
            _srun_flags.append("--no-container-mount-home")
            _srun_flags.append("--mpi=pmix")
            _srun_flags.append(f"-A={self.executor.account}")
            _srun_flags.append(f"-p={self.executor.partition}")
            if self.executor.gres:
                _srun_flags.append(f"--gres={self.executor.gres}")
            elif self.executor.gpus_per_node:
                _srun_flags.append(f"--gres=gpu:{self.executor.gpus_per_node}")
            else:
                _srun_flags.append("--gres=gpu:8")

            if self.nemo_run_dir:
                new_mounts = copy.deepcopy(mounts)
                for i, mount in enumerate(new_mounts):
                    if mount.startswith(RUNDIR_SPECIAL_NAME):
                        new_mounts[i] = mount.replace(RUNDIR_SPECIAL_NAME, self.nemo_run_dir, 1)

                new_mounts.append(f"{self.nemo_run_dir}:/{RUNDIR_NAME}")
            else:
                new_mounts = mounts

            _srun_flags += ["--container-mounts", ",".join(new_mounts)]
            container_workdir = self.workdir or self.cluster_dir
            _srun_flags.append(f"--container-workdir={container_workdir}")

            return " ".join(_srun_flags)

        vars_to_fill = {
            "sbatch_flags": sbatch_flags,
            "cluster_dir": self.cluster_dir,
            "log_dir": os.path.join(self.cluster_dir, "logs"),
            "uv_cache_dir": os.path.join(self.cluster_dir, "uv_cache"),
            "num_retries": max(1, self.executor.retries),
            "env_vars": env_vars,
            "setup_lines": self.executor.setup_lines,
            "common_srun_args": get_srun_flags(
                self.executor.container_mounts, self.executor.container_image
            ),
            "command": self.command,
            "command_workdir": self.workdir,
        }

        if self.pre_ray_start_commands:
            vars_to_fill["pre_ray_start_commands"] = "\n".join(self.pre_ray_start_commands)

        sbatch_script = fill_template(
            self.template_name,
            vars_to_fill,
            template_dir=self.template_dir
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
        )
        return sbatch_script

    def __repr__(self) -> str:
        return f"""
#----------------
# SBATCH_SCRIPT
#----------------

{self.materialize()}"""


class SlurmRayCluster:
    EXECUTOR_CLS = SlurmExecutor

    def __init__(self):
        self.cluster_map = {}

    def _get_ray_cluster_info(self, name: str, executor: SlurmExecutor) -> Dict[str, Any]:
        executor.tunnel.connect()
        cluster_dir = os.path.join(executor.tunnel.job_dir, name)
        cmd = f"test -f {cluster_dir}/ray_cluster_info.json && cat {cluster_dir}/ray_cluster_info.json"
        result = executor.tunnel.run(cmd, warn=True)

        if result.return_code == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Ray cluster info for '{name}'")
                return {}
        return {}

    def get_ray_cluster_status(
        self,
        name: str,
        executor: SlurmExecutor,
    ) -> Dict[str, Union[str, bool, None]]:
        logger.info(f"Getting Ray cluster status for '{name}'")
        executor.tunnel.connect()

        # Try to find the job by name
        job_name = SlurmRayRequest.get_job_name(executor, name)

        cmd = f"squeue -n {job_name} -h -o %A"
        result = executor.tunnel.run(cmd)

        job_id = result.stdout.strip()

        # If job not found in running jobs, check if it's in cluster_map
        if not job_id and name in self.cluster_map:
            job_id = self.cluster_map[name]
            # Verify this job_id exists
            cmd = f"squeue -j {job_id} -h -o %A"
            result = executor.tunnel.run(cmd)
            if not result.stdout.strip():
                # Job might be completed, check sacct
                cmd = f"sacct -j {job_id} --format=State --noheader --parsable2"
                result = executor.tunnel.run(cmd)
                if result.stdout.strip():
                    state = result.stdout.strip().split("\n")[0]
                    return {"state": state, "job_id": job_id, "ray_ready": state == "COMPLETED"}
                # Job not found in sacct either, so it doesn't exist
                return {"state": "NOT_FOUND", "job_id": None, "ray_ready": False}

        if not job_id:
            return {"state": "NOT_FOUND", "job_id": None, "ray_ready": False}

        # Store job_id in cluster_map for future reference
        self.cluster_map[name] = job_id

        # Check job status
        cmd = f"squeue -j {job_id} -h -o %T"
        result = executor.tunnel.run(cmd)

        if not result.stdout.strip():
            # Job not found in squeue, check sacct
            cmd = f"sacct -j {job_id} --format=State --noheader --parsable2"
            result = executor.tunnel.run(cmd)
            status = result.stdout.strip().split("\n")[0] if result.stdout.strip() else "UNKNOWN"

            return {"state": status, "job_id": job_id, "ray_ready": status == "COMPLETED"}

        status = result.stdout.strip()

        # When running, also check if ray is actually ready
        ray_ready = False
        if status == "RUNNING":
            ray_cluster_info = self._get_ray_cluster_info(name, executor)
            if ray_cluster_info:
                ray_ready = True

        return {"state": status, "job_id": job_id, "ray_ready": ray_ready}

    def create_ray_cluster(
        self,
        name: str,
        executor: SlurmExecutor,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
        command: Optional[str] = None,
        workdir: Optional[str] = None,
    ) -> Any:
        cluster_dir = os.path.join(executor.tunnel.job_dir, name)
        ray_sbatch = SlurmRayRequest(
            name=name,
            cluster_dir=cluster_dir,
            template_name="ray.sub.j2",
            executor=executor,
            pre_ray_start_commands=pre_ray_start_commands,
            command=command,
            workdir=workdir,
            launch_cmd=["sbatch", "--requeue", "--parsable"],
        ).materialize()

        if dryrun:
            logger.info(f"Dry run: Ray cluster '{name}'")
            print(ray_sbatch)
            return

        logger.info(f"Creating Ray cluster '{name}'")
        # Check if a cluster with this name already exists
        status = self.get_ray_cluster_status(name, executor)

        if status["job_id"] is not None:
            job_state = status["state"]
            if job_state in ["PENDING", "RUNNING", "CONFIGURING"]:
                logger.info(
                    f"Ray cluster '{name}' already exists with job ID {status['job_id']} "
                    f"and is currently in {job_state} state. "
                    f"Skipping creation."
                )
                return None
            elif job_state not in [
                "COMPLETING",
                "COMPLETED",
                "CANCELLED",
                "FAILED",
                "TIMEOUT",
                "NOT_FOUND",
            ]:
                logger.warning(
                    f"Ray cluster '{name}' exists with job ID {status['job_id']} "
                    f"in state {job_state}. Creating new cluster anyway."
                )

        executor.tunnel.connect()
        executor.tunnel.run(f"mkdir -p {cluster_dir}")

        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
            f.write(ray_sbatch)
            f.flush()
            os.fsync(f.fileno())
            ray_sbatch_path = f.name
            executor.tunnel.put(ray_sbatch_path, os.path.join(cluster_dir, "ray.sub"))

        sbatch_cmd = ["sbatch", "--parsable", os.path.join(cluster_dir, "ray.sub")]
        job_id = executor.tunnel.run(" ".join(sbatch_cmd)).stdout.strip()

        # Store job_id in cluster_map
        self.cluster_map[name] = job_id

        logger.info(f"Slurm job for Ray cluster '{name}' created with job ID {job_id}")

        return job_id

        return None

    def schedule_ray_job(
        self,
        name: str,
        executor: SlurmExecutor,
        command: str,
        workdir: Optional[str] = None,
        pre_ray_start_commands: Optional[list[str]] = None,
        runtime_env_yaml: Optional[str] = None,
        dryrun: bool = False,
    ):
        remote_workdir = None
        if workdir:
            if isinstance(executor.tunnel, SSHTunnel):
                # Rsync workdir honoring .gitignore
                remote_workdir = os.path.join(executor.tunnel.job_dir, name, "code")
                if not dryrun:
                    executor.tunnel.connect()
                    assert executor.tunnel.session is not None, "Tunnel session is not connected"
                    rsync(
                        executor.tunnel.session,
                        workdir,
                        remote_workdir,
                        rsync_opts="--filter=':- .gitignore'",
                    )
            else:
                remote_workdir = workdir
        elif executor.packager:
            if not dryrun:
                if isinstance(executor.tunnel, SSHTunnel):
                    package_dir_ref = tempfile.TemporaryDirectory()
                    package_dir = package_dir_ref.name
                else:
                    package_dir_ref = None
                    package_dir = os.path.join(executor.tunnel.job_dir, name)

                if isinstance(executor.packager, GitArchivePackager):
                    output = subprocess.run(
                        ["git", "rev-parse", "--show-toplevel"],
                        check=True,
                        stdout=subprocess.PIPE,
                    )
                    path = output.stdout.splitlines()[0].decode()
                    base_path = Path(path).absolute()
                else:
                    base_path = Path(os.getcwd()).absolute()

                local_tar_file = executor.packager.package(base_path, package_dir, name)
                local_code_extraction_path = os.path.join(package_dir, "code")
                os.makedirs(local_code_extraction_path, exist_ok=True)
                subprocess.run(
                    f"tar -xvzf {local_tar_file} -C {local_code_extraction_path} --ignore-zeros",
                    shell=True,
                    check=True,
                )

                if isinstance(executor.tunnel, SSHTunnel):
                    remote_workdir = os.path.join(executor.tunnel.job_dir, name, "code")
                    executor.tunnel.connect()
                    assert executor.tunnel.session is not None, "Tunnel session is not connected"
                    rsync(
                        executor.tunnel.session,
                        os.path.join(local_code_extraction_path, ""),
                        remote_workdir,
                        rsync_opts="--filter=':- .gitignore'",
                    )
                else:
                    remote_workdir = local_code_extraction_path

        assert remote_workdir is not None, "workdir is not set"
        job_id = self.create_ray_cluster(
            name,
            executor,
            pre_ray_start_commands=pre_ray_start_commands,
            dryrun=dryrun,
            command=command,
            workdir=remote_workdir,
        )
        return job_id

    def wait_until_ray_cluster_running(
        self,
        name: str,
        executor: SlurmExecutor,
        timeout: int = 600,
        delay_between_attempts: int = 30,
    ) -> bool:
        logger.info(f"Waiting until Ray cluster '{name}' is running")
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_ray_cluster_status(name, executor)

            if status["ray_ready"]:
                logger.info(f"Ray cluster '{name}' is ready.")
                return True

            # If job failed or was cancelled, return False
            if status["state"] in ["FAILED", "CANCELLED", "TIMEOUT", "NOT_FOUND"]:
                logger.error(f"Ray cluster '{name}' failed to start. Job state: {status['state']}")
                return False

            logger.info(f"Ray cluster '{name}' is not ready, waiting for it to be ready...")
            time.sleep(delay_between_attempts)

        logger.info(f"Ray cluster '{name}' is not ready after {timeout} seconds")
        return False

    def delete_ray_cluster(
        self,
        name: str,
        executor: SlurmExecutor,
        wait: bool = False,
        timeout: int = 60,
        poll_interval: int = 5,
    ) -> bool:
        logger.info(f"Deleting Ray cluster '{name}'")
        status = self.get_ray_cluster_status(name, executor)

        if status["job_id"] is None:
            logger.warning(f"Ray cluster '{name}' does not exist or is already deleted")
            return True

        job_id = status["job_id"]

        # If job is already completed or failed, no need to cancel
        if any(
            state in status["state"]  # type: ignore
            for state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NOT_FOUND"]
        ):
            logger.info(f"Ray cluster '{name}' job {job_id} is already in state {status['state']}")
            # Remove from cluster_map
            if name in self.cluster_map:
                del self.cluster_map[name]
            return True
        # Cancel the job
        executor.tunnel.connect()
        cmd = f"scancel {job_id}"
        logger.info(f"Cancelling Ray cluster '{name}' job {job_id}")

        try:
            executor.tunnel.run(cmd)
        except Exception as e:
            logger.error(f"Failed to cancel Ray cluster '{name}' job {job_id}: {e}")
            return False

        # Remove from cluster_map if it exists
        if name in self.cluster_map:
            del self.cluster_map[name]

        # Wait for job to be fully terminated if requested
        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout:
                status = self.get_ray_cluster_status(name, executor)

                # If job is not found anymore, it's been successfully cancelled
                if status["job_id"] is None:
                    logger.info(
                        f"Ray cluster '{name}' job {job_id} has been successfully cancelled"
                    )
                    if name in self.cluster_map:
                        del self.cluster_map[name]
                    return True

                # If job is in a terminated state, success
                if any(state in status["state"] for state in ["CANCELLED", "FAILED", "TIMEOUT"]):  # type: ignore
                    logger.info(
                        f"Ray cluster '{name}' job {job_id} is now in state {status['state']}"
                    )
                    if name in self.cluster_map:
                        del self.cluster_map[name]
                    return True

                logger.info(f"Waiting for Ray cluster '{name}' job {job_id} to terminate...")
                time.sleep(poll_interval)

            logger.warning(f"Timed out waiting for Ray cluster '{name}' job {job_id} to terminate")
            return False

        return True

    def port_forward(
        self,
        name: str,
        port: int,
        target_port: int,
        executor: SlurmExecutor,
        wait: bool = False,
    ):
        """Port forward to a Ray cluster using SSH tunnel.

        When you want to stop the forwarding:
            forward_thread.stop_forwarding()  # Call this method to stop forwarding

        If wait=True, this function will block until interrupted (e.g., with Ctrl+C).

        Parameters:
        - name (str): The name of the Ray cluster.
        - port (int): The local port to use for forwarding.
        - target_port (int): The target port on the Ray cluster to forward to.
        - executor (SlurmExecutor): The executor containing the tunnel configuration.
        - wait (bool, optional): If True, block indefinitely until interrupted. Defaults to False.

        Returns:
        - ForwardingThread: A thread object with stop_forwarding method.

        Raises:
        - RuntimeError: If the Ray cluster info cannot be found or is incomplete.
        - TimeoutError: If port forwarding fails to establish within the timeout period.
        """
        # Check if cluster exists and is running
        status = self.get_ray_cluster_status(name, executor)
        if status["job_id"] is None:
            raise RuntimeError(f"Could not find Ray cluster {name}")

        if not status["ray_ready"]:
            raise RuntimeError(f"Ray cluster {name} is not running or not ready yet")

        # Get cluster info
        ray_cluster_info = self._get_ray_cluster_info(name, executor)
        if not ray_cluster_info:
            raise RuntimeError(f"Could not find Ray cluster info for {name}")

        if "head_ip" not in ray_cluster_info:
            raise RuntimeError(f"Ray cluster info for {name} does not contain head_ip")

        head_ip = ray_cluster_info["head_ip"]

        # Use a queue for thread communication
        status_queue = queue.Queue()
        stop_event = threading.Event()

        class ForwardingThread(threading.Thread):
            def __init__(self, daemon=True):
                super().__init__(daemon=daemon)
                self._stop_event = stop_event
                self._ssh_process = None

            def run(self):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.bind(("localhost", port))
                    sock.close()
                except socket.error:
                    sock.close()
                    raise RuntimeError(f"Port {port} is already in use locally")

                self._ssh_process = None
                ssh_cmd_list_for_error_reporting = []
                ssh_cmd_list = []

                try:
                    ssh_cmd_list = ["ssh"]
                    ssh_cmd_list.extend(["-L", f"{port}:localhost:{target_port}"])
                    ssh_cmd_list.extend(
                        [
                            "-N",
                            "-o",
                            "StrictHostKeyChecking=no",
                            "-o",
                            "UserKnownHostsFile=/dev/null",
                            "-o",
                            "ExitOnForwardFailure=yes",
                            "-o",
                            "ConnectTimeout=10",
                            "-o",
                            "IdentitiesOnly=yes",
                        ]
                    )

                    jump_arg_str = f"{executor.tunnel.user}@{executor.tunnel.host}"
                    raw_jump_identity = getattr(executor.tunnel, "identity", None)
                    jump_identity_path_for_proxy = None
                    if raw_jump_identity:
                        expanded_path = os.path.expanduser(str(raw_jump_identity))
                        if os.path.isfile(expanded_path):
                            jump_identity_path_for_proxy = expanded_path
                            logger.debug(
                                f"Using jump identity {jump_identity_path_for_proxy} for ProxyCommand to {jump_arg_str}"
                            )
                        else:
                            logger.warning(
                                f"Jump host identity path {expanded_path} (from {raw_jump_identity}) not found."
                            )
                    logger.debug(f"Using jump host spec for ProxyCommand: {jump_arg_str}")

                    if jump_arg_str:
                        proxy_ssh_parts = ["ssh"]
                        if jump_identity_path_for_proxy:
                            proxy_ssh_parts.extend(["-i", jump_identity_path_for_proxy])
                            ssh_cmd_list.extend(["-i", jump_identity_path_for_proxy])
                        proxy_ssh_parts.extend(
                            [
                                "-o",
                                "StrictHostKeyChecking=no",
                                "-o",
                                "UserKnownHostsFile=/dev/null",
                                "-o",
                                "ConnectTimeout=10",
                            ]
                        )
                        proxy_ssh_parts.extend(["-W", "%h:%p", jump_arg_str])
                        proxy_command_value = shlex.join(proxy_ssh_parts)
                        ssh_cmd_list.extend(["-o", f"ProxyCommand={proxy_command_value}"])
                        logger.debug(f"Using ProxyCommand: {proxy_command_value}")

                    target_user = getattr(executor.tunnel, "user", None)
                    if target_user:
                        target_spec = f"{str(target_user)}@{head_ip}"
                    else:
                        target_spec = head_ip
                        logger.warning(
                            f"No explicit user for target {head_ip}, SSH will use default."
                        )
                    ssh_cmd_list.append(target_spec)

                    ssh_cmd_list = [
                        p for p in ssh_cmd_list if isinstance(p, str) and p.strip() != ""
                    ]

                    if not ssh_cmd_list or "ssh" not in ssh_cmd_list[0]:
                        err_msg_empty_cmd = "SSH command list is invalid or empty before Popen. Cannot start forwarding."
                        logger.error(err_msg_empty_cmd)
                        status_queue.put(("error", err_msg_empty_cmd))
                        return

                    ssh_cmd_list_for_error_reporting = list(ssh_cmd_list)
                    logger.debug(
                        f"Constructed SSH command: {' '.join(shlex.quote(p) for p in ssh_cmd_list)}"
                    )

                    self._ssh_process = subprocess.Popen(
                        ssh_cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )

                    status_queue.put(("success", None))
                    pid_info = str(self._ssh_process.pid) if self._ssh_process else "unknown"
                    logger.info(
                        f"SSH tunnel process (PID: {pid_info}) launched. "
                        f"Forwarding localhost:{port} to {head_ip}:{target_port}"
                    )

                    while not self._stop_event.is_set() and self._ssh_process:
                        process_return_code = self._ssh_process.poll()
                        if process_return_code is not None:
                            stdout_bytes, stderr_bytes = self._ssh_process.communicate()
                            decoded_stderr = stderr_bytes.decode(errors="replace")
                            decoded_stdout = stdout_bytes.decode(errors="replace")
                            logger.error(
                                f"SSH tunnel process terminated unexpectedly. "
                                f"Return code: {process_return_code}.\\n"
                                f"Command: {' '.join(shlex.quote(p) for p in ssh_cmd_list_for_error_reporting)}\\n"
                                f"Stdout: {decoded_stdout}\\n"
                                f"Stderr: {decoded_stderr}"
                            )
                            self._ssh_process = None
                            break
                        time.sleep(0.5)

                except Exception as e:
                    logger.error(
                        f"Exception in port forwarding thread run method: {str(e)}", exc_info=True
                    )
                    cmd_for_report = (
                        " ".join(shlex.quote(p) for p in ssh_cmd_list_for_error_reporting)
                        if ssh_cmd_list_for_error_reporting
                        else "[command construction failed]"
                    )
                    error_detail = f"Error starting or managing SSH tunnel: {str(e)}. Command (if available): {cmd_for_report}"
                    try:
                        status_queue.put_nowait(("error", error_detail))
                    except queue.Full:
                        logger.warning(
                            "Status queue was full when trying to report SSH setup/Popen error."
                        )
                    except Exception as q_e:
                        logger.error(f"Failed to put error on status_queue: {q_e}")

                finally:
                    self._cleanup()

            def _cleanup(self):
                if hasattr(self, "_ssh_process") and self._ssh_process:
                    process = self._ssh_process
                    pid_info = "unknown"  # Default pid_info
                    try:
                        # Check if PID exists before trying to access it, in case Popen failed partially
                        if hasattr(process, "pid") and process.pid is not None:
                            pid_info = str(process.pid)
                    except Exception:  # Broad catch if .pid access itself errors
                        pass  # pid_info remains "unknown"

                    if process.poll() is None:  # Process is still running
                        logger.debug(f"Attempting to stop SSH tunnel process (PID: {pid_info})...")
                        process.terminate()  # SIGTERM
                        try:
                            process.wait(timeout=2)  # Short wait for graceful termination
                            logger.debug(
                                f"SSH tunnel process (PID: {pid_info}) terminated gracefully (SIGTERM), exit code: {process.returncode}."
                            )
                        except subprocess.TimeoutExpired:
                            logger.warning(
                                f"SSH tunnel process (PID: {pid_info}) did not respond to SIGTERM within 2s. Sending SIGKILL."
                            )
                            process.kill()  # SIGKILL
                            try:
                                process.wait(timeout=1)  # Shorter wait for SIGKILL
                                logger.debug(
                                    f"SSH tunnel process (PID: {pid_info}) killed (SIGKILL), exit code: {process.returncode}."
                                )
                            except subprocess.TimeoutExpired:
                                logger.error(
                                    f"SSH tunnel process (PID: {pid_info}) did not terminate even after SIGKILL and 1s wait."
                                )
                        except Exception as e:
                            # Catch other exceptions during wait, e.g., if process died between poll() and wait()
                            logger.error(
                                f"Exception while waiting for SSH process (PID: {pid_info}) termination: {e}"
                            )
                            if process.poll() is not None:
                                logger.debug(
                                    f"SSH tunnel process (PID: {pid_info}) had already exited with code: {process.returncode} during exception handling."
                                )
                    else:  # Process had already exited before cleanup explicitly tried to stop it
                        # communicate() might have been called already if termination was handled in the run loop.
                        # Calling it again can lead to errors if pipes are closed.
                        # Just log that it was already stopped.
                        logger.debug(
                            f"SSH tunnel process (PID: {pid_info}) was already stopped. Exit code: {process.returncode}."
                        )
                    self._ssh_process = None  # Ensure it's cleared

            def stop_forwarding(self):
                logger.info("Stopping port forwarding")
                self._stop_event.set()

        # Create and start the forwarding thread
        forward_thread = ForwardingThread()
        forward_thread.start()

        # Wait for port forwarding to establish or fail with a timeout
        try:
            status, error_msg = status_queue.get(timeout=30)
            if status == "error":
                raise RuntimeError(f"Failed to establish port forwarding: {error_msg}")
        except queue.Empty:
            stop_event.set()
            time.sleep(0.2)  # Give it time to clean up
            raise TimeoutError("Timed out waiting for port forwarding to establish")

        # If wait option is set, block indefinitely until interrupted
        if wait:
            try:
                # Set up signal handlers for graceful shutdown
                import signal

                original_sigint_handler = signal.getsignal(signal.SIGINT)
                original_sigterm_handler = signal.getsignal(signal.SIGTERM)

                def signal_handler(sig, frame):
                    logger.info(f"Received signal {sig} to stop port forwarding")
                    stop_event.set()

                    # Restore original signal handlers
                    signal.signal(signal.SIGINT, original_sigint_handler)
                    signal.signal(signal.SIGTERM, original_sigterm_handler)

                # Set up signal handlers
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                logger.info("Port forwarding is active. Press Ctrl+C to stop...")
                while not stop_event.is_set():
                    if not forward_thread.is_alive():
                        logger.error(
                            "Port forwarding thread died unexpectedly after successful start."
                        )
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.debug("Keyboard interrupt received, stopping port forwarding")
            finally:
                logger.debug("Wait loop for port forwarding ended. Ensuring stop event is set.")
                stop_event.set()
                forward_thread.join(timeout=10)
                if forward_thread.is_alive():
                    logger.warning(
                        "Port forwarding thread did not terminate in time after stop signal."
                    )
                    if (
                        hasattr(forward_thread, "_ssh_process")
                        and forward_thread._ssh_process
                        and forward_thread._ssh_process.poll() is None
                    ):
                        pid_info = (
                            str(forward_thread._ssh_process.pid)
                            if forward_thread._ssh_process
                            else "unknown"
                        )
                        logger.warning(
                            f"SSH process (PID: {pid_info}) appears to be still running. Attempting to kill."
                        )
                        forward_thread._ssh_process.kill()
                        try:
                            forward_thread._ssh_process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.error(
                                f"SSH process (PID: {forward_thread._ssh_process.pid}) did not respond to kill."
                            )
        return forward_thread
