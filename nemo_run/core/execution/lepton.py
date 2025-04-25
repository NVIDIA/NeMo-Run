import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

from invoke.context import Context
from leptonai.api.v1.client import APIClient
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvVar, LeptonContainer, Mount
from leptonai.api.v1.types.job import LeptonJob, LeptonJobState, LeptonJobUserSpec
from leptonai.api.v1.types.replica import Replica

from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LeptonExecutor(Executor):
    """
    Dataclass to configure a Lepton Executor.

    This executor integrates with a Lepton endpoint for launching jobs
    via a REST API. It acquires an auth token, identifies the project/cluster,
    and launches jobs with a specified command. It can be adapted to meet user
    authentication and job-submission requirements on Lepton.
    """

    container_image: str
    nemo_run_dir: str
    launched_from_cluster: bool = False
    nodes: int = 1
    gpus_per_node: int = 0
    nprocs_per_node: int = 1
    shared_memory_size: int = 65536
    resource_shape: str = ""
    node_group: str = ""
    lepton_job_dir: str = field(init=False, default="")
    custom_spec: dict[str, Any] = field(default_factory=dict)

    def stop_job(self, job_id: str):
        """
        Send a stop signal to the requested job
        """
        client = APIClient()

        if not self.status(job_id) == LeptonJobState.Running:
            logger.info(f"Job {job_id} is not currently running. No action taken.")
            return

        # Send a "Stopped" signal to the job
        client.job.update(job_id, spec={"spec": {"stopped": True}})
        logger.info(f"Job {job_id} stopped successfully.")

    def move_data(self, sleep: float = 10) -> None:
        """
        Moves job directory into PVC and deletes the workload after completion
        """
        client = APIClient()
        client.storage.create_dir(additional_path=self.lepton_job_dir)

        # Create all sub-directories in the directory tree
        # Then, copy all files to the storage
        for root, dirs, files in os.walk(self.job_dir):
            # Create the sub-directories
            for dir in dirs:
                abs_path = os.path.join(root, dir)
                relative_path = os.path.join(self.lepton_job_dir, abs_path.replace(self.job_dir, "").lstrip("/"))
                client.storage.create_dir(additional_path=relative_path)
            # Copy the files in each sub-directory to the remote filesystem
            for file in files:
                abs_path = os.path.join(root, file)
                relative_path = os.path.join(self.lepton_job_dir, abs_path.replace(self.job_dir, "").lstrip("/"))
                client.storage.create_file(
                    local_path=abs_path,
                    remote_path=relative_path
                )

    def setup_distributed_pytorch(self) -> str:
        """
        Runs a custom script from Lepton to setup the distributed PyTorch
        environment variables required for distributed PyTorch jobs.
        """
        distributed_command = (
            "wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh && "
            "chmod +x init.sh && "
            "source init.sh"
        )
        return distributed_command

    def create_lepton_job(self, name: str):
        """
        Creates a distributed PyTorch job using the provided project/cluster IDs.
        """
        client = APIClient()

        envs = [EnvVar(name=key, value=value) for key, value in self.env_vars.items()]

        cmd = [
            "/bin/bash",
            "-c",
            f"chmod +x {self.lepton_job_dir}/launch_script.sh && bash {self.lepton_job_dir}/launch_script.sh"
        ]

        # Get node groups
        node_groups = client.nodegroup.list_all()
        node_group_map = {ng.metadata.name: ng for ng in node_groups}
        node_group_id = node_group_map[self.node_group]

        # Get node IDs
        valid_node_ids = set()
        node_ids = client.nodegroup.list_nodes(node_group_id)
        for node in node_ids:
            valid_node_ids.add(node.metadata.id_)

        job_spec = LeptonJobUserSpec(
            resource_shape=self.resource_shape,
            affinity=LeptonResourceAffinity(
                allowed_dedicated_node_groups=[node_group_id.metadata.id_],
                allowed_nodes_in_node_group=list(valid_node_ids),
            ),
            container=LeptonContainer(image=self.container_image, command=cmd),
            shared_memory_size=self.shared_memory_size,
            completions=self.nodes,
            parallelism=self.nodes,
            max_failure_retry=None,
            max_job_failure_retry=None,
            envs=envs,
            mounts=[
                Mount(path="/nemo-workspace", mount_path="/nemo-workspace")
            ],
            image_pull_secrets=[],
            ttl_seconds_after_finished=None,
            intra_job_communication=True,
            privileged=False,
            metrics=None,
            log=None,
            queue_config=None,
            stopped=None,
            reservation_config=None,
        )
        job = LeptonJob(spec=job_spec, metadata=Metadata(id=name))

        created_job = client.job.create(job)
        return created_job

    def launch(self, name: str, cmd: list[str]) -> tuple[str, str]:
        name = name.replace("_", "-").replace(".", "-")  # to meet K8s requirements
        launch_script = f"""
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh
ln -s {self.lepton_job_dir}/ /nemo_run
cd /nemo_run/code
{" ".join(cmd)}
"""

        with open(os.path.join(self.job_dir, "launch_script.sh"), "w+") as f:
            f.write(launch_script)

        logger.info("Copying experiment directory to remote filesystem")
        self.move_data()

        logger.info("Creating distributed workload")
        job = self.create_lepton_job(name)
        if not job:
            raise RuntimeError(f"Failed to create Lepton job")

        job_id = job.metadata.id_
        status = self.status(job_id)
        return job_id, status

    def nnodes(self) -> int:
        return self.nodes

    def nproc_per_node(self) -> int:
        # Default to the number of GPUs specified per node
        # If user doesn't want GPUs, can run multiple processes with CPU only
        if self.gpus_per_node:
            return self.gpus_per_node
        elif self.nprocs_per_node:
            return self.nprocs_per_node
        return 1

    def status(self, job_id: str) -> Optional[LeptonJobState]:
        client = APIClient()
        job = client.job.get(job_id)

        if not job:
            return LeptonJobState.Unknown

        # Lepton marks a job as Running when at least one pod is running
        # which can cause issues as all pods need to be running in order
        # to query it. Override the job state to check if all pods are
        # actually running. If not, set the status to Starting and wait
        # until all pods are ready.
        if job.status.state == LeptonJobState.Running:
            if job.status.ready < job.status.active:
                return LeptonJobState.Starting
        return job.status.state

    def cancel(self, job_id: str):
        client = APIClient()
        client.job.delete(job_id)
        logger.info(f"Successfully cancelled job {job_id} on Lepton")

    @classmethod
    def logs(cls: Type["LeptonExecutor"], app_id: str, fallback_path: Optional[str]):
        client = APIClient()

        # Get the first replica from the job which contains the job logs
        def _first_replica(job_id: str) -> Replica:
            client = APIClient()
            first_replica = None

            replicas = client.job.get_replicas(job_id)

            for replica in replicas:
                replica_id = replica.metadata.id_
                # The first replica has the pattern <job-id>-0-xxxxx
                # where xxxxx is a unique ID for each worker. Subsequent
                # workers increase the number between <job-id> and the
                # unique ID. For example, if a job-id is "my-nemo-job"
                # the first replica would be "my-nemo-job-0-xxxxx",
                # the second would be "my-nemo-job-1-yyyyy", and so on.
                if replica_id.replace(job_id, "").startswith("-0"):
                    first_replica = replica
            if not first_replica:
                raise RuntimeError(f"Unable to retrieve workers for job {job_id}")
            return first_replica

        def _status(job_id: str):
            client = APIClient()
            job = client.job.get(job_id)

            if not job:
                return LeptonJobState.Unknown

            # Lepton marks a job as Running when at least one pod is running
            # which can cause issues as all pods need to be running in order
            # to query it. Override the job state to check if all pods are
            # actually running. If not, set the status to Starting and wait
            # until all pods are ready.
            if job.status.state == LeptonJobState.Running:
                if job.status.ready < job.status.active:
                    return LeptonJobState.Starting
            return job.status.state

        # Regex pattern to remove everything up to and including the second '___'
        job_id = re.sub(r"^(?:.*?___){2}", "", app_id)

        # Wait for the job to be in the Running state prior to reading the logs
        while _status(job_id) != LeptonJobState.Running:
            time.sleep(1)
        replica = _first_replica(job_id)
        logs = client.job.get_log(id_or_job=job_id, replica=replica)

        for line in logs:
            print(line)

    def cleanup(self, handle: str): ...

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        self.job_name = task_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)

        # setting linked PVC job directory
        nemo_run_home = get_nemorun_home()
        job_subdir = self.job_dir[len(nemo_run_home) + 1 :]  # +1 to remove the initial backslash
        self.lepton_job_dir = os.path.join(self.nemo_run_dir, job_subdir)

        logger.info(
            "Lepton job directory set as:  %s",
            self.lepton_job_dir,
        )
        self.experiment_id = exp_id

    def get_launcher_prefix(self) -> Optional[list[str]]:
        launcher = self.get_launcher()
        if launcher.nsys_profile:
            return launcher.get_nsys_prefix(profile_dir="/nemo_run")

    def package_configs(self, *cfgs: tuple[str, str]) -> list[str]:
        filenames = []
        basepath = os.path.join(self.job_dir, "configs")
        for name, cfg in cfgs:
            filename = os.path.join(basepath, name)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(cfg)
            filenames.append(
                os.path.join(
                    "/nemo_run/configs",
                    name,
                )
            )
        return filenames

    def package(self, packager: Packager, job_name: str):
        assert self.experiment_id, "Executor not assigned to an experiment."
        if isinstance(packager, GitArchivePackager):
            output = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                stdout=subprocess.PIPE,
            )
            path = output.stdout.splitlines()[0].decode()
            base_path = Path(path).absolute()
        else:
            base_path = Path(os.getcwd()).absolute()

        local_pkg = packager.package(base_path, self.job_dir, job_name)
        local_code_extraction_path = os.path.join(self.job_dir, "code")
        ctx = Context()
        ctx.run(f"mkdir -p {local_code_extraction_path}")

        if self.get_launcher().nsys_profile:
            remote_nsys_extraction_path = os.path.join(
                self.job_dir, self.get_launcher().nsys_folder
            )
            ctx.run(f"mkdir -p {remote_nsys_extraction_path}")
        if local_pkg:
            ctx.run(
                f"tar -xvzf {local_pkg} -C {local_code_extraction_path} --ignore-zeros", hide=True
            )

    def macro_values(self) -> Optional[ExecutorMacros]:
        return None

    def _default_headers(self, token: Optional[str] = None) -> dict:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
