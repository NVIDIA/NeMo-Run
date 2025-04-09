import base64
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import tempfile
from typing import Any, Optional, Type

import requests
from invoke.context import Context

from nemo_run.core.execution.base import Executor, ExecutorMacros
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager
from nemo_run.config import get_nemorun_home

logger = logging.getLogger(__name__)


class DGXCloudState(Enum):
    CREATING = "Creating"
    INITIALIZING = "Initializing"
    RESUMING = "Resuming"
    PENDING = "Pending"
    DELETING = "Deleting"
    RUNNING = "Running"
    UPDATING = "Updating"
    STOPPED = "Stopped"
    STOPPING = "Stopping"
    DEGRADED = "Degraded"
    FAILED = "Failed"
    COMPLETED = "Completed"
    TERMINATING = "Terminating"
    UNKNOWN = "Unknown"


@dataclass(kw_only=True)
class DGXCloudExecutor(Executor):
    """
    Dataclass to configure a DGX Executor.

    This executor integrates with a DGX cloud endpoint for launching jobs
    via a REST API. It acquires an auth token, identifies the project/cluster,
    and launches jobs with a specified command. It can be adapted to meet user
    authentication and job-submission requirements on DGX.
    """

    base_url: str
    app_id: str
    app_secret: str
    project_name: str
    container_image: str
    nodes: int = 1
    gpus_per_node: int = 0
    nprocs_per_node: int = 1
    pvc_nemo_run_dir: str
    pvc_job_dir: str = field(init=False, default="")
    pvcs: list[dict[str, Any]] = field(default_factory=list)
    distributed_framework: str = "PyTorch"
    custom_spec: dict[str, Any] = field(default_factory=dict)

    def get_auth_token(self) -> Optional[str]:
        url = f"{self.base_url}/token"
        payload = {
            "grantType": "app_token",
            "appId": self.app_id,
            "appSecret": self.app_secret,
        }

        response = requests.post(url, json=payload, headers=self._default_headers())
        response_text = response.text.strip()
        auth_token = json.loads(response_text).get("accessToken", None)  # [1]
        if not auth_token:
            logger.error("Failed to retrieve auth token; response was: %s", response_text)
            return None

        logger.debug("Retrieved auth token from %s", url)
        return auth_token

    def get_project_and_cluster_id(self, token: str) -> tuple[Optional[str], Optional[str]]:
        url = f"{self.base_url}/org-unit/projects"
        headers = self._default_headers(token=token)
        response = requests.get(url, headers=headers)
        projects = json.loads(response.text.strip()).get("projects", [])
        project_id, cluster_id = None, None
        for prj in projects:
            if not self.project_name or prj["name"] == self.project_name:  # [2]
                project_id, cluster_id = prj["id"], prj["clusterId"]
                logger.debug(
                    "Found project '%s' (%s) on cluster '%s'", prj["name"], project_id, cluster_id
                )
                break
        return project_id, cluster_id

    def copy_directory_data_command(self, local_dir_path, dest_path):
        # Create a temporary directory to hold the tarball
        directory_name=os.path.basename(local_dir_path)

        with tempfile.TemporaryDirectory() as temp_dir:

            tarball_path = os.path.join(temp_dir, "archive.tar.gz")

            # Create a tarball of the directory
            subprocess.run(f"tar -czf {tarball_path} -C {local_dir_path} .", shell=True, check=True)

            # Read the tarball
            with open(tarball_path, "rb") as file:
                file_data = file.read()

            # Encode the tarball data to base64
            encoded_data = base64.b64encode(file_data).decode("utf-8")

            # Delete and recreate directory if it already exists, command to decode base64 data, save to a file, and extract inside the pod
            cmd = f"rm -rf {dest_path} && mkdir -p {dest_path} && echo {encoded_data} | base64 -d > {dest_path}/archive.tar.gz && tar -xzf {dest_path}/archive.tar.gz -C {dest_path} && rm {dest_path}/archive.tar.gz"
            return cmd

    def create_data_mover_workload(
        self, token: str, project_id: str, cluster_id: str, name: str
    ):
        """
        Creates an cpu only workload to move job directory into PVC using the provided project/cluster IDs.
        """

        cmd=self.copy_directory_data_command(self.job_dir, self.pvc_job_dir)

        url = f"{self.base_url}/workloads/workspaces"
        headers = self._default_headers(token=token)

        payload = {
            "name": name,
            "useGivenNameAsPrefix": True,
            "projectId": project_id,
            "clusterId": cluster_id,
            "spec": {
                "command": "sh -c",
                "args": f"'{cmd}'",
                "image": "busybox:1.37.0",
                "storage": {"pvc": self.pvcs},
            }
        }

        response = requests.post(url, json=payload, headers=headers)

        logger.debug(
            "Created workload; response code=%s, content=%s",
            response.status_code,
            response.text.strip(),
        )

        return response

    def workload_completed(
            self, token: str, workload_id: str
    ):
        """
        Checks the status of the interactive workload
        """

        url = f"{self.base_url}/workloads/{workload_id}"
        headers = self._default_headers(token=token)

        response = requests.get(url, headers=headers)

        if response.status_code not in [200, 202]:
            raise RuntimeError(f"Failed to get workload status, status_code={response.status_code}")

        phase = response.json()["phase"]

        if phase == "Completed":
            return True
        else:
            return False

    def delete_workload(
            self, token: str, workload_id: str, sleep: int = 10
    ):
        while not self.workload_completed(token, workload_id):
            time.sleep(sleep)

        url = f"{self.base_url}/workloads/workspaces/{workload_id}"
        headers = self._default_headers(token=token)

        response = requests.delete(url, headers=headers)

        logger.debug(
            "Delete interactive workspace; response code=%s, content=%s",
            response.status_code,
            response.text.strip(),
        )
        return response

    def create_distributed_job(
        self, token: str, project_id: str, cluster_id: str, name: str
    ):
        """
        Creates a distributed PyTorch job using the provided project/cluster IDs.
        """
        url = f"{self.base_url}/workloads/distributed"
        headers = self._default_headers(token=token)

        payload = {
            "name": name,
            "useGivenNameAsPrefix": True,
            "projectId": project_id,
            "clusterId": cluster_id,
            "spec": {
                "command": f"/bin/bash {self.pvc_job_dir}/launch_script.sh",
                "image": self.container_image,
                "distributedFramework": self.distributed_framework,
                "minReplicas": self.nodes,
                "maxReplicas": self.nodes,
                "numWorkers": self.nodes,
                "compute": {"gpuDevicesRequest": self.gpus_per_node},
                "storage": {"pvc": self.pvcs},
                "environmentVariables": [
                    {"name": key, "value": value} for key, value in self.env_vars.items()
                ],
                **self.custom_spec,
            },
        }

        response = requests.post(url, json=payload, headers=headers)
        logger.debug(
            "Created distributed job; response code=%s, content=%s",
            response.status_code,
            response.text.strip(),
        )
        return response

    def launch(self, name: str, cmd: list[str]) -> tuple[str, str]:
        name = name.replace("_", "-").replace(".", "-")  # to meet K8s requirements
        token = self.get_auth_token()
        if not token:
            raise RuntimeError("Failed to get auth token")

        project_id, cluster_id = self.get_project_and_cluster_id(token)
        if not project_id or not cluster_id:
            raise RuntimeError("Unable to determine project/cluster IDs for job submission")
        
        #prepare launch script and move data to PVC 
        launch_script = f"""
ln -s {self.pvc_job_dir}/ /nemo_run
cd /nemo_run/code
{" ".join(cmd)}
"""
        with open(os.path.join(self.job_dir, "launch_script.sh"), "w+") as f:
            f.write(launch_script)

        logger.info("Creating data movement workload")
        resp = self.create_data_mover_workload(token, project_id, cluster_id, "data-mover")
        if resp.status_code not in [200, 202]:
            raise RuntimeError(f"Failed to copy data to PVC, status_code={resp.status_code}")

        workload_id= resp.json()["workloadId"]
        resp = self.delete_workload(token, workload_id)
        if resp.status_code >= 200 and resp.status_code < 300:
            logger.info(
                "Successfully delete data movement workload %s on DGX with response code %d",
                workload_id,
                resp.status_code,
            )
        else:
            logger.error(
                "Failed to delete data movement workload %s, response code=%d, reason=%s",
                workload_id,
                resp.status_code,
                resp.text,
            )

        logger.info("Creating distributed workload")
        resp = self.create_distributed_job(token, project_id, cluster_id, name)
        if resp.status_code not in [200, 202]:
            raise RuntimeError(f"Failed to create job, status_code={resp.status_code}")

        r_json = resp.json()
        job_id = r_json["workloadId"]
        status = r_json["actualPhase"]
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

    def status(self, job_id: str) -> Optional[DGXCloudState]:
        url = f"{self.base_url}/workloads/distributed/{job_id}"
        token = self.get_auth_token()
        if not token:
            logger.error("Failed to retrieve auth token for cancellation request.")
            return None

        headers = self._default_headers(token=token)
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return DGXCloudState("Unknown")

        r_json = response.json()
        return DGXCloudState(r_json["actualPhase"])

    def cancel(self, job_id: str):
        # Retrieve the authentication token for the REST calls
        token = self.get_auth_token()
        if not token:
            logger.error("Failed to retrieve auth token for cancellation request.")
            return

        # Build the DELETE request to cancel the job
        url = f"{self.base_url}/workloads/distributed/{job_id}/suspend"
        headers = self._default_headers(token=token)

        response = requests.get(url, headers=headers)
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(
                "Successfully cancelled job %s on DGX with response code %d",
                job_id,
                response.status_code,
            )
        else:
            logger.error(
                "Failed to cancel job %s, response code=%d, reason=%s",
                job_id,
                response.status_code,
                response.text,
            )

    @classmethod
    def logs(cls: Type["DGXCloudExecutor"], app_id: str, fallback_path: Optional[str]):
        logger.warning(
            "Logs not available for DGXCloudExecutor based jobs. Please visit the cluster UI to view the logs."
        )

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
        job_subdir = self.job_dir[len(get_nemorun_home())+1:] # +1 to remove the initial backslash
        self.pvc_job_dir = os.path.join(self.pvc_nemo_run_dir, job_subdir)

        logger.info(
                "PVC job directory set as:  %s",
                self.pvc_job_dir,
            )
        self.experiment_id = exp_id

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