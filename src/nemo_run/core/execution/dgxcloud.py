import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from invoke.context import Context

from nemo_run.core.execution.base import (
    Executor,
    ExecutorMacros,
)
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DGXCloudExecutor(Executor):
    """
    Dataclass to configure a DGX Executor.

    This executor integrates with a DGX cloud endpoint for launching jobs
    via a REST API. It acquires an auth token, identifies the project/cluster,
    and launches jobs with a specified command. It can be adapted to meet user
    authentication and job-submission requirements on DGX.

    Example usage might include specifying the environment variables or secrets
    needed to create new distributed training jobs and storing user-specified
    configuration (cluster URL, project name, application secrets, etc.).
    """

    base_url: str
    app_id: str
    app_secret: str
    project_name: str
    job_name: str
    container_image: str
    nodes: int = 1
    gpus_per_node: int = 8
    pvcs: list[dict[str, Any]] = field(default_factory=list)
    distributed_framework: str = "PyTorch"
    custom_spec: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.job_name = self.job_name.replace("_", "-")

    def get_auth_token(self) -> Optional[str]:
        """
        Retrieves the authorization token from the endpoint. Required for subsequent
        calls to create distributed jobs on the DGX platform.
        """
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
        """
        Retrieves the project ID and cluster ID by matching the user-provided
        project_name to the result from the DGX API. Returns (project_id, cluster_id).
        """
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

    def create_distributed_job(self, token: str, project_id: str, cluster_id: str):
        """
        Creates a distributed PyTorch job using the provided project/cluster IDs.
        """
        url = f"{self.base_url}/workloads/distributed"
        headers = self._default_headers(token=token)
        payload = {
            "name": self.job_name,
            "useGivenNameAsPrefix": True,
            "projectId": project_id,
            "clusterId": cluster_id,
            "spec": {
                "command": "echo 'hello' && sleep 60 && echo 'goodbye'",
                #                 "args": f"""
                # # ln -s {self.job_dir} /nemo_run
                # echo "Hello"
                # sleep 600
                # echo "Goodbye"
                # """,
                "image": self.container_image,
                # "workingDir": "/nemo_run/code",
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

    def launch(self, *args, **kwargs) -> tuple[Optional[str], Optional[str]]:
        """
        Core entry point to create a token, get the project/cluster, and launch
        the distributed job on the DGX platform.
        Returns (job_id, handle) to align with the typical Nemo-Run Executor pattern.
        """
        token = self.get_auth_token()
        if not token:
            logger.error("Cannot proceed without auth token")
            return None, None

        project_id, cluster_id = self.get_project_and_cluster_id(token)
        if not project_id or not cluster_id:
            logger.error("Unable to determine project/cluster IDs for job submission")
            return None, None

        resp = self.create_distributed_job(token, project_id, cluster_id)
        if resp.status_code not in [200, 202]:
            logger.error("Failed to create job, status_code=%s", resp.status_code)
            return None, None

        # For demonstration, parse out some job ID from the response if available
        try:
            r_json = resp.json()
            job_id = r_json.get("id", "dgx_job_id")  # Example ID key
        except Exception:  # If the response is not valid JSON or no "id"
            job_id = "dgx_job_id"

        # Typically in Nemo-Run, "handle" can store information for references
        handle = f"dgx://{job_id}"
        return job_id, handle

    def status(self, app_id: str) -> tuple[Optional[str], Optional[dict]]:
        """
        Return the job status from the DGX platform. The app_id might be used
        to query the job ID stored at creation time. For demonstration, this is
        left abstract, as the API for status queries can be matched to user needs.
        """
        logger.debug("Getting status for app_id=%s", app_id)  # [1]
        # If a specialized endpoint exists, you would call it here, e.g.:
        # GET <base_url>/workloads/<job_id>
        return None, None

    def cancel(self, app_id: str):
        """
        Cancels the job on the DGX platform. Typically, you'd parse the job_id
        from app_id and call the relevant REST endpoint to delete/cancel the job.
        """
        logger.debug("Attempt to cancel job for app_id=%s", app_id)

    def logs(self, app_id: str, fallback_path: Optional[str]):
        """
        Prints or fetches logs for the job. Typically, you'd parse the job_id
        from app_id and query a logs endpoint. Fallback logic can be implemented
        if logs must be fetched from a known file path.
        """

    def cleanup(self, handle: str):
        """
        Performs any necessary cleanup after the job has completed.
        """

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        """
        Assigns the job to a specific experiment run directory in Nemo-Run.
        """
        self.job_name = task_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        self.experiment_id = exp_id
        os.makedirs(self.job_dir, exist_ok=True)
        assert any(
            map(lambda x: Path(self.job_dir).relative_to(Path(x["path"])), self.pvcs)
        ), f"Need to specify atleast one PVC matching {self.job_dir}"

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
        """
        Returns environment macros for distributed training. Not strictly used in this
        example, but can configure advanced key-value pairs for the job environment.
        """
        return None

    def _default_headers(self, token: Optional[str] = None) -> dict:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
