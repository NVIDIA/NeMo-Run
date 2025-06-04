import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
)
from torchx.specs import (
    AppDef,
    AppState,
    ReplicaStatus,
    Role,
    RoleStatus,
    runopts,
)

from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.runai import RunAIExecutor, RunAIState
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

# Local placeholder for storing NVIDIA Run:ai job states
RUNAI_JOB_DIRS = os.path.join(get_nemorun_home(), ".runai_jobs.json")

# Example mapping from some NVIDIA Run:ai statuses to the TorchX AppState
RUNAI_STATES: dict[RunAIState, AppState] = {
    RunAIState.CREATING: AppState.PENDING,
    RunAIState.INITIALIZING: AppState.SUBMITTED,
    RunAIState.RESUMING: AppState.RUNNING,
    RunAIState.PENDING: AppState.PENDING,
    RunAIState.DELETING: AppState.RUNNING,
    RunAIState.RUNNING: AppState.RUNNING,
    RunAIState.UPDATING: AppState.RUNNING,
    RunAIState.STOPPED: AppState.CANCELLED,
    RunAIState.STOPPING: AppState.CANCELLED,
    RunAIState.DEGRADED: AppState.FAILED,
    RunAIState.FAILED: AppState.FAILED,
    RunAIState.COMPLETED: AppState.SUCCEEDED,
    RunAIState.TERMINATING: AppState.RUNNING,
    RunAIState.UNKNOWN: AppState.FAILED,
}

log = logging.getLogger(__name__)


@dataclass
class RunAIRequest:
    """
    Wrapper around the torchx AppDef and the NVIDIA Run:ai executor.
    This object is used to store job submission info for the scheduler.
    """

    app: AppDef
    executor: RunAIExecutor
    cmd: list[str]
    name: str


class RunAICloudScheduler(SchedulerMixin, Scheduler[dict[str, str]]):  # type: ignore
    def __init__(self, session_name: str) -> None:
        super().__init__("runai", session_name)

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "job_dir",
            type_=str,
            help="The directory to place the job code and outputs."
            " The directory must not exist and will be created.",
        )
        return opts

    def _submit_dryrun(  # type: ignore
        self,
        app: AppDef,
        cfg: Executor,
    ) -> AppDryRunInfo[RunAIRequest]:
        assert isinstance(cfg, RunAIExecutor), (
            f"{cfg.__class__} not supported for NVIDIA Run:ai Cloud scheduler."
        )
        executor = cfg

        assert len(app.roles) == 1, "Only single-role apps are supported."
        role = app.roles[0]
        values = cfg.macro_values()
        if values:
            role = values.apply(role)

        cmd = [role.entrypoint] + role.args
        return AppDryRunInfo(
            RunAIRequest(app=app, executor=executor, cmd=cmd, name=role.name),
            # Minimal function to show the config, if any
            lambda req: f"NVIDIA Run:ai job for app: {req.app.name}, cmd: {' '.join(cmd)}, executor: {executor}",
        )

    def schedule(self, dryrun_info: AppDryRunInfo[RunAIRequest]) -> str:
        """
        Launches a job on NVIDIA Run:ai using the RunAIExecutor. Returns an app_id
        used by TorchX for subsequent queries/cancellations.
        """
        req = dryrun_info.request
        executor = req.executor

        # If needed, package or prepare code (executor may no-op).
        executor.package(executor.packager, job_name=executor.job_name)

        # The RunAIExecutor's launch call typically returns (job_id, handle).
        # We'll call it without additional parameters here.
        job_id, status = executor.launch(name=req.name, cmd=req.cmd)
        if not job_id:
            raise RuntimeError("Failed scheduling run on NVIDIA Run:ai: no job_id returned")

        # Example app_id format:
        # <experiment_id>___<role-name>___<job_id>
        # If we have no explicit role name, we fall back to the app name.
        role_name = req.app.roles[0].name
        # If experiment_id is not set, fake one for demonstration:
        experiment_id = getattr(executor, "experiment_id", "runai_experiment")
        app_id = f"{experiment_id}___{role_name}___{job_id}"

        # Store a status entry or logs path if available
        # Currently, the RunAIExecutor status is placeholder, but we keep the pattern
        _save_job_dir(app_id, job_status=status, executor=executor)

        return app_id

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        """
        Returns information about the job. If there's no recognized job,
        returns None.
        """
        # We split out the stored values from the JSON file
        stored_data = _get_job_dirs()
        job_info = stored_data.get(app_id)
        _, role_name, job_id = app_id.split("___")
        roles = [Role(name=role_name, image="", num_replicas=1)]
        roles_statuses = [
            RoleStatus(
                role_name,
                replicas=[
                    ReplicaStatus(id=0, role=role_name, state=AppState.SUBMITTED, hostname="")
                ],
            )
        ]

        if not job_info:
            return None

        executor: RunAIExecutor = job_info.get("executor", None)  # type: ignore
        if not executor:
            return None

        runai_state = executor.status(job_id) or RunAIState.UNKNOWN
        app_state = RUNAI_STATES.get(runai_state, AppState.UNKNOWN)
        roles_statuses[0].replicas[0].state = app_state

        return DescribeAppResponse(
            app_id=app_id,
            roles=roles,
            roles_statuses=roles_statuses,
            state=app_state,
            msg="",
            ui_url=f"{executor.base_url}/workloads/distributed/{job_id}",
        )

    def _cancel_existing(self, app_id: str) -> None:
        """
        Cancels the job by calling the NVIDIA Run:ai Executor's cancel method.
        """
        stored_data = _get_job_dirs()
        job_info = stored_data.get(app_id)
        _, _, job_id = app_id.split("___")
        executor: RunAIExecutor = job_info.get("executor", None)  # type: ignore
        if not executor:
            return None
        executor.cancel(job_id)

    def list(self) -> list[ListAppResponse]: ...

    def _validate(self, app: AppDef, scheduler: str) -> None:
        # For demonstration, skip validation
        pass


def create_scheduler(session_name: str, **kwargs: Any) -> RunAICloudScheduler:
    return RunAICloudScheduler(session_name=session_name)


def _save_job_dir(app_id: str, job_status: str, executor: RunAIExecutor) -> None:
    """
    Saves or updates local record of job status in JSON for demonstration.
    """
    original_apps = {}
    os.makedirs(os.path.dirname(RUNAI_JOB_DIRS), exist_ok=True)
    if not os.path.isfile(RUNAI_JOB_DIRS):
        os.makedirs(os.path.dirname(RUNAI_JOB_DIRS), exist_ok=True)
        Path(RUNAI_JOB_DIRS).touch()

    serializer = ZlibJSONSerializer()
    with open(RUNAI_JOB_DIRS, "r+") as f:
        try:
            original_apps = json.load(f)
        except Exception:
            original_apps = {}

        app = {
            "job_status": job_status,
            "executor": serializer.serialize(
                fdl_dc.convert_dataclasses_to_configs(executor, allow_post_init=True)
            ),
        }
        original_apps[app_id] = app

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
            json.dump(original_apps, fp)
            temp_path = fp.name

        f.close()
        shutil.move(temp_path, RUNAI_JOB_DIRS)


def _get_job_dirs() -> dict[str, dict[str, str]]:
    """
    Retrieves local record of job status in JSON for demonstration.
    """
    if not os.path.isfile(RUNAI_JOB_DIRS):
        return {}
    with open(RUNAI_JOB_DIRS, "r") as f:
        data = json.load(f)

    serializer = ZlibJSONSerializer()
    for app in data.values():
        try:
            app["executor"] = fdl.build(serializer.deserialize(app["executor"]))
        except Exception as e:
            log.debug(f"Failed to deserialize executor: {e}")
            continue

    return data
