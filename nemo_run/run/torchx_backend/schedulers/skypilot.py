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

"""
This contains the Skypiloy scheduler which can be used to run TorchX
components using Skypilot.
"""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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

from nemo_run.config import NEMORUN_HOME
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.skypilot import _SKYPILOT_AVAILABLE, SkypilotExecutor
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

try:
    import fcntl

    FCNTL_AVAILABLE = True
except ModuleNotFoundError:
    fcntl = None
    FCNTL_AVAILABLE = False

SKYPILOT_STATES = {}
try:
    import sky.task as skyt
    from sky.skylet import job_lib

    SKYPILOT_STATES: dict[job_lib.JobStatus, AppState] = {
        job_lib.JobStatus.PENDING: AppState.PENDING,
        job_lib.JobStatus.INIT: AppState.SUBMITTED,
        job_lib.JobStatus.SETTING_UP: AppState.RUNNING,
        job_lib.JobStatus.RUNNING: AppState.RUNNING,
        job_lib.JobStatus.SUCCEEDED: AppState.SUCCEEDED,
        job_lib.JobStatus.FAILED: AppState.FAILED,
        job_lib.JobStatus.FAILED_SETUP: AppState.FAILED,
        job_lib.JobStatus.CANCELLED: AppState.CANCELLED,
    }
except ImportError:
    ...

log: logging.Logger = logging.getLogger(__name__)
SKYPILOT_JOB_DIRS = os.path.join(NEMORUN_HOME, ".skypilot_jobs.json")


@dataclass
class SkypilotRequest:
    task: "skyt.Task"
    executor: SkypilotExecutor


class SkypilotScheduler(SchedulerMixin, Scheduler[dict[str, str]]):  # type: ignore
    def __init__(self, session_name: str) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("skypilot", session_name)
        assert (
            _SKYPILOT_AVAILABLE
        ), "Skypilot is not installed. Please install it using `pip install nemo_run[skypilot]"

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "job_dir",
            type_=str,
            help="""The directory to place the job code and outputs. The
            directory must not exist and will be created.
            """,
        )
        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[SkypilotRequest]) -> str:
        req = dryrun_info.request
        task = req.task
        executor = req.executor
        executor.package(executor.packager, job_name=executor.job_name)
        job_id, handle = executor.launch(task)
        assert (
            job_id and handle
        ), f"Failed scheduling run on Skypilot. Job id: {job_id}, Handle: {handle}"
        app_id = f"{executor.experiment_id}___{handle.get_cluster_name()}___{task.name}___{job_id}"
        _, task_details = SkypilotExecutor.status(app_id=app_id)
        if task_details:
            _save_job_dir(
                app_id,
                job_status=task_details["status"].value,
                log_dir=task_details["log_path"],
            )

        return app_id

    def _submit_dryrun(  # type: ignore
        self, app: AppDef, cfg: Executor
    ) -> AppDryRunInfo[SkypilotRequest]:
        from sky.utils import common_utils

        assert isinstance(
            cfg, SkypilotExecutor
        ), f"{cfg.__class__} not supported for skypilot scheduler."
        executor = cfg

        assert len(app.roles) == 1, "Only 1 role supported for Skypilot executor."
        role = app.roles[0]
        values = executor.macro_values()
        if values:
            role = values.apply(role)

        cmd = [role.entrypoint] + role.args
        task = cfg.to_task(name=role.name, cmd=cmd, env_vars=role.env)

        req = SkypilotRequest(task=task, executor=cfg)
        return AppDryRunInfo(req, lambda req: common_utils.dump_yaml_str(req.task.to_yaml_config()))

    def _validate(self, app: AppDef, scheduler: str) -> None:
        # Skip validation step for skypilot
        pass

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        from sky.skylet import job_lib

        cluster_name, task_name, _ = SkypilotExecutor.parse_app(app_id=app_id)
        cluster_status, task_details = SkypilotExecutor.status(app_id=app_id)

        roles = [Role(name=task_name, image="", num_replicas=1)]
        roles_statuses = [
            RoleStatus(
                task_name,
                replicas=[
                    ReplicaStatus(
                        id=0,
                        role=task_name,
                        state=AppState.SUBMITTED,
                        hostname=cluster_name,
                    )
                ],
            )
        ]

        if not cluster_status and not task_details:
            past_apps = _get_job_dirs()
            if app_id in past_apps and "job_status" in past_apps[app_id]:
                job_status = job_lib.JobStatus[past_apps[app_id]["job_status"]]
                app_state = SKYPILOT_STATES[job_status]
                roles_statuses[0].replicas[0].state = app_state
                return DescribeAppResponse(
                    app_id=app_id,
                    roles=roles,
                    roles_statuses=roles_statuses,
                    state=app_state,
                    msg="",
                    ui_url=past_apps[app_id]["log_dir"] if "log_dir" in past_apps[app_id] else None,
                )
            else:
                return None
        elif cluster_status and not task_details:
            return DescribeAppResponse(
                app_id=app_id,
                roles=roles,
                roles_statuses=roles_statuses,
                state=AppState.SUBMITTED,
                msg="",
            )
        elif task_details:
            app_state = SKYPILOT_STATES[task_details["status"]]
            _save_job_dir(
                app_id,
                job_status=task_details["status"].value,
                log_dir=task_details["log_path"],
            )
            roles_statuses[0].replicas[0].state = app_state
            return DescribeAppResponse(
                app_id=app_id,
                roles=roles,
                roles_statuses=roles_statuses,
                state=app_state,
                msg="",
                ui_url=task_details["log_path"],
            )
        else:
            return None

    def _cancel_existing(self, app_id: str) -> None:
        SkypilotExecutor.cancel(app_id=app_id)

    def list(self) -> list[ListAppResponse]: ...


def create_scheduler(session_name: str, **kwargs: Any) -> SkypilotScheduler:
    return SkypilotScheduler(
        session_name=session_name,
    )


def _save_job_dir(app_id: str, job_status: str, log_dir: str) -> None:
    original_apps = {}
    if not os.path.isfile(SKYPILOT_JOB_DIRS):
        Path(SKYPILOT_JOB_DIRS).touch()

    with open(SKYPILOT_JOB_DIRS, "r+") as f:
        if FCNTL_AVAILABLE:
            assert fcntl
            fcntl.flock(f, fcntl.LOCK_EX)

        try:
            try:
                original_apps = json.load(f)
            except Exception:
                original_apps = {}

            app = {
                "job_status": job_status,
                "log_dir": log_dir,
            }
            original_apps[app_id] = app

            with tempfile.NamedTemporaryFile(mode="w+") as fp:
                json.dump(original_apps, fp)
                fp.flush()

                shutil.copy(fp.name, SKYPILOT_JOB_DIRS)
                fp.close()
        finally:
            if FCNTL_AVAILABLE:
                assert fcntl
                fcntl.flock(f, fcntl.LOCK_UN)


def _get_job_dirs() -> dict[str, dict[str, str]]:
    try:
        with open(SKYPILOT_JOB_DIRS, "r") as f:
            apps: dict[str, dict[str, str]] = json.load(f)
    except FileNotFoundError:
        return {}

    return apps
