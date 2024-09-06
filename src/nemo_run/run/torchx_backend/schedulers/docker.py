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

import glob
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import fiddle as fdl
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    Stream,
    filter_regex,
    split_lines,
    split_lines_iterator,
)
from torchx.schedulers.docker_scheduler import DockerScheduler, _to_str
from torchx.schedulers.ids import make_unique
from torchx.schedulers.local_scheduler import (
    LogIterator,
)
from torchx.specs.api import (
    AppDef,
    AppState,
    ReplicaStatus,
    Role,
    RoleStatus,
    is_terminal,
)

from nemo_run.config import NEMORUN_HOME, RUNDIR_NAME
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.docker import (
    DockerContainer,
    DockerExecutor,
    DockerJobRequest,
)
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

try:
    import fcntl

    FCNTL_AVAILABLE = True
except ModuleNotFoundError:
    fcntl = None
    FCNTL_AVAILABLE = False

DOCKER_JOB_DIRS = os.path.join(NEMORUN_HOME, ".docker_jobs.json")


class PersistentDockerScheduler(SchedulerMixin, DockerScheduler):  # type: ignore
    def __init__(self, session_name: str) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__(session_name)

    def _submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo[DockerJobRequest]:  # type: ignore
        assert isinstance(
            cfg, DockerExecutor
        ), f"{cfg.__class__} not supported for docker scheduler."
        executor = cfg

        if len(app.roles) > 1:
            assert len(app.roles) == len(executor.resource_group)

        values = executor.macro_values()

        if values:
            executor.env_vars = {
                key: values.substitute(arg) for key, arg in executor.env_vars.items()
            }
            for resource_req in executor.resource_group:
                resource_req.env_vars = {
                    key: values.substitute(arg) for key, arg in resource_req.env_vars.items()
                }

        containers = []
        for i, role in enumerate(app.roles):
            _current_executor = executor.resource_group[i] if executor.run_as_group else executor
            if values:
                role = values.apply(role)
            cmd = [role.entrypoint] + role.args
            containers.append(
                DockerContainer(
                    name=role.name, command=cmd, executor=_current_executor, extra_env=role.env
                )
            )

        basename = Path(executor.job_dir).name
        app_id = make_unique(basename)
        req = DockerJobRequest(id=app_id, executor=executor, containers=containers)
        return AppDryRunInfo(req, repr)

    def schedule(self, dryrun_info: AppDryRunInfo[DockerJobRequest]) -> str:  # type: ignore
        client = self._docker_client
        req = dryrun_info.request
        basename = Path(req.executor.job_dir).name
        req.executor.package(packager=req.executor.packager, job_name=basename)

        for container in req.containers:
            container.executor.volumes.append(f"{req.executor.job_dir}:/{RUNDIR_NAME}")

        req.run(client=client)
        _save_req(app_id=req.id, req=req)
        return req.id

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        req = _get_app_req(app_id=app_id)
        if not req:
            return DescribeAppResponse(
                app_id=app_id,
                state=AppState.UNKNOWN,
            )

        roles = {}
        roles_statuses = {}

        states = []
        for container in req.containers:
            role = container.name

            if role not in roles:
                roles[role] = Role(
                    name=role,
                    num_replicas=0,
                    image=container.executor.container_image,
                )
                roles_statuses[role] = RoleStatus(role, [])
            roles[role].num_replicas += 1

            c = container.get_container(client=self._docker_client, id=app_id)
            state = self._get_app_state(c) if c else AppState.SUCCEEDED

            roles_statuses[role].replicas.append(
                ReplicaStatus(
                    id=0,
                    role=role,
                    state=state,
                    hostname=container.name,
                )
            )
            states.append(state)

        state = AppState.UNKNOWN
        if all(is_terminal(state) for state in states):
            if all(state == AppState.SUCCEEDED for state in states):
                state = AppState.SUCCEEDED
            else:
                state = AppState.FAILED
        else:
            state = next(state for state in states if not is_terminal(state))

        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=state,
        )

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        req = _get_app_req(app_id=app_id)
        if not req:
            return [""]

        try:
            container = next(filter(lambda c: c.name == role_name, req.containers))
            if not container:
                return [""]

            c = container.get_container(client=self._docker_client, id=app_id)
            if not c:
                existing_log_files = glob.glob(
                    os.path.join(container.executor.job_dir, f"*{role_name}*.out")
                )
                if not existing_log_files:
                    return [""]

                log_file = existing_log_files[0]
                if not os.path.isfile(log_file):
                    raise RuntimeError(
                        f"app: {app_id} was not configured to log into a file."
                        f" Did you run it with log_dir set in Dict[str, CfgVal]?"
                    )
                iterator = LogIterator(app_id, log_file, self)
                # sometimes there's multiple lines per logged line
                iterator = split_lines_iterator(iterator)
                if regex:
                    iterator = filter_regex(regex, iterator)
                return iterator
        except StopIteration:
            return [""]

        logs = c.logs(
            since=since,
            until=until,
            stream=should_tail,
            stderr=streams != Stream.STDOUT,
            stdout=streams != Stream.STDERR,
        )  # type: ignore

        if isinstance(logs, (bytes, str)):
            logs = _to_str(logs)

            if len(logs) == 0:
                logs = []
            else:
                logs = split_lines(logs)

        logs = map(_to_str, logs)

        if regex:
            return filter_regex(regex, logs)
        else:
            return logs


def create_scheduler(
    session_name: str,
) -> PersistentDockerScheduler:
    return PersistentDockerScheduler(
        session_name=session_name,
    )


def _save_req(app_id: str, req: DockerJobRequest) -> None:
    apps = {}
    if not os.path.isfile(DOCKER_JOB_DIRS):
        Path(DOCKER_JOB_DIRS).touch()

    with open(DOCKER_JOB_DIRS, "r+") as f:
        if FCNTL_AVAILABLE:
            assert fcntl
            fcntl.flock(f, fcntl.LOCK_EX)

        try:
            try:
                apps = json.load(f)
            except Exception:
                apps = {}

            apps[app_id] = ZlibJSONSerializer().serialize(req.to_config())
            with tempfile.NamedTemporaryFile(mode="w+") as fp:
                json.dump(apps, fp)
                fp.flush()

                shutil.copy(fp.name, DOCKER_JOB_DIRS)
                fp.close()
        finally:
            if FCNTL_AVAILABLE:
                assert fcntl
                fcntl.flock(f, fcntl.LOCK_UN)


def _get_app_req(app_id: str) -> Optional[DockerJobRequest]:
    try:
        with open(DOCKER_JOB_DIRS, "r") as f:
            apps: dict[str, str] = json.load(f)
    except FileNotFoundError:
        return None

    for _id, req_str in apps.items():
        if _id == app_id:
            req: DockerJobRequest = fdl.build(ZlibJSONSerializer().deserialize(req_str))  # type: ignore
            return req

    return None
