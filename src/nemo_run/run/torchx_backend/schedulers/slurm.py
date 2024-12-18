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
This contains the Nemo Slurm scheduler which can be used to run TorchX
components on a Nemo compatible Slurm cluster.
"""

import csv
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    ListAppResponse,
    Stream,
    filter_regex,
    split_lines_iterator,
)
from torchx.schedulers.local_scheduler import LogIterator
from torchx.schedulers.slurm_scheduler import (
    SLURM_STATES,
    SlurmScheduler,
)
from torchx.specs import (
    AppDef,
    AppState,
    ReplicaStatus,
    Role,
    RoleStatus,
)
from torchx.specs.api import is_terminal

from nemo_run.config import NEMORUN_HOME, from_dict
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.slurm import SlurmBatchRequest, SlurmExecutor, SlurmJobDetails
from nemo_run.core.tunnel.client import LocalTunnel, SSHTunnel, Tunnel
from nemo_run.run import experiment as run_experiment
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

log: logging.Logger = logging.getLogger(__name__)
SLURM_JOB_DIRS = os.path.join(NEMORUN_HOME, ".slurm_jobs")


class SlurmTunnelScheduler(SchedulerMixin, SlurmScheduler):  # type: ignore
    def __init__(self, session_name: str) -> None:
        self.tunnel: Optional[Tunnel] = None
        super().__init__(session_name)

    # TODO: Move this into the SlurmExecutor
    def _initialize_tunnel(self, tunnel: SSHTunnel | LocalTunnel):
        if self.tunnel == tunnel:
            return

        experiment = run_experiment._current_experiment.get(None)
        if experiment and tunnel.key in experiment.tunnels:
            self.tunnel = experiment.tunnels[tunnel.key]
            return

        self.tunnel = tunnel

        if experiment:
            experiment.tunnels[tunnel.key] = self.tunnel

    def _submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo[Any]:  # type: ignore
        assert isinstance(cfg, SlurmExecutor), f"{cfg.__class__} not supported for slurm scheduler."
        executor = cfg

        # Reuse tunnel if already exists
        self._initialize_tunnel(executor.tunnel)
        assert self.tunnel
        executor.tunnel = self.tunnel  # type: ignore

        job_dir = executor.job_dir
        assert isinstance(job_dir, str), "job_dir must be str"

        partition = executor.partition
        assert partition is None or isinstance(partition, str), "partition must be str"

        executor.package(packager=executor.packager, job_name=Path(job_dir).name)

        srun_cmds: list[list[str]] = []
        jobs = []
        envs = {}
        values = executor.macro_values()

        if values:
            executor.env_vars = {
                key: values.substitute(arg) for key, arg in executor.env_vars.items()
            }
            for resource_req in executor.resource_group:
                resource_req.env_vars = {
                    key: values.substitute(arg) for key, arg in resource_req.env_vars.items()
                }

        for role in app.roles:
            if values:
                role = values.apply(role)
            srun_cmd = [role.entrypoint] + role.args
            srun_cmds.append([" ".join(srun_cmd)])
            jobs.append(role.name)
            envs |= role.env

        cmd = ["sbatch", "--requeue", "--parsable"]
        req = SlurmBatchRequest(
            cmd=cmd,
            jobs=jobs,
            command_groups=srun_cmds,
            slurm_config=executor,
            max_retries=min(role.max_retries for role in app.roles),
            extra_env=envs,
            launcher=executor.get_launcher(),
        )

        # Write and copy sbatch script
        sbatch_dir = executor.experiment_dir
        path = os.path.join(sbatch_dir, f"{executor.job_name}_sbatch.sh")
        script = req.materialize()

        with open(path, "w") as f:
            f.write(script)

        return AppDryRunInfo(req, repr)

    def schedule(self, dryrun_info: AppDryRunInfo[SlurmBatchRequest]) -> str:  # type: ignore
        # Setup
        req = dryrun_info.request
        slurm_executor = dryrun_info.request.slurm_config
        assert slurm_executor.experiment_id, "Executor not assigned to experiment."

        job_dir = slurm_executor.job_dir
        tunnel = slurm_executor.tunnel
        assert tunnel, f"Tunnel required for {self.__class__}"
        assert job_dir, f"Need to provide job_dir for {self.__class__}"

        self._initialize_tunnel(tunnel)
        assert self.tunnel, f"Cannot initialize tunnel {tunnel}"

        dst_path = os.path.join(self.tunnel.job_dir, f"{slurm_executor.job_name}_sbatch.sh")

        if slurm_executor.dependencies:
            cmd = ["sbatch", "--requeue", "--parsable"]
            slurm_deps = slurm_executor.parse_deps()
            cmd.append(f"--dependency={slurm_executor.dependency_type}:{':'.join(slurm_deps)}")
            req.cmd = cmd

        # Run sbatch script
        req.cmd += [dst_path]
        job_id = self.tunnel.run(" ".join(req.cmd)).stdout.strip()

        # Save metadata
        _save_job_dir(job_id, job_dir, tunnel, slurm_executor.job_details.ls_term)
        return job_id

    def _cancel_existing(self, app_id: str) -> None:
        job_dirs = _get_job_dirs()
        if app_id in job_dirs:
            _, tunnel_cfg, _ = job_dirs[app_id]
            self._initialize_tunnel(tunnel_cfg)
        else:
            return None

        assert self.tunnel, "Tunnel is None."
        self.tunnel.run(f"scancel {app_id}", hide=False)

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        job_dirs = _get_job_dirs()
        if app_id in job_dirs:
            _, tunnel_cfg, _ = job_dirs[app_id]
            self._initialize_tunnel(tunnel_cfg)
        else:
            return None

        assert self.tunnel, "Tunnel is None."
        p = self.tunnel.run(
            f"sacct --parsable2 -j {app_id}",
        )
        output = p.stdout.strip().split("\n")

        if len(output) <= 1:
            return None

        reader = csv.DictReader(output, delimiter="|")

        roles = {}
        roles_statuses = {}
        msg = ""
        app_state = AppState.UNKNOWN
        for i, row in enumerate(reader):
            job_id, *parts = row["JobID"].split("+")
            if job_id != app_id:
                continue
            if len(parts) > 0 and "." in parts[0]:
                # we only care about the worker not the child jobs
                continue

            state = row["State"]
            msg = state
            state_enum = None
            for slurm_state, app_state_enum in SLURM_STATES.items():
                if state.startswith(slurm_state):
                    state_enum = app_state_enum
                    break

            assert state_enum, f"failed to translate slurm state {state} to torchx state"
            app_state = state_enum

            _, _, role = row["JobName"].rpartition(".")
            if not role:
                # name should always have at least 3 parts but sometimes sacct
                # is slow to update
                continue
            if role not in roles:
                roles[role] = Role(name=role, num_replicas=0, image="")
                roles_statuses[role] = RoleStatus(role, [])
            roles[role].num_replicas += 1
            roles_statuses[role].replicas.append(
                ReplicaStatus(id=i, role=role, state=app_state, hostname=""),
            )

        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=app_state,
            msg=msg,
        )

    def list(self) -> list[ListAppResponse]:
        # By default sacct only returns accounting information of jobs launched on the current day
        # To return all jobs launched, set starttime to one second past unix epoch time
        # Starttime will be modified when listing jobs by timeframe is supported
        assert self.tunnel, "Tunnel is None."
        p = self.tunnel.run("sacct --json -S1970-01-01-00:00:01")
        output_json = json.loads(p.stdout.strip())
        return [
            ListAppResponse(app_id=str(job["job_id"]), state=SLURM_STATES[job["state"]["current"]])
            for job in output_json["jobs"]
        ]

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
        if since or until:
            log.warning(
                f"since and/or until times specified for {self.__class__}.log_iter."
                " These will be ignored and all log lines will be returned"
            )

        job_dirs = _get_job_dirs()
        if app_id in job_dirs:
            local_dir, tunnel_cfg, ls_term = job_dirs[app_id]
            self._initialize_tunnel(tunnel_cfg)

            local_paths = SlurmJobDetails(folder=local_dir, job_name=role_name)
            local_file = str(local_paths.stdout).replace("%j", app_id)

            iterator = TunnelLogIterator(
                app_id,
                local_file,
                os.path.join(tunnel_cfg.job_dir, Path(local_dir).name),
                self,
                should_tail=should_tail,
                role_name=role_name,
                ls_term=ls_term,
            )
            # sometimes there's multiple lines per logged line
            iterator = split_lines_iterator(iterator)
            if regex:
                iterator = filter_regex(regex, iterator)
            return iterator
        else:
            return [f"Failed getting logs for {app_id}"]

    def close(self) -> None: ...


class TunnelLogIterator(LogIterator):
    def __init__(
        self,
        app_id: str,
        local_log_file: str,
        remote_dir: str,
        scheduler: SlurmTunnelScheduler,
        should_tail: bool = True,
        role_name: Optional[str] = None,
        is_local: bool = False,
        ls_term: Optional[str] = None,
    ) -> None:
        self._app_id: str = app_id
        self._log_file: str = local_log_file
        self._remote_dir: str = remote_dir
        self._log_fp: Optional[TextIO] = None
        self._scheduler: SlurmTunnelScheduler = scheduler  # type: ignore
        self._app_finished: bool = not should_tail
        self._role_name = role_name
        self._is_local = is_local
        self._ls_term = ls_term

    def _check_finished(self) -> None:
        # either the app (already finished) was evicted from the LRU cache
        # -- or -- the app reached a terminal state (and still in the cache)
        desc = self._scheduler.describe(self._app_id)
        if not desc or is_terminal(desc.state):
            self._app_finished = True
        else:
            self._app_finished = False

        if self._scheduler.tunnel:
            try:
                for _ in range(5):
                    extension = os.path.splitext(self._log_file)[1]
                    ls_term = (
                        self._ls_term
                        if self._ls_term
                        else os.path.join(self._remote_dir, f"log*{extension}")
                    )
                    ls_output = self._scheduler.tunnel.run(
                        f"ls -1 {ls_term} 2> /dev/null",
                        warn=True,
                    ).stdout.strip()
                    remote_log_files = ls_output.split("\n") if ls_output else []
                    if len(remote_log_files) >= 1:
                        if self._is_local:
                            self._log_file = remote_log_files[0]
                        else:
                            self._scheduler.tunnel.get(remote_log_files[0], self._log_file)
                        break
                    time.sleep(1)
            except Exception as e:
                log.warning(
                    f"Failed fetching logs from remote (will display logs from previous fetch): {e}"
                )


def create_scheduler(session_name: str, **kwargs: Any) -> SlurmTunnelScheduler:
    return SlurmTunnelScheduler(
        session_name=session_name,
    )


def _save_job_dir(
    job_id: str, local_job_dir: str, tunnel: SSHTunnel | LocalTunnel, ls_term: str
) -> None:
    with open(SLURM_JOB_DIRS, "a+") as f:
        f.write(
            f"{job_id} = {ls_term},{local_job_dir},{tunnel.__class__.__name__},{json.dumps(asdict(tunnel))}\n"
        )


def _get_job_dirs() -> dict[str, tuple[str, SSHTunnel | LocalTunnel, str]]:
    try:
        with open(SLURM_JOB_DIRS, "rt") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {}

    out = {}
    for line in lines:
        first, _, second = line.partition("=")
        if not first or not second:
            continue
        value = second.strip().split(",", maxsplit=3)
        if len(value) not in [3, 4]:
            continue

        if len(value) == 4:
            ls_term = value.pop(0)
        else:
            ls_term = ""

        local_dir = value[0]
        tunnel_cls = SSHTunnel if value[1] == SSHTunnel.__name__ else LocalTunnel
        try:
            tunnel = from_dict(json.loads(value[2]), tunnel_cls)
        except Exception:
            continue

        out[first.strip()] = (local_dir, tunnel, ls_term)
    return out
