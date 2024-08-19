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

import json
import os
import pprint
import shutil
import tempfile
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    Stream,
    filter_regex,
    split_lines_iterator,
)
from torchx.schedulers.local_scheduler import (
    COMBINED_LOG,
    STDERR_LOG,
    STDOUT_LOG,
    AppId,
    CWDImageProvider,
    ImageProvider,
    LocalOpts,
    LocalScheduler,
    LogIterator,
    PopenRequest,
    _LocalAppDef,
)
from torchx.specs.api import AppDef, AppState, Role

from nemo_run.config import NEMORUN_HOME
from nemo_run.core.execution.base import Executor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.run.torchx_backend.schedulers.api import SchedulerMixin

try:
    import fcntl

    FCNTL_AVAILABLE = True
except ModuleNotFoundError:
    fcntl = None
    FCNTL_AVAILABLE = False

LOCAL_JOB_DIRS = os.path.join(NEMORUN_HOME, ".local_jobs.json")


class PersistentLocalScheduler(SchedulerMixin, LocalScheduler):  # type: ignore
    def __init__(
        self,
        session_name: str,
        image_provider_class: Callable[[LocalOpts], ImageProvider],
        cache_size: int = 100,
        extra_paths: Optional[list[str]] = None,
    ) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        self.backend = "local"
        self.session_name = session_name

        self._apps: dict[AppId, _LocalAppDef] = {}
        self._image_provider_class = image_provider_class

        if cache_size <= 0:
            raise ValueError("cache size must be greater than zero")
        self._cache_size = cache_size

        self._extra_paths: list[str] = extra_paths or []

        # sets lazily on submit or dryrun based on log_dir cfg
        self._base_log_dir: Optional[str] = None
        self._created_tmp_log_dir: bool = False

    def _submit_dryrun(self, app: AppDef, cfg: Executor) -> AppDryRunInfo[PopenRequest]:  # type: ignore
        assert isinstance(cfg, LocalExecutor), f"{cfg.__class__} not supported for local scheduler."
        # Hack for inline scripts
        for role in app.roles:
            if len(role.args) == 2 and role.args[0] == "-c":
                role.args[1] = role.args[1][1:-1]

        cfg_dict = asdict(cfg)
        cfg_dict["log_dir"] = cfg_dict.pop("job_dir")
        request = self._to_popen_request(app, cfg_dict)  # type: ignore
        return AppDryRunInfo(request, lambda p: pprint.pformat(asdict(p), indent=2, width=80))

    def schedule(self, dryrun_info: AppDryRunInfo[PopenRequest]) -> str:
        app_id = super().schedule(dryrun_info=dryrun_info)
        _save_job_dir(self._apps)
        return app_id

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        resp = super().describe(app_id=app_id)

        if resp:
            _save_job_dir(self._apps)
            return resp

        saved_apps = _get_job_dirs()
        if (app_id not in self._apps) and (app_id in saved_apps):
            resp = DescribeAppResponse()
            resp.app_id = app_id
            resp.roles = [
                Role(name=name, image="") for name in saved_apps[app_id].role_replicas.keys()
            ]
            resp.state = saved_apps[app_id].state
            resp.num_restarts = 0
            resp.ui_url = f"file://{saved_apps[app_id].log_dir}"
            return resp

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
            warnings.warn(
                "Since and/or until times specified for LocalScheduler.log_iter."
                " These will be ignored and all log lines will be returned"
            )

        if app_id in self._apps:
            app = self._apps[app_id]
        else:
            saved_apps = _get_job_dirs()
            app = saved_apps[app_id]

        STREAM_FILES = {
            None: COMBINED_LOG,
            Stream.COMBINED: COMBINED_LOG,
            Stream.STDOUT: STDOUT_LOG,
            Stream.STDERR: STDERR_LOG,
        }
        log_file = os.path.join(app.log_dir, role_name, str(k), STREAM_FILES[streams])

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


def create_scheduler(
    session_name: str,
    cache_size: int = 100,
    extra_paths: Optional[list[str]] = None,
    **kwargs: Any,
) -> PersistentLocalScheduler:
    return PersistentLocalScheduler(
        session_name=session_name,
        image_provider_class=CWDImageProvider,
        cache_size=cache_size,
        extra_paths=extra_paths,
    )


def _save_job_dir(apps: dict[str, _LocalAppDef]) -> None:
    original_apps = {}
    if not os.path.isfile(LOCAL_JOB_DIRS):
        Path(LOCAL_JOB_DIRS).touch()

    with open(LOCAL_JOB_DIRS, "r+") as f:
        if FCNTL_AVAILABLE:
            assert fcntl
            fcntl.flock(f, fcntl.LOCK_EX)

        try:
            try:
                original_apps = json.load(f)
            except Exception:
                original_apps = {}

            new_apps = {
                app_id: (
                    app_def.state.name,
                    app_def.id,
                    app_def.log_dir,
                    list(app_def.role_replicas.keys()),
                )
                for app_id, app_def in apps.items()
            }
            final_apps = original_apps | new_apps

            with tempfile.NamedTemporaryFile(mode="w+") as fp:
                json.dump(final_apps, fp)
                fp.flush()

                shutil.copy(fp.name, LOCAL_JOB_DIRS)
                fp.close()
        finally:
            if FCNTL_AVAILABLE:
                assert fcntl
                fcntl.flock(f, fcntl.LOCK_UN)


def _get_job_dirs() -> dict[str, _LocalAppDef]:
    try:
        with open(LOCAL_JOB_DIRS, "r") as f:
            apps: dict[str, tuple[str, str, str, list[str]]] = json.load(f)
    except FileNotFoundError:
        return {}

    out = {}
    for app_id, app_details in apps.items():
        if len(app_details) != 4:
            continue

        state_name = app_details[0]
        state = AppState[state_name]
        app_id = app_details[1]
        log_dir = app_details[2]
        role_names = app_details[3]
        app_def = _LocalAppDef(id=app_id, log_dir=log_dir)
        for role in role_names:
            app_def.role_replicas[role] = []
        app_def.set_state(state)

        out[app_id] = app_def
    return out
