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

import logging
from typing import Any, Optional

from torchx.runner.api import Runner as TorchXRunner
from torchx.runner.events import log_event
from torchx.schedulers import get_scheduler_factories
from torchx.specs import (
    AppDef,
    AppDryRunInfo,
    AppHandle,
    make_app_handle,
    parse_app_handle,
)
from torchx.util.types import none_throws

from nemo_run.core.execution.base import Executor

logger: logging.Logger = logging.getLogger(__name__)


class Runner(TorchXRunner):
    def dryrun(  # type: ignore
        self,
        app: AppDef,
        scheduler: str,
        cfg: Optional[Executor] = None,
        workspace: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> AppDryRunInfo:
        # input validation
        if not app.roles:
            raise ValueError(
                f"No roles for app: {app.name}. Did you forget to add roles to AppDef?"
            )

        for role in app.roles:
            if not role.entrypoint:
                raise ValueError(
                    f"No entrypoint for role: {role.name}."
                    f" Did you forget to call role.runs(entrypoint, args, env)?"
                )
            if role.num_replicas <= 0:
                raise ValueError(
                    f"Non-positive replicas for role: {role.name}."
                    f" Did you forget to set role.num_replicas?"
                )

        with log_event("dryrun", scheduler):
            sched = self._scheduler(scheduler)

            sched._validate(app, scheduler)
            dryrun_info = sched.submit_dryrun(app, cfg)
            dryrun_info._scheduler = scheduler
            return dryrun_info

    def run(  # type: ignore
        self,
        app: AppDef,
        scheduler: str,
        cfg: Optional[Executor] = None,
        workspace: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> AppHandle:
        with log_event(api="run", workspace=workspace) as ctx:
            dryrun_info = self.dryrun(
                app,
                scheduler,
                cfg=cfg,
                workspace=workspace,
                parent_run_id=parent_run_id,
            )
            handle = self.schedule(dryrun_info)
            ctx._torchx_event.scheduler = none_throws(dryrun_info._scheduler)
            ctx._torchx_event.app_image = none_throws(dryrun_info._app).roles[0].image
            ctx._torchx_event.app_id = parse_app_handle(handle)[2]
            return handle

    def schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        scheduler = none_throws(dryrun_info._scheduler)
        app_image = none_throws(dryrun_info._app).roles[0].image
        with log_event(
            "schedule",
            scheduler,
            app_image=app_image,
        ) as ctx:
            sched = self._scheduler(scheduler)
            app_id = sched.schedule(dryrun_info)
            app_handle = make_app_handle(scheduler, self._name, app_id)
            app = none_throws(dryrun_info._app)
            self._apps[app_handle] = app
            _, _, app_id = parse_app_handle(app_handle)
            ctx._torchx_event.app_id = app_id
            return app_handle


def get_runner(
    component_defaults: Optional[dict[str, dict[str, str]]] = None,
    **scheduler_params: Any,
) -> Runner:
    """
    Convenience method to construct and get a Runner object. Usage:

    .. code-block:: python

      with get_runner() as runner:
        app_handle = runner.run(component(args), scheduler="kubernetes", runcfg)
        print(runner.status(app_handle))

    Alternatively,

    .. code-block:: python

     runner = get_runner()
     try:
        app_handle = runner.run(component(args), scheduler="kubernetes", runcfg)
        print(runner.status(app_handle))
     finally:
        runner.close()

    Args:
        name: human readable name that will be included as part of all launched
            jobs.
        scheduler_params: extra arguments that will be passed to the constructor
            of all available schedulers.


    """
    name = "nemo_run"

    scheduler_factories = get_scheduler_factories()
    return Runner(name, scheduler_factories, component_defaults, scheduler_params=scheduler_params)
