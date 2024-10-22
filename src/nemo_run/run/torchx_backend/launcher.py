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

import contextvars
import logging
import sys
import threading
import time
from typing import Literal, Optional, overload

import torchx.specs as specs
from torchx.schedulers.api import Stream
from torchx.specs import AppDef, AppHandle
from torchx.specs.api import parse_app_handle

from nemo_run.core.execution.base import Executor
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.exceptions import UnknownStatusError
from nemo_run.run.logs import get_logs
from nemo_run.run.torchx_backend.runner import Runner, get_runner

logger: logging.Logger = logging.getLogger(__name__)


@overload
def launch(
    executable: AppDef,
    executor_name: str,
    executor: Executor,
    dryrun: Literal[True],
    wait: bool = False,
    log: bool = False,
    parent_run_id: Optional[str] = None,
    runner: Runner | None = None,
    log_dryrun: bool = ...,
) -> tuple[None, None]: ...


@overload
def launch(
    executable: AppDef,
    executor_name: str,
    executor: Executor,
    dryrun: Literal[False],
    wait: bool = False,
    log: bool = False,
    parent_run_id: Optional[str] = None,
    runner: Runner | None = None,
    log_dryrun: bool = ...,
) -> tuple[str, specs.AppStatus]: ...


@overload
def launch(
    executable: AppDef,
    executor_name: str,
    executor: Executor,
    dryrun: bool = False,
    wait: bool = False,
    log: bool = False,
    parent_run_id: Optional[str] = None,
    runner: Runner | None = None,
    log_dryrun: bool = False,
) -> tuple[str | None, specs.AppStatus | None]: ...


def launch(
    executable: AppDef,
    executor_name: str,
    executor: Executor,
    dryrun: bool = False,
    wait: bool = False,
    log: bool = False,
    parent_run_id: Optional[str] = None,
    runner: Runner | None = None,
    log_dryrun: bool = False,
) -> tuple[str | None, specs.AppStatus | None]:
    runner = runner or get_runner()

    if dryrun:
        dryrun_info = runner.dryrun(
            executable,
            executor_name,
            cfg=executor,
            parent_run_id=parent_run_id,
        )
        if log_dryrun:
            CONSOLE.log("\n=== APPLICATION ===\n")
            CONSOLE.log(dryrun_info)

        return None, None
    else:
        app_handle = runner.run(
            executable,
            executor_name,
            cfg=executor,  # type: ignore
            parent_run_id=parent_run_id,
        )
        logger.info(f"Launched app: {app_handle}")
        app_status = runner.status(app_handle)
        if app_status:
            logger.info(app_status.format())
        if wait:
            app_status = wait_and_exit(runner=runner, app_handle=app_handle, log=log)

        return app_handle, app_status


def wait_and_exit(
    *,
    app_handle: AppHandle,
    log: bool,
    runner: Runner | None = None,
    timeout: int = 10,
) -> specs.AppStatus:
    if runner is None:
        runner = get_runner()

    _, _, app_id = parse_app_handle(app_handle=app_handle)
    logger.info(f"Waiting for job {app_id} to finish [log={log}]...")

    log_thread = None
    if log:
        log_thread = ContextThread(
            target=get_logs,
            kwargs={
                "file": sys.stdout,
                "runner": runner,
                "identifier": app_handle,
                "regex": None,
                "should_tail": True,
                "streams": Stream.COMBINED,
            },
        )
        log_thread.daemon = True
        log_thread.start()

    tries = 0
    status = None
    while tries < timeout:
        status = runner.wait(app_handle, wait_interval=2)
        if status:
            break
        tries += 1
        time.sleep(1)

    if not status:
        raise UnknownStatusError(f"unknown status, wait returned {status}")

    logger.info(f"Job {app_id} finished: {status.state}")

    return status


class ContextThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        self.ctx = contextvars.copy_context()
        super().__init__(*args, **kwargs)

    def run(self):
        self.ctx.run(super().run)
