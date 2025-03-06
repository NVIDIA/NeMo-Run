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
import sys
import threading
import time
from queue import Queue
from typing import Optional, TextIO

from torchx.cli.cmd_log import _prefix_line, find_role_replicas, validate
from torchx.cli.colors import ENDC, GREEN
from torchx.schedulers.api import Stream
from torchx.specs.api import is_started, parse_app_handle
from torchx.specs.builders import make_app_handle
from torchx.util.types import none_throws

from nemo_run.core.execution.base import LogSupportedExecutor
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.run.torchx_backend.runner import Runner, get_runner
from nemo_run.run.torchx_backend.schedulers.api import (
    REVERSE_EXECUTOR_MAPPING,
)

logger: logging.Logger = logging.getLogger(__name__)


def print_log_lines(
    file: TextIO,
    runner: Runner,
    app_handle: str,
    role_name: str,
    replica_id: int,
    regex: str,
    should_tail: bool,
    exceptions: "Queue[Exception]",
    streams: Optional[Stream],
    log_path: Optional[str] = None,
) -> None:
    try:
        scheduler_backend, _, app_id = parse_app_handle(app_handle=app_handle)
        executor_cls = REVERSE_EXECUTOR_MAPPING[scheduler_backend]
        if issubclass(executor_cls, LogSupportedExecutor):
            executor_cls.logs(app_id, fallback_path=log_path)
        else:
            for line in runner.log_lines(
                app_handle,
                role_name,
                replica_id,
                regex,
                should_tail=should_tail,
                streams=streams,
            ):
                prefix = f"{GREEN}{role_name[-10:]}/{replica_id}{ENDC} "
                print(_prefix_line(prefix, line), file=file, end="", flush=True)
                file.flush()
    except Exception as e:
        exceptions.put(e)
        raise


# This is a patched version of torchx get_logs to avoid infinite loop when waiting for app state response
def get_logs(
    file: TextIO,
    identifier: str,
    regex: Optional[str],
    should_tail: bool = False,
    runner: Optional[Runner] = None,
    streams: Optional[Stream] = None,
    wait_timeout: int = 10,
) -> None:
    validate(identifier)
    scheduler_backend, _, path_str = identifier.partition("://")

    # path is of the form ["", "app_id", "master", "0"]
    path = path_str.split("/")
    session_name = path[0] or "default"
    app_id = path[1]
    role_name = path[2] if len(path) > 2 else None

    if not runner:
        runner = get_runner()

    app_handle = make_app_handle(scheduler_backend, session_name, app_id)

    display_waiting = True
    tries = 0
    while True:
        status = runner.status(app_handle)
        if status and is_started(status.state):
            break
        elif display_waiting:
            CONSOLE.log("Waiting for app state response before fetching logs...")
            display_waiting = False
        tries += 1
        if tries >= wait_timeout:
            break
        time.sleep(1)

    app = none_throws(
        runner.describe(app_handle),
        f"Unable to find {app_handle}. Please see the FAQ on Github or file an issue if the error persists.",
    )
    # print all replicas for the role
    replica_ids = find_role_replicas(app, role_name)

    if not replica_ids:
        valid_ids = "\n".join(
            [
                f"  {idx}: {scheduler_backend}://{app_id}/{role.name}"
                for idx, role in enumerate(app.roles)
            ]
        )

        CONSOLE.log(
            f"[bold red]No role [{role_name}] found for app: {app.name}."
            f" Did you mean one of the following:\n{valid_ids}",
        )
        sys.exit(1)

    exceptions = Queue()
    threads = []
    for role_name, replica_id in replica_ids:
        thread = threading.Thread(
            target=print_log_lines,
            args=(
                file,
                runner,
                app_handle,
                role_name,
                replica_id,
                regex,
                should_tail,
                exceptions,
                streams,
            ),
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Retrieve all exceptions, print all except one and raise the first recorded exception
    threads_exceptions = []
    while not exceptions.empty():
        threads_exceptions.append(exceptions.get())

    if len(threads_exceptions) > 0:
        for i in range(1, len(threads_exceptions)):
            logger.error(threads_exceptions[i])

        raise threads_exceptions[0]
