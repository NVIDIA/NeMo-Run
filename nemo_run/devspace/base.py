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

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import fiddle as fdl

from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.core.tunnel.client import Callback

if TYPE_CHECKING:
    from nemo_run.core.execution.base import Executor


class DevSpace:
    def __init__(
        self,
        name: str,
        executor: "Executor",
        cmd: str = "launch_devspace",
        use_packager: bool = False,
        env_vars: Optional[Dict[str, str]] = None,
        add_workspace_to_pythonpath: bool = True,
    ):
        self.name = name
        self.executor = executor
        self.cmd = cmd
        self.use_packager = use_packager
        self.env_vars = env_vars
        self.add_workspace_to_pythonpath = add_workspace_to_pythonpath

    @classmethod
    def connect(cls, host: str, path: str):
        from nemo_run.core.tunnel.client import SSHTunnel

        user, hostname = host.split("@")

        tunnel = SSHTunnel(host=hostname, user=user, job_dir=path)
        tunnel.connect()

        try:
            space_zlib = tunnel.run(f"cat {path}/devspace.zlib").stdout.strip()
        except Exception as e:
            raise ValueError(f"Could not find the devspace at {host}:{path}. {e}")

        space_cfg: fdl.Buildable = ZlibJSONSerializer().deserialize(space_zlib)
        space: DevSpace = fdl.build(space_cfg)
        space.__io__ = space_cfg
        space.executor.tunnel = tunnel

        executor_callback = space.executor.connect_devspace(space, tunnel_dir=Path(path))

        tunnel.keep_alive(executor_callback)

    def launch(self):
        if self.use_packager:
            self.executor.packager.setup()
        self.execute_cmd()

    def execute_cmd(self):
        if hasattr(self.executor, self.cmd):
            if self.cmd == "launch_devspace":
                executor_callback = self.executor.launch_devspace(
                    self,
                    env_vars=self.env_vars,
                    add_workspace_to_pythonpath=self.add_workspace_to_pythonpath,
                )

                if isinstance(executor_callback, Callback):
                    self.executor.tunnel.keep_alive(executor_callback)
            else:
                getattr(self.executor, self.cmd)()
