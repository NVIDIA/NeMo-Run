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

import fiddle as fdl
import typer
from rich.console import Console

from nemo_run import devspace
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.core.tunnel import server

console = Console()


def sshserver(space_zlib: str, verbose: bool = False):
    space_cfg: fdl.Buildable = ZlibJSONSerializer().deserialize(space_zlib)
    space: devspace.DevSpace = fdl.build(space_cfg)

    server_dir = server.server_dir(space.executor.job_dir, space.name)
    server_dir.mkdir(parents=True, exist_ok=True)
    (server_dir / "devspace.zlib").write_text(space_zlib)

    console.print(f"[bold green]Tunnel location:[/bold green] {server_dir}")

    user = space.executor.tunnel.user
    hostname = space.executor.tunnel.host

    console.print("\n")
    console.rule("[bold green]Local connection", characters="*")
    console.print("To connect to the tunnel, run the following command on your local machine:")
    console.print("\n")
    console.print(f"nemorun devspace connect {user}@{hostname} {server_dir}")
    console.rule("[bold green]", characters="*")

    server.launch(server_dir, workspace_name=space.name, verbose=verbose)


def launch(space: devspace.DevSpace):
    space.__io__ = launch.__io__.space
    space.executor.__io__ = space.__io__.executor
    space.launch()


def connect(host: str, path: str):
    devspace.DevSpace.connect(host, path)


def create() -> typer.Typer:
    app = typer.Typer(pretty_exceptions_enable=False)

    from nemo_run.cli.api import Entrypoint

    Entrypoint(launch, "devspace", enable_executor=False).cli(app)
    app.command(
        "sshserver",
        context_settings={"allow_extra_args": False},
    )(sshserver)
    app.command(
        "connect",
        context_settings={"allow_extra_args": False},
    )(connect)

    return app
