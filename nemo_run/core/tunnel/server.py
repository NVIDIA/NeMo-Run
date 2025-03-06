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

import atexit
import json
import os
import signal
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from invoke.context import Context
from rich.console import Console

from nemo_run.core.tunnel import client


def server_dir(job_dir, name) -> Path:
    return Path(job_dir) / ".nemo_run" / ".tunnels" / name


def launch(
    path: Path,
    workspace_name: str,
    verbose: bool = False,
    hostname: Optional[str] = None,
):
    ctx = Context()
    console = Console()

    def signal_handler(sig, frame):
        console.print("\n[bold red]SSH Tunnel interrupted by user.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    user = os.environ["USER"]
    hostname = hostname or socket.gethostname()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    metadata = TunnelMetadata(
        user=user, workspace_name=workspace_name, hostname=hostname, port=port
    )
    metadata.save(path)
    atexit.register(client.delete_tunnel_dir, path)
    ctx.run('echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config.d/custom.conf')

    if verbose:
        ctx.run('echo "LogLevel DEBUG3" >> /etc/ssh/sshd_config.d/custom.conf')

    ctx.run(f"/usr/sbin/sshd -D -p {port}", pty=True, hide=not verbose)


@dataclass
class TunnelMetadata:
    user: str
    hostname: str
    port: int
    workspace_name: str

    def save(self, path: Path):
        """Save the metadata to a JSON file.

        Args:
            path (Path): The directory path where the metadata file will be saved.

        Example:
            metadata = TunnelMetadata(user="user", hostname="host", port=22)
            metadata.save(Path("/path/to/dir"))
        """
        tunnel_file = path / "metadata.json"
        with tunnel_file.open("w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def restore(cls, path: Path, tunnel=None) -> "TunnelMetadata":
        """Restore the metadata from a JSON file.

        Args:
            path (Path): The directory path where the metadata file is located.

        Returns:
            TunnelMetadata: The restored TunnelMetadata object.

        Example:
            metadata = TunnelMetadata.restore(Path("/path/to/dir"))
        """
        tunnel_file = path / "metadata.json"

        if tunnel:
            data = json.loads(tunnel.run(f"cat {tunnel_file}", hide="out").stdout.strip())
        else:
            with tunnel_file.open("r") as f:
                data = json.load(f)

        return cls(**data)
