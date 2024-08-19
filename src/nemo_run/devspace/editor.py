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

import os
import shutil
from pathlib import Path

from invoke.context import Context

from nemo_run.core.frontend.console.api import CONSOLE


def launch_editor(tunnel: str, path: str):
    """Launch a code editor for the specified SSH tunnel.

    Args:
        tunnel (str): The name of the SSH tunnel.

    Raises:
        EnvironmentError: If the specified editor is not installed.
    """
    local = Context()

    from InquirerPy import inquirer

    editor = inquirer.select(  # type: ignore
        message="Which code editor would you like to launch?",
        choices=["code", "cursor", "none"],
    ).execute()

    if editor != "none":
        CONSOLE.rule(f"[bold green]Launching {editor}", characters="*")
        if editor == "code":
            if not shutil.which("code"):
                raise EnvironmentError(
                    "VS Code is not installed. Please install it from https://code.visualstudio.com/"
                )

            code_cli = "code"

            # If we're running in WSL. Launch code from the executable directly.
            # This avoids the code launch script activating the WSL remote extension
            # which enables us to specify the ssh tunnel as the remote
            if os.name == "posix" and "WSL" in os.uname().release:
                code_cli = (
                    (Path(shutil.which("code")).parent.parent / "Code.exe")
                    .as_posix()
                    .replace(" ", "\\ ")
                )

            cmd = f"{code_cli} --new-window --remote ssh-remote+tunnel.{tunnel} {path}"
            CONSOLE.print(cmd)
            local.run(f"NEMO_EDITOR=vscode {cmd}")
        elif editor == "cursor":
            local.run(f"NEMO_EDITOR=cursor cursor --remote ssh-remote+tunnel.{tunnel} {path}")
