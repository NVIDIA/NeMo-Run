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
import platform
import shutil
from pathlib import Path

from invoke.context import Context

from nemo_run.core.frontend.console.api import CONSOLE


def find_editor_executable(base_executable_name):
    """Find the proper executable path for an editor, especially in WSL environments.

    Args:
        base_executable_name (str): The base name of the executable (e.g., 'code', 'cursor')

    Returns:
        str: The path to the executable

    Raises:
        ValueError: If the editor is not supported
        EnvironmentError: If the editor is not installed or Windows executable not found in WSL
    """
    # Define supported editors
    SUPPORTED_EDITORS = {
        "code": {
            "display_name": "VS Code",
            "download_url": "https://code.visualstudio.com/",
            "exe_name": "Code.exe",
        },
        "cursor": {
            "display_name": "Cursor",
            "download_url": "https://www.cursor.com/",
            "exe_name": "Cursor.exe",
        },
        # Add new editors here
    }

    # Check if the editor is supported
    if base_executable_name not in SUPPORTED_EDITORS:
        supported_list = ", ".join(SUPPORTED_EDITORS.keys())
        raise ValueError(
            f"Editor '{base_executable_name}' is not supported. "
            f"Supported editors are: {supported_list}"
        )

    editor_config = SUPPORTED_EDITORS[base_executable_name]

    # Check if the editor is installed
    executable_path = shutil.which(base_executable_name)
    if not executable_path:
        raise EnvironmentError(
            f"{editor_config['display_name']} is not installed. "
            f"Please install it from {editor_config['download_url']}"
        )

    # Default editor command is the base executable
    editor_cmd = base_executable_name

    # If we're running in WSL, find the Windows executable
    if os.name == "posix" and "WSL" in os.uname().release:
        # Start from the executable directory
        current_path = Path(executable_path).parent
        exe_found = False

        # Walk up to 5 levels to find the Windows .exe
        for _ in range(5):
            potential_exe = current_path / editor_config["exe_name"]
            if potential_exe.exists():
                editor_cmd = potential_exe.as_posix().replace(" ", "\\ ")
                exe_found = True
                break
            # Move up one directory
            parent_path = current_path.parent
            if parent_path == current_path:  # Reached root
                break
            current_path = parent_path

        # Raise an error if we couldn't find the Windows executable in WSL
        if not exe_found:
            raise EnvironmentError(
                f"Running in WSL but couldn't find {editor_config['exe_name']} in the "
                f"directory structure. For proper WSL integration, ensure {editor_config['display_name']} "
                f"is installed in Windows and properly configured for WSL. "
                f"See the documentation for {editor_config['display_name']} WSL integration."
            )

    return editor_cmd


def launch_editor(tunnel: str, path: str):
    """Launch a code editor for the specified SSH tunnel.

    Args:
        tunnel (str): The name of the SSH tunnel.
        path (str): The path to open in the editor.

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

        # Find the proper executable
        editor_cmd = find_editor_executable(editor)

        # Execute the editor command
        cmd = f"{editor_cmd} --new-window --remote ssh-remote+tunnel.{tunnel} {path}"
        CONSOLE.print(cmd)

        if platform.system() == "Windows":
            local.run(f"set NEMO_EDITOR={editor} && {cmd}")
        else:
            local.run(f"NEMO_EDITOR={editor} {cmd}")
