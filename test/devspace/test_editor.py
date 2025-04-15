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
import pytest
from pathlib import Path
import shutil
from unittest.mock import MagicMock

from nemo_run.devspace.editor import find_editor_executable


class TestFindEditorExecutable:
    def test_unsupported_editor(self):
        """Test that unsupported editors raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            find_editor_executable("unsupported_editor")

    def test_editor_not_installed(self, monkeypatch):
        """Test that missing editors raise EnvironmentError."""
        # Monkeypatch shutil.which to return None (simulate editor not found)
        monkeypatch.setattr(shutil, "which", lambda x: None)

        with pytest.raises(EnvironmentError, match="is not installed"):
            find_editor_executable("code")

        with pytest.raises(EnvironmentError, match="is not installed"):
            find_editor_executable("cursor")

    def test_non_wsl_environment(self, tmp_path, monkeypatch):
        """Test editor detection in non-WSL environment using real file."""
        # Create a fake editor executable in a temp directory
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        code_exec = bin_dir / "code"
        code_exec.touch(mode=0o755)  # Make it executable

        cursor_exec = bin_dir / "cursor"
        cursor_exec.touch(mode=0o755)

        # Add our temp directory to PATH
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}:{old_path}"

        try:
            # Monkeypatch os.uname to return a non-WSL environment
            if hasattr(os, "uname"):  # Skip on Windows
                monkeypatch.setattr(os, "uname", lambda: MagicMock(release="Linux 5.10.0"))

            # Test with actual executables in path
            assert find_editor_executable("code") == "code"
            assert find_editor_executable("cursor") == "cursor"
        finally:
            # Restore PATH
            os.environ["PATH"] = old_path

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="WSL tests only relevant on Unix systems"
    )
    def test_wsl_environment(self, tmp_path, monkeypatch):
        """Test editor detection in WSL environment."""
        # Create directory structure with both Linux and "Windows" executables
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        # Linux executables
        code_exec = bin_dir / "code"
        code_exec.touch(mode=0o755)

        cursor_exec = bin_dir / "cursor"
        cursor_exec.touch(mode=0o755)

        # Windows .exe files at various levels
        exe_dir = tmp_path / "winbin"
        exe_dir.mkdir()
        code_exe = exe_dir / "Code.exe"
        code_exe.touch(mode=0o755)

        cursor_exe_dir = tmp_path / "apps" / "cursor"
        cursor_exe_dir.mkdir(parents=True)
        cursor_exe = cursor_exe_dir / "Cursor.exe"
        cursor_exe.touch(mode=0o755)

        # Add our temp directory to PATH
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}:{old_path}"

        try:
            # Mock WSL environment
            monkeypatch.setattr(os, "name", "posix")
            monkeypatch.setattr(os, "uname", lambda: MagicMock(release="Microsoft-WSL"))

            # Test cases with different configurations

            # 1. Case where we find the .exe file at a specific location
            with monkeypatch.context() as m:

                def mock_which(cmd):
                    if cmd == "code":
                        return str(code_exec)
                    elif cmd == "cursor":
                        return str(cursor_exec)
                    return None

                # Only need to mock exists for the specific paths we want to test
                original_exists = Path.exists

                def mock_exists(self):
                    if (
                        self == code_exe
                        or self.name == "Code.exe"
                        and str(self).startswith(str(tmp_path))
                    ):
                        return True
                    if (
                        self == cursor_exe
                        or self.name == "Cursor.exe"
                        and str(self).startswith(str(tmp_path))
                    ):
                        return True
                    return original_exists(self)

                m.setattr(shutil, "which", mock_which)
                m.setattr(Path, "exists", mock_exists)

                # Test code with .exe available
                result = find_editor_executable("code")
                assert "Code.exe" in result

                # Test cursor with .exe available
                result = find_editor_executable("cursor")
                assert "Cursor.exe" in result

            # 2. Case where we don't find the .exe file (should now raise error)
            with monkeypatch.context() as m:

                def mock_which(cmd):
                    if cmd == "code":
                        return str(code_exec)
                    elif cmd == "cursor":
                        return str(cursor_exec)
                    return None

                # Make exists always return False for .exe files
                def mock_exists(self):
                    if ".exe" in str(self).lower():
                        return False
                    return original_exists(self)

                m.setattr(shutil, "which", mock_which)
                m.setattr(Path, "exists", mock_exists)

                # Test code with no .exe available (should now raise error)
                with pytest.raises(EnvironmentError, match="Running in WSL but couldn't find"):
                    find_editor_executable("code")

                # Test cursor with no .exe available (should now raise error)
                with pytest.raises(EnvironmentError, match="Running in WSL but couldn't find"):
                    find_editor_executable("cursor")

        finally:
            # Restore PATH
            os.environ["PATH"] = old_path
