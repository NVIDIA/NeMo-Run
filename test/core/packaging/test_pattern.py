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

import filecmp
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from nemo_run.core.packaging.pattern import PatternPackager
from test.conftest import MockContext


@patch("nemo_run.core.packaging.pattern.Context", MockContext)
def test_package_with_include_pattern_rel_path(tmpdir):
    # Create extra files in a separate directory
    (tmpdir / "extra").mkdir()
    with open(tmpdir / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(tmpdir / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    packager = PatternPackager(include_pattern=str(tmpdir / "extra/*"), relative_path=str(tmpdir))
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(tmpdir), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(tmpdir, "extra"),
            os.path.join(job_dir, "extracted_output", "extra"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.pattern.Context", MockContext)
def test_package_with_multi_include_pattern_rel_path(tmpdir):
    # Create extra files in a separate directory
    (tmpdir / "extra").mkdir()
    with open(tmpdir / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(tmpdir / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    include_pattern = [str(tmpdir / "extra/extra_file1.txt"), str(tmpdir / "extra/extra_file2.txt")]
    relative_path = [str(tmpdir), str(tmpdir)]

    packager = PatternPackager(include_pattern=include_pattern, relative_path=relative_path)
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(tmpdir), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(tmpdir, "extra"),
            os.path.join(job_dir, "extracted_output", "extra"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files
