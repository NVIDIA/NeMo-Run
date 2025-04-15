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

import invoke
import pytest

from nemo_run.core.packaging.git import GitArchivePackager
from test.conftest import MockContext


def mock_check_call(cmd, *args, **kwargs):
    cmd = " ".join(cmd)
    if "git archive" in cmd:
        return
    elif "pip install" in cmd:
        return
    else:
        raise subprocess.CalledProcessError(1, cmd)


@pytest.fixture
def temp_repo(tmpdir):
    repo_path = tmpdir.mkdir("repo")
    os.chdir(str(repo_path))
    subprocess.check_call(["git", "init", "--initial-branch=main"])
    # Create some files
    open("file1.txt", "w").write("Hello")
    open("file2.txt", "w").write("World")
    subprocess.check_call(["git", "add", "."])
    subprocess.check_call(["git", "commit", "-m", "Initial commit"])
    return repo_path


@pytest.fixture
def packager(temp_repo):
    return GitArchivePackager(basepath=str(temp_repo), subpath="", ref="HEAD")


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package(packager, temp_repo):
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(temp_repo, os.path.join(job_dir, "extracted_output"))
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_subpath(packager, temp_repo):
    temp_repo = Path(temp_repo)
    (temp_repo / "subdir").mkdir()
    open(temp_repo / "subdir" / "file3.txt", "w").write("Subdir file")
    subprocess.check_call(["git", "add", "."])
    subprocess.check_call(["git", "commit", "-m", "Add subdir"])

    packager = GitArchivePackager(basepath=str(temp_repo), subpath="subdir", ref="HEAD")
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "subdir"), os.path.join(job_dir, "extracted_output")
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_subpath_with_basepath(packager, temp_repo):
    temp_repo = Path(temp_repo)
    (temp_repo / "subdir").mkdir()
    (temp_repo / "subdir" / "subdir2").mkdir()
    open(temp_repo / "subdir" / "subdir2" / "file3.txt", "w").write("Subdir file")
    subprocess.check_call(["git", "add", "."])
    subprocess.check_call(["git", "commit", "-m", "Add subdir"])

    packager = GitArchivePackager(
        basepath=os.path.join(temp_repo, "subdir"), subpath="subdir/subdir2", ref="HEAD"
    )
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "subdir", "subdir2"),
            os.path.join(job_dir, "extracted_output"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_invalid_ref(packager, temp_repo):
    packager.ref = "invalid_ref"
    with pytest.raises(invoke.exceptions.UnexpectedExit):
        with tempfile.TemporaryDirectory() as job_dir:
            packager.package(Path(temp_repo), job_dir, "test_package")


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_nonexistent_basepath(packager, temp_repo):
    packager.basepath = str(Path(temp_repo) / "nonexistent_path")
    with pytest.raises(subprocess.CalledProcessError):
        with tempfile.TemporaryDirectory() as job_dir:
            packager.package(Path(temp_repo), job_dir, "test_package")


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_include_pattern(packager, temp_repo):
    temp_repo = Path(temp_repo)
    # Create extra files
    (temp_repo / "extra").mkdir()
    with open(temp_repo / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(temp_repo / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    packager = GitArchivePackager(ref="HEAD", include_pattern="extra")
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "extra"),
            os.path.join(job_dir, "extracted_output", "extra"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_include_pattern_and_subpath(packager, temp_repo):
    temp_repo = Path(temp_repo)
    # Create extra files
    (temp_repo / "extra").mkdir()
    with open(temp_repo / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(temp_repo / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    # Create extra files
    (temp_repo / "extra2").mkdir()
    with open(temp_repo / "extra2" / "extra2_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(temp_repo / "extra2" / "extra2_file2.txt", "w") as f:
        f.write("Extra file 2")
    subprocess.check_call(
        [f"cd {temp_repo} && git add extra2 && git commit -m 'Extra2 commit'"], shell=True
    )

    packager = GitArchivePackager(ref="HEAD", include_pattern="extra", subpath="extra2")
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "extra"),
            os.path.join(job_dir, "extracted_output", "extra"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files

        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "extra2"),
            os.path.join(job_dir, "extracted_output"),
            ignore=["extra"],
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_include_pattern_multiple_directories(packager, temp_repo):
    temp_repo = Path(temp_repo)
    # Create extra files
    (temp_repo / "extra").mkdir()
    with open(temp_repo / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(temp_repo / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    (temp_repo / "extra_1").mkdir()
    with open(temp_repo / "extra_1" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(temp_repo / "extra_1" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    packager = GitArchivePackager(ref="HEAD", include_pattern="extra extra_1")
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "extra"),
            os.path.join(job_dir, "extracted_output", "extra"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files

        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "extra_1"),
            os.path.join(job_dir, "extracted_output", "extra_1"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_include_pattern_rel_path(packager, temp_repo, tmpdir):
    temp_repo = Path(temp_repo)
    # Create extra files in a separate directory
    (tmpdir / "extra").mkdir()
    with open(tmpdir / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(tmpdir / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    packager = GitArchivePackager(
        include_pattern=str(tmpdir / "extra/*"), include_pattern_relative_path=str(tmpdir)
    )
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
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


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_multi_include_pattern_rel_path(packager, temp_repo, tmpdir):
    temp_repo = Path(temp_repo)
    # Create extra files in a separate directory
    (tmpdir / "extra").mkdir()
    with open(tmpdir / "extra" / "extra_file1.txt", "w") as f:
        f.write("Extra file 1")
    with open(tmpdir / "extra" / "extra_file2.txt", "w") as f:
        f.write("Extra file 2")

    include_pattern = [str(tmpdir / "extra/extra_file1.txt"), str(tmpdir / "extra/extra_file2.txt")]
    relative_path = [str(tmpdir), str(tmpdir)]

    packager = GitArchivePackager(
        include_pattern=include_pattern, include_pattern_relative_path=relative_path
    )
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
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


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_check_uncommitted_changes(packager, temp_repo):
    temp_repo = Path(temp_repo)
    with open(temp_repo / "file1.txt", "w") as f:
        f.write("Modified content")

    packager = GitArchivePackager(ref="HEAD", check_uncommitted_changes=True)
    with pytest.raises(RuntimeError, match="Your repo has uncommitted changes"):
        packager.package(temp_repo, str(temp_repo), "test_package")


def test_untracked_files_raises_exception(temp_repo):
    packager = GitArchivePackager(check_untracked_files=True)
    Path(temp_repo / "untracked.txt").touch()
    with open(temp_repo / "untracked.txt", "w") as f:
        f.write("Untracked file")
    with pytest.raises(AssertionError, match="Your repo has untracked files"):
        packager.package(temp_repo, str(temp_repo), "test")


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_include_submodules(packager, temp_repo):
    temp_repo = Path(temp_repo)
    # Create first submodule
    submodule_path = temp_repo / "submodule"
    submodule_path.mkdir()
    os.chdir(str(submodule_path))
    subprocess.check_call(["git", "init", "--initial-branch=main"])
    open("submodule_file.txt", "w").write("Submodule file")
    subprocess.check_call(["git", "add", "."])
    subprocess.check_call(["git", "commit", "-m", "Initial submodule commit"])

    # Create second submodule
    submodule2_path = temp_repo / "submodule2"
    submodule2_path.mkdir()
    os.chdir(str(submodule2_path))
    subprocess.check_call(["git", "init", "--initial-branch=main"])
    open("submodule2_file.txt", "w").write("Second submodule file")
    subprocess.check_call(["git", "add", "."])
    subprocess.check_call(["git", "commit", "-m", "Initial submodule2 commit"])

    os.chdir(str(temp_repo))
    subprocess.check_call(["git", "submodule", "add", str(submodule_path)])
    subprocess.check_call(["git", "submodule", "add", str(submodule2_path)])
    subprocess.check_call(["git", "commit", "-m", "Add submodules"])

    packager = GitArchivePackager(ref="HEAD", include_submodules=True)
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        # Check first submodule
        cmp = filecmp.dircmp(
            os.path.join(temp_repo, "submodule"),
            os.path.join(job_dir, "extracted_output", "submodule"),
        )
        assert cmp.left_list == cmp.right_list
        assert not cmp.diff_files

        # Check second submodule
        cmp2 = filecmp.dircmp(
            os.path.join(temp_repo, "submodule2"),
            os.path.join(job_dir, "extracted_output", "submodule2"),
        )
        assert cmp2.left_list == cmp2.right_list
        assert not cmp2.diff_files


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_without_include_submodules(packager, temp_repo):
    temp_repo = Path(temp_repo)
    # Create a submodule
    submodule_path = temp_repo / "submodule"
    submodule_path.mkdir()
    os.chdir(str(submodule_path))
    subprocess.check_call(["git", "init", "--initial-branch=main"])
    open("submodule_file.txt", "w").write("Submodule file")
    subprocess.check_call(["git", "add", "."])
    subprocess.check_call(["git", "commit", "-m", "Initial submodule commit"])
    os.chdir(str(temp_repo))
    subprocess.check_call(["git", "submodule", "add", str(submodule_path)])
    subprocess.check_call(["git", "commit", "-m", "Add submodule"])

    packager = GitArchivePackager(ref="HEAD", include_submodules=False)
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        subprocess.check_call(shlex.split(f"mkdir -p {os.path.join(job_dir, 'extracted_output')}"))
        subprocess.check_call(
            shlex.split(
                f"tar -xvzf {output_file} -C {os.path.join(job_dir, 'extracted_output')} --ignore-zeros"
            ),
        )
        assert len(os.listdir(os.path.join(job_dir, "extracted_output", "submodule"))) == 0


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_uncommitted_changes(packager, temp_repo):
    temp_repo = Path(temp_repo)
    with open(temp_repo / "file1.txt", "w") as f:
        f.write("Modified content")

    packager = GitArchivePackager(ref="HEAD", include_uncommitted=True)
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        extract_dir = os.path.join(job_dir, "extracted_output")
        subprocess.check_call(shlex.split(f"mkdir -p {extract_dir}"))
        subprocess.check_call(
            shlex.split(f"tar -xvzf {output_file} -C {extract_dir} --ignore-zeros"),
        )

        # Verify that the modified file was included with changes
        with open(os.path.join(extract_dir, "file1.txt"), "r") as f:
            content = f.read()
        assert content == "Modified content"


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_untracked_files(packager, temp_repo):
    temp_repo = Path(temp_repo)
    # Add an untracked file
    with open(temp_repo / "untracked.txt", "w") as f:
        f.write("Untracked content")

    packager = GitArchivePackager(ref="HEAD", include_untracked=True)
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        extract_dir = os.path.join(job_dir, "extracted_output")
        subprocess.check_call(shlex.split(f"mkdir -p {extract_dir}"))
        subprocess.check_call(
            shlex.split(f"tar -xvzf {output_file} -C {extract_dir} --ignore-zeros"),
        )

        # Verify that the untracked file was included
        assert os.path.exists(os.path.join(extract_dir, "untracked.txt"))
        with open(os.path.join(extract_dir, "untracked.txt"), "r") as f:
            content = f.read()
        assert content == "Untracked content"


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_with_uncommitted_and_untracked(packager, temp_repo):
    temp_repo = Path(temp_repo)
    with open(temp_repo / "file1.txt", "w") as f:
        f.write("Modified content")

    # Add an untracked file
    with open(temp_repo / "untracked.txt", "w") as f:
        f.write("Untracked content")

    packager = GitArchivePackager(ref="HEAD", include_uncommitted=True, include_untracked=True)
    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        extract_dir = os.path.join(job_dir, "extracted_output")
        subprocess.check_call(shlex.split(f"mkdir -p {extract_dir}"))
        subprocess.check_call(
            shlex.split(f"tar -xvzf {output_file} -C {extract_dir} --ignore-zeros"),
        )

        # Verify that the modified file was included with changes
        with open(os.path.join(extract_dir, "file1.txt"), "r") as f:
            content = f.read()
        assert content == "Modified content"

        # Verify that the untracked file was included
        assert os.path.exists(os.path.join(extract_dir, "untracked.txt"))
        with open(os.path.join(extract_dir, "untracked.txt"), "r") as f:
            content = f.read()
        assert content == "Untracked content"


@patch("nemo_run.core.packaging.git.Context", MockContext)
def test_package_subpath_with_uncommitted_and_untracked(packager, temp_repo):
    temp_repo = Path(temp_repo)
    # Create a subdir
    (temp_repo / "subdir").mkdir()

    # Add a file in subdir and commit
    with open(temp_repo / "subdir" / "committed.txt", "w") as f:
        f.write("Committed content")
    subprocess.check_call(["git", "add", "."], cwd=str(temp_repo))
    subprocess.check_call(["git", "commit", "-m", "Add subdir"], cwd=str(temp_repo))

    # Make an uncommitted change to the file in subdir
    with open(temp_repo / "subdir" / "committed.txt", "w") as f:
        f.write("Modified committed content")

    # Add an untracked file in the subdir
    with open(temp_repo / "subdir" / "untracked.txt", "w") as f:
        f.write("Untracked content in subdir")

    # Add a file outside of subdir that should not be included
    with open(temp_repo / "outside.txt", "w") as f:
        f.write("Outside content")

    # list all files in temp_repo recursively
    print(f"Files in temp_repo recursively:")
    for root, dirs, files in os.walk(temp_repo):
        for file in files:
            print(os.path.join(root, file))

    packager = GitArchivePackager(
        ref="HEAD",
        subpath="subdir",
        include_uncommitted=True,
        include_untracked=True,
    )

    with tempfile.TemporaryDirectory() as job_dir:
        output_file = packager.package(Path(temp_repo), job_dir, "test_package")
        assert os.path.exists(output_file)
        extract_dir = os.path.join(job_dir, "extracted_output")
        subprocess.check_call(shlex.split(f"mkdir -p {extract_dir}"))

        subprocess.check_call(
            shlex.split(f"tar -xvzf {output_file} -C {extract_dir} --ignore-zeros"),
        )

        # Verify that the modified file in subdir was included with changes
        with open(os.path.join(extract_dir, "committed.txt"), "r") as f:
            content = f.read()
        assert content == "Modified committed content"

        # Verify that the untracked file in subdir was included
        assert os.path.exists(os.path.join(extract_dir, "untracked.txt"))
        with open(os.path.join(extract_dir, "untracked.txt"), "r") as f:
            content = f.read()
        assert content == "Untracked content in subdir"

        # Verify that files outside the subpath are not included
        assert not os.path.exists(os.path.join(extract_dir, "outside.txt"))
