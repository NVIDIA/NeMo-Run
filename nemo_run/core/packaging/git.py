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
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from invoke.context import Context

from nemo_run.core.packaging.base import Packager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GitArchivePackager(Packager):
    """
    Uses `git archive <https://git-scm.com/docs/git-archive>`_ for packaging your code.

    At a high level, it works in the following way:

    #. base_path = ``git rev-parse --show-toplevel``.
    #. Optionally define a subpath as ``base_path/self.subpath`` by setting ``subpath`` attribute.
    #. ``cd base_path`` && ``git archive --format=tar.gz --output={output_file} {self.ref}:{subpath}``
    #. This extracted tar file becomes the working directory for your job.

    .. note::
        By default, git archive will only package code committed in the specified ref.
        You can use include_uncommitted=True and include_untracked=True to package
        uncommitted changes and untracked files respectively.
    """

    basepath: str = ""
    #: Relative subpath in your repo to package code from.
    #: For eg, if your repo has three folders a, b and c
    #: and you specify a as the subpath, only files inside a
    #: will be packaged. In your job, the root workdir will be
    #: a/.
    subpath: str = ""

    #: Git ref to use for archiving the code.
    #: Can be a branch name or a commit ref like HEAD.
    ref: str = "HEAD"

    #: Include submodules in the archive.
    include_submodules: bool = True

    #: Include extra files in the archive which matches include_pattern
    #: This str will be included in the command as: find {include_pattern} -type f to get the list of extra files to include in the archive
    include_pattern: str | list[str] = ""

    #: Relative path to use as tar -C option - need to be consistent with include_pattern
    #: If not provided, will use git base path.
    include_pattern_relative_path: str | list[str] = ""

    check_uncommitted_changes: bool = False
    check_untracked_files: bool = False

    #: Include uncommitted changes in the archive
    include_uncommitted: bool = False

    #: Include untracked files in the archive
    include_untracked: bool = False

    def package(self, path: Path, job_dir: str, name: str) -> str:
        output_file = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(output_file):
            return output_file

        if self.basepath:
            path = Path(self.basepath)

        subprocess.check_call(f"cd {str(path)} && git rev-parse", shell=True)
        output = subprocess.run(
            f"cd {str(path)} && git rev-parse --show-toplevel",
            check=True,
            stdout=subprocess.PIPE,
            shell=True,
        )
        git_base_path = Path(output.stdout.splitlines()[0].decode().strip())
        git_sub_path = os.path.join(self.subpath, "")

        if self.check_uncommitted_changes:
            try:
                subprocess.check_call(
                    f"cd {shlex.quote(str(git_base_path))} && git diff-index --quiet HEAD --",
                    shell=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "Your repo has uncommitted changes. Please commit your changes or set check_uncommitted_changes to False to proceed with packaging."
                ) from e

        if self.check_untracked_files:
            untracked_files = subprocess.run(
                f"cd {shlex.quote(str(git_base_path))} && git ls-files --others --exclude-standard",
                shell=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert not bool(untracked_files), (
                "Your repo has untracked files. Please track your files via git or set check_untracked_files to False to proceed with packaging."
            )

        ctx = Context()
        # we first add git files into an uncompressed archive
        # then we add submodule files into that archive
        # then we add an extra files from pattern to that archive
        # finally we compress it (cannot compress right away, since adding files is not possible)
        git_archive_cmd = (
            f"git archive --format=tar --output={output_file}.tmp {self.ref}:{git_sub_path}"
        )
        if os.uname().sysname == "Linux":
            tar_submodule_cmd = f"tar Af {output_file}.tmp $sha1.tmp && rm $sha1.tmp"
        else:
            tar_submodule_cmd = f"cat $sha1.tmp >> {output_file}.tmp && rm $sha1.tmp"

        git_submodule_cmd = f"""git submodule foreach --recursive \
'git archive --format=tar --prefix=$sm_path/ --output=$sha1.tmp HEAD && {tar_submodule_cmd}'"""

        with ctx.cd(git_base_path):
            ctx.run(git_archive_cmd)
            if self.include_submodules:
                ctx.run(git_submodule_cmd)

        # Handle uncommitted changes
        if self.include_uncommitted:
            unstaged_file_id = uuid.uuid4()
            unstaged_tar_file = f"unstaged_{unstaged_file_id}.tmp"

            with ctx.cd(git_base_path):
                # Get the list of files with unstaged changes
                changed_files = (
                    subprocess.run(
                        "git diff --name-only", shell=True, capture_output=True, text=True
                    )
                    .stdout.strip()
                    .split("\n")
                )

                if changed_files and changed_files[0]:  # Check if non-empty
                    # Filter files in subpath if specified
                    if self.subpath:
                        filtered_files = []
                        for f in changed_files:
                            if f.startswith(self.subpath):
                                filtered_files.append(f)
                        changed_files = filtered_files

                    if changed_files:
                        # Create the tarfile in the right way based on subpath
                        if self.subpath:
                            # We need to change to the subpath dir and create the tar without the prefix
                            subpath_dir = os.path.join(git_base_path, self.subpath)

                            # Create list of files relative to the subpath
                            relative_files = []
                            for f in changed_files:
                                relative_files.append(
                                    os.path.relpath(os.path.join(git_base_path, f), subpath_dir)
                                )

                            # Create tar from the subpath directory
                            tar_path = os.path.join(git_base_path, unstaged_tar_file)
                            with ctx.cd(subpath_dir):
                                if relative_files:  # Only create tar if there are files
                                    ctx.run(f"tar -cf {tar_path} {' '.join(relative_files)}")
                                else:
                                    # Create empty tar file to avoid errors
                                    ctx.run(f"tar -cf {tar_path} --files-from /dev/null")
                        else:
                            # No subpath, create tar from the git base path
                            if changed_files:  # Only create tar if there are files
                                ctx.run(f"tar -cf {unstaged_tar_file} {' '.join(changed_files)}")
                            else:
                                # Create empty tar file to avoid errors
                                ctx.run(f"tar -cf {unstaged_tar_file} --files-from /dev/null")

                        # Add to the main archive - use a more compatible approach
                        # Instead of using 'tar Af', use the extract-and-create approach for all platforms
                        temp_dir = f"temp_extract_unstaged_{unstaged_file_id}"
                        ctx.run(f"mkdir -p {temp_dir}")
                        ctx.run(f"tar xf {output_file}.tmp -C {temp_dir}")
                        ctx.run(f"tar xf {unstaged_tar_file} -C {temp_dir}")
                        ctx.run(f"tar cf {output_file}.tmp -C {temp_dir} .")
                        ctx.run(f"rm -rf {temp_dir}")
                        ctx.run(f"rm {unstaged_tar_file}")

        # Handle untracked files
        if self.include_untracked:
            untracked_file_id = uuid.uuid4()
            untracked_tar_file = f"untracked_{untracked_file_id}.tmp"

            with ctx.cd(git_base_path):
                # Get the list of untracked files
                untracked_files = (
                    subprocess.run(
                        "git ls-files --others --exclude-standard",
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    .stdout.strip()
                    .split("\n")
                )

                if untracked_files and untracked_files[0]:  # Check if non-empty
                    # Filter files in subpath if specified
                    if self.subpath:
                        filtered_files = []
                        for f in untracked_files:
                            if f.startswith(self.subpath):
                                filtered_files.append(f)
                        untracked_files = filtered_files

                    if untracked_files:
                        # Create the tarfile in the right way based on subpath
                        if self.subpath:
                            # We need to change to the subpath dir and create the tar without the prefix
                            subpath_dir = os.path.join(git_base_path, self.subpath)

                            # Create list of files relative to the subpath
                            relative_files = []
                            for f in untracked_files:
                                relative_files.append(
                                    os.path.relpath(os.path.join(git_base_path, f), subpath_dir)
                                )

                            # Create tar from the subpath directory
                            tar_path = os.path.join(git_base_path, untracked_tar_file)
                            with ctx.cd(subpath_dir):
                                ctx.run(f"tar -cf {tar_path} {' '.join(relative_files)}")
                        else:
                            # No subpath, create tar from the git base path
                            ctx.run(f"tar -cf {untracked_tar_file} {' '.join(untracked_files)}")

                        # Add to the main archive - use a more compatible approach
                        # Instead of using 'tar Af', use the extract-and-create approach for all platforms
                        temp_dir = f"temp_extract_untracked_{untracked_file_id}"
                        ctx.run(f"mkdir -p {temp_dir}")
                        ctx.run(f"tar xf {output_file}.tmp -C {temp_dir}")
                        ctx.run(f"tar xf {untracked_tar_file} -C {temp_dir}")
                        ctx.run(f"tar cf {output_file}.tmp -C {temp_dir} .")
                        ctx.run(f"rm -rf {temp_dir}")
                        ctx.run(f"rm {untracked_tar_file}")

        # Process include_pattern files
        if isinstance(self.include_pattern, str):
            self.include_pattern = [self.include_pattern]

        if isinstance(self.include_pattern_relative_path, str):
            self.include_pattern_relative_path = [self.include_pattern_relative_path]

        if len(self.include_pattern) != len(self.include_pattern_relative_path):
            raise ValueError(
                "include_pattern and include_pattern_relative_path should have the same length"
            )

        for include_pattern, include_pattern_relative_path in zip(
            self.include_pattern, self.include_pattern_relative_path
        ):
            pattern_file_id = uuid.uuid4()
            pattern_tar_file_name = f"additional_{pattern_file_id}.tmp"

            if include_pattern == "":
                continue
            include_pattern_relative_path = include_pattern_relative_path or shlex.quote(
                str(git_base_path)
            )
            relative_include_pattern = os.path.relpath(
                include_pattern, include_pattern_relative_path
            )
            pattern_tar_file_name = os.path.join(git_base_path, pattern_tar_file_name)
            include_pattern_cmd = (
                f"find {relative_include_pattern} -type f | tar -cf {pattern_tar_file_name} -T -"
            )

            with ctx.cd(include_pattern_relative_path):
                ctx.run(include_pattern_cmd)

            with ctx.cd(git_base_path):
                if os.uname().sysname == "Linux":
                    # On Linux, directly concatenate tar files
                    ctx.run(f"tar Af {output_file}.tmp {pattern_tar_file_name}")
                    ctx.run(f"rm {pattern_tar_file_name}")
                else:
                    # Extract and repack approach for other platforms
                    temp_dir = f"temp_extract_{pattern_file_id}"
                    ctx.run(f"mkdir -p {temp_dir}")
                    ctx.run(f"tar xf {output_file}.tmp -C {temp_dir}")
                    ctx.run(f"tar xf {pattern_tar_file_name} -C {temp_dir}")
                    ctx.run(f"tar cf {output_file}.tmp -C {temp_dir} .")
                    ctx.run(f"rm -rf {temp_dir} {pattern_tar_file_name}")

        gzip_cmd = f"gzip -c {output_file}.tmp > {output_file}"
        rm_cmd = f"rm {output_file}.tmp"

        with ctx.cd(git_base_path):
            ctx.run(gzip_cmd)
            ctx.run(rm_cmd)

        return output_file


__all__ = ["GitArchivePackager"]
