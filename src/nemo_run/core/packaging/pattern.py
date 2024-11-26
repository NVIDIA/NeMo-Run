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
from dataclasses import dataclass
from pathlib import Path

from invoke.context import Context

from nemo_run.core.packaging.base import Packager


@dataclass(kw_only=True)
class PatternPackager(Packager):
    """
    Will package all the files from the specified pattern.
    """

    #: Include files in the archive which matches include_pattern
    #: This str will be included in the command as:
    #: find {include_pattern} -type f to get the list of extra files to include in the archive
    #: best to use an absolute path here and a proper relative path argument to pass to tar
    include_pattern: str | list[str]

    #: Relative path to use as tar -C option.
    relative_path: str | list[str]

    def package(self, path: Path, job_dir: str, name: str) -> str:
        output_file = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(output_file):
            return output_file

        if isinstance(self.include_pattern, str):
            self.include_pattern = [self.include_pattern]

        if isinstance(self.relative_path, str):
            self.relative_path = [self.relative_path]

        if len(self.include_pattern) != len(self.relative_path):
            raise ValueError("include_pattern and relative_path should have the same length")

        # Create initial empty tar file
        ctx = Context()
        ctx.run(f"tar -cf {output_file}.tmp --files-from /dev/null")

        for include_pattern, relative_path in zip(self.include_pattern, self.relative_path):
            if include_pattern == "":
                continue

            relative_include_pattern = os.path.relpath(include_pattern, relative_path)

            with ctx.cd(relative_path):
                # Append files directly to the main tar archive
                cmd = f"find {relative_include_pattern} -type f -print0 | xargs -0 tar -rf {output_file}.tmp"
                ctx.run(cmd)

        # Gzip the final result
        gzip_cmd = f"gzip -c {output_file}.tmp > {output_file}"
        rm_cmd = f"rm {output_file}.tmp"

        ctx.run(gzip_cmd)
        ctx.run(rm_cmd)

        return output_file


__all__ = ["PatternPackager"]
