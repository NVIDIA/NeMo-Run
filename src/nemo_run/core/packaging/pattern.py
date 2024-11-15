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
from typing import List, Union

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
    include_pattern: Union[str, List[str]]

    #: Relative path to use as tar -C option.
    relative_path: Union[str, List[str]]

    def package(self, path: Path, job_dir: str, name: str) -> str:
        output_file = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(output_file):
            return output_file

        if isinstance(self.include_pattern, str):
            self.include_pattern = [self.include_pattern]

        if isinstance(self.relative_path, str):
            self.relative_path = [self.relative_path]

        if not len(self.relative_path) == len(self.include_pattern):
            raise ValueError("`relative_path` and `include_pattern` should have the same number of arguments.")

        cmd = f"tar -czf {output_file}"
        for pattern_index in range(len(self.include_pattern)):
            relative_path = self.relative_path[pattern_index]
            include_pattern = self.include_pattern[pattern_index]
            relative_include_pattern = os.path.relpath(include_pattern, relative_path)
            cmd += f" -C {relative_path} -T <(cd {relative_path} && find {relative_include_pattern} -type f)"

        ctx = Context()
        ctx.run(cmd)
        return output_file


__all__ = ["PatternPackager"]
