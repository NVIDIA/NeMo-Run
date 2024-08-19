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
from dataclasses import dataclass
from pathlib import Path

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc

from nemo_run.config import Config

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Packager:
    """
    Base class for packaging your code.

    The packager is generally used as part of an Executor
    and provides the executor with information on how to package your code.

    It can also include information on how to run your code.
    For example, a packager can determine whether to use torchrun
    or whether to use debug flags.

    .. note::
        This class can also be used independently as a passthrough packager.
        This is useful in cases where you do not need to package code.
        For example, a local executor which uses your current working directory
        or an executor that uses a docker image that has all the code included.
    """

    #: Uses component or executor specific debug flags if set to True.
    debug: bool = False

    def to_config(self) -> Config:
        return fdl.cast(Config, fdl_dc.convert_dataclasses_to_configs(self, allow_post_init=True))

    def _repr_svg_(self):
        return self.to_config()._repr_svg_()

    def package(self, path: Path, job_dir: str, name: str) -> str: ...

    def setup(self):
        """
        This is run on the executor before starting your job.
        """
        ...


__all__ = ["Packager"]
