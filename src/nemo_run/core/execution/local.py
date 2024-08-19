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
from dataclasses import dataclass

from nemo_run.core.execution.base import Executor

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LocalExecutor(Executor):
    """
    Dataclass to configure local executor.

    Example:

    .. code-block:: python

        run.LocalExecutor()

    """

    #: Used by components like torchrun to deduce the number of tasks to launch.
    ntasks_per_node: int = 1

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        self.experiment_id = exp_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        os.makedirs(self.job_dir, exist_ok=True)

    def nnodes(self) -> int:
        return 1

    def nproc_per_node(self) -> int:
        return self.ntasks_per_node
