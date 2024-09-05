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

from dataclasses import dataclass, field


from nemo_run.config import ConfigurableMixin, Partial, Script
from nemo_run.core.execution.base import Executor


@dataclass(kw_only=True)
class ExperimentPlugin(ConfigurableMixin):
    """
    A base class for plugins that can be used to modify experiments, tasks, and executors.
    """

    experiment_id: str = field(init=False, default="")

    def assign(self, experiment_id: str):
        self.experiment_id = experiment_id

    def setup(self, task: Partial | Script, executor: Executor):
        """
        A hook method for setting up tasks and executors together.

        This method is intended to be overridden by subclasses to perform
        custom setup.
        """
        pass
