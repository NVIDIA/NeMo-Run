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
import tempfile

from src.nemo_run.core.execution.local import LocalExecutor


def test_local_executor_init():
    executor = LocalExecutor(ntasks_per_node=2)
    assert executor.ntasks_per_node == 2
    assert executor.experiment_id is None


def test_local_executor_assign():
    executor = LocalExecutor()
    with tempfile.TemporaryDirectory() as tmp_dir:
        executor.assign(
            exp_id="test_exp",
            exp_dir=tmp_dir,
            task_id="test_task",
            task_dir="test_task",
        )

        assert executor.experiment_id == "test_exp"
        assert executor.job_dir == os.path.join(tmp_dir, "test_task")
        assert os.path.exists(executor.job_dir)


def test_local_executor_nnodes():
    executor = LocalExecutor()
    assert executor.nnodes() == 1


def test_local_executor_nproc_per_node():
    executor = LocalExecutor(ntasks_per_node=3)
    assert executor.nproc_per_node() == 3
