# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tempfile

import pytest
from torchx.specs import AppDef, Role

from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.run.torchx_backend.schedulers.skypilot import (
    SkypilotScheduler,
    create_scheduler,
)


@pytest.fixture
def mock_app_def():
    return AppDef(name="test_app", roles=[Role(name="test_role", image="")])


@pytest.fixture
def skypilot_executor():
    return SkypilotExecutor(
        job_dir=tempfile.mkdtemp(),
        gpus="V100",
        gpus_per_node=1,
        cloud="aws",
    )


@pytest.fixture
def skypilot_scheduler():
    return create_scheduler(session_name="test_session")


def test_create_scheduler():
    scheduler = create_scheduler(session_name="test_session")
    assert isinstance(scheduler, SkypilotScheduler)
    assert scheduler.session_name == "test_session"


def test_skypilot_scheduler_methods(skypilot_scheduler):
    # Test that basic methods exist
    assert hasattr(skypilot_scheduler, "_submit_dryrun")
    assert hasattr(skypilot_scheduler, "schedule")
    assert hasattr(skypilot_scheduler, "describe")
    assert hasattr(skypilot_scheduler, "_validate")
