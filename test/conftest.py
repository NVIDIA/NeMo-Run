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
import sys
from typing import Any, Optional

import pytest
from invoke.config import Config
from invoke.context import Context

os.environ["INCLUDE_WORKSPACE_FILE"] = "false"


class MockContext(Context):
    def __init__(self, config: Optional[Config] = None) -> None:
        defaults = Config.global_defaults()
        defaults["run"]["pty"] = True
        defaults["run"]["in_stream"] = False
        super().__init__(config=config)

    def run(self, command: str, **kwargs: Any):
        kwargs["in_stream"] = False
        runner = self.config.runners.local(self)
        return self._run(runner, command, **kwargs)


@pytest.fixture(autouse=True)
def add_test_to_pythonpath():
    """Add the test directory to PYTHONPATH for all tests."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if test_dir not in sys.path:
        sys.path.append(test_dir)
    yield
    if test_dir in sys.path:
        sys.path.remove(test_dir)


@pytest.fixture(autouse=True)
def reset_nemorun_skip_confirmation():
    from nemo_run.cli import api

    """Reset NEMORUN_SKIP_CONFIRMATION to None before each test."""
    api.NEMORUN_SKIP_CONFIRMATION = None
    yield
    # Optionally, reset after the test as well
    api.NEMORUN_SKIP_CONFIRMATION = None
