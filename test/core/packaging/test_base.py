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

import pytest
from nemo_run.config import Config

from src.nemo_run.core.packaging.base import Packager


@pytest.fixture
def packager():
    return Packager()


def test_default_init(packager):
    assert packager.debug is False


def test_to_config(packager):
    config = packager.to_config()
    assert isinstance(config, Config)
    assert config.debug is False
