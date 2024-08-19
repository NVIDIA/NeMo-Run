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
from unittest.mock import Mock, patch

import pytest
from nemo_run.core.frontend.console.api import configure_logging
from rich.logging import RichHandler


@pytest.fixture
def mock_console():
    with patch("nemo_run.core.frontend.console.api.CONSOLE", Mock()) as mocked_console:
        yield mocked_console


@pytest.fixture
def mock_is_jupyter():
    with patch("nemo_run.core.frontend.console.api._is_jupyter") as mocked_is_jupyter:
        yield mocked_is_jupyter


def test_configure_logging(mock_console, mock_is_jupyter):
    # Test with a valid logging level
    mock_is_jupyter.return_value = False
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO

    mock_is_jupyter.return_value = True
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_invalid_level(mock_console, mock_is_jupyter):
    # Test with an invalid logging level
    mock_is_jupyter.return_value = False
    with pytest.raises(ValueError):
        configure_logging("INVALID_LEVEL")

    mock_is_jupyter.return_value = True
    with pytest.raises(ValueError):
        configure_logging("INVALID_LEVEL_2")


def test_configure_logging_jupyter(mock_console, mock_is_jupyter):
    # Test when _is_jupyter() returns True
    mock_is_jupyter.return_value = True
    configure_logging("INFO")
    assert all(map(lambda x: not isinstance(x, RichHandler), logging.getLogger().handlers))
