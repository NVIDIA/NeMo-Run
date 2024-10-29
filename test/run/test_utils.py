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

import sys
from unittest.mock import MagicMock

import pytest

from nemo_run.run.experiment import Experiment
from nemo_run.run.utils import TeeStdoutStderr, _Tee


class TestTeeStdoutStderr:
    def test__Tee_write_without_file_raises_exception(self):
        tee = _Tee("test.txt", sysout=sys.stderr)
        with pytest.raises(ValueError):
            tee.write("test")

    def test__Tee_flush_without_file_raises_exception(self):
        tee = _Tee("test.txt", sysout=sys.stdout)
        with pytest.raises(ValueError):
            tee.flush()

    def test__Tee_flush(self):
        tee = _Tee("test.txt", sysout=sys.stdout)
        tee.file = MagicMock()
        tee.flush()
        tee.file.flush.assert_called_once()

    def test_TeeStdoutStderr_context_manager(self, tmp_path, capsys):
        temp_file = tmp_path / "test.txt"
        with TeeStdoutStderr(str(temp_file)):
            print("output_err", file=sys.stderr)
            print("output_out", file=sys.stdout)

        print("output_err", file=sys.stderr)
        print("output_out", file=sys.stdout)

        with open(str(temp_file), "r") as f:
            content = f.read()
            assert content == "output_err\noutput_out\n"
        captured = capsys.readouterr()
        assert captured.out == "output_out\noutput_out\n"
        assert captured.err == "output_err\noutput_err\n"


def test_list_experiments_handles_missing():
    assert Experiment.catalog("test_experiment") == []
