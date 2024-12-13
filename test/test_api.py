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

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

import nemo_run as run
from nemo_run.api import dryrun_fn


@dataclass
class Model:
    hidden: int


@dataclass
class Trainer:
    model: Model


class Test_autoconvert:
    def test_decorator(self):
        @run.autoconvert
        def my_model(hidden: int = 1000) -> Model:
            return Model(hidden)

        assert my_model() == run.Config(Model, hidden=1000)
        assert my_model.wrapped() == Model(hidden=1000)

    def test_decorator_partial(self):
        @run.autoconvert(partial=True)
        def my_model(hidden: int = 1000) -> Model:
            return Model(hidden)

        assert my_model(10) == run.Partial(Model, hidden=10)

    def test_decorator_with_returntype_config(self):
        @run.autoconvert
        def dummy_fn() -> run.Config[int]:
            return run.Config(int, 3)

        assert dummy_fn() == run.Config(int, 3)

        @run.autoconvert(partial=True)
        def dummy_fn_2() -> run.Partial[int]:
            return run.Partial(int, 3)

        assert dummy_fn_2() == run.Partial(int, 3)

        @run.autoconvert(partial=True)
        def dummy_fn_3() -> run.Config[int]:
            return run.Config(int, 3)

        assert dummy_fn_3() == run.Partial(int, 3)

    def test_nesting(self):
        @run.autoconvert
        def my_model(hidden: int = 1000) -> Model:
            return Model(hidden)

        @run.autoconvert
        def my_trainer(hidden: int = 1000) -> Trainer:
            return Trainer(model=my_model(hidden))

        assert my_trainer() == run.Config(Trainer, model=run.Config(Model, hidden=1000))

    def test_warning_for_config_return(self):
        @run.autoconvert
        def my_model(hidden: int = 1000) -> run.Config[Model]:
            return run.Config(Model, hidden=hidden)

        # Verify that the function is returned as-is
        result = my_model(500)
        assert isinstance(result, run.Config)
        assert result == run.Config(Model, hidden=500)


class TestDryRun:
    @pytest.fixture
    def configured_fn(self):
        def some_fn(arg1: str) -> str:
            return "hello world"

        return run.Partial(some_fn, arg1="value1")

    def test_dryrun_fn_invalid_input(
        self,
    ):
        with pytest.raises(TypeError):
            dryrun_fn("not a configured function")  # type: ignore

    def test_dryrun_fn_no_executor_build(self, capsys, configured_fn):
        dryrun_fn(configured_fn)

        captured = capsys.readouterr()
        assert "Dry run for task test.test_api:some_fn" in captured.out
        assert "Resolved Arguments" in captured.out
        assert "arg1" in captured.out
        assert "value1" in captured.out
        assert "Executor" not in captured.out

    def test_dryrun_fn_with_executor(self, capsys, configured_fn):
        dryrun_fn(configured_fn, executor=run.LocalExecutor())

        captured = capsys.readouterr()
        assert "Dry run for task test.test_api:some_fn" in captured.out
        assert (
            "LocalExecutor(packager=Packager(debug=False, symlink_from_remote_dir=None)"
            in captured.out
        )

    def test_dryrun_fn_with_build(self, mocker, configured_fn):
        build_mock = Mock()
        mocker.patch("fiddle.build", build_mock)

        dryrun_fn(configured_fn, build=True)
        build_mock.assert_called_once_with(configured_fn)
