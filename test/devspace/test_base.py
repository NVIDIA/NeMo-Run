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

import pytest
from nemo_run.core.execution.base import Executor
from nemo_run.core.tunnel.client import SSHTunnel
from nemo_run.devspace.base import DevSpace


@dataclass(kw_only=True)
class DummyExecutor(Executor):
    tunnel: SSHTunnel

    def connect_devspace(self, space, tunnel_dir=None): ...


class TestDevSpace:
    def test_devspace_init(self, mocker):
        executor_mock = mocker.Mock()
        space = DevSpace("test", executor_mock)
        assert space.name == "test"
        assert space.executor == executor_mock
        assert space.cmd == "launch_devspace"
        assert space.use_packager is False
        assert space.env_vars is None
        assert space.add_workspace_to_pythonpath is True

    def test_devspace_connect(self, mocker):
        tunnel_mock = mocker.patch("nemo_run.core.tunnel.client.SSHTunnel")
        tunnel_mock.return_value.run.return_value.stdout = "Test Zlib String"
        mocker.patch("nemo_run.devspace.base.ZlibJSONSerializer").deserialize.return_value = {
            "name": "test",
            "executor": "mock_executor",
        }
        mocker.patch("fiddle.build").return_value = DevSpace(
            "test", executor=DummyExecutor(tunnel=tunnel_mock)
        )

        DevSpace.connect("user@host", "/path")
        assert tunnel_mock.called
        assert tunnel_mock().run.called

    def test_devspace_connect_cat_exception(self, mocker):
        tunnel_mock = mocker.patch("nemo_run.core.tunnel.client.SSHTunnel")
        tunnel_mock.return_value.run.side_effect = Exception("Mocked cat exception")

        with pytest.raises(ValueError) as e:
            DevSpace.connect("user@host", "/path")
        assert "Could not find the devspace at user@host:/path." in str(e.value)

    def test_devspace_launch(self, mocker):
        executor_mock = mocker.Mock()

        space = DevSpace("test", executor_mock)
        space.use_packager = True
        space.execute_cmd = mocker.Mock()

        space.launch()
        assert executor_mock.packager.setup.called
        assert space.execute_cmd.called

        space.use_packager = False
        space.executor = mocker.Mock()
        space.launch()
        assert not space.executor.packager.setup.called
        assert space.execute_cmd.called

    def test_execute_cmd_exists(self, mocker):
        executor_mock = mocker.Mock()
        space = DevSpace("test", executor_mock)

        space.execute_cmd()
        assert executor_mock.launch_devspace.called_with(
            space,
            env_vars=space.env_vars,
            add_workspace_to_pythonpath=space.add_workspace_to_pythonpath,
        )

        space.cmd = "something_else"
        space.execute_cmd()
        assert executor_mock.something_else.called_with()

    def test_execute_cmd_not_exists(self, mocker):
        tunnel_mock = mocker.patch("nemo_run.core.tunnel.client.SSHTunnel")
        executor_mock = DummyExecutor(tunnel=tunnel_mock)
        space = DevSpace("test", executor_mock)
        space.cmd = "nonexistent_command"

        space.execute_cmd()
        assert not hasattr(executor_mock, "nonexistent_command")

    # def test_execute_cmd_callback(self, mocker):
    #     executor_mock = mocker.Mock()
    #     tunnel_mock = mocker.Mock()
    #     executor_mock.launch_devspace.return_value = mocker.Mock()
    #     executor_mock.tunnel = tunnel_mock

    #     space = DevSpace("test", executor_mock)

    #     space.execute_cmd()
    #     assert tunnel_mock.keep_alive.called

    # def test_devspace_connect_empty_host_path(self, mocker):
    #     with pytest.raises(ValueError):
    #         DevSpace.connect("", "/path")

    #     with pytest.raises(ValueError):
    #         DevSpace.connect("user@host", "")

    # def test_devspace_connect_no_file(self, mocker):
    #     tunnel_mock = mocker.patch("nemo_run.core.tunnel.client.SSHTunnel")
    #     tunnel_mock.return_value.run.return_value.stdout = ""
    #     mocker.patch(
    #         "nemo_run.devspace.base.ZlibJSONSerializer"
    #     ).deserialize.return_value = {"name": "test", "executor": "mock_executor"}

    #     with pytest.raises(ValueError):
    #         DevSpace.connect("user@host", "/path")
