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
from torchx import specs

from nemo_run.config import Partial, Script
from nemo_run.core.execution.base import Executor, FaultTolerance, Torchrun
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.packaging.base import Packager
from nemo_run.run.torchx_backend.packaging import (
    merge_executables,
    package,
)


@dataclass(kw_only=True)
class MockExecutor(Executor):
    nodes: int = 1
    ntasks_per_node: int = 1

    def _setup_launcher(self):
        return None

    def nnodes(self) -> int:
        return self.nodes

    def nproc_per_node(self) -> int:
        return self.ntasks_per_node


@pytest.fixture
def mock_executor():
    return MockExecutor(packager=Packager())


def dummy_add(a: int, b: int) -> int:
    return a + b


def test_package_partial(mock_executor):
    fn_or_script = Partial(dummy_add, a=1, b=2)
    mock_executor.retries = 3
    app_def = package(
        name="test",
        fn_or_script=fn_or_script,
        executor=mock_executor,
    )
    assert app_def.name == "test"
    assert len(app_def.roles) == 1
    assert app_def.roles[0].name == "test"
    role = app_def.roles[0]
    assert role.entrypoint == "python"
    assert role.max_retries == 3
    assert role.args == [
        "-m",
        "nemo_run.core.runners.fdl_runner",
        "-n",
        "test",
        "eJzdlEtPwzAMgP_KlMuGhKp1RwRIcOOAxIHbNEVuk21haVKlzrRq2n8nDuto9wSxE5eqduLPdvxYM2ctsrvemmFdyvDDnJyy2x5byJok4Yui5iAET9kmqG32IXOsvix8qWXQt6y_MWW9BRVWeB1VmVcalalIa6CIusiIZIWyIO54zF6MkKuBou_D8IauA5vc9roHaTzI2GRCTiSCAIRgb7zWxBMqxz8GR4hubHu-rpr3sTx2iYz-QSJkLiALPYMOltJV0vHm3i8qNVVCaJnwyuVJbs1UzdrxPDdO3hsfr00oe132hOgGZPbQnxpuHc911aemOusdrcvnK55BvpBGJCgr5GUQYKZMJ5Dd5LBN7N2WO3AzX0iDnMR9n935a2bsNANhdh6xHYTThLmqQlb1ZcgoQE41znUrFfu-tXp-2htGFpY7b464ewOHCvSZLoC9F9ASIn4J2pMiDf8l4DxasnvanI9J2EwHL5tdAI2OgTICnXzdbjUuTNLmCD_QSR04ufXmYIOn7Y25E0Zb4eLkpgf1SskbXVXWUMjDZJiEyD4BcFMKTQ==",
    ]


def test_package_partial_to_file(tmpdir):
    fn_or_script = Partial(dummy_add, a=1, b=2)
    app_def = package(
        name="test",
        fn_or_script=fn_or_script,
        executor=LocalExecutor(job_dir=tmpdir),
        serialize_to_file=True,
    )
    assert app_def.name == "test"
    assert len(app_def.roles) == 1
    assert app_def.roles[0].name == "test"
    role = app_def.roles[0]
    assert role.entrypoint == "python"
    assert role.args == [
        "-m",
        "nemo_run.core.runners.fdl_runner",
        "-n",
        "test",
        f"{tmpdir}/configs/test_fn_or_script",
    ]


def test_package_script(mock_executor):
    fn_or_script = Script(
        path="test.py",
        args=["arg1", "arg2"],
        env={"ENV_VAR": "value"},
    )
    app_def = package(
        name="test",
        fn_or_script=fn_or_script,
        executor=mock_executor,
    )
    assert app_def.name == "test"
    assert len(app_def.roles) == 1
    assert app_def.roles[0].name == "test"
    role = app_def.roles[0]
    assert role.entrypoint == "bash"
    assert role.args == ["test.py", "arg1", "arg2"]
    assert role.env == {"ENV_VAR": "value"}


@pytest.mark.parametrize(
    "inline, expected",
    [
        ("echo 'Hello World Mock Test'", ["/nemo_run/scripts/test.sh"]),
        (
            """echo \"Hello World Mock Test\"""",
            ["/nemo_run/scripts/test.sh"],
        ),
    ],
)
def test_package_script_inline(mock_executor, inline, expected):
    fn_or_script = Script(inline=inline)
    app_def = package(
        name="test",
        fn_or_script=fn_or_script,
        executor=mock_executor,
    )
    assert app_def.name == "test"
    assert len(app_def.roles) == 1
    assert app_def.roles[0].name == "test"
    role = app_def.roles[0]
    assert role.entrypoint == "bash"
    assert role.args == expected


def test_package_torchrun(mock_executor):
    mock_executor.get_launcher = lambda: Torchrun(rdzv_backend="etcd", rdzv_port=2379)
    fn_or_script = Script(
        path="test.py",
        args=["arg1", "arg2"],
        env={"ENV_VAR": "value"},
    )
    app_def = package(
        name="test",
        fn_or_script=fn_or_script,
        executor=mock_executor,
    )
    assert app_def.name == "test"
    role = app_def.roles[0]

    # Hack: overwrite random id for now
    role.args[5] = "1"

    assert role.entrypoint == "torchrun"
    assert role.args == [
        "--rdzv-backend",
        "etcd",
        "--rdzv-endpoint",
        "localhost:0",
        "--rdzv-id",
        "1",
        "--nnodes",
        "1",
        "--nproc-per-node",
        "1",
        "--node-rank",
        "0",
        "--tee",
        "3",
        "--no-python",
        "test.py",
        "arg1",
        "arg2",
    ]

    mock_executor.nodes = 2
    mock_executor.packager.debug = True

    fn_or_script.m = True
    app_def = package(
        name="test",
        fn_or_script=fn_or_script,
        executor=mock_executor,
    )
    assert app_def.name == "test"
    role = app_def.roles[0]

    # Hack: overwrite random id for now
    role.args[5] = "1"

    assert role.entrypoint == "torchrun"
    assert role.args == [
        "--rdzv-backend",
        "etcd",
        "--rdzv-endpoint",
        "$$${head_node_ip_var}:2379",
        "--rdzv-id",
        "1",
        "--nnodes",
        "$$${num_nodes_var}",
        "--nproc-per-node",
        "1",
        "--node-rank",
        "$$${node_rank_var}",
        "--tee",
        "3",
        "-m",
        "test.py",
        "arg1",
        "arg2",
    ]
    assert role.env == {
        "ENV_VAR": "value",
        "LOGLEVEL": "INFO",
        "CUDA_LAUNCH_BLOCKING": "1",
        "NCCL_DESYNC_DEBUG": "1",
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
    }

    with pytest.raises(ValueError):
        fn_or_script.m = None
        fn_or_script.path = None
        app_def = package(
            name="test",
            fn_or_script=fn_or_script,
            executor=mock_executor,
        )


def test_package_fault_tolerance(mock_executor):
    mock_executor.get_launcher = lambda: FaultTolerance(
        rdzv_backend="etcd",
        rdzv_port=2379,
        workload_check_interval=10,
        initial_rank_heartbeat_timeout=5,
        rank_heartbeat_timeout=5,
        rank_termination_signal="SIGINT",
        log_level="INFO",
    )
    fn_or_script = Script(
        path="test.py",
        args=["arg1", "arg2"],
        env={"ENV_VAR": "value"},
    )
    app_def = package(
        name="test",
        fn_or_script=fn_or_script,
        executor=mock_executor,
    )
    assert app_def.name == "test"
    role = app_def.roles[0]
    # Hack: overwrite random id for now
    role.args[15] = "1"

    assert role.entrypoint == "ft_launcher"
    assert role.args == [
        "--ft-param-workload_check_interval",
        "10",
        "--ft-param-initial_rank_heartbeat_timeout",
        "5",
        "--ft-param-rank_heartbeat_timeout",
        "5",
        "--ft-param-rank_termination_signal",
        "SIGINT",
        "--ft-param-log_level",
        "INFO",
        "--rdzv-backend",
        "etcd",
        "--rdzv-endpoint",
        "localhost:0",
        "--rdzv-id",
        "1",
        "--nnodes",
        "1",
        "--nproc-per-node",
        "1",
        "--node-rank",
        "0",
        "--tee",
        "3",
        "--no-python",
        "test.py",
        "arg1",
        "arg2",
    ]


def test_merge_executables():
    app_def1 = specs.AppDef(name="app1", roles=[specs.Role(name="role1", image="")])
    app_def2 = specs.AppDef(name="app2", roles=[specs.Role(name="role2", image="")])
    merged_app_def = merge_executables([app_def1, app_def2], name="merged")  # type: ignore
    assert merged_app_def.name == "merged"
    assert len(merged_app_def.roles) == 2
    assert merged_app_def.roles[0].name == "role1"
    assert merged_app_def.roles[1].name == "role2"
