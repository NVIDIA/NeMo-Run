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
from typing import List

import nemo_run as run


@dataclass
class DummyModel:
    hidden: int = 100
    activation: str = "relu"


@dataclass
class NestedModel:
    dummy: DummyModel


@dataclass(kw_only=True)
class DummyPlugin(run.Plugin):
    some_arg: int = 10


@dataclass(kw_only=True)
class AnotherPlugin(run.Plugin):
    another_arg: int = 10


@run.cli.factory
@run.autoconvert
def dummy_factory_for_entrypoint() -> DummyModel:
    return DummyModel(hidden=1000)


@run.cli.factory
def dummy_model_config() -> run.Config[DummyModel]:
    return run.Config(DummyModel, hidden=2000, activation="tanh")


@run.cli.factory
@run.autoconvert
def my_dummy_model(hidden=2000) -> DummyModel:
    return DummyModel(hidden=hidden, activation="tanh")


@run.cli.entrypoint(namespace="dummy", skip_confirmation=True)
def dummy_entrypoint(dummy: DummyModel):
    print(dummy)
    NestedModel(dummy=dummy)


@run.cli.factory(target=dummy_entrypoint)
def dummy_recipe() -> run.Partial[dummy_entrypoint]:
    return run.Partial(dummy_entrypoint, dummy=dummy_model_config())


@run.cli.factory
@run.autoconvert
def local_executor() -> run.Executor:
    return run.LocalExecutor()


@run.cli.factory
@run.autoconvert
def dummy_plugin(some_arg: int = 20) -> run.Plugin:
    return DummyPlugin(some_arg=some_arg)


@run.cli.factory
@run.autoconvert
def plugin_list(arg: int = 20) -> List[run.Plugin]:
    return [
        dummy_plugin(arg),
        AnotherPlugin(another_arg=arg),
    ]


if __name__ == "__main__":
    run.cli.main(dummy_entrypoint)
