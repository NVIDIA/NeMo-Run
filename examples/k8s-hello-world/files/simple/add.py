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

import nemo_run as run


def add(a: int, b: int) -> int:
    print(f"Adding {a} to {b} returns {a + b}")
    return a + b


@dataclass
class SomeObject:
    value_1: int
    value_2: int
    value_3: int


def add_object(obj_1: SomeObject, obj_2: SomeObject) -> SomeObject:
    result = SomeObject(
        value_1=obj_1.value_1 + obj_2.value_1,
        value_2=obj_1.value_2 + obj_2.value_2,
        value_3=obj_1.value_3 + obj_2.value_3,
    )
    print(f"{result = }")

    return result


@run.autoconvert
def commonly_used_object() -> SomeObject:
    return SomeObject(
        value_1=5,
        value_2=10,
        value_3=15,
    )


@run.autoconvert
def commonly_used_object_2() -> SomeObject:
    return SomeObject(
        value_1=500,
        value_2=1000,
        value_3=1500,
    )
