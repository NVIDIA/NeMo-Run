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

from simple.add import SomeObject, add_object, commonly_used_object

import nemo_run as run

# This script defines an experiment that invokes three tasks in parallel, two scripts and a run.Partial.
# The example demonstrates how you can use scripts and run.Partial.
if __name__ == "__main__":
    script = run.Script("./scripts/echo.sh")
    inline_script = run.Script(
        inline="""
env
echo "Hello 1"
echo "Hello 2"
"""
    )
    fn = run.Partial(
        add_object,
        obj_1=commonly_used_object(),
        obj_2=run.Config(SomeObject, value_1=10, value_2=20, value_3=30),
    )
    executor = run.LocalExecutor()

    with run.Experiment("experiment_with_scripts", executor=executor, log_level="WARN") as exp:
        exp.add(script, tail_logs=True)
        exp.add(inline_script, tail_logs=True)
        exp.add(fn, tail_logs=True)
        exp.run(detach=False)
