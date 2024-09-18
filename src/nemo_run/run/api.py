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

from typing import List, Optional, Union

from fiddle import Buildable

from nemo_run.config import Script, get_type_namespace
from nemo_run.core.execution.base import Executor
from nemo_run.run.experiment import Experiment
from nemo_run.run.plugin import ExperimentPlugin as Plugin
from nemo_run.run.task import direct_run_fn


def run(
    fn_or_script: Union[Buildable, Script],
    executor: Optional[Executor] = None,
    plugins: Optional[Union[Plugin, List[Plugin]]] = None,
    name: str = "",
    dryrun: bool = False,
    direct: bool = False,
    detach: bool = False,
    tail_logs: bool = True,
    log_level: str = "INFO",
):
    """
    Runs a single configured function on the specified executor.
    If no executor is specified, it runs the run.Partial function directly
    i.e. equivalent to calling the python function directly.

    Examples
    --------
    .. code-block:: python

        import nemo_run as run

        # Run it directly in the same process
        run.run(configured_fn)

        # Do a dryrun
        run.run(configured_fn, dryrun=True)

        # Specify a custom executor
        local_executor = LocalExecutor()
        run.run(configured_fn, executor=local_executor)

        slurm_executor = run.SlurmExecutor(...)
        run.run(configured_fn, executor=slurm_executor)

    """
    if not isinstance(fn_or_script, (Buildable, Script)):
        raise TypeError(f"Need a configured Buildable or run.Script. Got {fn_or_script}.")

    if direct or executor is None:
        direct_run_fn(fn_or_script, dryrun=dryrun)
        return

    if plugins:
        plugins = [plugins] if not isinstance(plugins, list) else plugins

    default_name = (
        fn_or_script.get_name()
        if isinstance(fn_or_script, Script)
        else get_type_namespace(fn_or_script.__fn_or_cls__)
    )
    name = name or default_name
    with Experiment(title=name, executor=executor, log_level=log_level) as exp:
        exp.add(fn_or_script, tail_logs=tail_logs, plugins=plugins)
        if dryrun:
            exp.dryrun()
            return

        exp.run(detach=detach)
