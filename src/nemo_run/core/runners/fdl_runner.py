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

import os
from pathlib import Path
import fiddle as fdl
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
import typer

from nemo_run.config import Partial
from nemo_run.core.packaging.base import Packager
from nemo_run.run.task import dryrun_fn

app = typer.Typer(pretty_exceptions_enable=False)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def direct_run(
    ctx: typer.Context,
    dryrun: bool = typer.Option(
        False,
        "--dryrun",
        help="Does not actually submit the app, just prints how the arguments will be resolved.",
    ),
    run_name: str = typer.Option(
        "run",
        "--name",
        "-n",
        help="Name of the run.",
    ),
    package_cfg: str = typer.Option(
        None,
        "--package-cfg",
        "-p",
        help="Serialized Package config of the run.",
    ),
    config: str = typer.Argument(..., help="Serialized fdl config."),
):
    if package_cfg:
        if os.path.isfile(package_cfg):
            package_cfg = Path(package_cfg).read_text()
        deser_package: fdl.Buildable = ZlibJSONSerializer().deserialize(package_cfg)
        package: Packager = fdl.build(deser_package)
        package.setup()

    if os.path.isfile(config):
        config = Path(config).read_text()

    fdl_fn: fdl.Buildable = ZlibJSONSerializer().deserialize(config)
    fdl_fn = fdl.cast(Partial, fdl_fn)
    if dryrun:
        dryrun_fn(fdl_fn, build=True)
        return

    fn = fdl.build(fdl_fn)
    fn()


if __name__ == "__main__":
    app()
