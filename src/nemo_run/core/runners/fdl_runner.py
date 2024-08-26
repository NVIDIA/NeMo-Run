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

import typer

fdl_runner_app = typer.Typer(pretty_exceptions_enable=False)


@fdl_runner_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def fdl_direct_run(
    fdl_ctx: typer.Context,
    fdl_dryrun: bool = typer.Option(
        False, "--dryrun", help="Print resolved arguments without running."
    ),
    fdl_run_name: str = typer.Option("run", "--name", "-n", help="Name of the run."),
    fdl_package_cfg: str = typer.Option(
        None, "--package-cfg", "-p", help="Serialized Package config."
    ),
    fdl_config: str = typer.Argument(..., help="Serialized fdl config."),
):
    import os
    from pathlib import Path

    import fiddle as fdl

    from nemo_run.config import Partial
    from nemo_run.core.packaging.base import Packager
    from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
    from nemo_run.run.experiment import maybe_load_external_main
    from nemo_run.run.task import dryrun_fn

    if fdl_package_cfg:
        if os.path.isfile(fdl_package_cfg):
            fdl_package_cfg = Path(fdl_package_cfg).read_text()
        fdl_deser_package: fdl.Buildable = ZlibJSONSerializer().deserialize(fdl_package_cfg)
        fdl_package: Packager = fdl.build(fdl_deser_package)
        fdl_package.setup()

    if os.path.isfile(fdl_config):
        maybe_load_external_main(Path(fdl_config).parent.parent.parent)
        fdl_config = Path(fdl_config).read_text()

    fdl_buildable: fdl.Buildable = ZlibJSONSerializer().deserialize(fdl_config)
    fdl_buildable = fdl.cast(Partial, fdl_buildable)
    if fdl_dryrun:
        dryrun_fn(fdl_buildable, build=True)
        return

    fdl_fn = fdl.build(fdl_buildable)
    fdl_fn()


if __name__ == "__main__":
    fdl_runner_app()
