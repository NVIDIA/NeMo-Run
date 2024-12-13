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

import logging
import os
from typing import Iterator, Optional, Type, Union

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
from torchx import specs

from nemo_run.config import SCRIPTS_DIR, Partial, Script
from nemo_run.core.execution.base import Executor, FaultTolerance, Torchrun
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.serialization.yaml import YamlSerializer
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.run.torchx_backend.components import ft_launcher, torchrun

log: logging.Logger = logging.getLogger(__name__)


def package(
    name: str,
    fn_or_script: Union[Partial, Script],
    executor: Executor,
    num_replicas: int = 1,
    cpu: int = -1,
    gpu: int = -1,
    memMB: int = 1024,
    h: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    mounts: Optional[list[str]] = None,
    serialize_to_file: bool = False,
):
    default_cmd = ["-m", "nemo_run.core.runners.fdl_runner"]

    env = env or {}
    env = env | executor.env_vars
    mounts = mounts or []

    if isinstance(fn_or_script, Partial):
        args = [
            "-n",
            name,
        ]

        packager_name = f"{name}_packager"
        fn_or_script_name = f"{name}_fn_or_script"
        if serialize_to_file:
            cfgs = [
                (packager_name, _serialize(executor.packager.to_config())),
                (fn_or_script_name, _serialize(fn_or_script)),
            ]
            packager_filename, fn_or_script_filename = executor.package_configs(*cfgs)

            try:
                yaml_cfgs = [
                    (
                        f"{name}_executor.yaml",
                        _serialize(executor.to_config(), serializer_cls=YamlSerializer),
                    ),
                    (
                        f"{name}_config.yaml",
                        _serialize(fn_or_script, serializer_cls=YamlSerializer),
                    ),
                ]
                executor.package_configs(*yaml_cfgs)
            except Exception as e:
                log.warning(f"Failed saving yaml configs due to: {e}")

            args.append(fn_or_script_filename)
        else:
            args.append(_serialize(fn_or_script))

        role_args = default_cmd + args
        m = default_cmd[1]
        no_python = False
        script = None
        entrypoint = "python"
    else:
        try:
            yaml_cfgs = [
                (
                    f"{name}_executor.yaml",
                    _serialize(executor.to_config(), serializer_cls=YamlSerializer),
                ),
                (
                    f"{name}_config.yaml",
                    _serialize(
                        fdl_dc.convert_dataclasses_to_configs(fn_or_script, allow_post_init=True),
                        serializer_cls=YamlSerializer,
                    ),
                ),
            ]
            executor.package_configs(*yaml_cfgs)
        except Exception as e:
            log.warning(f"Failed saving yaml configs due to: {e}")

        args = fn_or_script.args
        role_args = fn_or_script.to_command(
            filename=os.path.join(executor.job_dir, SCRIPTS_DIR, f"{name}.sh"),
            is_local=True if isinstance(executor, LocalExecutor) else False,
        )
        m = fn_or_script.path if fn_or_script.m else None
        no_python = fn_or_script.entrypoint != "python"
        script = fn_or_script.path if not fn_or_script.m else None
        env = env | fn_or_script.env
        entrypoint = fn_or_script.entrypoint

    launcher = executor.get_launcher()
    if launcher and isinstance(launcher, Torchrun):
        app_def = torchrun.torchrun(
            *args,
            script=script,
            name=name,
            m=m,
            no_python=no_python,
            image="",
            h=h,
            cpu=cpu,
            gpu=gpu,
            memMB=memMB,
            j=f"{executor.nnodes()}x{executor.nproc_per_node()}",
            rdzv_backend=launcher.rdzv_backend,
            rdzv_port=launcher.rdzv_port,
            env=env,
            mounts=mounts,
            debug=executor.packager.debug,
            max_retries=executor.retries,
        )
    elif launcher and isinstance(launcher, FaultTolerance):
        app_def = ft_launcher.ft_launcher(
            *args,
            script=script,
            name=name,
            m=m,
            no_python=no_python,
            image="",
            h=h,
            cpu=cpu,
            gpu=gpu,
            memMB=memMB,
            j=f"{executor.nnodes()}x{executor.nproc_per_node()}",
            rdzv_backend=launcher.rdzv_backend,
            rdzv_port=launcher.rdzv_port,
            env=env,
            mounts=mounts,
            debug=executor.packager.debug,
            workload_check_interval=launcher.workload_check_interval,
            initial_rank_heartbeat_timeout=launcher.initial_rank_heartbeat_timeout,
            rank_heartbeat_timeout=launcher.rank_heartbeat_timeout,
            rank_termination_signal=launcher.rank_termination_signal,
            log_level=launcher.log_level,
            max_retries=executor.retries,
            max_restarts=launcher.max_restarts,
        )
    else:
        app_def = specs.AppDef(
            name=name,
            roles=[
                specs.Role(
                    name=name,
                    image="",
                    entrypoint=entrypoint,
                    num_replicas=num_replicas,
                    resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                    args=role_args,
                    env=env,
                    mounts=specs.parse_mounts(mounts),
                    max_retries=executor.retries,
                )
            ],
        )

    if launcher and launcher.nsys_profile:
        role = app_def.roles[0]
        nsys_prefix = executor.get_launcher_prefix()
        if nsys_prefix:
            role.args = [role.entrypoint] + role.args
            role.entrypoint = "nsys"
            role.args = nsys_prefix + role.args

    return app_def


def merge_executables(app_defs: Iterator[specs.AppDef], name: str) -> specs.AppDef:
    result = specs.AppDef(name=name, roles=[])
    for app_def in app_defs:
        result.roles.extend(app_def.roles)
    return result


def _serialize(
    buildable: fdl.Partial | fdl.Config,
    serializer_cls: Type[ZlibJSONSerializer | YamlSerializer] = ZlibJSONSerializer,
) -> str:
    serialized = serializer_cls().serialize(buildable)
    return serialized
