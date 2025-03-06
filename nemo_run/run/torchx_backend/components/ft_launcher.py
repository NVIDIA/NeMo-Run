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

import shlex
from typing import Optional

import torchx
import torchx.specs as specs
from torchx.components import dist as torchx_dist

from nemo_run.run.torchx_backend.components import torchrun


# Adapted from torchrun component
def ft_launcher(
    *script_args: str,
    script: Optional[str] = None,
    m: Optional[str] = None,
    no_python: bool = False,
    image: str = torchx.IMAGE,
    name: str = "/",
    h: Optional[str] = None,
    cpu: int = 2,
    gpu: int = 0,
    memMB: int = 1024,
    j: str = "1x2",
    env: Optional[dict[str, str]] = None,
    max_retries: int = 0,
    rdzv_port: int = 49450,
    rdzv_backend: str = "c10d",
    mounts: Optional[list[str]] = None,
    debug: bool = False,
    workload_check_interval: Optional[float] = None,
    initial_rank_heartbeat_timeout: Optional[float] = None,
    rank_heartbeat_timeout: Optional[float] = None,
    rank_termination_signal: Optional[str] = None,
    log_level: Optional[str] = None,
    max_restarts: Optional[int] = None,
) -> specs.AppDef:
    torchrun_component = torchrun.torchrun(
        *script_args,
        script=script,
        name=name,
        m=m,
        no_python=no_python,
        image="",
        h=h,
        cpu=cpu,
        gpu=gpu,
        memMB=memMB,
        j=j,
        rdzv_backend=rdzv_backend,
        rdzv_port=rdzv_port,
        env=env,
        mounts=mounts,
        debug=debug,
        max_retries=max_retries,
    )

    ft_args = []

    if any(
        map(
            lambda arg: arg is not None,
            [
                workload_check_interval,
                initial_rank_heartbeat_timeout,
                rank_heartbeat_timeout,
                rank_termination_signal,
                log_level,
                max_restarts,
            ],
        )
    ):
        if workload_check_interval:
            ft_args += [
                "--ft-param-workload_check_interval",
                str(workload_check_interval),
            ]

        if initial_rank_heartbeat_timeout:
            ft_args += [
                "--ft-param-initial_rank_heartbeat_timeout",
                str(initial_rank_heartbeat_timeout),
            ]

        if rank_heartbeat_timeout:
            ft_args += [
                "--ft-param-rank_heartbeat_timeout",
                str(rank_heartbeat_timeout),
            ]

        if rank_termination_signal:
            ft_args += ["--ft-param-rank_termination_signal", rank_termination_signal]

        if log_level:
            ft_args += ["--ft-param-log_level", log_level]

        if max_restarts:
            ft_args += ["--max-restarts", str(max_restarts)]

    else:
        ft_args = ["--ignore-missing-fault-tol-cfg"]

    ft_args = list(
        map(
            lambda arg: arg if isinstance(arg, torchx_dist._noquote) else shlex.quote(arg),
            ft_args,
        )
    )

    torchrun_component.roles[0].entrypoint = "ft_launcher"
    torchrun_component.roles[0].args = ft_args + torchrun_component.roles[0].args  # type: ignore

    return torchrun_component
