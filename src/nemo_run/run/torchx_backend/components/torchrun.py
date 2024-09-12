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
import random
import shlex
from typing import Optional

import torchx
import torchx.specs as specs
from torchx.components import dist as torchx_dist

from nemo_run.core.execution.base import ExecutorMacros

_TORCH_DEBUG_FLAGS: dict[str, str] = {
    "CUDA_LAUNCH_BLOCKING": "1",
    "NCCL_DESYNC_DEBUG": "1",
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
}
"""
These are commonly set environment variables to debug PyTorch execution.

* ``CUDA_LAUNCH_BLOCKING``: Read more `here <https://docs.nvidia.com/cuda/cuda-gdb/index.html#set-cuda-launch-blocking>`__.
* ``NCCL_DESYNC_DEBUG``
* ``TORCH_DISTRIBUTED_DEBUG``: Read more `here <https://pytorch.org/docs/stable/distributed.html#torch-distributed-debug>`__.
* ``TORCH_SHOW_CPP_STACKTRACES``: Read more `here <https://pytorch.org/docs/stable/distributed.html#torch-distributed-debug>`__.
"""


# Adapted from https://github.com/pytorch/torchx/blob/main/torchx/components/dist.py
def torchrun(
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
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
    to launch and coordinate PyTorch worker processes. Defaults to using ``c10d`` rendezvous backend
    on rendezvous_endpoint ``$rank_0_host:$rdzv_port``. Note that ``rdzv_port`` parameter is ignored
    when running on single node, and instead we use port 0 which instructs torchelastic to chose
    a free random port on the host.

    Note: (cpu, gpu, memMB) parameters are mutually exclusive with ``h`` (named resource) where
          ``h`` takes precedence if specified for setting resource requirements.
          See `registering named resources <https://pytorch.org/torchx/latest/advanced.html#registering-named-resources>`_.

    Args:
        script_args: arguments to the main module
        script: script or binary to run within the image
        m: the python module path to run
        image: image (e.g. docker)
        name: job name override in the following format: ``{experimentname}/{runname}`` or ``{experimentname}/`` or ``/{runname}`` or ``{runname}``.
            Uses the script or module name if ``{runname}`` not specified.
        cpu: number of cpus per replica
        gpu: number of gpus per replica
        memMB: cpu memory in MB per replica
        h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
        j: [{min_nnodes}:]{nnodes}x{nproc_per_node}, for gpu hosts, nproc_per_node must not exceed num gpus
        env: environment variables to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3)
        max_retries: the number of scheduler retries allowed
        rdzv_port: the port on rank0's host to use for hosting the c10d store used for rendezvous.
                   Only takes effect when running multi-node. When running single node, this parameter
                   is ignored and a random free port is chosen.
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                See scheduler documentation for more info.
        debug: whether to run with preset debug flags enabled
    """
    if (script is None) == (m is None):
        raise ValueError("exactly one of --script and -m must be specified")

    # nnodes: number of nodes or minimum nodes for elastic launch
    # max_nnodes: maximum number of nodes for elastic launch
    # nproc_per_node: number of processes on each node
    min_nnodes, max_nnodes, nproc_per_node, nnodes_rep = torchx_dist.parse_nnodes(j)

    if max_nnodes == 1:
        # using port 0 makes elastic chose a free random port which is ok
        # for single-node jobs since all workers run under a single agent
        # When nnodes is 0 and max_nnodes is 1, it's still a single node job
        # but pending until the resources become available
        rdzv_endpoint = "localhost:0"
        num_nodes = nnodes_rep
        nproc_per_node = str(nproc_per_node)
        node_rank = "0"
    else:
        # for multi-node, rely on the rank0_env environment variable set by
        # the schedulers (see scheduler implementation for the actual env var this maps to)
        # some schedulers (e.g. aws batch) make the rank0's ip-addr available on all BUT on rank0
        # so default to "localhost" if the env var is not set or is empty
        # rdzv_endpoint bash resolves to something to the effect of
        # ${TORCHX_RANK0_HOST:=localhost}:29500
        # use $$ in the prefix to escape the '$' literal (rather than a string Template substitution argument)
        rdzv_endpoint = torchx_dist._noquote(f"$${ExecutorMacros.HEAD_NODE_IP_VAR}:{rdzv_port}")
        num_nodes = torchx_dist._noquote(f"$${ExecutorMacros.NUM_NODES_VAR}")
        nproc_per_node = str(nproc_per_node)
        node_rank = torchx_dist._noquote(f"$${ExecutorMacros.NODE_RANK_VAR}")

    if env is None:
        env = {}

    env.setdefault("LOGLEVEL", os.getenv("LOGLEVEL", "INFO"))
    if debug:
        env.update(_TORCH_DEBUG_FLAGS)

    cmd = [
        "--rdzv-backend",
        rdzv_backend,
        "--rdzv-endpoint",
        rdzv_endpoint,
        "--rdzv-id",
        f"{random.randint(1, 10000)}",
        "--nnodes",
        num_nodes,
        "--nproc-per-node",
        nproc_per_node,
        "--node-rank",
        node_rank,
        "--tee",
        "3",
        # "--role",
        # "",
    ]
    if script is not None:
        if no_python:
            cmd += ["--no-python"]
        cmd += [script]
    elif m is not None:
        cmd += ["-m", m]
    cmd += script_args
    return specs.AppDef(
        name=name,
        roles=[
            specs.Role(
                name=name,
                image=image,
                entrypoint="torchrun",
                num_replicas=1,
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                args=list(
                    map(
                        lambda arg: arg
                        if isinstance(arg, torchx_dist._noquote)
                        else shlex.quote(arg),
                        cmd,
                    )
                ),
                env=env,
                port_map={
                    "c10d": rdzv_port,
                },
                max_retries=max_retries,
                mounts=specs.parse_mounts(mounts) if mounts else [],
            )
        ],
    )
