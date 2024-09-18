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

from nemo_run import cli
from nemo_run.api import autoconvert, dryrun_fn
from nemo_run.config import Config, ConfigurableMixin, Partial, Script
from nemo_run.core.execution.base import Executor, ExecutorMacros, FaultTolerance, Torchrun
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.execution.skypilot import SkypilotExecutor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.git import GitArchivePackager
from nemo_run.core.packaging.pattern import PatternPackager
from nemo_run.core.tunnel.client import LocalTunnel, SSHTunnel
from nemo_run.devspace.base import DevSpace
from nemo_run.help import help
from nemo_run.lazy import LazyEntrypoint, lazy_imports
from nemo_run.run.api import run
from nemo_run.run.experiment import Experiment
from nemo_run.run.plugin import ExperimentPlugin as Plugin

__all__ = [
    "autoconvert",
    "cli",
    "dryrun_fn",
    "lazy_imports",
    "LazyEntrypoint",
    "Config",
    "ConfigurableMixin",
    "DevSpace",
    "DockerExecutor",
    "dryrun_fn",
    "Executor",
    "ExecutorMacros",
    "Experiment",
    "FaultTolerance",
    "GitArchivePackager",
    "PatternPackager",
    "help",
    "LocalExecutor",
    "LocalTunnel",
    "Packager",
    "Partial",
    "Plugin",
    "run",
    "Script",
    "SkypilotExecutor",
    "SlurmExecutor",
    "SSHTunnel",
    "Torchrun",
]

try:
    from nemo_run._version import __version__
except Exception:
    __version__ = "0.0.1"
