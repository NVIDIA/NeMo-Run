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

import copy
import importlib.util
import os
from dataclasses import asdict, dataclass, field
from string import Template
from typing import Optional, Protocol, Union, runtime_checkable

import fiddle as fdl
from torchx.specs import Role
from typing_extensions import Self

from nemo_run.config import ConfigurableMixin, get_nemorun_home
from nemo_run.core.execution.launcher import LAUNCHER_MAP, Launcher
from nemo_run.core.packaging.base import Packager


@dataclass(kw_only=True)
class ExecutorMacros(ConfigurableMixin):
    """
    Defines macros.
    """

    HEAD_NODE_IP_VAR = "${head_node_ip_var}"
    NPROC_PER_NODE_VAR = "${nproc_per_node_var}"
    NUM_NODES_VAR = "${num_nodes_var}"
    NODE_RANK_VAR = "${node_rank_var}"

    FT_LAUNCHER_CFG_PATH_VAR = "${ft_launcher_cfg_path_var}"

    head_node_ip_var: str
    nproc_per_node_var: str
    num_nodes_var: str
    node_rank_var: str
    het_group_host_var: str
    ft_launcher_cfg_path_var: str = "FAULT_TOL_CFG_PATH"

    @staticmethod
    def group_host(index: int):
        return f"$$${{het_group_host_var}}_{index}"

    def apply(self, role: Role) -> Role:
        """
        apply applies the values to a copy the specified role and returns it.
        """

        role = copy.deepcopy(role)
        role.args = [self.substitute(arg) for arg in role.args]
        role.env = {key: self.substitute(arg) for key, arg in role.env.items()}
        return role

    def substitute(self, arg: str) -> str:
        """
        substitute applies the values to the template arg.
        """
        return Template(arg).safe_substitute(**asdict(self))


@runtime_checkable
class LogSupportedExecutor(Protocol):
    @classmethod
    def logs(cls, app_id: str, fallback_path: Optional[str]): ...


@dataclass(kw_only=True)
class Executor(ConfigurableMixin):
    """
    Base dataclass for configuration of an executor.
    This cannot be used independently but
    you can use this as the base type to register executor factories.

    See :class:`LocalExecutor` and :class:`SlurmExecutor` for examples.
    """

    packager: Packager = field(default_factory=lambda: Packager())
    launcher: Optional[Union[Launcher, str]] = None
    env_vars: dict[str, str] = field(default_factory=dict)
    retries: int = 0
    #: Set by run.Experiment
    experiment_id: Optional[str] = None

    #: Directory that will store metadata for your run.
    #: This is set automatically if using run.Experiment
    job_dir: str = ""
    experiment_dir: str = field(init=False, default="")
    _launcher_setup: bool = field(init=False, default=False)

    def info(self) -> str:
        return self.__class__.__qualname__

    def clone(self) -> Self:
        return fdl.build(self.to_config())

    def get_launcher(self) -> Launcher:
        if not self._launcher_setup:
            self._setup_launcher()
            self._launcher_setup = True

        assert self.launcher is None or isinstance(self.launcher, Launcher), (
            f"{self.info()} could not setup the launcher."
        )
        if self.launcher is None:
            self.launcher = Launcher()

        return self.launcher

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ) -> None:
        """
        This function will be called by run.Experiment
        to assign the executor for the specific experiment.
        """
        raise NotImplementedError

    def nnodes(self) -> int:
        """
        Helper function called by torchrun component
        to determine --nnodes.
        """
        raise NotImplementedError

    def nproc_per_node(self) -> int:
        """
        Helper function called by torchrun component
        to determine --nproc-per-node.
        """
        raise NotImplementedError

    def macro_values(self) -> Optional[ExecutorMacros]:
        """
        Get macro values specific to the executor.
        This allows replacing common macros with executor specific vars for node ips, etc.
        """
        return None

    def _setup_launcher(self):
        if not self.launcher:
            return None

        if isinstance(self.launcher, str):
            self.launcher = LAUNCHER_MAP[self.launcher]()

    def get_launcher_prefix(self) -> Optional[list[str]]:
        launcher = self.get_launcher()
        if launcher.nsys_profile:
            os.makedirs(os.path.join(self.job_dir, launcher.nsys_folder), exist_ok=True)
            return launcher.get_nsys_prefix(profile_dir=self.job_dir)

    def get_nsys_entrypoint(self) -> str:
        return ("nsys", "")

    def supports_launcher_transform(self) -> bool:
        return False

    def package_configs(self, *cfgs: tuple[str, str]) -> list[str]:
        filenames = []
        basepath = os.path.join(self.job_dir, "configs")
        os.makedirs(basepath, exist_ok=True)
        for name, cfg in cfgs:
            filename = os.path.join(basepath, name)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(cfg)
            filenames.append(filename)
        return filenames

    def create_job_dir(self):
        os.makedirs(self.job_dir, exist_ok=True)

    def cleanup(self, handle: str): ...


def import_executor(name: str, file_path: Optional[str] = None, call: bool = True, **kwargs) -> Executor:
    """
    Retrieves an executor instance by name from a specified or default Python file.
    The file must contain either a function or executor instance by the provided name.

    This function dynamically imports the file_path, searches for the name attr
    and returns the value corresponding to the given name, and optionally calls the value if call is True.

    This functionality allows you to define all your executors in a single file which lives separately from your codebase.
    It is similar to ~/.ssh/config and allows you to use executors across your projects without having to redefine them.

    Example:
        executor = import_executor("local", file_path="path/to/executors.py")
        executor = import_executor("gpu")  # Uses the default location of os.path.join(get_nemorun_home(), "executors.py")

    Args:
        name (str): The name of the executor to retrieve.
        file_path (Optional[str]): The path to the Python file containing the executor definitions.
            Defaults to None, in which case the default location of os.path.join(get_nemorun_home(), "executors.py") is used.

            The file_path is expected to be a string representing a file path with the following structure:
            - It should be a path to a Python file (with a .py extension).
            - The file should contain a dictionary named `EXECUTOR_MAP` that maps executor names to their corresponding instances.
            - The file can be located anywhere in the file system, but if not provided, it defaults to `get_nemorun_home()/executors.py`.
        call (bool): If True, the value from the module is called with the rest of the given kwargs.

    Returns:
        Executor: The executor instance corresponding to the given name.
    """

    if not file_path:
        file_path = os.path.join(get_nemorun_home(), "executors.py")

    spec = importlib.util.spec_from_file_location("executors", file_path)
    assert spec
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    executor_fn = getattr(module, name)
    if not callable(executor_fn):
        return executor_fn
    return executor_fn(**kwargs)  # type: ignore
