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

import inspect
import importlib
import yaml
from fiddle._src import config as config_lib
from fiddle._src import partial

from nemo_run.config import Config, Partial


def _config_representer(dumper, data, type_name="Config"):
    """Returns a YAML representation of `data`."""
    value = dict(data.__arguments__)

    # We put __fn_or_cls__ into __arguments__, so "__fn_or_cls__" must be a new
    # key that doesn't exist in __arguments__. It would be pretty rare for this
    # to be an issue.
    if "__fn_or_cls__" in value:
        raise ValueError(
            "It is not supported to dump objects of functions/classes "
            "that have a __fn_or_cls__ parameter."
        )

    value["_target_"] = (
        f"{inspect.getmodule(config_lib.get_callable(data)).__name__}.{config_lib.get_callable(data).__qualname__}"  # type: ignore
    )
    if type_name == "Partial":
        value["_partial_"] = True

    return dumper.represent_data(value)


def _partial_representer(dumper, data):
    return _config_representer(dumper, data, type_name="Partial")


def _function_representer(dumper, data):
    value = {
        "_target_": f"{inspect.getmodule(data).__name__}.{data.__qualname__}",  # type: ignore
        "_call_": False,
    }
    return dumper.represent_data(value)


yaml.SafeDumper.add_representer(config_lib.Config, _config_representer)
yaml.SafeDumper.add_representer(partial.Partial, _partial_representer)
yaml.SafeDumper.add_representer(Config, _config_representer)
yaml.SafeDumper.add_representer(Partial, _partial_representer)
yaml.SafeDumper.add_representer(type(lambda: ...), _function_representer)
yaml.SafeDumper.add_representer(type(object), _function_representer)

try:
    import torch  # type: ignore

    def _torch_dtype_representer(dumper, data):
        value = {
            "_target_": str(data),
            "_call_": False,
        }
        return dumper.represent_data(value)

    yaml.SafeDumper.add_representer(torch.dtype, _torch_dtype_representer)
except ModuleNotFoundError:
    ...


class YamlSerializer:
    """Serializer that handles YAML format for Config objects."""

    def serialize(self, cfg: config_lib.Buildable, stream=None) -> str:
        """Serialize a Buildable to YAML format.

        Args:
            cfg: The Buildable object to serialize.
            stream: Optional stream to write to.

        Returns:
            YAML string representation of the Buildable.
        """
        if getattr(cfg, "is_lazy", False):
            cfg = cfg.resolve()

        return yaml.safe_dump(cfg, stream=stream)

    def deserialize(
        self,
        serialized: str,
    ) -> config_lib.Buildable:
        """Deserialize a YAML string into a Buildable object.

        Args:
            serialized: YAML string to deserialize.

        Returns:
            A Buildable object (Config or Partial).
        """
        from nemo_run.cli.cli_parser import parse_cli_args
        from nemo_run.cli.lazy import dictconfig_to_dot_list
        from omegaconf import OmegaConf

        # Load the YAML into a DictConfig
        config_dict = yaml.safe_load(serialized)

        # Extract the target from _target_
        if "_target_" not in config_dict:
            raise ValueError("Missing '_target_' key in serialized YAML")

        target_path = config_dict.pop("_target_")
        is_partial = config_dict.pop("_partial_", False)

        # Import the target function/class
        module_name, func_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        target = getattr(module, func_name)

        # Convert the DictConfig to dot list arguments
        dot_list = dictconfig_to_dot_list(OmegaConf.create(config_dict))
        args = [f"{name}{op}{value}" for name, op, value in dot_list]

        # Use parse_cli_args to build the Config/Partial
        from nemo_run.config import Config, Partial

        output_type = Partial if is_partial else Config

        return parse_cli_args(target, args, output_type=output_type)
