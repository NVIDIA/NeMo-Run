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

from typing import Any, Dict
import json
import toml
from pathlib import Path
import yaml
from fiddle._src import config as config_lib

from nemo_run.core.serialization.yaml import YamlSerializer


class ConfigSerializer:
    """Serializer that supports multiple formats (YAML, JSON, TOML) for Config objects.

    This class provides a unified interface for serializing and deserializing Buildable
    objects (Config/Partial) to and from various formats. All non-YAML formats are
    converted through YAML as an intermediate format:

    For serialization:
        Buildable → YAML → JSON/TOML

    For deserialization:
        JSON/TOML → YAML → Buildable

    The conversion flow ensures consistent behavior across formats by leveraging the
    robust YAML serialization logic for all formats. This approach provides several benefits:

    1. Code reuse - The core serialization/deserialization logic is maintained in one place
    2. Consistency - All formats benefit from the same underlying implementation
    3. Maintenance - Updates to the serialization logic automatically apply to all formats

    The class provides methods for string serialization/deserialization as well as
    convenience methods for directly writing to files (dump_*).

    Examples:
        # Create a serializer
        serializer = ConfigSerializer()

        # Serialize to different formats
        yaml_str = serializer.serialize_yaml(config)
        json_str = serializer.serialize_json(config)
        toml_str = serializer.serialize_toml(config)

        # Write directly to files
        serializer.dump_yaml(config, "config.yaml")
        serializer.dump_json(config, "config.json")
        serializer.dump_toml(config, "config.toml")

        # Deserialize from different formats
        config_from_yaml = serializer.deserialize_yaml(yaml_str)
        config_from_json = serializer.deserialize_json(json_str)
        config_from_toml = serializer.deserialize_toml(toml_str)
    """

    def __init__(self):
        """Initialize the ConfigSerializer with a YamlSerializer for YAML operations."""
        self._yaml_serializer = YamlSerializer()

    def serialize_yaml(self, cfg: config_lib.Buildable, stream=None) -> str:
        """Serialize a Buildable to YAML format.

        Args:
            cfg: The Buildable object to serialize.
            stream: Optional stream to write to.

        Returns:
            YAML string representation of the Buildable.
        """
        return self._yaml_serializer.serialize(cfg, stream)

    def dump_yaml(self, cfg: config_lib.Buildable, output_path: str | Path) -> None:
        """Serialize a Buildable to YAML and write it to a file.

        Args:
            cfg: The Buildable object to serialize.
            output_path: Path to the output file (str or Path).
        """
        from pathlib import Path

        yaml_str = self.serialize_yaml(cfg)
        Path(output_path).write_text(yaml_str)

    def deserialize_yaml(self, serialized: str) -> config_lib.Buildable:
        """Deserialize a YAML string into a Buildable object.

        Args:
            serialized: YAML string to deserialize.

        Returns:
            A Buildable object (Config or Partial).
        """
        return self._yaml_serializer.deserialize(serialized)

    def serialize_json(self, cfg: config_lib.Buildable, stream=None) -> str:
        """Serialize a Buildable to JSON format.

        Args:
            cfg: The Buildable object to serialize.
            stream: Optional stream to write to.

        Returns:
            JSON string representation of the Buildable.
        """
        yaml_str = self.serialize_yaml(cfg, stream=None)
        return yaml_to_json(yaml_str)

    def dump_json(self, cfg: config_lib.Buildable, output_path: str | Path) -> None:
        """Serialize a Buildable to JSON and write it to a file.

        Args:
            cfg: The Buildable object to serialize.
            output_path: Path to the output file (str or Path).
        """
        from pathlib import Path

        json_str = self.serialize_json(cfg)
        Path(output_path).write_text(json_str)

    def deserialize_json(self, serialized: str) -> config_lib.Buildable:
        """Deserialize a JSON string into a Buildable object.

        Args:
            serialized: JSON string to deserialize.

        Returns:
            A Buildable object (Config or Partial).
        """
        yaml_str = json_to_yaml(serialized)
        return self.deserialize_yaml(yaml_str)

    def serialize_toml(self, cfg: config_lib.Buildable, stream=None) -> str:
        """Serialize a Buildable to TOML format.

        Args:
            cfg: The Buildable object to serialize.
            stream: Optional stream to write to.

        Returns:
            TOML string representation of the Buildable.
        """
        yaml_str = self.serialize_yaml(cfg, stream=None)
        return yaml_to_toml(yaml_str)

    def dump_toml(self, cfg: config_lib.Buildable, output_path: str | Path) -> None:
        """Serialize a Buildable to TOML and write it to a file.

        Args:
            cfg: The Buildable object to serialize.
            output_path: Path to the output file (str or Path).
        """
        from pathlib import Path

        toml_str = self.serialize_toml(cfg)
        Path(output_path).write_text(toml_str)

    def deserialize_toml(self, serialized: str) -> config_lib.Buildable:
        """Deserialize a TOML string into a Buildable object.

        Args:
            serialized: TOML string to deserialize.

        Returns:
            A Buildable object (Config or Partial).
        """
        yaml_str = toml_to_yaml(serialized)
        return self.deserialize_yaml(yaml_str)

    def load_yaml(self, input_path: str | Path) -> config_lib.Buildable:
        """Load a YAML file and deserialize it into a Buildable object.

        Args:
            input_path: Path to the YAML file (str or Path).

        Returns:
            A Buildable object (Config or Partial).
        """
        from pathlib import Path

        yaml_str = Path(input_path).read_text()
        return self.deserialize_yaml(yaml_str)

    def load_json(self, input_path: str | Path) -> config_lib.Buildable:
        """Load a JSON file and deserialize it into a Buildable object.

        Args:
            input_path: Path to the JSON file (str or Path).

        Returns:
            A Buildable object (Config or Partial).
        """
        from pathlib import Path

        json_str = Path(input_path).read_text()
        return self.deserialize_json(json_str)

    def load_toml(self, input_path: str | Path) -> config_lib.Buildable:
        """Load a TOML file and deserialize it into a Buildable object.

        Args:
            input_path: Path to the TOML file (str or Path).

        Returns:
            A Buildable object (Config or Partial).
        """
        from pathlib import Path

        toml_str = Path(input_path).read_text()
        return self.deserialize_toml(toml_str)

    def load(self, input_path: str | Path) -> config_lib.Buildable:
        """Load a file and deserialize it into a Buildable object based on file extension.

        Args:
            input_path: Path to the file (str or Path).

        Returns:
            A Buildable object (Config or Partial).

        Raises:
            ValueError: If the file extension is not supported.
        """
        path = Path(input_path)
        extension = path.suffix.lower()

        if extension in (".yaml", ".yml"):
            return self.load_yaml(path)
        elif extension == ".json":
            return self.load_json(path)
        elif extension == ".toml":
            return self.load_toml(path)
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}. Supported extensions are: .yaml, .yml, .json, .toml"
            )

    def dump(self, cfg: config_lib.Buildable, output_path: str | Path) -> None:
        """Serialize a Buildable to a file based on the file extension.

        Args:
            cfg: The Buildable object to serialize.
            output_path: Path to the output file (str or Path).

        Raises:
            ValueError: If the file extension is not supported.
        """
        from pathlib import Path

        path = Path(output_path)
        extension = path.suffix.lower()

        if extension in (".yaml", ".yml"):
            self.dump_yaml(cfg, path)
        elif extension == ".json":
            self.dump_json(cfg, path)
        elif extension == ".toml":
            self.dump_toml(cfg, path)
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}. Supported extensions are: .yaml, .yml, .json, .toml"
            )

    def load_dict(self, input_path: str | Path) -> dict:
        """Load a configuration file and return the raw dictionary data without resolving to a Buildable.

        Args:
            input_path: Path to the configuration file (str or Path).

        Returns:
            dict: The raw dictionary data from the configuration file.

        Raises:
            ValueError: If the file extension is not supported.
        """
        path = Path(input_path)
        extension = path.suffix.lower()

        if extension in (".yaml", ".yml"):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        elif extension == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif extension == ".toml":
            with open(path, "r") as f:
                return yaml_to_dict(toml_to_yaml(f.read()))
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}. Supported extensions are: .yaml, .yml, .json, .toml"
            )

    def dump_dict(
        self, data: dict, output_path: str | Path, format: str = None, section: str = None
    ) -> None:
        """Write a dictionary to a configuration file in the specified format.

        Args:
            data: The dictionary to serialize.
            output_path: Path to the output file (str or Path).
            format: Optional format override ('yaml', 'json', 'toml'). If None,
                    format is determined from file extension.
            section: Optional section key to extract from the data dictionary.
                    If specified, only this section will be serialized.

        Raises:
            ValueError: If the file extension or specified format is not supported.
            KeyError: If the specified section doesn't exist in the data dictionary.
        """
        from pathlib import Path

        path = Path(output_path)

        # Extract section if specified
        if section:
            if section not in data:
                raise KeyError(f"Section '{section}' not found in configuration")
            data = data[section]

        # Handle potential section specifier in output_path
        if ":" in str(path):
            # Split off any section specifier from the path
            path_str, section = str(path).split(":", 1)
            path = Path(path_str)

            # Extract the specified section from data
            if section not in data:
                raise KeyError(f"Section '{section}' not found in configuration")
            data = data[section]

        # Determine format from explicit parameter or file extension
        if format:
            file_format = format.lower()
        else:
            extension = path.suffix.lower()
            if extension in (".yaml", ".yml"):
                file_format = "yaml"
            elif extension == ".json":
                file_format = "json"
            elif extension == ".toml":
                file_format = "toml"
            else:
                raise ValueError(
                    f"Unsupported file extension: {extension}. "
                    f"Supported extensions are: .yaml, .yml, .json, .toml"
                )

        # Write to the file using the appropriate format
        with open(path, "w") as f:
            if file_format == "yaml":
                yaml.safe_dump(data, f)
            elif file_format == "json":
                json.dump(data, f, indent=2)
            elif file_format == "toml":
                import toml

                f.write(toml.dumps(data))
            else:
                raise ValueError(
                    f"Unsupported format: {file_format}. Supported formats are: yaml, json, toml"
                )


def yaml_to_dict(yaml_str: str) -> Dict[str, Any]:
    """Convert YAML string to Python dictionary."""
    return yaml.safe_load(yaml_str)


def dict_to_yaml(data: Dict[str, Any]) -> str:
    """Convert Python dictionary to YAML string."""
    return yaml.dump(data, default_flow_style=False)


def yaml_to_json(yaml_str: str) -> str:
    """Convert YAML string to JSON string."""
    data = yaml_to_dict(yaml_str)
    return json.dumps(data, indent=2)


def json_to_yaml(json_str: str) -> str:
    """Convert JSON string to YAML string."""
    data = json.loads(json_str)
    return dict_to_yaml(data)


def yaml_to_toml(yaml_str: str) -> str:
    """Convert YAML string to TOML string."""
    data = yaml_to_dict(yaml_str)
    return toml.dumps(data)


def toml_to_yaml(toml_str: str) -> str:
    """Convert TOML string to YAML string."""
    data = toml.loads(toml_str)
    return dict_to_yaml(data)


def json_to_toml(json_str: str) -> str:
    """Convert JSON string to TOML string."""
    data = json.loads(json_str)
    return toml.dumps(data)


def toml_to_json(toml_str: str) -> str:
    """Convert TOML string to JSON string."""
    data = toml.loads(toml_str)
    return json.dumps(data, indent=2)
