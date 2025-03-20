# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import toml
import yaml
from typing import Any, Dict


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