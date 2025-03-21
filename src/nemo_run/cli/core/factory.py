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
import functools
from typing import Callable, Optional, Type, TypeVar, get_args, overload
from dataclasses import dataclass

import catalogue
import fiddle as fdl

from nemo_run.cli.cache import cache_factory_metadata
from nemo_run.config import Config, Partial, get_type_namespace, get_underlying_types
from nemo_run.cli.core.workspace import load_entrypoints, load_workspace

F = TypeVar("F", bound=Callable[..., object])
T = TypeVar("T")
POPULATE_CACHE = False  # This should be set appropriately based on your application logic
DEFAULT_NAME = "default"  # Constants from original file


@dataclass
class FactoryRegistration:
    namespace: str
    name: str


def register_factory(
    fn: Callable,
    target: Optional[Type],
    name: Optional[str] = None,
    namespace: Optional[str] = None,
    target_arg: Optional[str] = None,
    is_target_default: bool = False,
) -> FactoryRegistration:
    _name: str = name or fn.__name__

    if namespace:
        _namespace = namespace
    elif target:
        _namespace = get_type_namespace(target)
    else:
        _return_type = get_return_type(fn)
        if isinstance(_return_type, (Config, Partial)) or (
            hasattr(_return_type, "__origin__")
            and issubclass(_return_type.__origin__, (Config, Partial))
        ):
            _return_type = get_args(_return_type)[0]

        _namespace = get_type_namespace(_return_type)

    if target_arg:
        assert target, "target_arg cannot be used without specifying a parent."
        _namespace = f"{_namespace}.{target_arg}"

    catalogue._set((_namespace, _name), fn)
    if is_target_default:
        catalogue._set((_namespace, DEFAULT_NAME), fn)

    return FactoryRegistration(namespace=_namespace, name=_name)


def get_return_type(fn: Callable) -> Type:
    return_type = inspect.signature(fn).return_annotation
    if return_type is inspect.Signature.empty:
        raise TypeError(f"Missing return type annotation for function '{fn}'")

    # Handle forward references
    if isinstance(return_type, str):
        return_type = eval(return_type)

    return return_type