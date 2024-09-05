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

import weakref
from typing import TYPE_CHECKING, Any, Dict, TypeVar

if TYPE_CHECKING:
    from nemo_run.config import Config

_T = TypeVar("_T")


class _ConfigRegistry:
    """
    A registry for storing and retrieving configuration objects.

    This class uses weak references to track object instances and a regular dictionary to store
    configurations, automatically removing entries when instances are garbage collected.

    Attributes:
        _objects (Dict[int, Config]): A dictionary to store configurations.
        _ref_map (weakref.WeakKeyDictionary): A weak key dictionary to map instances to their IDs.
    """

    def __init__(self):
        """Initializes the ConfigRegistry with empty dictionaries."""
        self._objects: Dict[int, "Config[_T]"] = {}
        self._ref_map = weakref.WeakKeyDictionary()
        self._strong_ref_map: Dict[Any, int] = {}  # New dictionary for non-weakref objects

    def register(self, instance: _T, cfg: "Config[_T]") -> None:
        """
        Registers a configuration for a given instance.

        Args:
            instance (_T): The instance to associate with the configuration.
            cfg (Config[_T]): The configuration object to register.

        Returns:
            None

        Example:
            >>> registry = ConfigRegistry()
            >>> cfg = SomeConfig()
            >>> instance = SomeClass()
            >>> registry.register(instance, cfg)
        """
        obj_id = id(instance)
        self._objects[obj_id] = cfg
        if self._is_weakref_able(instance):
            self._ref_map[instance] = obj_id
        else:
            self._strong_ref_map[instance] = obj_id

    def _is_weakref_able(self, obj: Any) -> bool:
        try:
            weakref.ref(obj)
            return True
        except TypeError:
            return False

    def get(self, obj: _T) -> "Config[_T]":
        """
        Retrieves the configuration for a given object.

        Args:
            obj (_T): The object to retrieve the configuration for.

        Returns:
            Config[_T]: The configuration associated with the object.

        Raises:
            ObjectNotFoundError: If no configuration is found for the given object.

        Example:
            >>> registry = ConfigRegistry()
            >>> instance = SomeClass()
            >>> cfg = registry.get(instance)
        """
        obj_id = id(obj)
        cfg = self._objects.get(obj_id)
        if cfg is None:
            raise ObjectNotFoundError(
                f"No configuration found for {obj} "
                f"with id {obj_id}. Total configs in registry: {len(self._objects)}."
            )
        return cfg

    def get_by_id(self, obj_id: int) -> "Config[_T]":
        """
        Retrieves the configuration for a given object id.

        Args:
            obj_id (int): The id of the object to retrieve the configuration for.

        Returns:
            Config[_T]: The configuration associated with the object id.

        Raises:
            ObjectNotFoundError: If no configuration is found for the given object id.

        Example:
            >>> registry = ConfigRegistry()
            >>> obj = SomeClass()
            >>> obj_id = id(obj)
            >>> cfg = registry.get_by_id(obj_id)
        """
        cfg = self._objects.get(obj_id)
        if cfg is None:
            raise ObjectNotFoundError(f"No config found for id {obj_id}")
        return cfg

    def __len__(self):
        """
        Returns the number of configurations stored in the registry.

        Returns:
            int: The number of configurations in the registry.

        Example:
            >>> registry = ConfigRegistry()
            >>> len(registry)
            0
        """
        return len(self._objects)

    def __contains__(self, obj) -> bool:
        """
        Checks if a configuration for the given object exists in the registry.

        Args:
            obj (_T): The object to check for.

        Returns:
            bool: True if a configuration for the object exists, False otherwise.

        Example:
            >>> registry = _ConfigRegistry()
            >>> instance = SomeClass()
            >>> registry.register(instance, SomeConfig())
            >>> instance in registry
            True
            >>> other_instance = SomeClass()
            >>> other_instance in registry
            False
        """
        return id(obj) in self._objects

    def cleanup(self):
        """
        Removes configurations for instances that have been garbage collected.

        This method should be called periodically to clean up the registry.

        Example:
            >>> registry = ConfigRegistry()
            >>> registry.cleanup()
        """
        active_ids = set(self._ref_map.values()) | set(self._strong_ref_map.values())
        to_remove = set(self._objects.keys()) - active_ids
        for obj_id in to_remove:
            del self._objects[obj_id]


class ObjectNotFoundError(Exception):
    """Custom exception for when an object is not found in the registry."""

    pass
