import weakref
from typing import TYPE_CHECKING, Dict, TypeVar


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
        self._ref_map[instance] = obj_id

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
        active_ids = set(self._ref_map.values())
        to_remove = set(self._objects.keys()) - active_ids
        for obj_id in to_remove:
            del self._objects[obj_id]


class ObjectNotFoundError(Exception):
    """Custom exception for when an object is not found in the registry."""

    pass
