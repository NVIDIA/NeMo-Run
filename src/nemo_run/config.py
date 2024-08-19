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

from __future__ import annotations

import copy
import dataclasses
import inspect
import os
import re
import sys
import typing
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Generic, Optional, Type, TypeVar, Union, get_args

import catalogue
import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import graphviz
from fiddle._src import config, daglish, daglish_extensions
from fiddle._src.casting import register_supported_cast
from fiddle._src.config import TypeOrCallableProducingT
from fiddle.graphviz import render, render_diff
from typing_extensions import Annotated, ParamSpec, Self

import nemo_run.exceptions as run_exceptions
from nemo_run.core.lark_parser import parse_args

Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")

_T = TypeVar("_T")
_BuildableT = TypeVar("_BuildableT", bound=fdl.Buildable)

RECURSIVE_TYPES = (typing.Union, typing.Optional)
NEMORUN_HOME = os.environ.get("NEMORUN_HOME", os.path.expanduser("~/.nemo_run"))


def get_type_namespace(typ: Type | Callable) -> str:
    """
    Get the namespace of a type or callable.

    Args:
        typ: The type or callable to get the namespace for.

    Returns:
        A string representing the namespace of the type or callable.

    Examples:
        >>> class MyClass:
        ...     pass
        >>> get_type_namespace(MyClass)
        'your_module.MyClass'
    """
    module = typ.__module__
    if module == "__main__":
        # Get the filename without extension
        main_module = sys.modules["__main__"]
        filename = os.path.basename(main_module.__file__)
        module = os.path.splitext(filename)[0]

    return f"{module}.{typ.__qualname__}"


def get_underlying_types(type_hint: typing.Any) -> typing.Set[typing.Type]:
    if isinstance(type_hint, typing._GenericAlias):  # type: ignore
        if str(type_hint).startswith("typing.Annotated"):
            origin = type_hint.__origin__.__origin__
        else:
            origin = type_hint.__origin__
        if origin in RECURSIVE_TYPES:
            types = set()
            for arg in type_hint.__args__:
                types.update(get_underlying_types(arg))
            return types
    return {type_hint}


def from_dict(raw_data: dict | list | str | float | int | bool, cls: Type[_T]) -> _T:
    if isinstance(raw_data, dict):
        underlying_types = get_underlying_types(cls)
        underlying_types = [tp for tp in underlying_types if tp is not type(None)]
        assert (
            len(underlying_types) == 1
        ), f"Unable to load {cls}. Nested union types are not currently supported."
        cls = underlying_types[0]  # type: ignore

    if dataclasses.is_dataclass(cls):
        fields_dict = {
            f.name: from_dict(raw_data.get(f.name), f.type)  # type: ignore
            for f in dataclasses.fields(cls)
            if f.init
        }
        return cls(**fields_dict)  # type: ignore
    elif isinstance(raw_data, list):
        return [from_dict(item, cls.__args__[0]) for item in raw_data]  # type: ignore
    else:
        return raw_data  # type: ignore


def set_value(cfg: config.Buildable, key: str, value: Any) -> None:
    """Set an attribute's value.

    Args:
      cfg: A `fdl.Buildable` whose attribute is to be overridden.
      assignment: String representing attribute's override expression. Of the form
        `attribute=value`.
    """
    *parents, last = _parse_path(key)

    walk = typing.cast(Any, cfg)
    try:
        for parent in parents:
            walk = parent.follow(walk)
    except Exception as e:
        raise run_exceptions.SetValueError(f'Invalid path "{key}".') from e

    try:
        if isinstance(last, daglish.Attr):
            setattr(walk, last.name, value)
        elif isinstance(last, daglish.Key):
            walk[last.key] = value
        else:
            raise run_exceptions.SetValueError(f"Unexpected path element {last}.")
    except Exception as e:
        raise run_exceptions.SetValueError(f'Could not set "{key}" to "{value}".') from e


class _CloneAndFNMixin:
    def clone(self):
        """Returns a deep clone of the object."""
        return copy.deepcopy(self)

    def walk(self: _BuildableT, **kwargs) -> _BuildableT:  # type: ignore
        """
        Recursively applies a transformation function to attributes within the configuration object
        and its children that match the keys provided in kwargs. Attributes not listed in kwargs
        are not modified.

        Args:
            **kwargs (dict): A dictionary where keys are attribute names and values are functions
                             that take the current attribute value and return a new value.

        Returns
        -------
            Config: A new Config instance with selectively modified attributes.

        Examples
        --------
            >>> config = Config(model=ModelConfig(seq_length=128))
            >>> new_config = config.walk(seq_length=lambda cfg: cfg.seq_length * 2)
            >>> new_config.model.seq_length
            256
        """
        return _try_set_all(self, _walk=True, **kwargs)

    def broadcast(self: _BuildableT, **kwargs) -> _BuildableT:  # type: ignore
        """
        Sets new values to attributes within the configuration object and its children that match
        the keys provided in kwargs. Attributes not listed in kwargs are not modified.

        Args:
            **kwargs (dict): A dictionary where keys are attribute names and values are the new
                             values to be set.

        Returns
        -------
            Config: A new Config instance with selectively updated attributes.

        Examples
        --------
            >>> config = Config(model=ModelConfig(tensor_model_parallel_size=1))
            >>> new_config = config.broadcast(tensor_model_parallel_size=2)
            >>> new_config.model.tensor_model_parallel_size
            2
        """
        return _try_set_all(self, **kwargs)


class _VisualizeMixin:
    def visualize(self, **kwargs) -> graphviz.Graph:
        return render(self, **kwargs)

    def diff(self, old: Self):
        return render_diff(old=old, new=self, trim=True)

    def save_config_img(self, path_str: str) -> None:
        """
        Saves the configuration to a file.

        Args:
            path (str): The file path where the configuration should be saved.
            fdl_fn (Partial): The function descriptor library function to save.

        Example:
            >>> save_config_img("path/to/dir", some_fdl_fn)
        """
        path: Path = Path(path_str)

        if not path.suffix:
            path = path / "config.png"
        elif path.suffix != ".png":
            raise ValueError("The file extension must be .png")

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            f.write(self.visualize().pipe("png"))

    def _repr_svg_(self):
        """Special method used by Jupyter to represent an object as SVG.

        Returns
        -------
            str: SVG representation of the Config object if Graphviz can render it.
            If Graphviz rendering fails or is not available, it returns None.
        """
        try:
            # Attempt to render using Graphviz and return SVG representation
            return self.visualize().pipe(format="svg").decode("utf-8")
        except Exception as e:
            # If rendering fails, log the exception or handle it as needed
            print(f"Graphviz rendering failed: {e}")
            return self.__repr__()


class Config(Generic[_T], fdl.Config[_T], _CloneAndFNMixin, _VisualizeMixin):
    """
    Wrapper around fdl.Config with nemo_run specific functionality.
    See `fdl.Config <https://fiddle.readthedocs.io/en/latest/api_reference/core.html#config>`_ for more.
    """

    def __init__(
        self,
        fn_or_cls: Union[fdl.Buildable[_T], TypeOrCallableProducingT[_T]],
        *args,
        bind_args: bool = True,
        **kwargs,
    ):
        new_kwargs = kwargs
        if bind_args and not isinstance(fn_or_cls, fdl.Buildable):
            try:
                new_kwargs = _bind_args(fn_or_cls, *args, **kwargs)
            except Exception:
                new_kwargs = kwargs

        super().__init__(fn_or_cls, *args, **new_kwargs)


class Partial(Generic[_T], fdl.Partial[_T], _CloneAndFNMixin, _VisualizeMixin):
    """
    Wrapper around fdl.Partial with nemo_run specific functionality.
    See `fdl.Partial <https://fiddle.readthedocs.io/en/latest/api_reference/core.html#partial>`_ for more.
    """

    def __init__(
        self,
        fn_or_cls: Union[fdl.Buildable[_T], TypeOrCallableProducingT[_T]],
        *args,
        bind_args: bool = True,
        **kwargs,
    ):
        new_kwargs = kwargs
        if bind_args and not isinstance(fn_or_cls, fdl.Buildable):
            try:
                new_kwargs = _bind_args(fn_or_cls, **kwargs)
            except Exception:
                new_kwargs = kwargs

        super().__init__(fn_or_cls, *args, **new_kwargs)


register_supported_cast(fdl.Config, Config)
register_supported_cast(fdl.Partial, Partial)
register_supported_cast(Config, Config)
register_supported_cast(Partial, Partial)


@dataclasses.dataclass
class Script:
    """
    Dataclass to configure raw scripts.

    Examples:

    .. code-block:: python

        file_based_script = run.Script("./scripts/echo.sh")

        inline_script = run.Script(
            inline=\"\"\"
        env
        echo "Hello 1"
        echo "Hello 2"
        \"\"\"
        )

    """

    #: Path to your script
    path: str = ""
    #: Inline contents of the script. Either path or inline needs to be set.
    inline: str = ""
    #: Args to pass to your scripts, only applicable when path is set.
    args: list[str] = dataclasses.field(default_factory=list)
    #: Environment variables to set when running the script.
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Shell to use, defaults to bash.
    shell: str = "bash"
    #: Whether to use a python binary to run your script, set shell="" to activate.
    python: str = "python"
    #: Whether to use ``python -m`` when executing via python.
    m: bool = False

    def __post_init__(self):
        assert self.path or self.inline
        assert self.shell or (self.m and self.python), "Need to specify shell or python"

    def get_name(self):
        if self.inline:
            name = self.inline.strip()[:10]
            return re.sub("[^0-9a-zA-Z]+", "_", name)
        else:
            return os.path.basename(self.path)

    def to_command(self, with_entrypoint: bool = False) -> list[str]:
        if self.inline:
            inline = self.inline.replace('"', '\\"')
            cmd = ["-c", f'"{inline}"']
            if with_entrypoint:
                cmd = [self.shell] + cmd

            return cmd

        args = [self.path] + self.args
        if self.m:
            cmd = ["-m"] + args
        else:
            cmd = args

        if with_entrypoint:
            if self.shell:
                cmd = [self.shell] + cmd
            elif self.m:
                cmd = [self.python] + cmd
            else:
                raise ValueError(
                    "Cannot use with_entrypoint=True without specifying shell or python"
                )

        return cmd


# A type alias for an optional type that is annotated with a Config.
# This is useful for when you want to specify a type is Optional but
# always want to provide a default config.
OptionalDefaultConfig = Annotated[Optional[_T], Config[_T]]
OptionalDefaultPartial = Annotated[Optional[_T], Partial[_T]]


def _parse_path(path: str) -> daglish.Path:
    """Parses a path into a list of either attributes or index lookups."""
    if not path.startswith("[") and not path.startswith("."):
        path = f".{path}"  # Add a leading `.` to make parsing work properly.

    return daglish_extensions.parse_path(path)


def _bind_args(
    fn_or_cls: TypeOrCallableProducingT,
    *fn_args: fdl.Config | str | Callable,
    **fn_kwargs: fdl.Config | str | Callable,
) -> dict[str, fdl.Config | str | Callable]:
    sig = inspect.signature(fn_or_cls)
    params = sig.parameters

    if set(fn_kwargs) > set(params):
        raise TypeError(
            f"{set(fn_kwargs) - set(params)} does not exist as args in {fn_or_cls.__module__}:{fn_or_cls.__name__}. Please remove them."
        )

    final_args = _construct_args(fn_or_cls, params, fn_kwargs)
    final_args = fn_kwargs | final_args
    sig.bind(*fn_args, **final_args)
    return final_args


def _construct_args(
    fn_or_cls: TypeOrCallableProducingT,
    params: MappingProxyType[str, inspect.Parameter],
    kwargs: dict[str, fdl.Config | str | Callable],
):
    final_args = {}

    primitive = [str, float, int, bool, bytes]
    primitive.extend([Optional[t] for t in primitive])

    from nemo_run.api import AutoConfigProtocol

    def _get_from_registry(val, annotation, name):
        if catalogue.check_exists(
            get_type_namespace(annotation),
            val,
        ):
            return catalogue._get(
                (
                    get_type_namespace(annotation),
                    val,
                )
            )

        namespace = f"{get_type_namespace(fn_or_cls)}.{name}"
        if catalogue.check_exists(namespace, val):
            return catalogue._get((namespace, val))

        return catalogue._get((str(annotation), val))

    for name, parameter in params.items():
        is_primitive = parameter.annotation in primitive
        arg = kwargs.get(name, None)

        if arg:
            if is_primitive:
                final_args[name] = arg
            else:
                if isinstance(arg, str):
                    types = get_underlying_types(parameter.annotation)
                    for t in types:
                        try:
                            arg = _get_from_registry(arg, t, name=name)
                            break
                        except catalogue.RegistryError:
                            ...

                if isinstance(arg, (Config, Partial, fdl.ArgFactory)):
                    # TODO: Check validity
                    final_args[name] = arg
                elif callable(arg):
                    if _is_config_or_partial_factory(arg) or isinstance(arg, AutoConfigProtocol):
                        final_args[name] = arg()
                    else:
                        final_args[name] = Partial(arg)
                else:
                    if dataclasses.is_dataclass(arg):
                        arg = fdl.cast(
                            Config,
                            fdl_dc.convert_dataclasses_to_configs(arg, allow_post_init=True),
                        )
                    final_args[name] = arg
        elif str(parameter.annotation).startswith("typing.Annotated"):
            args = get_args(parameter.annotation)
            if str(args[0]).startswith("typing.Optional") and len(args) > 1:
                cfg_type = get_args(args[0])[0]
                buildable = args[1].__origin__
                if issubclass(buildable, fdl.Buildable):
                    final_args[name] = buildable(cfg_type)

    return final_args


ConfigT = TypeVar("ConfigT", Config, Partial)


def _try_set_all(config: _BuildableT, _walk: bool = False, **kwargs) -> _BuildableT:
    for key, val in kwargs.items():
        if hasattr(config, key):
            _val = val(config) if _walk else val
            setattr(config, key, _val)

        for attr_name in dir(config):
            try:
                if hasattr(config, attr_name):
                    attr = getattr(config, attr_name)
                    if isinstance(attr, (fdl.Config, fdl.Partial)):
                        _try_set_all(attr, _walk=_walk, **kwargs)
            except ValueError:
                pass

    return config


def enable_overrides(fn_or_cfg: Any, prefix: str = ""):
    """
    Enables hydra style overrides via the CLI for a configured function in a script.

    This works on either configured objects or on objects which allow setattr/getattr on dot paths.
    An optional prefix specifies the namespace for the passed in object.

    .. note::
        This is experimental, and prone to errors.

    .. note::
        Only to be used inside a main function in a python script because it gets the arguments from sys.argv.

    Examples
    --------
    .. code-block:: python

        # in script
        if __name__ == "__main__":
            fn = run.Partial(
                add_object,
                obj_1="commonly_used_object",
                obj_2=run.Config(SomeObject, value_1=10, value_2=20, value_3=30),
            )
            run.enable_overrides(fn)
            executor = run.LocalExecutor()
            run.enable_overrides(executor, prefix="executor")

        # from cli
        >>> python script.py obj_1.value_1=500 executor.launcher="torchrun"

    """
    possible_overrides = sys.argv[1:]
    parsed_args, parsed_overrides = parse_args(possible_overrides)
    parsed_overrides = parsed_args | parsed_overrides
    for key, value in parsed_overrides.items():
        try:
            if prefix:
                if key.startswith(f"{prefix}."):
                    key = key.partition(f"{prefix}.")[-1]
                    set_value(fn_or_cfg, key, value)
            else:
                set_value(fn_or_cfg, key, value)
        except run_exceptions.SetValueError:
            ...


def _is_config_or_partial_factory(func):
    sig = inspect.signature(func)
    return_annotation = sig.return_annotation
    has_only_kwargs = all(param.kind == param.KEYWORD_ONLY for param in sig.parameters.values())
    return (
        has_only_kwargs
        and return_annotation in (Config, Partial)
        or (
            hasattr(return_annotation, "__origin__")
            and return_annotation.__origin__ in (Config, Partial)
        )
    )
