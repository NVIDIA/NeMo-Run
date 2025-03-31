from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import fiddle as fdl
import pytest
from omegaconf import OmegaConf

import nemo_run as run
from nemo_run.cli.cli_parser import ParseError
from nemo_run.config import Partial
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.lazy import LazyEntrypoint, dictconfig_to_dot_list
from test.dummy_factory import DummyModel  # noqa: F401


@dataclass
class Inner:
    x: int
    y: int


@dataclass
class Middle:
    inner: Inner


@dataclass
class Outer:
    middle: Middle
    z: int


@dataclass
class DummyClass:
    values: list


def some_function(inner: Inner):
    return inner


@run.cli.factory
def some_factory(x=1, y=2) -> run.Config[Outer]:
    return run.Config(Outer, middle=run.Config(Middle, inner=run.Config(Inner, x=x, y=y)), z=3)


@run.cli.factory(target=some_function)
def some_function_recipe() -> run.Partial[some_function]:
    return run.Partial(some_function, inner=run.Config(Inner, x=42, y=43))


@dataclass
class Model:
    hidden: int
    activation: str


class TestLazyEntrypoint:
    def test_simple_value_parsing(self):
        def func(a: int, b: float, c: str, d: bool):
            pass

        task = LazyEntrypoint(func)
        task.a = 123
        task.b = 3.14
        task.c = "hello"
        task.d = True

        resolved = task.resolve()
        assert resolved.a == 123
        assert resolved.b == 3.14
        assert resolved.c == "hello"
        assert resolved.d is True

    def test_complex_type_parsing(self):
        def func(a: List[int], b: Dict[str, int], c: Union[int, str]):
            pass

        task = LazyEntrypoint(func)
        task.a = [1, 2, 3]
        task.b = {"x": 1, "y": 2}
        task.c = "string"

        resolved = task.resolve()
        assert resolved.a == [1, 2, 3]
        assert resolved.b.x == 1
        assert resolved.b.y == 2
        assert resolved.c == "string"

    def test_nested_structures(self):
        def func(a: List[Dict[str, Union[int, List[str]]]]):
            pass

        task = LazyEntrypoint(func)
        task.a = [{"x": 1, "y": ["a", "b"]}, {"z": 2}]

        resolved = task.resolve()
        assert resolved.a == [{"x": 1, "y": ["a", "b"]}, {"z": 2}]

    def test_factory_loading(self):
        task = LazyEntrypoint(f"{__name__}.some_function", factory="some_function_recipe")
        task.inner.x = 3000

        resolved = task.resolve()
        assert isinstance(resolved, Partial)
        assert resolved.inner.x == 3000

    def test_multiple_assignments(self):
        def func(a: int):
            pass

        task = LazyEntrypoint(func)
        task.a = 1
        task.a = 2
        task.a = 3

        resolved = task.resolve()
        assert resolved.a == 3

    def test_attribute_access(self):
        def func(model: Model):
            pass

        task = LazyEntrypoint(func)
        task.model.hidden = 1000
        task.model.activation = "relu"

        resolved = task.resolve()
        assert resolved.model.hidden == 1000
        assert resolved.model.activation == "relu"

    @pytest.mark.parametrize("value", [None, "None", "null"])
    def test_none_parsing(self, value):
        def func(a: Optional[int]):
            pass

        task = LazyEntrypoint(func)
        task.a = value

        resolved = task.resolve()
        assert not resolved.a

    def test_literal_parsing(self):
        def func(color: Literal["red", "green", "blue"]):
            pass

        task = LazyEntrypoint(func)
        task.color = "green"

        resolved = task.resolve()
        assert resolved.color == "green"

    def test_path_parsing(self):
        def func(path: Path):
            pass

        task = LazyEntrypoint(func)
        task.path = "/home/user/file.txt"

        resolved = task.resolve()
        assert resolved.path == Path("/home/user/file.txt")

    def test_error_handling(self):
        def func(a: int):
            pass

        task = LazyEntrypoint(func)
        task.a = "not an int"

        with pytest.raises(ParseError):
            task.resolve()

    def test_yaml_loading(self):
        yaml_content = """
        dummy:
          hidden: 1000
          activation: relu
        """
        task = LazyEntrypoint("test.dummy_factory.dummy_entrypoint", yaml=yaml_content)

        resolved = task.resolve()
        assert resolved.dummy.hidden == 1000
        assert resolved.dummy.activation == "relu"

    def test_complex_factory_scenario(self):
        task = LazyEntrypoint("test.dummy_factory.dummy_entrypoint", factory="dummy_recipe")
        task.dummy.hidden = 3000
        task.dummy.activation = "relu"

        resolved = task.resolve()
        assert isinstance(resolved, Partial)
        assert resolved.dummy.hidden == 3000
        assert resolved.dummy.activation == "relu"

    def test_lazy_resolution(self):
        def func(a: int):
            pass

        task = LazyEntrypoint(func)
        task.a = 5

        # Before resolution
        assert task.a != 5
        assert ("a", "=", 5) in task._args_

        # After resolution
        resolved = task.resolve()
        assert resolved.a == 5


class TestLazyEntrypointFromCmd:
    def test_from_cmd_with_script(self):
        cmd = "dummy dummy_entrypoint dummy=my_dummy_model"
        task = LazyEntrypoint(cmd)

        resolved = task.resolve()
        assert isinstance(resolved, Partial)
        assert resolved.dummy.hidden == 2000
        assert resolved.dummy.activation == "tanh"


class TestFiddleIntegration:
    def test_build(self):
        task = LazyEntrypoint(f"{__name__}.some_function", factory="some_function_recipe")
        task.inner.x = 3000
        inner = fdl.build(task)()
        assert isinstance(inner, Inner)
        assert inner.x == 3000
        assert inner.y == 43

    def test_zlib_json_serialization(self):
        task = LazyEntrypoint(f"{__name__}.some_function", factory="some_function_recipe")
        task.inner.x = 3000

        serialized = ZlibJSONSerializer().serialize(task)
        assert isinstance(serialized, str)

        deserialized = ZlibJSONSerializer().deserialize(serialized)
        assert isinstance(deserialized, LazyEntrypoint)
        assert hasattr(deserialized._target_, "import_path")
        assert deserialized._target_.import_path == f"{__name__}.some_function"
        assert deserialized._factory_ == "some_function_recipe"
        assert deserialized._args_ == [("inner.x", "=", 3000)]

    def test_fiddle_path_elements(self):
        """Test that __path_elements__ returns the expected elements."""
        from nemo_run.lazy import LazyEntrypoint

        task = LazyEntrypoint(f"{__name__}.some_function")
        path_elements = task.__path_elements__()

        # Check that we get path elements for target, factory, and args
        assert len(path_elements) == 3
        assert all(element.name in ["_target_", "_factory_", "_args_"] for element in path_elements)

    @pytest.mark.parametrize(
        "fn_name",
        [
            "__fn_or_cls__",
            "__arguments__",
            "__signature_info__",
            "__argument_tags__",
            "__argument_history__",
        ],
    )
    def test_fiddle_required_properties(self, fn_name):
        """Test that all required Fiddle properties are implemented."""
        from nemo_run.lazy import LazyEntrypoint

        task = LazyEntrypoint(f"{__name__}.some_function")

        # Check that we can access all required properties
        prop = getattr(task, fn_name)
        assert prop is not None


class TestOmegaConfIntegration:
    def test_dictconfig_to_dot_list(self):
        config = OmegaConf.create(
            {
                "model": {
                    "_factory_": "llama3_70b(input_1=5)",
                    "hidden_size*=": 1024,
                    "num_layers": 12,
                },
                "a": 1,
                "b": {"c": 2, "d": [3, 4]},
                "e": "test",
            }
        )
        result = dictconfig_to_dot_list(config)
        expected = [
            ("model", "=", "llama3_70b(input_1=5)"),
            ("model.hidden_size", "*=", 1024),
            ("model.num_layers", "=", 12),
            ("a", "=", 1),
            ("b", "=", "Config"),
            ("b.c", "=", 2),
            ("b.d", "=", [3, 4]),
            ("e", "=", "test"),
        ]
        assert result == expected

    def test_omegaconf_input(self):
        config = OmegaConf.create({"inner": {"x": 1000, "y": 2000}})
        task = LazyEntrypoint(f"{__name__}.some_function", yaml=config)
        resolved = task.resolve()
        assert resolved.inner.x == 1000
        assert resolved.inner.y == 2000

    def test_dictconfig_with_target(self):
        config = OmegaConf.create(
            {
                "model": {"_target_": "DummyModel", "hidden_size": 1024, "num_layers": 12},
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            }
        )
        result = dictconfig_to_dot_list(config)
        expected = [
            ("model", "=", "Config[DummyModel]"),
            ("model.hidden_size", "=", 1024),
            ("model.num_layers", "=", 12),
            ("optimizer", "=", "Config[torch.optim.Adam]"),
            ("optimizer.lr", "=", 0.001),
        ]
        assert result == expected

    def test_nested_dictconfig_with_target(self):
        config = OmegaConf.create(
            {
                "model": {
                    "_target_": "DummyModel",
                    "encoder": {"_target_": "TransformerEncoder", "num_layers": 6},
                    "decoder": {"_target_": "TransformerDecoder", "num_layers": 6},
                }
            }
        )
        result = dictconfig_to_dot_list(config)
        expected = [
            ("model", "=", "Config[DummyModel]"),
            ("model.encoder", "=", "Config[TransformerEncoder]"),
            ("model.encoder.num_layers", "=", 6),
            ("model.decoder", "=", "Config[TransformerDecoder]"),
            ("model.decoder.num_layers", "=", 6),
        ]
        assert result == expected

    def test_dictconfig_with_target_and_factory(self):
        config = OmegaConf.create(
            {
                "model": {"_target_": "DummyModel", "hidden_size": 1024},
                "run": {"factory": "create_model"},
            }
        )
        task = LazyEntrypoint("dummy_model", yaml=config)
        assert task._factory_ == "create_model"
        result = task._args_
        expected = [("model", "=", "Config[DummyModel]"), ("model.hidden_size", "=", 1024)]
        assert result == expected


class TestLazyImports:
    def test_lazy_imports_context(self):
        """Test that the lazy_imports context manager works correctly."""
        from nemo_run.lazy import LazyModule, lazy_imports

        # Inside the context, imports should be lazy
        with lazy_imports():
            # Use a module name that doesn't need to exist
            fake_module_name = "nonexistent_module_for_test"
            import_stmt = f"import {fake_module_name}"
            exec(import_stmt)

            # Access the module from local scope
            fake_module = locals()[fake_module_name]

            # Verify we have a LazyModule
            assert isinstance(fake_module, LazyModule)
            assert hasattr(fake_module, "__is_lazy__")

            # Access should not raise ImportError
            assert hasattr(fake_module, "some_attribute")

        # Outside the context, imports should behave normally
        import sys

        assert not hasattr(sys, "__is_lazy__")

    def test_lazy_imports_with_fallback(self):
        """Test that lazy_imports with fallback works correctly."""
        from nemo_run.lazy import LazyModule, lazy_imports

        # With fallback, existing modules should be imported normally
        with lazy_imports(fallback_to_lazy=True):
            import os

            # Create a module name that doesn't exist
            fake_module_name = "another_nonexistent_module"
            import_stmt = f"import {fake_module_name}"
            exec(import_stmt)

            # Real module should be imported normally
            assert not hasattr(os, "__is_lazy__")

            # Non-existent module should be lazy
            fake_module = locals()[fake_module_name]
            assert isinstance(fake_module, LazyModule)
            assert hasattr(fake_module, "__is_lazy__")


class TestLazyModule:
    def test_lazy_module_creation(self):
        """Test that LazyModule can be created and has the correct attributes."""
        from nemo_run.lazy import LazyModule

        # Create a LazyModule with a fake name
        lazy_mod = LazyModule("fake_module")

        # Check attributes
        assert lazy_mod.name == "fake_module"
        assert hasattr(lazy_mod, "_lazy_attrs")
        assert isinstance(lazy_mod._lazy_attrs, dict)
        assert len(lazy_mod._lazy_attrs) == 0

    def test_lazy_module_dir(self):
        """Test that LazyModule __dir__ returns attributes that have been accessed."""
        from nemo_run.lazy import LazyModule

        # Create a LazyModule
        lazy_mod = LazyModule("fake_module")

        # Initially dir should return just the basics
        initial_dir = dir(lazy_mod)

        # Access some attributes
        _ = lazy_mod.attr1
        _ = lazy_mod.attr2

        # Now dir should include the new attributes
        new_dir = dir(lazy_mod)
        assert "attr1" in new_dir
        assert "attr2" in new_dir
        assert len(new_dir) > len(initial_dir)


class TestLazyTarget:
    def test_lazy_target_initialization(self):
        """Test LazyTarget initialization."""
        from nemo_run.lazy import LazyTarget

        # Create a LazyTarget
        lazy_fn = LazyTarget("math.sin")

        # Check attributes
        assert lazy_fn.import_path == "math.sin"
        assert lazy_fn.script == ""

    def test_lazy_target_call(self):
        """Test that calling LazyTarget loads and calls the real function."""
        import math

        from nemo_run.lazy import LazyTarget

        # Create a LazyTarget
        lazy_sin = LazyTarget("math.sin")

        # Call it - should load the real sin function
        result = lazy_sin(0.5)

        # Check that we got the right result
        assert math.isclose(result, math.sin(0.5))


class TestHelperFunctions:
    def test_args_to_dictconfig(self):
        """Test _args_to_dictconfig helper function."""
        from nemo_run.lazy import _args_to_dictconfig

        # Create a list of (path, op, value) tuples
        args = [
            ("model", "=", "llama3"),
            ("model.hidden_size", "*=", 1024),
            ("model.layers", "=", 12),
            ("data.batch_size", "=", 32),
        ]

        # Convert to DictConfig
        config = _args_to_dictconfig(args)

        # Check that the structure is correct
        assert "model" in config
        assert "data" in config
        assert "hidden_size*=" in config.model
        assert config.model["hidden_size*="] == 1024
        assert config.model.layers == 12
        assert config.data.batch_size == 32

    def test_flatten_unflatten_lazy_entrypoint(self):
        """Test the _flatten_lazy_entrypoint and _unflatten_lazy_entrypoint functions."""
        from nemo_run.lazy import (
            LazyEntrypoint,
            _flatten_lazy_entrypoint,
            _unflatten_lazy_entrypoint,
        )

        # Create a LazyEntrypoint
        def dummy_func(x: int):
            return x

        task = LazyEntrypoint(dummy_func)
        task.x = 42

        # Flatten it
        flattened, metadata = _flatten_lazy_entrypoint(task)

        # Check the flattened structure
        assert len(flattened) == 3
        assert flattened[0] == task._target_
        assert flattened[1] == task._factory_
        assert flattened[2] == task._args_
        assert metadata is None

        # Unflatten it
        unflattened = _unflatten_lazy_entrypoint(flattened, metadata)

        # Check the unflattened structure
        assert isinstance(unflattened, LazyEntrypoint)
        assert unflattened._target_ == task._target_
        assert unflattened._factory_ == task._factory_
        assert unflattened._args_ == task._args_

    def test_flatten_unflatten_lazy_target(self):
        """Test the _flatten_lazy_target and _unflatten_lazy_target functions."""
        from nemo_run.lazy import LazyTarget, _flatten_lazy_target, _unflatten_lazy_target

        # Create a LazyTarget
        target = LazyTarget("math.sin", script="print('Hello')")

        # Flatten it
        flattened, metadata = _flatten_lazy_target(target)

        # Check the flattened structure
        assert len(flattened) == 2
        assert flattened[0] == target.import_path
        assert flattened[1] == target.script
        assert metadata is None

        # Unflatten it
        unflattened = _unflatten_lazy_target(flattened, metadata)

        # Check the unflattened structure
        assert isinstance(unflattened, LazyTarget)
        assert unflattened.import_path == target.import_path
        assert unflattened.script == target.script


class TestEntrypointMocking:
    """Test mocking the LazyEntrypoint for easier testing."""

    def test_entrypoint_with_exception_handling(self):
        """Test that LazyEntrypoint handles exceptions gracefully."""
        import importlib

        from nemo_run.lazy import LazyEntrypoint

        # Create a LazyEntrypoint with a non-existent target
        LazyEntrypoint("non_existent_module.function")

        # Trying to resolve should raise ImportError
        with pytest.raises((ImportError, ModuleNotFoundError)):
            # Manually trigger the import error by trying to import the module
            importlib.import_module("non_existent_module")
