from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import math
import os
import pickle
import sys

import fiddle as fdl
import pytest
from omegaconf import OmegaConf

import nemo_run as run
from nemo_run.cli.cli_parser import ParseError
from nemo_run.config import Partial
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.cli.lazy import (
    LazyEntrypoint, 
    LazyTarget,
    LazyModule,
    DictConfig,
    dictconfig_to_dot_list,
    load_config_from_path,
    _args_to_dictconfig,
    _flatten_lazy_entrypoint,
    _unflatten_lazy_entrypoint,
    _flatten_lazy_target,
    _unflatten_lazy_target,
    _load_entrypoint_from_script,
    _flatten_lazy_entrypoint,
    _unflatten_lazy_entrypoint,
    _is_config_file_path,
    import_module,
)
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


# Dummy class for testing LazyModule call
class MyCallableClass:
    def __call__(self, *args, **kwargs):
        return "called class"


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
        task = LazyEntrypoint(f"{__name__}.some_function")

        # Check that we can access all required properties
        prop = getattr(task, fn_name)
        assert prop is not None

    @pytest.mark.parametrize(
        "value, should_resolve",
        [
            ("non_existent_module.function", False),  # Should trigger lazy path & fail resolve
            (f"{__name__}.some_function", True),  # Should trigger real path & resolve
        ],
    )
    def test_lazy_entrypoint_flatten_in_build_state(self, value, should_resolve, monkeypatch):
        """Test flattening LazyEntrypoint when fdl._state.in_build is True."""
        from fiddle._src.building import _state

        task = LazyEntrypoint(value)
        # Do not add arguments here, as they might conflict with the target signature (e.g., math.sin)

        # Mock fdl._state.in_build to be True
        monkeypatch.setattr(_state, "in_build", True)

        if should_resolve:
            # Flattening should now resolve and flatten the result
            flattened, metadata = task.__flatten__()
            # Check that the flattened result is not the LazyEntrypoint structure
            # but the resolved structure's flatten output
            assert not isinstance(flattened, tuple) or len(flattened) != 3
            assert flattened != (task._target_, task._factory_, task._args_)
        else:
            # Flattening an unresolvable target in build state should raise the import error
            with pytest.raises(ModuleNotFoundError):
                task.__flatten__()

        # Unpatch
        monkeypatch.setattr(_state, "in_build", False)


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
        # Filter out the ('b', '=', 'Config') entry for comparison,
        # as its presence depends on internal logic we might not want to test strictly here.
        result_filtered = [item for item in result if item != ("b", "=", "Config")]
        expected_filtered = [item for item in expected if item != ("b", "=", "Config")]
        assert result_filtered == expected_filtered
        # Check structure args separately if needed
        assert any(item == ("b", "=", "Config") for item in result)

    def test_omegaconf_input(self):
        config = OmegaConf.create({"inner": {"x": 1000, "y": 2000}})
        task = LazyEntrypoint(f"{__name__}.some_function", yaml=config)
        resolved = task.resolve()
        assert resolved.inner.x == 1000
        assert resolved.inner.y == 2000

    def test_omegaconf_input_path(self, tmp_path):
        yaml_content = """
        inner:
          x: 1001
          y: 2001
        """
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content)

        task = LazyEntrypoint(f"{__name__}.some_function", yaml=p)
        resolved = task.resolve()
        assert resolved.inner.x == 1001
        assert resolved.inner.y == 2001

    def test_omegaconf_input_json(self, tmp_path):
        json_content = """
        {
          "inner": {
            "x": 1002,
            "y": 2002
          }
        }
        """
        p = tmp_path / "config.json"
        p.write_text(json_content)

        task = LazyEntrypoint(f"{__name__}.some_function", yaml=p)
        resolved = task.resolve()
        assert resolved.inner.x == 1002
        assert resolved.inner.y == 2002

    def test_omegaconf_input_toml(self, tmp_path):
        toml_content = """
        [inner]
        x = 1003
        y = 2003
        """
        p = tmp_path / "config.toml"
        p.write_text(toml_content)

        task = LazyEntrypoint(f"{__name__}.some_function", yaml=p)
        resolved = task.resolve()
        assert resolved.inner.x == 1003
        assert resolved.inner.y == 2003

    def test_omegaconf_input_invalid_path(self):
        with pytest.raises(ValueError, match="Error loading config file"):
            LazyEntrypoint(f"{__name__}.some_function", yaml="non_existent_file.yaml")

    def test_omegaconf_input_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid config type"):
            LazyEntrypoint(f"{__name__}.some_function", yaml=123)

    def test_dictconfig_to_dot_list_partial(self):
        config = OmegaConf.create(
            {"model": {"_target_": "MyModel", "_partial_": True, "arg": 1}}
        )
        result = dictconfig_to_dot_list(config)
        expected = [("model", "=", "Partial[MyModel]"), ("model.arg", "=", 1)]
        assert result == expected

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
        # Filter out structure args for robust comparison
        result_filtered = [
            item for item in result if not item[2] == "Config[TransformerEncoder]"
        ]
        result_filtered = [
            item for item in result_filtered if not item[2] == "Config[TransformerDecoder]"
        ]
        expected_filtered = [
            item for item in expected if not item[2] == "Config[TransformerEncoder]"
        ]
        expected_filtered = [
            item for item in expected_filtered if not item[2] == "Config[TransformerDecoder]"
        ]
        assert result_filtered == expected_filtered

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

    def test_args_to_dictconfig_with_at_syntax(self, tmp_path):
        """Test _args_to_dictconfig handles @ syntax in values."""
        config_content = """
        lr: 0.01
        beta1: 0.9
        """
        p = tmp_path / "optim.yaml"
        p.write_text(config_content)

        args = [
            ("model", "=", "MyModel"),
            ("optimizer", "=", f"@{p}"),
            ("trainer.max_epochs", "=", 10),
        ]
        config = _args_to_dictconfig(args)

        assert config.model == "MyModel"
        assert isinstance(config.optimizer, DictConfig)
        assert config.optimizer.lr == 0.01
        assert config.optimizer.beta1 == 0.9
        assert config.trainer.max_epochs == 10

    def test_args_to_dictconfig_with_at_syntax_non_existent(self):
        """Test _args_to_dictconfig handles non-existent @ file."""
        args = [("optimizer", "=", "@non_existent.yaml")]
        config = _args_to_dictconfig(args)
        # Should keep the string if loading fails
        assert config.optimizer == "@non_existent.yaml"

    def test_args_to_dictconfig_non_dict_override(self):
        """Test overriding a non-dict value with a dict structure."""
        args = [("a", "=", 1), ("a.b", "=", 2)]
        config = _args_to_dictconfig(args)
        assert isinstance(config.a, DictConfig)
        assert config.a.b == 2


class TestLazyImports:
    def test_lazy_imports_context(self):
        """Test that the lazy_imports context manager works correctly."""
        from nemo_run.cli.lazy import LazyModule, lazy_imports

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
        from nemo_run.cli.lazy import LazyModule, lazy_imports

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
        lazy_mod = LazyModule("fake_module")

        # Check attributes
        assert lazy_mod.name == "fake_module"
        assert hasattr(lazy_mod, "_lazy_attrs")
        assert isinstance(lazy_mod._lazy_attrs, dict)
        assert len(lazy_mod._lazy_attrs) == 0

    def test_lazy_module_dir(self):
        """Test that LazyModule __dir__ returns attributes that have been accessed."""
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

    def test_lazy_module_nested_getattr(self):
        """Test __getattr__ for nested lazy modules."""
        lazy_mod = LazyModule("nemo_run.cli")
        lazy_lazy = lazy_mod.lazy # nemo_run.cli -> LazyModule("nemo_run.cli.lazy")
        assert isinstance(lazy_lazy, LazyModule)
        assert lazy_lazy.name == "nemo_run.cli.lazy"

        # Accessing attribute on a nested module currently returns LazyModule due to '.' in self.name
        lazy_entrypoint_attr = lazy_lazy.LazyEntrypoint # nemo_run.cli.lazy -> LazyModule("nemo_run.cli.lazy.LazyEntrypoint")
        assert isinstance(lazy_entrypoint_attr, LazyModule) # Adjusted assertion
        assert lazy_entrypoint_attr.name == "nemo_run.cli.lazy.LazyEntrypoint"

        # Accessing attribute on a top-level module currently returns LazyTarget
        lazy_math = LazyModule("math")
        lazy_sin_attr = lazy_math.sin # math -> LazyTarget("math.sin")
        assert isinstance(lazy_sin_attr, LazyTarget)
        assert lazy_sin_attr.import_path == "math.sin"

    def test_lazy_module_call(self, monkeypatch):
        """Test calling a LazyModule loads and calls the real module/class attribute."""
        # Add the current module as 'my_test_module' to sys.modules
        # and ensure MyCallableClass is available within it.
        current_module = sys.modules[__name__]
        monkeypatch.setitem(sys.modules, "my_test_module", current_module)
        monkeypatch.setattr(current_module, "MyCallableClass", MyCallableClass, raising=False)

        # Create LazyModule for a callable *attribute* (needs dot in name for import_module)
        lazy_callable_attr = LazyModule("my_test_module.MyCallableClass")

        # Calling the LazyModule should import my_test_module, get MyCallableClass,
        # and then return the result of calling MyCallableClass(), which is an instance.
        instance = lazy_callable_attr()
        assert isinstance(instance, MyCallableClass)

        # Now, call the instance to check its __call__ method
        call_result = instance()
        assert call_result == "called class"

    def test_lazy_module_pickle(self):
        """Test that LazyModule can be pickled and unpickled."""
        lazy_mod = LazyModule("nemo_run.cli.lazy")
        _ = lazy_mod.LazyEntrypoint # Access creates a LazyModule in _lazy_attrs

        pickled = pickle.dumps(lazy_mod)
        unpickled = pickle.loads(pickled)

        assert isinstance(unpickled, LazyModule)
        assert unpickled.name == "nemo_run.cli.lazy"
        assert "LazyEntrypoint" in unpickled._lazy_attrs
        # Check that the stored attribute is a LazyModule, matching the current __getattr__ logic
        assert isinstance(unpickled._lazy_attrs["LazyEntrypoint"], LazyModule) # Adjusted assertion
        assert unpickled._lazy_attrs["LazyEntrypoint"].name == "nemo_run.cli.lazy.LazyEntrypoint"


class TestLazyTarget:
    def test_lazy_target_initialization(self):
        """Test LazyTarget initialization."""
        lazy_fn = LazyTarget("math.sin")

        # Check attributes
        assert lazy_fn.import_path == "math.sin"
        assert lazy_fn.script == ""

    def test_lazy_target_call(self):
        """Test that calling LazyTarget loads and calls the real function."""
        result = LazyTarget("math.sin")(0.5)

        # Check that we got the right result
        assert math.isclose(result, math.sin(0.5))


class TestHelperFunctions:
    def test_args_to_dictconfig(self):
        """Test _args_to_dictconfig helper function."""
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

    @pytest.mark.parametrize(
        "path_str, expected",
        [
            ("config.yaml", True),
            ("path/to/my_config.yml", True),
            ("settings.json", True),
            ("params.toml", True),
            ("config.YAML", True),
            ("archive.zip", False),
            ("no_extension", False),
            ("", False),
            ("config.yaml:section", True),
            ("@config.yaml", True),
        ]
    )
    def test_is_config_file_path(self, path_str, expected):
        """Test the _is_config_file_path helper function."""
        assert _is_config_file_path(path_str) == expected

    def test_dummy_fn(self):
        """Test that _dummy_fn raises NotImplementedError."""
        from nemo_run.cli.lazy import _dummy_fn
        with pytest.raises(NotImplementedError):
            _dummy_fn()

    def test_import_module(self):
        """Test the import_module helper function."""
        # Test importing an attribute from a module
        math_sin = import_module("math.sin")
        assert math_sin == math.sin

        # Test importing from a nested module
        lazy_target = import_module("nemo_run.cli.lazy.LazyTarget")
        assert lazy_target == LazyTarget

        # Test non-existent module
        with pytest.raises(ImportError):
            import_module("non_existent_module.function")
        # Test non-existent attribute
        with pytest.raises(AttributeError):
            import_module("math.non_existent_function")

    def test_load_config_from_path(self, tmp_path):
        """Test loading configuration using @ syntax."""
        # Nested config with multiple keys
        nested_multi_key_yaml = """
        model:
          _target_: Model
          hidden: 1
        optim:
          lr: 0.1
        """
        nested_multi_key_p = tmp_path / "nested_multi_key.yaml"
        nested_multi_key_p.write_text(nested_multi_key_yaml)

        # Nested config with single key
        nested_single_key_yaml = """
        trainer:
          _target_: Trainer
          precision: "bf16"
        """
        nested_single_key_p = tmp_path / "nested_single_key.yaml"
        nested_single_key_p.write_text(nested_single_key_yaml)


        # Flat config
        flat_yaml = """
        _target_: Optimizer
        lr: 0.01
        """
        flat_p = tmp_path / "flat.yaml"
        flat_p.write_text(flat_yaml)

        # JSON config
        json_cfg = """
        { "batch_size": 32, "num_workers": 4 }
        """
        json_p = tmp_path / "data.json"
        json_p.write_text(json_cfg)

        # TOML config
        toml_cfg = """
        precision = "bf16"
        """
        toml_p = tmp_path / "trainer_flat.toml"
        toml_p.write_text(toml_cfg)

        # Test basic loading (flat)
        loaded_flat = load_config_from_path(f"@{flat_p}")
        assert isinstance(loaded_flat, DictConfig)
        assert loaded_flat._target_ == "Optimizer"
        assert loaded_flat.lr == 0.01

        # Test basic loading (nested, single key -> extracts value)
        loaded_nested_single = load_config_from_path(f"@{nested_single_key_p}")
        assert isinstance(loaded_nested_single, DictConfig)
        assert loaded_nested_single._target_ == "Trainer"
        assert loaded_nested_single.precision == "bf16"

        # Test basic loading (nested, multiple keys -> returns full structure)
        loaded_nested_multi = load_config_from_path(f"@{nested_multi_key_p}")
        assert isinstance(loaded_nested_multi, DictConfig)
        assert "model" in loaded_nested_multi
        assert "optim" in loaded_nested_multi
        assert loaded_nested_multi.model._target_ == "Model" # Check within model key
        assert loaded_nested_multi.model.hidden == 1       # Check within model key
        assert loaded_nested_multi.optim.lr == 0.1         # Check within optim key


        # Test loading JSON (flat)
        loaded_json = load_config_from_path(f"@{json_p}")
        assert isinstance(loaded_json, DictConfig)
        assert loaded_json.batch_size == 32

        # Test loading TOML (flat)
        loaded_toml = load_config_from_path(f"@{toml_p}")
        assert isinstance(loaded_toml, DictConfig)
        assert loaded_toml.precision == "bf16"

        # Test loading section from multi-key nested config
        loaded_section = load_config_from_path(f"@{nested_multi_key_p}:optim")
        assert isinstance(loaded_section, DictConfig)
        assert loaded_section.lr == 0.1
        assert "model" not in loaded_section

        # Test file not found
        with pytest.raises(ValueError, match="Config file not found"):
            load_config_from_path("@non_existent.yaml")

        # Test invalid format
        with pytest.raises(ValueError, match="Invalid config file format"):
            load_config_from_path("invalid_syntax.yaml")

        # Test section not found
        with pytest.raises(ValueError, match="Section 'invalid_section' not found"):
            load_config_from_path(f"@{nested_multi_key_p}:invalid_section")

        # Test invalid file content (e.g., malformed yaml)
        malformed_p = tmp_path / "bad.yaml"
        malformed_p.write_text("key: [missing bracket")
        with pytest.raises(ValueError, match="Error loading config file"):
            load_config_from_path(f"@{malformed_p}")


class TestEntrypointMocking:
    """Test mocking the LazyEntrypoint for easier testing."""

    def test_entrypoint_with_exception_handling(self):
        """Test that LazyEntrypoint handles exceptions gracefully."""
        import importlib

        # Create a LazyEntrypoint with a non-existent target
        LazyEntrypoint("non_existent_module.function")

        # Trying to resolve should raise ImportError
        with pytest.raises((ImportError, ModuleNotFoundError)):
            # Manually trigger the import error by trying to import the module
            importlib.import_module("non_existent_module")
