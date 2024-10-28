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
class TestClass:
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
        assert deserialized._target_.import_path == f"{__name__}.some_function"
        assert deserialized._factory_ == "some_function_recipe"
        assert deserialized._args_ == [("inner.x", "=", 3000)]


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
