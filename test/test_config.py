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

from dataclasses import dataclass
from typing import Optional, Union
from unittest.mock import Mock, mock_open, patch

import fiddle as fdl
import graphviz
import pytest
from typing_extensions import Annotated

import nemo_run as run
from nemo_run.config import OptionalDefaultConfig, Script, from_dict, set_value
from nemo_run.exceptions import SetValueError


@dataclass
class DummyModel:
    hidden: int = 100
    activation: str = "relu"
    seq_length: int = 10


@dataclass
class Optimizer:
    lr: float = 0.1


@run.autoconvert
def dummy_model() -> DummyModel:
    return DummyModel()


@run.autoconvert
def optimizer() -> Optimizer:
    return Optimizer()


def train(model: DummyModel, optim: OptionalDefaultConfig[Optimizer] = None):
    return optim


def train_manual(
    model: DummyModel,
    optim: Annotated[Optional[Optimizer], run.Config[Optimizer]] = None,
):
    return optim


@dataclass
class Data:
    name: str
    value: int


@dataclass
class NestedData:
    data: Data


@dataclass
class ListData:
    data: list[Data]


class TestFromDict:
    def test_from_dict_with_dataclass(self):
        input_dict = {"name": "test", "value": 123}
        output = from_dict(input_dict, Data)
        assert output.name == "test"
        assert output.value == 123

    def test_from_dict_with_nested_dataclass(self):
        input_dict = {"data": {"name": "test", "value": 123}}
        output = from_dict(input_dict, NestedData)
        assert output.data.name == "test"
        assert output.data.value == 123

    def test_from_dict_with_list(self):
        input_list = {"data": [{"name": "test1", "value": 123}, {"name": "test2", "value": 456}]}
        output = from_dict(input_list, ListData)
        assert len(output.data) == 2
        assert output.data[0].name == "test1"
        assert output.data[0].value == 123
        assert output.data[1].name == "test2"
        assert output.data[1].value == 456

    def test_from_dict_with_simple_types(self):
        input_data = "test"
        output = from_dict(input_data, str)
        assert output == "test"

        input_data = 123
        output = from_dict(input_data, int)
        assert output == 123

        input_data = True
        output = from_dict(input_data, bool)
        assert output is True

    def test_from_dict_with_invalid_union(self):
        with pytest.raises(AssertionError):
            from_dict({"name": "test"}, Union[str, int])  # type: ignore


class TestSetValue:
    @pytest.fixture
    def dummy_config(self) -> run.Config:
        @dataclass
        class TestConfig:
            attribute: str = "default"

        return run.Config(TestConfig)

    def test_set_value_with_valid_key(self, dummy_config: run.Config):
        cfg = dummy_config
        key = "attribute"
        value = "value"
        set_value(cfg, key, value)
        assert cfg.attribute == value

    def test_set_value_with_invalid_attr(self, dummy_config: run.Config):
        cfg = dummy_config
        key = "non_existent_attribute"
        value = "value"
        with pytest.raises(SetValueError):
            set_value(cfg, key, value)

    def test_set_value_on_dataclass(self, dummy_config: run.Config):
        cfg = fdl.build(dummy_config)
        key = "attribute"
        value = "new_value"
        set_value(cfg, key, value)
        assert cfg.attribute == value

    def test_set_value_on_dict(self):
        cfg = {"key": "old_value"}
        key = "key"
        value = "new_value"
        with pytest.raises(SetValueError):
            set_value(cfg, key, value)  # type: ignore

    def test_set_value_with_invalid_key(self, dummy_config: run.Config):
        key = "non_existent_key"
        value = "value"
        with pytest.raises(SetValueError):
            set_value(dummy_config, key, value)

    def test_set_value_with_invalid_path(self, dummy_config):
        key = "non_existent.attribute"
        value = "value"
        with pytest.raises(SetValueError, match="Invalid path"):
            set_value(dummy_config, key, value)

    def test_set_value_with_invalid_path_element(self, dummy_config: run.Config):
        key = "attribute."
        value = "value"
        with pytest.raises(ValueError):
            set_value(dummy_config, key, value)


def fake_visualize(self, **kwargs):
    @dataclass
    class Graph:
        def pipe(self, format):
            return str.encode("fake graph")

    return Graph()


def fake_visualize_error(self, **kwargs):
    raise Exception


class TestConfigDict:
    def test_config_dict(self):
        cfg = run.Config({}, a=1, b=2)
        assert cfg.a == 1
        assert cfg.b == 2


class TestPartial:
    def test_resolve_optional(self):
        partial = run.Partial(train, model=dummy_model(), optim=optimizer())
        fn = fdl.build(partial)

        assert fn() == fdl.build(optimizer())

    def test_extra_args(self):
        with pytest.raises(TypeError):
            run.Partial(train, model=dummy_model(), optim=optimizer(), something="else")

    def test_dataclass_arg(self):
        partial = run.Partial(NestedData, data=Data(name="test", value=123))
        assert isinstance(partial.data, run.Config)
        assert partial.data.name == "test"
        assert partial.data.value == 123

    @pytest.mark.parametrize("train_func", [train, train_manual])
    def test_default_optional(self, train_func):
        partial = run.Partial(train_func, model=dummy_model())
        fn = fdl.build(partial)

        assert fn() == fdl.build(optimizer())

    def test_clone(self):
        partial = run.Partial(train, model=dummy_model(), optim=optimizer())
        clone = partial.clone()

        assert id(partial) != id(clone)
        assert partial == clone

    def test_walk(self):
        partial = run.Partial(train, model=dummy_model(), optim=optimizer())
        new_partial = partial.walk(seq_length=lambda cfg: cfg.seq_length * 2)
        assert new_partial.model.seq_length == 20

    def test_broadcast(self):
        partial = run.Partial(train, model=dummy_model(), optim=optimizer())
        new_partial = partial.broadcast(hidden=2)
        assert new_partial.model.hidden == 2

    def test_visualize(self):
        partial = run.Partial(train, model=dummy_model(), optim=optimizer())
        graph = partial.visualize()
        assert isinstance(graph, graphviz.Graph)

    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch.object(run.Partial, "visualize", fake_visualize)
    def test_save_config_img(self, mocked_open):
        mock_graph = graphviz.Graph()
        mock_graph.pipe = Mock(name="pipe", return_value="")

        partial = run.Partial(train, model=dummy_model, optim=optimizer())
        partial.save_config_img("test/config")
        mocked_open.assert_called_once_with("wb")

        with pytest.raises(ValueError):
            partial.save_config_img("test/config.invalid_extension")

    @patch.object(run.Partial, "visualize", fake_visualize)
    @patch("builtins.print")
    def test_repr_svg_success(self, mocked_print):
        partial = run.Partial(train, model=dummy_model, optim=optimizer())
        output = partial._repr_svg_()
        assert output == "fake graph"
        mocked_print.assert_not_called()

    @patch("builtins.print")
    @patch.object(run.Partial, "visualize", fake_visualize_error)
    def test_repr_svg_failure(self, mocked_print):
        partial = run.Partial(train, model=dummy_model, optim=optimizer())
        partial._repr_svg_()
        mocked_print.assert_called_once()


class TestScript:
    def test_script_initialization(
        self,
    ):
        script = Script(path="test_path")
        assert script.path == "test_path"
        assert script.inline == ""
        assert script.args == []
        assert script.env == {}
        assert script.entrypoint == "bash"
        assert script.m is False

    def test_script_post_init_assertion(
        self,
    ):
        with pytest.raises(AssertionError):
            Script()

        with pytest.raises(AssertionError):
            Script("test.py", entrypoint="")

    def test_get_name_from_inline(self):
        script = Script(inline="echo hello world")
        assert script.get_name() == "echo_hello"

    def test_get_name_from_inline_with_special_characters(self):
        script = Script(inline='echo "hi world!"')
        assert script.get_name() == "echo_hi_w"

    def test_get_name_from_inline_long_string(self):
        script = Script(inline="this is a very long inline script that should be truncated")
        assert script.get_name() == "this_is_a_"

    def test_get_name_from_path(self):
        script = Script(path="/path/to/my/script.py")
        assert script.get_name() == "script.py"

    def test_get_name_from_path_with_extension(self):
        script = Script(path="/path/to/my/script.sh")
        assert script.get_name() == "script.sh"

    def test_get_name_from_path_without_extension(self):
        script = Script(path="/path/to/my/script")
        assert script.get_name() == "script"

    def test_script_to_command_with_inline(
        self,
    ):
        script = Script(inline="echo 'test'")
        command = script.to_command()
        assert command == ["-c", "\"echo 'test'\""]

    def test_script_to_command_with_path(
        self,
    ):
        script = Script(path="test.py")
        command = script.to_command()
        assert command == ["test.py"]

    def test_script_to_command_with_args(
        self,
    ):
        script = Script(path="test.py", args=["--arg1", "value1"])
        command = script.to_command()
        assert command == ["test.py", "--arg1", "value1"]

    def test_script_to_command_with_m(
        self,
    ):
        script = Script(path="test.py", m=True, entrypoint="python")
        command = script.to_command()
        assert command == ["-m", "test.py"]

    @pytest.mark.parametrize(
        "args, expected",
        [
            ({"with_entrypoint": True, "entrypoint": "bash"}, ["bash", "test.py"]),
            (
                {
                    "with_entrypoint": True,
                    "entrypoint": "python3",
                    "m": True,
                },
                ["python3", "-m", "test.py"],
            ),
            (
                {
                    "with_entrypoint": True,
                    "entrypoint": "python3",
                    "m": True,
                    "args": ["--arg1", "value1"],
                },
                ["python3", "-m", "test.py", "--arg1", "value1"],
            ),
            ({"with_entrypoint": False}, ["test.py"]),
        ],
    )
    def test_script_to_command_with_entrypoint(self, args: dict, expected):
        with_entrypoint = args.pop("with_entrypoint")
        script = Script(path="test.py", **args)
        command = script.to_command(with_entrypoint=with_entrypoint)
        assert command == expected

    def test_script_to_command_error(self):
        script = Script(path="test.py")
        script.entrypoint = ""
        with pytest.raises(ValueError):
            script.to_command(with_entrypoint=True)

    def test_inline_script(self):
        script = Script(inline="echo 'test'")
        assert script.to_command(with_entrypoint=False) == ["-c", "\"echo 'test'\""]
        assert script.to_command(with_entrypoint=True) == [
            "bash",
            "-c",
            "\"echo 'test'\"",
        ]
