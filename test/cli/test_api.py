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

from configparser import ConfigParser
from dataclasses import dataclass
from test.dummy_factory import DummyModel, dummy_entrypoint
from typing import Optional, Union
from unittest.mock import ANY, MagicMock, Mock, patch

import fiddle as fdl
import pytest
from importlib_metadata import EntryPoint, EntryPoints
from rich.console import Console
from typer.testing import CliRunner

import nemo_run as run
from nemo_run import cli, config
from nemo_run.cli.api import Entrypoint, create_cli
from nemo_run.core.execution.base import Executor

_RUN_FACTORIES_ENTRYPOINT: str = """
[nemo_run.cli]
dummy = test.dummy_factory
"""


# Helper methods taken from https://github.com/pytorch/torchx/blob/main/torchx/util/test/entrypoints_test.py
def EntryPoint_from_config(config: ConfigParser) -> list[EntryPoint]:
    # from stdlib, Copyright (c) Python Authors
    return [
        EntryPoint(name, value, group)
        for group in config.sections()
        for name, value in config.items(group)
    ]


def EntryPoint_from_text(text: str) -> list[EntryPoint]:
    # from stdlib, Copyright (c) Python Authors
    config = ConfigParser(delimiters="=")
    config.read_string(text)
    return EntryPoint_from_config(config)


_ENTRY_POINTS: EntryPoints = EntryPoints(
    EntryPoint_from_text(_RUN_FACTORIES_ENTRYPOINT)
)


@dataclass
class Optimizer:
    lr: float = 0.1


@cli.factory
@run.autoconvert
def dummy_model() -> DummyModel:
    return DummyModel()


@cli.factory
def _dummy_model_config() -> run.Config[DummyModel]:
    return run.Config(DummyModel, hidden=2000, activation="tanh")


@cli.factory
@run.autoconvert
def optimizer() -> Optimizer:
    return Optimizer()


class TestEntrypoint:
    @pytest.fixture
    def sample_function(self):
        def func(a: int, b: str, c: float = 1.0):
            return a, b, c
        return func

    def test_entrypoint_initialization(self, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test_namespace")
        assert entrypoint.fn == sample_function
        assert entrypoint.namespace == "test_namespace"
        assert entrypoint.name == "func"
        assert entrypoint.enable_executor == True
        assert entrypoint.require_conformation == True
        assert entrypoint.type == "task"

    def test_entrypoint_initialization_with_custom_params(self, sample_function):
        entrypoint = Entrypoint(
            sample_function,
            namespace="custom_namespace",
            name="custom_name",
            help_str="Custom help",
            enable_executor=False,
            require_conformation=False,
            type="sequential_experiment"
        )
        assert entrypoint.namespace == "custom_namespace"
        assert entrypoint.name == "custom_name"
        assert "Custom help" in entrypoint.help_str
        assert entrypoint.enable_executor == False
        assert entrypoint.require_conformation == False
        assert entrypoint.type == "sequential_experiment"

    def test_entrypoint_call(self, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test")
        result = entrypoint(1, "test", 2.0)
        assert result == (1, "test", 2.0)

    def test_entrypoint_configure(self, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test")
        entrypoint.configure(a=5, b="configured")
        assert entrypoint._configured_fn is not None
        assert entrypoint._configured_fn.a == 5
        assert entrypoint._configured_fn.b == "configured"

    def test_parse_partial(self, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test")
        partial = entrypoint.parse_partial(["a=10", "b=hello"])
        assert partial.a == 10
        assert partial.b == "hello"
        assert partial.c == 1.0  # Default value

    def test_parse_executor(self):
        entrypoint = Entrypoint(lambda: None, namespace="test")
        executor = entrypoint.parse_executor("local_executor", ["num_gpus=2"])
        assert executor.__fn_or_cls__ == run.LocalExecutor
        assert executor.num_gpus == 2

    @patch("nemo_run.cli.api.typer.Typer")
    def test_cli_add_command(self, mock_typer):
        entrypoint = Entrypoint(lambda: None, namespace="test")
        entrypoint.cli(mock_typer)
        mock_typer.command.assert_called_once()

    @patch("nemo_run.cli.api.Console")
    @patch("nemo_run.cli.api.parse_cli_args")
    def test_execute_simple(self, mock_parse_cli_args, mock_console, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test")
        mock_parse_cli_args.return_value = run.Partial(sample_function, a=1, b="test")
        entrypoint._execute_simple(["a=1", "b=test"], mock_console)
        mock_parse_cli_args.assert_called_once()

    @patch("nemo_run.cli.api.run")
    @patch("nemo_run.cli.api.Console")
    def test_execute_with_executor(self, mock_console, mock_run, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test")
        mock_executor = MagicMock(spec=Executor)

        with patch.object(entrypoint, '_get_executor', return_value=mock_executor):
            entrypoint._execute_with_executor(
                ["a=1", "b=test"],
                mock_console,
                load=None,
                factory=None,
                wait=True,
                dryrun=False,
                run_name="test_run",
                direct=False,
                strict=False
            )

        mock_run.run.assert_called_once()

    def test_parse_executor_args(self):
        entrypoint = Entrypoint(lambda: None, namespace="test")
        executor_name, executor_args, filtered_args = entrypoint._parse_executor_args([
            "a=1",
            "executor=local_executor",
            "executor.num_gpus=2",
            "b=test"
        ])
        assert executor_name == "local_executor"
        assert executor_args == ["num_gpus=2"]
        assert filtered_args == ["a=1", "b=test"]

    @patch("nemo_run.cli.api.typer.confirm", return_value=True)
    def test_should_continue(self, mock_confirm):
        entrypoint = Entrypoint(lambda: None, namespace="test", require_conformation=True)
        assert entrypoint._should_continue() == True
        mock_confirm.assert_called_once()

    @patch("nemo_run.cli.api.run.help")
    def test_help(self, mock_help, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test")
        console = Console()
        entrypoint.help(console)
        mock_help.assert_called_once_with(sample_function, console=console, with_docs=True)

    def test_path(self, sample_function):
        entrypoint = Entrypoint(sample_function, namespace="test_namespace")
        assert entrypoint.path == "test_namespace.func"

    def test_entrypoint_with_invalid_experiment_type(self, sample_function):
        with pytest.raises(ValueError, match="Unknown entrypoint type: invalid_type"):
            Entrypoint(sample_function, namespace="test", type="invalid_type")

    def test_entrypoint_with_missing_executor_for_experiment(self):
        def experiment_function(experiment):
            pass

        with pytest.raises(ValueError, match="The function must have an argument named `executor`"):
            Entrypoint(experiment_function, namespace="test", type="sequential_experiment")

    def test_entrypoint_with_missing_experiment_for_experiment(self):
        def experiment_function(executor):
            pass

        with pytest.raises(ValueError, match="The function must have an argument named `experiment`"):
            Entrypoint(experiment_function, namespace="test", type="sequential_experiment")

    def test_entrypoint_with_executor_for_task(self):
        def task_function(executor):
            pass

        with pytest.raises(ValueError, match="The function cannot have an argument named `executor`"):
            Entrypoint(task_function, namespace="test", type="task")

    def test_entrypoint_with_custom_name(self):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", name="custom_name")
        assert entrypoint.name == "custom_name"

    def test_entrypoint_with_help_str(self):
        def sample_func():
            """This is a sample function."""
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", help_str="Custom help")
        assert "Custom help" in entrypoint.help_str
        assert "This is a sample function." in entrypoint.help_str

    def test_entrypoint_configure(self):
        def sample_func(a: int, b: str):
            pass

        entrypoint = Entrypoint(sample_func, namespace="test")
        entrypoint.configure(a=5, b="test")
        assert entrypoint._configured_fn is not None
        assert entrypoint._configured_fn.a == 5
        assert entrypoint._configured_fn.b == "test"

    @patch("nemo_run.cli.api.typer.Typer")
    def test_entrypoint_cli_with_executor(self, mock_typer):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", enable_executor=True)
        entrypoint.cli(mock_typer)
        mock_typer.command.assert_called_once()

    @patch("nemo_run.cli.api.typer.Typer")
    def test_entrypoint_cli_without_executor(self, mock_typer):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", enable_executor=False)
        entrypoint.cli(mock_typer)
        mock_typer.command.assert_called_once()

    @patch("nemo_run.cli.api.run.dryrun_fn")
    @patch("nemo_run.cli.api.Console")
    def test_execute_with_executor_dryrun(self, mock_console, mock_dryrun_fn):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test")
        entrypoint._execute_with_executor(
            [],
            mock_console,
            load=None,
            factory=None,
            wait=False,
            dryrun=True,
            run_name="test_run",
            direct=False,
            strict=False
        )
        mock_dryrun_fn.assert_called_once()

    @patch("nemo_run.cli.api.run.run")
    @patch("nemo_run.cli.api.Console")
    def test_execute_with_executor_direct(self, mock_console, mock_run):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test")
        entrypoint._execute_with_executor(
            [],
            mock_console,
            load=None,
            factory=None,
            wait=False,
            dryrun=False,
            run_name="test_run",
            direct=True,
            strict=False
        )
        mock_run.assert_called_once_with(
            fn_or_script=ANY,
            name="test_run",
            executor=None,
            direct=True,
            wait=False
        )

    def test_parse_executor_args(self):
        entrypoint = Entrypoint(lambda: None, namespace="test")
        args = ["a=1", "executor=local", "executor.gpus=2", "b=3"]
        executor_name, executor_args, filtered_args = entrypoint._parse_executor_args(args)
        assert executor_name == "local"
        assert executor_args == ["gpus=2"]
        assert filtered_args == ["a=1", "b=3"]

    @patch("nemo_run.cli.api.typer.confirm", return_value=False)
    def test_should_continue_false(self, mock_confirm):
        entrypoint = Entrypoint(lambda: None, namespace="test", require_conformation=True)
        assert not entrypoint._should_continue()

    def test_should_continue_no_confirmation(self):
        entrypoint = Entrypoint(lambda: None, namespace="test", require_conformation=False)
        assert entrypoint._should_continue()

    @patch("nemo_run.cli.api.typer.Typer")
    @patch("nemo_run.cli.api.Entrypoint._add_command")
    def test_main(self, mock_add_command, mock_typer):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test")
        entrypoint.main()
        mock_add_command.assert_called_once()
        mock_typer.return_value.assert_called_once()

    def test_entrypoint_with_experiment_type(self):
        def experiment_func(experiment, executor):
            pass

        entrypoint = Entrypoint(experiment_func, namespace="test", type="sequential_experiment")
        assert entrypoint.type == "sequential_experiment"

    @patch("nemo_run.cli.api.Experiment")
    @patch("nemo_run.cli.api.Console")
    def test_execute_with_executor_experiment(self, mock_console, mock_experiment):
        def experiment_func(experiment, executor):
            pass

        entrypoint = Entrypoint(experiment_func, namespace="test", type="sequential_experiment")
        entrypoint._execute_with_executor(
            [],
            mock_console,
            load=None,
            factory=None,
            wait=False,
            dryrun=False,
            run_name="test_experiment",
            direct=False,
            strict=False
        )
        mock_experiment.assert_called_once()
        mock_experiment.return_value.__enter__.return_value.run.assert_called_once_with(
            sequential=True, detach=True
        )


@dataclass
class SomeObject:
    value_1: int
    value_2: int
    value_3: int


class TestFactoryAndResolve:
    @patch("nemo_run.cli.api.metadata.entry_points", return_value=_ENTRY_POINTS)
    def test_factory_without_arguments(
        self,
        mock_entry_points,
    ):
        @cli.factory
        @run.autoconvert
        def commonly_used_object() -> SomeObject:
            return SomeObject(
                value_1=5,
                value_2=10,
                value_3=15,
            )

        obj = run.cli.resolve_factory(SomeObject, "commonly_used_object")()
        assert isinstance(obj, run.Config)
        obj = fdl.build(obj)
        assert isinstance(obj, SomeObject)
        assert obj.value_1 == 5
        assert obj.value_2 == 10
        assert obj.value_3 == 15

    def test_factory_with_target_and_arg(
        self,
    ):
        @dataclass
        class ChildObject:
            value: str

        @dataclass
        class ParentObject:
            child: Union[ChildObject, str]

        @cli.factory(target=ParentObject, target_arg="child")
        def common_factory(value="random") -> ChildObject:
            return ChildObject(value=value)

        cfg = run.Config.from_cli(ParentObject, "child=common_factory")
        assert fdl.build(cfg).child.value == "random"

        cfg = run.Config.from_cli(ParentObject, "child=common_factory(custom)")
        assert fdl.build(cfg).child.value == "custom"

    def test_factory_default(
        self,
    ):
        @dataclass
        class DefaultObject:
            value: str = "default"

        @dataclass
        class ParentObject:
            obj: DefaultObject

        @cli.factory(is_target_default=True)
        @run.autoconvert
        def default_factory() -> DefaultObject:
            return DefaultObject()

        cfg = run.Config.from_cli(ParentObject, "obj=default")
        assert fdl.build(cfg).obj.value == "default"

    def test_factory_raises_error_without_return_annotation(
        self,
    ):
        with pytest.raises(TypeError, match="Missing return type annotation"):

            @cli.factory
            def no_return_annotation_function():
                pass

    def test_factory_raises_error_without_parent_when_arg_is_used(
        self,
    ):
        @dataclass
        class TestObject:
            pass

        with pytest.raises(
            ValueError, match="`target_arg` cannot be used without specifying a `target`."
        ):

            @cli.factory(target_arg="test")
            def test_function() -> TestObject:
                return TestObject()

    def test_factory_with_unsupported_input(self):
        @dataclass
        class TestObject:
            pass

        with pytest.raises(TypeError):
            obj = TestObject()
            obj.__name__ = "abcd"
            cli.factory(fn=obj)

    def test_factory_with_namespace(self):
        @dataclass
        class CustomObject:
            value: str

        @cli.factory(target=CustomObject, namespace="custom_namespace")
        @run.autoconvert
        def custom_factory() -> CustomObject:
            return CustomObject(value="custom")

        assert (
            run.cli.resolve_factory("custom_namespace", "custom_factory")().value
            == "custom"
        )

    def test_resolve(self):
        dummy_model = run.cli.resolve_factory(DummyModel, "dummy_model")()
        assert dummy_model.hidden == 100
        assert dummy_model.activation == "relu"
        assert isinstance(dummy_model, run.Config)

        dummy_model_config = run.cli.resolve_factory(DummyModel, "dummy_model_config")()
        assert dummy_model_config.hidden == 2000
        assert dummy_model_config.activation == "tanh"
        assert isinstance(dummy_model_config, run.Config)

    def test_resolve_optional(self):
        optim = run.cli.resolve_factory(Optional[Optimizer], "optimizer")
        assert optim() == optimizer()

    def test_resolve_union(self):
        model = run.cli.resolve_factory(Union[DummyModel, Optimizer], "dummy_model")
        assert model() == dummy_model()

        with pytest.raises(ValueError):
            run.cli.resolve_factory(Union[DummyModel, Optimizer], "dummy_model_123")

    def test_resolve_entrypoints(self):
        assert (
            run.cli.resolve_factory(DummyModel, "dummy_factory_for_entrypoint")().hidden
            == 1000
        )

    def test_help(self):
        from nemo_run import api

        registry_details = []
        for t in config.get_underlying_types(Optional[Optimizer]):
            namespace = config.get_type_namespace(t)
            registry_details.extend(run.cli.list_factories(namespace))

        assert len(registry_details) == 1
        assert registry_details[0] == optimizer

    def test_factory_for_entrypoint(self):
        cfg = run.cli.resolve_factory(dummy_entrypoint, "dummy_recipe")()
        assert cfg.dummy.hidden == 2000


class TestListEntrypoints:
    @dataclass
    class DummyTask:
        path: str

    @pytest.fixture
    def mock_get_all(self):
        mock_get_all = Mock()

        mock_get_all.return_value = {
            "namespace1.task1": self.DummyTask(path="namespace1.task1"),
            "namespace2.task1": self.DummyTask(path="namespace2.task1"),
            "namespace2.task2": self.DummyTask(path="namespace2.task2"),
        }

        return mock_get_all

    def test_list_entrypoints_without_namespace(self, mocker, mock_get_all):
        mocker.patch("catalogue._get_all", mock_get_all)
        result = cli.list_entrypoints()
        mock_get_all.assert_called_once_with(("nemo_run.cli.entrypoints",))

        expected_result = {
            "namespace1": {"task1": self.DummyTask(path="namespace1.task1")},
            "namespace2": {
                "task1": self.DummyTask(path="namespace2.task1"),
                "task2": self.DummyTask(path="namespace2.task2"),
            },
        }
        assert result == expected_result

    def test_list_entrypoints_with_namespace(self, mocker, mock_get_all):
        mocker.patch("catalogue._get_all", mock_get_all)
        result = cli.list_entrypoints(namespace="hello")
        mock_get_all.assert_called_once_with(("nemo_run.cli.entrypoints", "hello"))

        expected_result = {
            "namespace1": {"task1": self.DummyTask(path="namespace1.task1")},
            "namespace2": {
                "task1": self.DummyTask(path="namespace2.task1"),
                "task2": self.DummyTask(path="namespace2.task2"),
            },
        }
        assert result == expected_result


class TestEntrypoint:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        return create_cli(add_verbose_callback=False, nested_entrypoints_creation=False)

    def test_dummy_entrypoint_cli(self, runner, app):
        with patch('test.dummy_factory.NestedModel') as mock_nested_model:
            result = runner.invoke(app, ["dummy", "dummy_entrypoint", "dummy=dummy_model_config"])
            assert result.exit_code == 0
            mock_nested_model.assert_called_once_with(dummy=DummyModel(hidden=2000, activation="tanh"))

    def test_parse_partial(self):
        entrypoint = Entrypoint(dummy_entrypoint, namespace="test")
        partial = entrypoint.parse_partial(["dummy=dummy_model_config"])
        assert isinstance(partial, run.Partial)
        assert partial.dummy.hidden == 2000
        assert partial.dummy.activation == "tanh"

    def test_parse_partial_function_call(self):
        entrypoint = Entrypoint(dummy_entrypoint, namespace="test")
        partial = entrypoint.parse_partial(["dummy=my_dummy_model(hidden=100)"])
        assert isinstance(partial, run.Partial)
        assert partial.dummy.hidden == 100
        assert partial.dummy.activation == "tanh"

    def test_parse_executor(self):
        entrypoint = Entrypoint(dummy_entrypoint, namespace="test")
        executor = entrypoint.parse_executor("local_executor", ["retries=3"])
        assert isinstance(executor, run.Config)
        assert executor.__fn_or_cls__ == run.LocalExecutor
        assert executor.retries == 3

    def test_parse_executor_with_multiple_args(self):
        entrypoint = Entrypoint(dummy_entrypoint, namespace="test")
        executor = entrypoint.parse_executor("local_executor", ["retries=3", "ntasks_per_node=60"])
        assert isinstance(executor, run.Config)
        assert executor.__fn_or_cls__ == run.LocalExecutor
        assert executor.retries == 3
        assert executor.ntasks_per_node == 60

    def test_parse_executor_invalid_name(self):
        entrypoint = Entrypoint(dummy_entrypoint, namespace="test")
        with pytest.raises(ValueError, match="No matching factory found for: invalid_executor"):
            entrypoint.parse_executor("invalid_executor", [])

    def test_experiment_entrypoint(self):
        def dummy_pretrain(log_dir: str):
            pass

        def dummy_finetune(log_dir: str):
            pass

        @run.cli.entrypoint(namespace="llm", type="sequential_experiment")
        def my_experiment(
            experiment: run.Experiment,
            executor: run.Executor,
            pretrain: run.Partial[dummy_pretrain] = run.Partial(dummy_pretrain, log_dir="/pretrain"),
            finetune: run.Partial[dummy_finetune] = run.Partial(dummy_finetune, log_dir="/finetune")
        ):
            pretrain.log_dir = f"/{experiment.name}/checkpoints"
            finetune.log_dir = f"/{experiment.name}/checkpoints"

            for i in range(1):
                experiment.add(
                    pretrain,
                    executor=executor,
                    name=experiment.name,
                    tail_logs=True if isinstance(executor, run.LocalExecutor) else False,
                )

                experiment.add(
                    finetune,
                    executor=executor,
                    name=experiment.name,
                    tail_logs=True if isinstance(executor, run.LocalExecutor) else False,
                )

            return experiment

        # Mock the necessary objects and methods
        mock_experiment = Mock(spec=run.Experiment)
        mock_experiment.name = "test_experiment"
        mock_executor = Mock(spec=run.LocalExecutor)
        mock_pretrain = Mock(spec=dummy_pretrain)
        mock_pretrain.log_dir = "/pretrain"
        mock_finetune = Mock(spec=dummy_finetune)
        mock_finetune.log_dir = "/finetune"

        # Call the entrypoint function
        result = my_experiment(
            experiment=mock_experiment,
            executor=mock_executor,
            pretrain=mock_pretrain,
            finetune=mock_finetune
        )

        # Assert that the experiment methods were called correctly
        assert result == mock_experiment
        assert mock_experiment.add.call_count == 2
        mock_experiment.add.assert_any_call(
            mock_pretrain,
            executor=mock_executor,
            name=mock_experiment.name,
            tail_logs=True
        )
        mock_experiment.add.assert_any_call(
            mock_finetune,
            executor=mock_executor,
            name=mock_experiment.name,
            tail_logs=True
        )

        assert mock_pretrain.log_dir == f"/{mock_experiment.name}/checkpoints"
        assert mock_finetune.log_dir == f"/{mock_experiment.name}/checkpoints"

    def test_entrypoint_with_custom_name(self):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", name="custom_name")
        assert entrypoint.name == "custom_name"

    def test_entrypoint_with_help_str(self):
        def sample_func():
            """This is a sample function."""
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", help_str="Custom help")
        assert entrypoint.name in entrypoint.help_str
        assert "Custom help" in entrypoint.help_str
        # The docstring is not automatically included in help_str
        assert "This is a sample function." not in entrypoint.help_str

    def test_entrypoint_configure(self):
        def sample_func(a: int, b: str):
            pass

        entrypoint = Entrypoint(sample_func, namespace="test")
        entrypoint.configure(a=5, b="test")
        assert entrypoint._configured_fn is not None
        assert entrypoint._configured_fn.a == 5
        assert entrypoint._configured_fn.b == "test"

    @patch("nemo_run.cli.api.typer.Typer")
    def test_entrypoint_cli_with_executor(self, mock_typer):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", enable_executor=True)
        entrypoint.cli(mock_typer)
        mock_typer.command.assert_called_once()

    @patch("nemo_run.cli.api.typer.Typer")
    def test_entrypoint_cli_without_executor(self, mock_typer):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test", enable_executor=False)
        entrypoint.cli(mock_typer)
        mock_typer.command.assert_called_once()

    @patch("nemo_run.dryrun_fn")
    @patch("nemo_run.cli.api.Console")
    def test_execute_with_executor_dryrun(self, mock_console, mock_dryrun_fn):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test")
        entrypoint._execute_with_executor(
            [],
            mock_console,
            load=None,
            factory=None,
            wait=False,
            dryrun=True,
            run_name="test_run",
            direct=False,
            strict=False
        )
        mock_dryrun_fn.assert_called_once()

    def test_parse_executor_args(self):
        entrypoint = Entrypoint(lambda: None, namespace="test")
        args = ["a=1", "executor=local", "executor.gpus=2", "b=3"]
        executor_name, executor_args, filtered_args = entrypoint._parse_prefixed_args(args, "executor")
        assert executor_name == "local"
        assert executor_args == ["gpus=2"]
        assert filtered_args == ["a=1", "b=3"]

    @patch("nemo_run.cli.api.typer.confirm", return_value=False)
    def test_should_continue_false(self, mock_confirm):
        entrypoint = Entrypoint(lambda: None, namespace="test", require_conformation=True)
        assert not entrypoint._should_continue()

    def test_should_continue_no_confirmation(self):
        entrypoint = Entrypoint(lambda: None, namespace="test", require_conformation=False)
        assert entrypoint._should_continue()

    @patch("nemo_run.cli.api.typer.Typer")
    @patch("nemo_run.cli.api.Entrypoint._add_command")
    def test_main(self, mock_add_command, mock_typer):
        def sample_func():
            pass

        entrypoint = Entrypoint(sample_func, namespace="test")
        entrypoint.main()
        mock_add_command.assert_called_once()
        mock_typer.return_value.assert_called_once()

    def test_entrypoint_with_experiment_type(self):
        def experiment_func(experiment, executor):
            pass

        entrypoint = Entrypoint(experiment_func, namespace="test", type="sequential_experiment")
        assert entrypoint.type == "sequential_experiment"

    @patch("nemo_run.Experiment")
    @patch("nemo_run.cli.api.Console")
    def test_execute_with_executor_experiment(self, mock_console, mock_experiment):
        def experiment_func(experiment, executor):
            pass

        entrypoint = Entrypoint(
            experiment_func,
            namespace="test",
            type="sequential_experiment",
            require_conformation=False
        )
        entrypoint._execute_with_executor(
            [],
            mock_console,
            load=None,
            factory=None,
            wait=False,
            dryrun=False,
            run_name="test_experiment",
            direct=False,
            strict=False
        )
        mock_experiment.assert_called_once()
        mock_experiment.return_value.__enter__.return_value.run.assert_called_once_with(
            sequential=True, detach=True
        )