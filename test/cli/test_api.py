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
from typing import List, Optional, Union
from unittest.mock import Mock, patch

import fiddle as fdl
import pytest
from importlib_metadata import EntryPoint, EntryPoints
from typer.testing import CliRunner

import nemo_run as run
from nemo_run import cli, config
from nemo_run.cli import api as cli_api
from nemo_run.cli.api import Entrypoint, RunContext, create_cli
from test.dummy_factory import DummyModel, dummy_entrypoint

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


_ENTRY_POINTS: EntryPoints = EntryPoints(EntryPoint_from_text(_RUN_FACTORIES_ENTRYPOINT))


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


class TestRunContext:
    @pytest.fixture
    def sample_function(self):
        def func(a: int, b: str, c: float = 1.0):
            return a, b, c

        return func

    @pytest.fixture
    def sample_experiment(self):
        def func(ctx, a: int, b: str, c: float = 1.0):
            return a, b, c

        return func

    def test_run_context_initialization(self):
        ctx = RunContext(name="test_run")
        assert ctx.name == "test_run"
        assert not ctx.direct
        assert not ctx.dryrun
        assert ctx.factory is None
        assert ctx.load is None
        assert not ctx.repl
        assert not ctx.detach
        assert not ctx.skip_confirmation
        assert not ctx.tail_logs

    def test_run_context_parse_args(self):
        ctx = RunContext(name="test_run")
        ctx.parse_args(
            ["executor=local_executor", "executor.ntasks_per_node=2", "plugins=dummy_plugin"]
        )
        assert isinstance(ctx.executor, run.LocalExecutor)
        assert ctx.executor.ntasks_per_node == 2
        assert ctx.plugins[0].some_arg == 20

    def test_run_context_plugin_list_factory(self):
        ctx = RunContext(name="test_run")
        ctx.parse_args(
            [
                "executor=local_executor",
                "executor.ntasks_per_node=2",
                "plugins=plugin_list",
                "plugins[0].some_arg=50",
            ]
        )
        assert isinstance(ctx.executor, run.LocalExecutor)
        assert ctx.executor.ntasks_per_node == 2
        assert len(ctx.plugins) == 2
        assert ctx.plugins[0].some_arg == 50

    def test_run_context_parse_fn(self, sample_function):
        ctx = RunContext(name="test_run")
        partial = ctx.parse_fn(sample_function, ["a=10", "b=hello"])
        assert partial.a == 10
        assert partial.b == "hello"
        assert partial.c == 1.0  # Default value

    @patch("nemo_run.dryrun_fn")
    @patch("nemo_run.run")
    def test_run_context_execute_task(self, mock_run, mock_dryrun_fn, sample_function):
        ctx = RunContext(name="test_run", skip_confirmation=True)
        ctx.cli_execute(sample_function, ["a=10", "b=hello"])
        mock_dryrun_fn.assert_called_once()
        mock_run.assert_called_once()

    def test_run_context_to_config(self):
        ctx = RunContext(name="test_run")
        config = ctx.to_config()
        assert isinstance(config, run.Config)
        assert config.name == "test_run"

    def test_run_context_parse_executor(self):
        ctx = RunContext(name="test_run")
        executor = ctx.parse_executor("local_executor", "ntasks_per_node=4")
        assert isinstance(executor, run.Config)
        assert executor.__fn_or_cls__ == run.LocalExecutor
        assert executor.ntasks_per_node == 4

    def test_run_context_parse_plugin(self):
        ctx = RunContext(name="test_run")
        plugin = ctx.parse_plugin("dummy_plugin", "some_arg=30")
        assert isinstance(plugin, run.Config)
        assert plugin.__fn_or_cls__.__name__ == "DummyPlugin"
        assert plugin.some_arg == 30

    def test_run_context_parse_args_with_invalid_executor(self):
        ctx = RunContext(name="test_run")
        with pytest.raises(ValueError, match="Executor invalid_executor not found"):
            ctx.parse_args(["executor=invalid_executor"])

    def test_run_context_parse_args_with_invalid_plugin(self):
        ctx = RunContext(name="test_run")
        with pytest.raises(ValueError, match="Plugin invalid_plugin not found"):
            ctx.parse_args(["plugins=invalid_plugin"])

    @patch("nemo_run.dryrun_fn")
    @patch("nemo_run.run")
    def test_run_context_execute_task_with_dryrun(self, mock_run, mock_dryrun_fn, sample_function):
        ctx = RunContext(name="test_run", dryrun=True, skip_confirmation=True)
        ctx.cli_execute(sample_function, ["a=10", "b=hello"])
        mock_dryrun_fn.assert_called_once()
        mock_run.assert_not_called()

    @patch("nemo_run.dryrun_fn")
    @patch("nemo_run.run")
    @patch("typer.confirm", return_value=False)
    def test_run_context_execute_task_with_confirmation_denied(
        self, mock_confirm, mock_run, mock_dryrun_fn, sample_function
    ):
        ctx = RunContext(name="test_run")
        cli_api.NEMORUN_SKIP_CONFIRMATION = None
        ctx.cli_execute(sample_function, ["a=10", "b=hello"])
        mock_dryrun_fn.assert_called_once()
        mock_confirm.assert_called_once()
        mock_run.assert_not_called()

    @patch("IPython.embed")
    def test_run_context_execute_task_with_repl(self, mock_embed, sample_function):
        ctx = RunContext(name="test_run", repl=True, skip_confirmation=True)
        ctx.cli_execute(sample_function, ["a=10", "b=hello"])
        mock_embed.assert_called_once()

    def test_run_context_parse_fn_with_factory(self, sample_function):
        ctx = RunContext(name="test_run", factory="dummy_factory")
        with patch("nemo_run.cli.cli_parser.parse_factory") as mock_parse_factory:
            mock_parse_factory.return_value = run.Partial(sample_function, a=20, b="world")
            partial = ctx.parse_fn(sample_function, [])
            assert partial.a == 20
            assert partial.b == "world"
            assert partial.c == 1.0  # Default value
            mock_parse_factory.assert_called_once()

    def test_run_context_with_invalid_entrypoint_type(self, sample_function):
        ctx = RunContext(name="test_run")
        with pytest.raises(ValueError, match="Unknown entrypoint type: invalid_type"):
            ctx.cli_execute(sample_function, [], entrypoint_type="invalid_type")

    @patch("nemo_run.cli.api.RunContext.cli_execute")
    def test_run_context_run_task(self, mock_run):
        ctx = RunContext(name="test_run")

        def sample_function(a, b):
            return None

        ctx.cli_execute(sample_function, ["a=10", "b=hello"])

        mock_run.assert_called_once_with(sample_function, ["a=10", "b=hello"])

    def test_run_context_run_with_detach(self):
        ctx = RunContext(name="test_run", skip_confirmation=True)

        def sample_function(a, b):
            return None

        ctx.cli_execute(sample_function, ["a=10", "b=hello", "run.detach=False"])
        assert not ctx.detach


@dataclass
class SomeObject:
    value_1: int
    value_2: int
    value_3: int


class TestFactoryAndResolve:
    @patch("nemo_run.cli.api.metadata.entry_points", return_value=_ENTRY_POINTS)
    def test_factory_without_arguments(self, mock_entry_points):
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

    def test_factory_with_target_and_arg(self):
        @dataclass
        class ChildObject:
            value: str

        @dataclass
        class ParentObject:
            child: Union[ChildObject, str]

        @cli.factory(target=ParentObject, target_arg="child")
        def common_factory(value="random") -> ChildObject:
            return ChildObject(value=value)

        cfg = cli.parse_config(ParentObject, "child=common_factory")
        assert fdl.build(cfg).child.value == "random"

        cfg = cli.parse_config(ParentObject, "child=common_factory(custom)")
        assert fdl.build(cfg).child.value == "custom"

    def test_factory_default(self):
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

        cfg = cli.parse_config(ParentObject, "obj=default")
        assert fdl.build(cfg).obj.value == "default"

    def test_factory_raises_error_without_return_annotation(self):
        with pytest.raises(TypeError, match="Missing return type annotation"):

            @cli.factory
            def no_return_annotation_function():
                pass

    def test_factory_raises_error_without_parent_when_arg_is_used(self):
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

        assert run.cli.resolve_factory("custom_namespace", "custom_factory")().value == "custom"

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
        assert run.cli.resolve_factory(DummyModel, "dummy_factory_for_entrypoint")().hidden == 1000

    def test_help(self):
        registry_details = []
        for t in config.get_underlying_types(Optional[Optimizer]):
            namespace = config.get_type_namespace(t)
            registry_details.extend(run.cli.list_factories(namespace))

        assert len(registry_details) == 2
        assert optimizer in registry_details

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


@dataclass
class Model:
    """Dummy model config"""

    hidden_size: int
    num_layers: int
    activation: str


@dataclass
class Optimizer:
    """Dummy optimizer config"""

    learning_rate: float
    weight_decay: float
    betas: List[float]


@run.cli.factory
@run.autoconvert
def my_model(hidden_size: int = 256, num_layers: int = 3, activation: str = "relu") -> Model:
    """Create a model configuration."""
    return Model(hidden_size=hidden_size, num_layers=num_layers, activation=activation)


@run.cli.factory
def my_optimizer(
    learning_rate: float = 0.001, weight_decay: float = 1e-5, betas: List[float] = [0.9, 0.999]
) -> run.Config[Optimizer]:
    """Create an optimizer configuration."""
    return run.Config(
        Optimizer, learning_rate=learning_rate, weight_decay=weight_decay, betas=betas
    )


def defaults() -> run.Partial["train_model"]:
    return run.Partial(
        train_model,
        model=my_model(),
        optimizer=my_optimizer(),
        epochs=40,
        batch_size=1024,
    )


@run.cli.entrypoint(
    default_factory=defaults,
    namespace="my_llm",
    skip_confirmation=True,
)
def train_model(
    model: Model,
    optimizer: Optimizer,
    epochs: int = 10,
    batch_size: int = 32,
):
    """
    Train a model using the specified configuration.

    Args:
        model (Model): Configuration for the model.
        optimizer (Optimizer): Configuration for the optimizer.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 32.
    """
    print("Training model with the following configuration:")
    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Simulating model training
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

    print("Training completed!")

    return {"model": model, "optimizer": optimizer, "epochs": epochs, "batch_size": batch_size}


@run.cli.factory(target=train_model)
def custom_defaults() -> run.Partial["train_model"]:
    return run.Partial(
        train_model,
        model=my_model(),
        optimizer=my_optimizer(),
        epochs=10,
        batch_size=1024,
    )


class TestEntrypointRunner:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        return create_cli(add_verbose_callback=False, nested_entrypoints_creation=False)

    def test_dummy_entrypoint_cli(self, runner, app):
        with patch("test.dummy_factory.NestedModel") as mock_nested_model:
            result = runner.invoke(
                app,
                [
                    "dummy",
                    "dummy_entrypoint",
                    "dummy=dummy_model_config",
                    "run.skip_confirmation=True",
                ],
            )
            assert result.exit_code == 0
            mock_nested_model.assert_called_once_with(
                dummy=DummyModel(hidden=2000, activation="tanh")
            )

    def test_parse_partial_function_call(self):
        entrypoint = Entrypoint(dummy_entrypoint, namespace="test")
        partial = entrypoint.parse_partial(["dummy=my_dummy_model(hidden=100)"])
        assert isinstance(partial, run.Partial)
        assert partial.dummy.hidden == 100
        assert partial.dummy.activation == "tanh"

    def test_with_factory(self, runner, app):
        # Test CLI execution with default factory
        result = runner.invoke(
            app,
            [
                "my_llm",
                "train_model",
                "--factory",
                "custom_defaults",
                "model.hidden_size=200",
                "--yes",
            ],
            env={"INCLUDE_WORKSPACE_FILE": "false"},
        )
        assert result.exit_code == 0

        output = result.stdout
        assert "Training model with the following configuration:" in output
        assert "Model: Model(hidden_size=200, num_layers=3, activation='relu')" in output

    def test_with_defaults(self, runner, app):
        # Test CLI execution with default factory
        result = runner.invoke(
            app,
            [
                "my_llm",
                "train_model",
                "model.hidden_size=1024",
                "optimizer.learning_rate=0.005",
                "epochs=30",
                "run.skip_confirmation=True",
            ],
            env={"INCLUDE_WORKSPACE_FILE": "false"},
        )
        assert result.exit_code == 0

        # Parse the output to check the values
        output = result.stdout
        assert "Training model with the following configuration:" in output
        assert "Model: Model(hidden_size=1024, num_layers=3, activation='relu')" in output
        assert (
            "Optimizer: Optimizer(learning_rate=0.005, weight_decay=1e-05, betas=[0.9, 0.999])"
            in output
        )
        assert "Epochs: 30" in output
        assert "Batch size: 1024" in output
        assert "Training completed!" in output

        # Check that all epochs were simulated
        for i in range(1, 31):
            assert f"Epoch {i}/30" in output

    def test_experiment_entrypoint(self):
        def dummy_pretrain(log_dir: str):
            pass

        def dummy_finetune(log_dir: str):
            pass

        @run.cli.entrypoint(namespace="llm", type="experiment")
        def my_experiment(
            ctx: run.cli.RunContext,
            pretrain: run.Partial[dummy_pretrain] = run.Partial(
                dummy_pretrain, log_dir="/pretrain"
            ),
            finetune: run.Partial[dummy_finetune] = run.Partial(
                dummy_finetune, log_dir="/finetune"
            ),
        ):
            pretrain.log_dir = f"/{ctx.experiment.name}/checkpoints"
            finetune.log_dir = f"/{ctx.experiment.name}/checkpoints"

            for i in range(1):
                ctx.experiment.add(
                    pretrain,
                    executor=ctx.executor,
                    name=ctx.experiment.name,
                    tail_logs=True if isinstance(ctx.executor, run.LocalExecutor) else False,
                )

                ctx.experiment.add(
                    finetune,
                    executor=ctx.executor,
                    name=ctx.experiment.name,
                    tail_logs=True if isinstance(ctx.executor, run.LocalExecutor) else False,
                )

            return ctx.experiment

        # Mock the necessary objects and methods
        mock_experiment = Mock(spec=run.Experiment)
        mock_experiment.name = "test_experiment"
        mock_executor = Mock(spec=run.LocalExecutor)

        mock_ctx = Mock(spec=run.cli.RunContext)
        mock_ctx.experiment = mock_experiment
        mock_ctx.executor = mock_executor

        mock_pretrain = Mock(spec=dummy_pretrain)
        mock_pretrain.log_dir = "/pretrain"
        mock_finetune = Mock(spec=dummy_finetune)
        mock_finetune.log_dir = "/finetune"

        # Call the entrypoint function
        result = my_experiment(ctx=mock_ctx, pretrain=mock_pretrain, finetune=mock_finetune)

        # Assert that the experiment methods were called correctly
        assert result == mock_experiment
        assert mock_experiment.add.call_count == 2
        mock_experiment.add.assert_any_call(
            mock_pretrain, executor=mock_executor, name=mock_experiment.name, tail_logs=True
        )
        mock_experiment.add.assert_any_call(
            mock_finetune, executor=mock_executor, name=mock_experiment.name, tail_logs=True
        )

        assert mock_pretrain.log_dir == f"/{mock_experiment.name}/checkpoints"
        assert mock_finetune.log_dir == f"/{mock_experiment.name}/checkpoints"

    @dataclass
    class SomeObject:
        value_1: int
        value_2: int


class TestDefaultFactory:
    def test_default_factory(self):
        # Test that the default factory is applied correctly
        partial = run.cli.resolve_factory(train_model, "default")()
        assert isinstance(partial, run.Partial)

        # Check that the default values are set correctly
        assert partial.model.hidden_size == 256
        assert partial.model.num_layers == 3
        assert partial.model.activation == "relu"
        assert partial.optimizer.learning_rate == 0.001
        assert partial.optimizer.weight_decay == 1e-5
        assert partial.optimizer.betas == [0.9, 0.999]
        assert partial.epochs == 40
        assert partial.batch_size == 1024

    def test_build_from_default_factory(self):
        # Test that we can build the configuration from the default factory
        partial = run.cli.resolve_factory(train_model, "default")()
        result = fdl.build(partial)()

        assert isinstance(result["model"], Model)
        assert isinstance(result["optimizer"], Optimizer)
        assert result["epochs"] == 40
        assert result["batch_size"] == 1024
