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

import os
import sys
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Annotated, List, Optional, Union, TYPE_CHECKING
from unittest.mock import Mock, patch

import fiddle as fdl
import pytest
import typer
from importlib_metadata import EntryPoint, EntryPoints
from typer.testing import CliRunner
from rich.console import Console

import nemo_run as run
from nemo_run import cli, config
from nemo_run.cli import api as cli_api
from nemo_run.config import Config
from nemo_run.cli.lazy import LazyEntrypoint
from nemo_run.cli.api import (
    Entrypoint,
    RunContext,
    EntrypointCommand,
    add_global_options,
    create_cli,
    _search_workspace_file,
    _load_workspace_file,
    _load_workspace,
    main as cli_main,
    extract_constituent_types,
)
from test.dummy_factory import DummyModel, dummy_entrypoint
import nemo_run.cli.cli_parser  # Import the module to mock its function

if TYPE_CHECKING:
    from test.dummy_type import RealType


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
    learning_rate: float = 0.1
    weight_decay: float = 1e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


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

    def test_run_context_cli_execute_load_not_implemented(self, sample_function):
        ctx = RunContext(name="test_run", load="some_dir")
        with pytest.raises(NotImplementedError, match="Load is not implemented yet"):
            ctx.cli_execute(sample_function, [])

    @patch("nemo_run.cli.api._serialize_configuration")
    def test_run_context_execute_task_export(self, mock_serialize, sample_function):
        ctx = RunContext(name="test_run", to_yaml="config.yaml", skip_confirmation=True)
        with patch("nemo_run.dryrun_fn"):  # Mock dryrun as it's called before export check
            ctx.cli_execute(sample_function, ["a=10"])
        mock_serialize.assert_called_once()
        assert mock_serialize.call_args[0][1] == "config.yaml"  # Check to_yaml path

    @patch("nemo_run.run")
    @patch("nemo_run.cli.api._serialize_configuration")
    def test_execute_lazy_export(self, mock_serialize, mock_run):
        # Mock sys.argv for lazy execution context
        original_argv = sys.argv
        sys.argv = ["nemo_run", "--lazy", "lazy_test", "arg1=1", "--to-yaml", "output.yaml"]
        os.environ["LAZY_CLI"] = "true"  # Ensure lazy mode is active

        # Create a dummy entrypoint for LazyEntrypoint
        @cli.entrypoint(namespace="test_lazy")
        def lazy_test_fn(arg1: int):
            pass

        # Directly test the execute_lazy method's export behavior
        ctx = RunContext(name="lazy_test", to_yaml="output.yaml", skip_confirmation=True)
        # We need executor and plugins initialized even if None, as execute_lazy accesses them
        ctx.executor = None
        ctx.plugins = []
        lazy_entry = LazyEntrypoint("test_lazy.lazy_test_fn arg1=1")

        # Mock parse_args as it's called within execute_lazy
        with patch("nemo_run.cli.api.RunContext.parse_args", return_value=["arg1=1"]):
            # Mock _should_continue to avoid interaction/torchrun checks
            with patch("nemo_run.cli.api.RunContext._should_continue", return_value=True):
                ctx.execute_lazy(lazy_entry, sys.argv, "lazy_test")

        mock_serialize.assert_called_once()
        # Check arguments passed to _serialize_configuration
        assert isinstance(mock_serialize.call_args[0][0], LazyEntrypoint)  # Check config object
        assert mock_serialize.call_args[0][1] == "output.yaml"  # Check to_yaml path
        assert mock_serialize.call_args[1].get("is_lazy") is True  # Check is_lazy kwarg
        mock_run.assert_not_called()  # Should not run if exporting

        del os.environ["LAZY_CLI"]
        sys.argv = original_argv  # Restore original argv

    def test_execute_lazy_error_cases(self):
        lazy_entry = LazyEntrypoint("dummy")
        # Dry run
        ctx_dry = RunContext(name="lazy_test", dryrun=True)
        with pytest.raises(ValueError, match="Dry run is not supported for lazy execution"):
            ctx_dry.execute_lazy(lazy_entry, [], "lazy_test")
        # REPL
        ctx_repl = RunContext(name="lazy_test", repl=True)
        with pytest.raises(
            ValueError, match="Interactive mode is not supported for lazy execution"
        ):
            ctx_repl.execute_lazy(lazy_entry, [], "lazy_test")
        # Direct
        ctx_direct = RunContext(name="lazy_test", direct=True)
        with pytest.raises(
            ValueError, match="Direct execution is not supported for lazy execution"
        ):
            ctx_direct.execute_lazy(lazy_entry, [], "lazy_test")

    @patch("nemo_run.cli.api._serialize_configuration")
    @patch("fiddle.build")
    def test_execute_experiment_export(self, mock_build, mock_serialize, sample_experiment):
        ctx = RunContext(name="test_exp", to_json="exp_config.json", skip_confirmation=True)
        with patch("nemo_run.dryrun_fn"):  # Mock dryrun
            ctx.cli_execute(sample_experiment, ["a=5"], entrypoint_type="experiment")
        mock_serialize.assert_called_once()
        assert mock_serialize.call_args[0][3] == "exp_config.json"  # Check to_json path
        assert "is_lazy" in mock_serialize.call_args[1]
        assert mock_serialize.call_args[1]["is_lazy"] is False
        mock_build.assert_not_called()  # Should not build if exporting

    @patch("fiddle.build")
    def test_execute_experiment_normal(self, mock_build, sample_experiment):
        ctx = RunContext(name="test_exp", skip_confirmation=True)
        # Mock the build process to avoid actual execution
        mock_partial = Mock()
        mock_build.return_value = mock_partial
        with (
            patch("nemo_run.dryrun_fn"),
            patch("typer.confirm", return_value=True),
        ):  # Mock dryrun and confirmation
            ctx.cli_execute(sample_experiment, ["a=5", "b='exp'"], entrypoint_type="experiment")
        mock_build.assert_called_once()
        mock_partial.assert_called_once_with()  # Check that the built object is called

    def test_run_context_get_help(self):
        help_text = RunContext.get_help()
        assert "Represents the context for executing a run" in help_text

    def test_run_context_cli_command_defaults(self):
        app = typer.Typer()
        defaults = {"dryrun": True, "verbose": True}

        # Mock the actual execution logic inside the command
        with patch.object(RunContext, "cli_execute") as mock_cli_execute:
            # Create the command with defaults
            RunContext.cli_command(app, "testcmd", lambda: None, cmd_defaults=defaults)

            # Simulate calling the command with no overrides
            runner = CliRunner()
            runner.invoke(app, ["testcmd"])

            # Check that cli_execute was called with the context reflecting defaults
            mock_cli_execute.assert_called_once()
            # Can't directly check ctx_instance attributes as it's created inside the closure,
            # but we can check if the options passed to _configure_global_options reflect defaults
            with patch("nemo_run.cli.api._configure_global_options") as mock_configure:
                runner.invoke(app, ["testcmd"])
                mock_configure.assert_called_with(
                    app, False, True, True, None, True
                )  # verbose=True expected


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

    def test_forward_ref_with_real_type_factory(self):
        """Test that ForwardRef works when factory is registered for the actual type."""

        # Function that uses ForwardRef to the module-level RealType class
        def func(param: Optional["RealType"] = None):
            pass

        from test.dummy_type import RealType as _RealType

        # Register the factory in the module's global namespace
        # The factory returns a RealType instance with a specific value
        @run.cli.factory
        @run.autoconvert
        def real_type_factory() -> _RealType:
            return _RealType(value=100)

        @run.cli.factory(target=func, target_arg="param")
        @run.autoconvert
        def other_factory() -> _RealType:
            return _RealType(value=200)

        try:
            # Now test parsing works using the factory name
            result = cli_api.parse_cli_args(func, ["param=real_type_factory"])
            assert isinstance(result.param, run.Config)
            assert result.param.value == 100

            result = cli_api.parse_cli_args(func, ["param=other_factory"])
            assert isinstance(result.param, run.Config)
            assert result.param.value == 200

        finally:
            # Clean up - remove the factory from registry
            if hasattr(sys.modules[__name__], "real_type_factory"):
                delattr(sys.modules[__name__], "real_type_factory")


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
class Trainer:
    """Dummy trainer config"""

    model: Model
    learning_rate: float = 0.001


@run.cli.factory
@run.autoconvert
def my_model(hidden_size: int = 256, num_layers: int = 3, activation: str = "relu") -> Model:
    """Create a model configuration."""
    return Model(hidden_size=hidden_size, num_layers=num_layers, activation=activation)


@run.cli.factory
@run.autoconvert
def my_other_model(hidden_size: int = 512, num_layers: int = 3, activation: str = "relu") -> Model:
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


@run.cli.entrypoint(
    namespace="my_llm",
    skip_confirmation=True,
)
def train_model_default_optimizer(
    model: Model,
    optimizer: Annotated[Optional[Optimizer], run.Config[Optimizer]] = None,
    epochs: int = 10,
    batch_size: int = 32,
):
    if optimizer is None:
        optimizer = Optimizer()

    return train_model(model, optimizer, epochs, batch_size)


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

    def test_with_defaults_no_optimizer(self, runner, app):
        # Test CLI execution with default factory
        result = runner.invoke(
            app,
            [
                "my_llm",
                "train_model_default_optimizer",
                "model=my_model(hidden_size=1024)",
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
        assert "Epochs: 30" in output
        assert "Batch size: 32" in output
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

    def test_with_factory_and_overwrite(self, runner, app):
        # Test CLI execution with factory and parameter overwrite
        result = runner.invoke(
            app,
            [
                "my_llm",
                "train_model",
                "model=my_other_model",
                "model.num_layers=10",
                "--yes",
            ],
            env={"INCLUDE_WORKSPACE_FILE": "false"},
        )
        assert result.exit_code == 0

        output = result.stdout
        assert "Training model with the following configuration:" in output
        # Check that my_model_2's default hidden_size (512) is used
        assert "Model: Model(hidden_size=512, num_layers=10, activation='relu')" in output


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


@pytest.fixture
def runner():
    return CliRunner()


class TestGlobalOptions:
    @pytest.fixture(autouse=True)
    def _setup(self):
        """Setup for all test cases"""
        # Store original environment for cleanup
        self.original_env = os.environ.copy()
        yield
        # Restore environment after each test
        os.environ.clear()
        os.environ.update(self.original_env)

    @pytest.fixture
    def app(self):
        app = typer.Typer()

        # Add test command that throws an error
        @app.command()
        def error_command():
            """Command that throws a test exception"""
            raise ValueError("Test error for exception handling")

        # Add global options to test app
        add_global_options(app)
        return app

    def test_verbose_logging(self, runner, app):
        """Test verbose logging functionality"""
        with patch("nemo_run.cli.api.configure_logging") as mock_configure:
            # Test enabled
            runner.invoke(app, ["-v", "error-command"])
            mock_configure.assert_called_once_with(True)

            # Test disabled
            mock_configure.reset_mock()
            runner.invoke(app, ["error-command"])
            mock_configure.assert_called_once_with(False)


class TestTorchrunAndConfirmation:
    """Test torchrun detection and confirmation behavior."""

    @patch("os.environ", {"WORLD_SIZE": "2"})
    def test_is_torchrun_true(self):
        """Test that _is_torchrun returns True when WORLD_SIZE > 1."""
        from nemo_run.cli.api import _is_torchrun

        assert _is_torchrun() is True

    @patch("os.environ", {})
    def test_is_torchrun_false_no_env(self):
        """Test that _is_torchrun returns False when WORLD_SIZE not in environment."""
        from nemo_run.cli.api import _is_torchrun

        assert _is_torchrun() is False

    @patch("os.environ", {"WORLD_SIZE": "1"})
    def test_is_torchrun_false_size_one(self):
        """Test that _is_torchrun returns False when WORLD_SIZE = 1."""
        from nemo_run.cli.api import _is_torchrun

        assert _is_torchrun() is False

    @patch("nemo_run.cli.api._is_torchrun", return_value=True)
    def test_should_continue_torchrun(self, mock_torchrun):
        """Test that _should_continue returns True under torchrun."""
        ctx = run.cli.RunContext(name="test")
        assert ctx._should_continue(False) is True
        mock_torchrun.assert_called_once()

    @patch("nemo_run.cli.api._is_torchrun", return_value=False)
    @patch("nemo_run.cli.api.NEMORUN_SKIP_CONFIRMATION", True)
    def test_should_continue_global_flag_true(self, mock_torchrun):
        """Test that _should_continue respects global NEMORUN_SKIP_CONFIRMATION flag."""
        ctx = run.cli.RunContext(name="test")
        assert ctx._should_continue(False) is True
        mock_torchrun.assert_called_once()

    @patch("nemo_run.cli.api._is_torchrun", return_value=False)
    @patch("nemo_run.cli.api.NEMORUN_SKIP_CONFIRMATION", False)
    def test_should_continue_global_flag_false(self, mock_torchrun):
        """Test that _should_continue respects global NEMORUN_SKIP_CONFIRMATION flag."""
        ctx = run.cli.RunContext(name="test")
        assert ctx._should_continue(False) is False
        mock_torchrun.assert_called_once()

    @patch("nemo_run.cli.api._is_torchrun", return_value=False)
    @patch("nemo_run.cli.api.NEMORUN_SKIP_CONFIRMATION", None)
    def test_should_continue_skip_confirmation(self, mock_torchrun):
        """Test that _should_continue respects skip_confirmation parameter."""
        ctx = run.cli.RunContext(name="test")
        assert ctx._should_continue(True) is True
        mock_torchrun.assert_called_once()

    @patch("nemo_run.cli.api._is_torchrun", return_value=False)
    @patch("nemo_run.cli.api.NEMORUN_SKIP_CONFIRMATION", None)
    @patch("typer.confirm", return_value=True)
    def test_should_continue_confirm_yes(self, mock_confirm, mock_torchrun):
        """Test _should_continue when user confirms."""
        ctx = run.cli.RunContext(name="test")
        cli_api.NEMORUN_SKIP_CONFIRMATION = None  # Reset global state
        assert ctx._should_continue(False) is True
        mock_torchrun.assert_called_once()
        mock_confirm.assert_called_once_with("Continue?")
        assert cli_api.NEMORUN_SKIP_CONFIRMATION is True

    @patch("nemo_run.cli.api._is_torchrun", return_value=False)
    @patch("nemo_run.cli.api.NEMORUN_SKIP_CONFIRMATION", None)
    @patch("typer.confirm", return_value=False)
    def test_should_continue_confirm_no(self, mock_confirm, mock_torchrun):
        """Test _should_continue when user denies."""
        ctx = run.cli.RunContext(name="test")
        cli_api.NEMORUN_SKIP_CONFIRMATION = None  # Reset global state
        assert ctx._should_continue(False) is False
        mock_torchrun.assert_called_once()
        mock_confirm.assert_called_once_with("Continue?")
        assert cli_api.NEMORUN_SKIP_CONFIRMATION is False


class TestRunContextLaunch:
    """Test RunContext.launch method."""

    def test_launch_with_dryrun(self):
        """Test launch with dryrun."""
        ctx = run.cli.RunContext(name="test_run", dryrun=True)
        mock_experiment = Mock(spec=run.Experiment)

        ctx.launch(mock_experiment)

        mock_experiment.dryrun.assert_called_once()
        mock_experiment.run.assert_not_called()

    def test_launch_normal(self):
        """Test launch without dryrun."""
        ctx = run.cli.RunContext(name="test_run", direct=True, tail_logs=True)
        mock_experiment = Mock(spec=run.Experiment)

        ctx.launch(mock_experiment)

        mock_experiment.run.assert_called_once_with(
            sequential=False, detach=False, direct=True, tail_logs=True
        )

    def test_launch_with_executor(self):
        """Test launch with executor specified."""
        ctx = run.cli.RunContext(name="test_run")
        ctx.executor = Mock(spec=run.LocalExecutor)
        mock_experiment = Mock(spec=run.Experiment)

        ctx.launch(mock_experiment)

        mock_experiment.run.assert_called_once_with(
            sequential=False, detach=False, direct=False, tail_logs=False
        )

    def test_launch_sequential(self):
        """Test launch with sequential=True."""
        ctx = run.cli.RunContext(name="test_run")
        # Initialize executor to None explicitly
        ctx.executor = None
        mock_experiment = Mock(spec=run.Experiment)

        ctx.launch(mock_experiment, sequential=True)

        mock_experiment.run.assert_called_once_with(
            sequential=True, detach=False, direct=True, tail_logs=False
        )


class TestParsePrefixedArgs:
    """Test _parse_prefixed_args function."""

    def test_parse_prefixed_args_simple(self):
        """Test parsing simple prefixed arguments."""
        from nemo_run.cli.api import _parse_prefixed_args

        args = ["executor=local", "other=value"]
        prefix_value, prefix_args, other_args = _parse_prefixed_args(args, "executor")

        assert prefix_value == "local"
        assert prefix_args == []
        assert other_args == ["other=value"]

    def test_parse_prefixed_args_with_dot_notation(self):
        """Test parsing prefixed arguments with dot notation."""
        from nemo_run.cli.api import _parse_prefixed_args

        args = ["executor=local", "executor.gpu=2", "other=value"]
        prefix_value, prefix_args, other_args = _parse_prefixed_args(args, "executor")

        assert prefix_value == "local"
        assert prefix_args == ["gpu=2"]
        assert other_args == ["other=value"]

    def test_parse_prefixed_args_with_brackets(self):
        """Test parsing prefixed arguments with bracket notation."""
        from nemo_run.cli.api import _parse_prefixed_args

        args = ["plugins=list", "plugins[0].name=test", "other=value"]
        prefix_value, prefix_args, other_args = _parse_prefixed_args(args, "plugins")

        assert prefix_value == "list"
        assert prefix_args == ["[0].name=test"]
        assert other_args == ["other=value"]

    def test_parse_prefixed_args_invalid_format(self):
        """Test parsing prefixed arguments with invalid format."""
        from nemo_run.cli.api import _parse_prefixed_args

        args = ["executorblah", "other=value"]
        with pytest.raises(ValueError, match="Executor overwrites must start with 'executor.'"):
            _parse_prefixed_args(args, "executor")

    def test_parse_prefixed_args_no_prefix(self):
        """Test parsing when no prefixed arguments are present."""
        from nemo_run.cli.api import _parse_prefixed_args

        args = ["arg1=value1", "arg2=value2"]
        prefix_value, prefix_args, other_args = _parse_prefixed_args(args, "executor")

        assert prefix_value is None
        assert prefix_args == []
        assert other_args == ["arg1=value1", "arg2=value2"]


class TestConfigExport:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path

    def test_yaml_export(self, temp_dir):
        @run.autoconvert
        def my_model(hidden_size: int = 1000) -> Model:
            return Model(hidden_size)

        config = my_model(hidden_size=2000)
        yaml_path = temp_dir / "config.yaml"

        from nemo_run.cli.api import _serialize_configuration

        with patch("rich.console.Console") as mock_console:
            _serialize_configuration(config, to_yaml=str(yaml_path), console=mock_console)

        # Verify the YAML file was created and contains correct content
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "hidden_size: 2000" in content
        assert "_target_: test.cli.test_api.Model" in content

    def test_json_export(self, temp_dir):
        @run.autoconvert
        def my_model(hidden_size: int = 1000) -> Model:
            return Model(hidden_size)

        config = my_model(hidden_size=2000)
        json_path = temp_dir / "config.json"

        from nemo_run.cli.api import _serialize_configuration

        with patch("rich.console.Console") as mock_console:
            _serialize_configuration(config, to_json=str(json_path), console=mock_console)

        # Verify the JSON file was created and contains correct content
        assert json_path.exists()
        import json

        with open(json_path) as f:
            data = json.load(f)
        assert data["hidden_size"] == 2000
        assert data["_target_"] == "test.cli.test_api.Model"

    def test_toml_export(self, temp_dir):
        @run.autoconvert
        def my_model(hidden_size: int = 1000) -> Model:
            return Model(hidden_size)

        config = my_model(hidden_size=2000)
        toml_path = temp_dir / "config.toml"

        from nemo_run.cli.api import _serialize_configuration

        with patch("rich.console.Console") as mock_console:
            _serialize_configuration(config, to_toml=str(toml_path), console=mock_console)

        # Verify the TOML file was created and contains correct content
        assert toml_path.exists()
        import toml

        with open(toml_path) as f:
            data = toml.load(f)
        assert data["hidden_size"] == 2000
        assert data["_target_"] == "test.cli.test_api.Model"

    def test_lazy_config_export(self, temp_dir):
        from nemo_run.cli.lazy import LazyEntrypoint

        # Create a lazy configuration with nested structure
        lazy_config = LazyEntrypoint("test.cli.test_api.Model", factory="my_model_factory")
        lazy_config.hidden_size = 3000
        lazy_config.model = LazyEntrypoint("test.cli.test_api.SubModel")
        lazy_config.model.layers = [1, 2, 3]
        lazy_config.model.activation = "relu"
        lazy_config.optimizer.learning_rate = 0.001

        yaml_path = temp_dir / "lazy_config.yaml"

        from nemo_run.cli.api import _serialize_configuration

        with patch("rich.console.Console") as mock_console:
            _serialize_configuration(
                lazy_config, to_yaml=str(yaml_path), is_lazy=True, console=mock_console
            )

        # Verify the YAML file was created with lazy configuration format
        assert yaml_path.exists()
        content = yaml_path.read_text()

        # The content should look like:
        # _factory_: my_model_factory
        # _target_: test.cli.test_api.Model
        # hidden_size: 3000
        # model:
        #   activation: relu
        #   layers:
        #   - 1
        #   - 2
        #   - 3
        # optimizer:
        #   learning_rate: 0.001

        assert "_factory_: my_model_factory" in content
        assert "_target_: test.cli.test_api.Model" in content
        assert "hidden_size: 3000" in content
        assert "model:" in content
        assert "  activation: relu" in content
        assert "  layers:" in content
        assert "  - 1" in content
        assert "  - 2" in content
        assert "  - 3" in content
        assert "optimizer:" in content
        assert "  learning_rate: 0.001" in content

    def test_section_export(self, temp_dir):
        @run.autoconvert
        def my_trainer(model: Model, learning_rate: float = 0.001) -> Trainer:
            return Trainer(model=model)

        model_config = run.Config(Model, hidden_size=2000, num_layers=3, activation="relu")
        config = my_trainer(model=model_config, learning_rate=0.01)

        # Export just the model section
        yaml_path = temp_dir / "model_section.yaml"

        from nemo_run.cli.api import _serialize_configuration

        with patch("rich.console.Console") as mock_console:
            _serialize_configuration(config, to_yaml=f"{yaml_path}:model", console=mock_console)

        # Verify only the model section was exported
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "hidden_size: 2000" in content
        assert "num_layers: 3" in content
        assert "activation: relu" in content
        assert "_target_: test.cli.test_api.Model" in content
        assert "learning_rate" not in content

    def test_multiple_formats_export(self, temp_dir):
        @run.autoconvert
        def my_model(hidden_size: int = 1000) -> Model:
            return Model(hidden_size)

        config = my_model(hidden_size=2000)
        yaml_path = temp_dir / "config.yaml"
        json_path = temp_dir / "config.json"
        toml_path = temp_dir / "config.toml"

        from nemo_run.cli.api import _serialize_configuration

        with patch("rich.console.Console") as mock_console:
            _serialize_configuration(
                config,
                to_yaml=str(yaml_path),
                to_json=str(json_path),
                to_toml=str(toml_path),
                console=mock_console,
            )

        # Verify all files were created with correct content
        assert all(p.exists() for p in [yaml_path, json_path, toml_path])

        # Check YAML content
        yaml_content = yaml_path.read_text()
        assert "hidden_size: 2000" in yaml_content
        assert "_target_: test.cli.test_api.Model" in yaml_content

        # Check JSON content
        import json

        with open(json_path) as f:
            json_data = json.load(f)
        assert json_data["hidden_size"] == 2000
        assert json_data["_target_"] == "test.cli.test_api.Model"

        # Check TOML content
        import toml

        with open(toml_path) as f:
            toml_data = toml.load(f)
        assert toml_data["hidden_size"] == 2000
        assert toml_data["_target_"] == "test.cli.test_api.Model"

    def test_invalid_section_export(self, temp_dir):
        @run.autoconvert
        def my_model(hidden_size: int = 1000) -> Model:
            return Model(hidden_size)

        config = my_model(hidden_size=2000)
        yaml_path = temp_dir / "config.yaml"

        from nemo_run.cli.api import _serialize_configuration

        # Try to export a non-existent section
        with pytest.raises(ValueError, match="Section 'invalid' not found in configuration"):
            _serialize_configuration(
                config,
                to_yaml=f"{yaml_path}:invalid",
            )

    def test_export_verbose(self, temp_dir):
        @run.autoconvert
        def my_model() -> Model:
            return Model(hidden_size=10, num_layers=1, activation="test")

        config = my_model()
        yaml_path = temp_dir / "verbose.yaml"
        json_path = temp_dir / "verbose.json"

        from nemo_run.cli.api import _serialize_configuration

        mock_console = Mock(spec=Console)

        _serialize_configuration(
            config,
            to_yaml=str(yaml_path),
            to_json=str(json_path),
            console=mock_console,
            verbose=True,
        )

        # Check that console print was called multiple times for verbose output
        assert mock_console.print.call_count > 2
        mock_console.print.assert_any_call(
            f"[bold green]Configuration exported to YAML:[/bold green] {yaml_path}"
        )
        mock_console.print.assert_any_call("[bold cyan]File contents:[/bold cyan]")
        mock_console.print.assert_any_call(
            f"[bold green]Configuration exported to JSON:[/bold green] {json_path}"
        )

    def test_export_error_handling(self, temp_dir):
        config = Config(Model, hidden_size=100)
        non_existent_path = temp_dir / "non_existent_dir" / "config.yaml"

        from nemo_run.cli.api import _serialize_configuration

        mock_console = Mock(spec=Console)

        with pytest.raises(Exception):  # Expecting FileNotFoundError or similar
            _serialize_configuration(
                config,
                to_yaml=str(non_existent_path),
                console=mock_console,
                verbose=True,  # Test error printing in verbose mode
            )

        # Check that error message was printed
        expected_error_msg = str(
            FileNotFoundError(f"[Errno 2] No such file or directory: '{str(non_existent_path)}'")
        )
        mock_console.print.assert_called_with(
            f"[bold red]Failed to export configuration to YAML:[/bold red] {expected_error_msg}"
        )

    def test_export_no_format_error(self):
        from nemo_run.cli.api import _serialize_configuration

        with pytest.raises(ValueError, match="At least one output format must be provided"):
            _serialize_configuration(Config(int))  # Dummy config


class TestWorkspaceLoading:
    @pytest.fixture(autouse=True)
    def _setup_teardown(self, tmp_path, monkeypatch):
        # Setup
        original_cwd = os.getcwd()
        nemorun_home_path = tmp_path / ".nemorun_home"
        nemorun_home_path.mkdir()
        nemorun_home_str = str(nemorun_home_path)

        monkeypatch.setenv("INCLUDE_WORKSPACE_FILE", "true")
        monkeypatch.setattr(cli_api, "get_nemorun_home", lambda: nemorun_home_str)

        # Change directory for test file creation AND for the function under test
        os.chdir(str(tmp_path))

        # Clear cache *before* test runs
        cli_api._load_workspace.cache_clear()

        yield tmp_path, nemorun_home_path

        # Teardown
        os.chdir(original_cwd)  # Restore original CWD
        cli_api._load_workspace.cache_clear()

    def test_search_workspace_file_not_found(self, _setup_teardown):
        # Fixture ensures CWD is tmp_path, no files created here
        assert _search_workspace_file() is None

    def test_search_workspace_file_disabled(self, _setup_teardown, monkeypatch):
        tmp_path, _ = _setup_teardown
        monkeypatch.setenv("INCLUDE_WORKSPACE_FILE", "false")
        ws_path = tmp_path / "workspace.py"
        ws_path.touch()  # Create a file that *would* be found otherwise
        assert _search_workspace_file() is None  # Should return None due to env var

    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_load_workspace_file(self, mock_module_from_spec, mock_spec_from_file, _setup_teardown):
        tmp_path, _ = _setup_teardown
        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        ws_path = tmp_path / "dummy_ws.py"
        ws_path.touch()
        _load_workspace_file(str(ws_path))
        mock_spec_from_file.assert_called_once_with("workspace", str(ws_path))
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_spec.loader.exec_module.assert_called_once_with(mock_module)

    @patch("nemo_run.cli.api._search_workspace_file")
    @patch("nemo_run.cli.api._load_workspace_file")
    def test_load_workspace(self, mock_load_file, mock_search_file, _setup_teardown):
        # Case 1: File found
        ws_path = "/fake/path/workspace.py"
        mock_search_file.return_value = ws_path
        _load_workspace()  # Call 1
        mock_search_file.assert_called_once()
        mock_load_file.assert_called_once_with(ws_path)

        # Reset mocks and *explicitly clear cache* before second call
        mock_search_file.reset_mock()
        mock_load_file.reset_mock()
        cli_api._load_workspace.cache_clear()  # Reset cache

        # Case 2: File not found
        mock_search_file.return_value = None
        _load_workspace()  # Call 2
        mock_search_file.assert_called_once()  # Should call search again
        mock_load_file.assert_not_called()

    @patch("nemo_run.cli.api._search_workspace_file")
    @patch("nemo_run.cli.api._load_workspace_file")
    def test_load_workspace_cached(self, mock_load_file, mock_search_file, _setup_teardown):
        # Test caching behavior
        ws_path = "/fake/path/workspace.py"
        mock_search_file.return_value = ws_path

        # First call
        _load_workspace()  # Call 1
        mock_search_file.assert_called_once()
        mock_load_file.assert_called_once_with(ws_path)

        # Reset counts but *not* the cache
        mock_search_file.reset_mock()
        mock_load_file.reset_mock()

        # Second call - should use cache
        _load_workspace()  # Call 2
        mock_search_file.assert_not_called()  # Should not call search
        mock_load_file.assert_not_called()  # Should not call load

    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    @patch("nemo_run.cli.api._search_workspace_file")
    def test_load_workspace_integration(
        self, mock_search, mock_module_from_spec, mock_spec_from_file, _setup_teardown, monkeypatch
    ):
        """Test integration of search and load."""
        tmp_path, _ = _setup_teardown
        ws_path = tmp_path / "workspace.py"
        ws_path.write_text("print('Workspace loaded!')")  # Create the file

        mock_search.return_value = str(ws_path)  # Make search find it

        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_spec_from_file.return_value = mock_spec
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module

        _load_workspace()  # Call under test

        mock_search.assert_called_once()
        mock_spec_from_file.assert_called_once_with("workspace", str(ws_path))
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_spec.loader.exec_module.assert_called_once_with(mock_module)


class TestMainFunction:
    """Tests for the main CLI function dispatcher."""

    @pytest.fixture
    def simple_func(self):
        """A plain function without decorators."""

        def _simple_func(arg: int):
            print(f"Simple func called with {arg}")

        return _simple_func

    @pytest.fixture
    def decorated_func(self):
        """A function already decorated with @entrypoint."""

        @cli.entrypoint(namespace="test_main", skip_confirmation=True)
        def _decorated_func(arg: int):
            print(f"Decorated func called with {arg}")

        return _decorated_func

    @pytest.fixture
    def mock_entrypoint_main(self):
        """Mocks the main method of the Entrypoint class."""
        with patch("nemo_run.cli.api.Entrypoint.main") as mock_main:
            yield mock_main

    @pytest.fixture(autouse=True)
    def reset_main_entrypoint(self):
        """Ensure MAIN_ENTRYPOINT is reset before/after tests."""
        original_main = cli_api.MAIN_ENTRYPOINT
        cli_api.MAIN_ENTRYPOINT = None
        yield
        cli_api.MAIN_ENTRYPOINT = original_main

    def test_main_basic_execution(self, decorated_func, mock_entrypoint_main):
        """Test calling main with a decorated function."""
        cmd_defaults = {"dryrun": True}
        cli_main(decorated_func, cmd_defaults=cmd_defaults)
        mock_entrypoint_main.assert_called_once_with(cmd_defaults)

    def test_main_applies_decorator(self, simple_func, mock_entrypoint_main):
        """Test that main applies @entrypoint if needed."""
        cmd_defaults = {"verbose": True}
        cli_main(simple_func, cmd_defaults=cmd_defaults)
        # Check that the function now has the cli_entrypoint attribute
        assert hasattr(simple_func, "cli_entrypoint")
        assert isinstance(simple_func.cli_entrypoint, Entrypoint)
        # Check that the entrypoint's main was called
        mock_entrypoint_main.assert_called_once_with(cmd_defaults)

    def test_main_applies_decorator_with_kwargs(self, simple_func, mock_entrypoint_main):
        """Test that main passes kwargs to the entrypoint decorator."""
        with patch("nemo_run.cli.api.entrypoint") as mock_entrypoint_decorator:
            # Need to mock the decorator to check its call args,
            # and also mock the returned object's main method.
            mock_entrypoint_instance = Mock()
            mock_entrypoint_decorator.return_value.return_value = mock_entrypoint_instance
            mock_entrypoint_instance.cli_entrypoint.main = Mock()  # Mock the main method here

            cli_main(simple_func, namespace="custom_ns", skip_confirmation=True)

            # Check decorator call
            mock_entrypoint_decorator.assert_called_once()
            call_kwargs = mock_entrypoint_decorator.call_args.kwargs
            assert call_kwargs.get("namespace") == "custom_ns"
            assert call_kwargs.get("skip_confirmation") is True

            # Check that the (mocked) entrypoint's main was called
            mock_entrypoint_instance.cli_entrypoint.main.assert_called_once_with(None)

    def test_main_default_overrides(self, decorated_func, mock_entrypoint_main):
        """Test overriding default factory, executor, plugins."""
        mock_factory = Mock()
        mock_executor = Mock(spec=Config)
        mock_plugins = [Mock(spec=Config)]

        original_factory = decorated_func.cli_entrypoint.default_factory
        original_executor = decorated_func.cli_entrypoint.default_executor
        original_plugins = decorated_func.cli_entrypoint.default_plugins

        # Mock the entrypoint's main to check attribute values *during* the call
        def check_defaults_and_restore(*args, **kwargs):
            assert decorated_func.cli_entrypoint.default_factory == mock_factory
            assert decorated_func.cli_entrypoint.default_executor == mock_executor
            assert decorated_func.cli_entrypoint.default_plugins == mock_plugins

        mock_entrypoint_main.side_effect = check_defaults_and_restore

        cli_main(
            decorated_func,
            default_factory=mock_factory,
            default_executor=mock_executor,
            default_plugins=mock_plugins,
        )

        # Check they were restored after the call
        assert decorated_func.cli_entrypoint.default_factory == original_factory
        assert decorated_func.cli_entrypoint.default_executor == original_executor
        assert decorated_func.cli_entrypoint.default_plugins == original_plugins
        mock_entrypoint_main.assert_called_once()  # Ensure it was called

    def test_main_lazy_cli_enabled_normal_func(
        self, decorated_func, mock_entrypoint_main, monkeypatch
    ):
        """Test lazy mode with a normal entrypoint."""
        monkeypatch.setenv("LAZY_CLI", "true")
        cli_main(decorated_func)
        assert cli_api.MAIN_ENTRYPOINT == decorated_func.cli_entrypoint
        mock_entrypoint_main.assert_not_called()  # main should not be called directly

    @patch("typer.Typer")
    @patch("nemo_run.cli.api.RunContext.cli_command")
    @patch("sys.argv", ["script.py", "lazy_cmd", "arg1=val1"])
    @patch(
        "nemo_run.cli.lazy.LazyTarget.__post_init__", lambda self: None
    )  # Prevent script path check
    def test_main_lazy_cli_disabled_lazy_func(
        self, mock_cli_command, mock_typer_cls, mock_entrypoint_main, monkeypatch
    ):
        """Test non-lazy mode with a LazyEntrypoint (should delegate to Typer)."""
        monkeypatch.setenv("LAZY_CLI", "false")  # Ensure it's off
        # Use a real path component even though __post_init__ is patched,
        # as the constructor might still do basic parsing.
        lazy_entry = LazyEntrypoint("some.module:func arg1=val1")  # This is the input fn to main
        mock_app_instance = Mock()
        mock_typer_cls.return_value = mock_app_instance

        cli_main(lazy_entry, default_factory=Mock(), default_executor=Mock())

        # Check that Typer app was created and cli_command was called to set it up
        mock_typer_cls.assert_called_once()
        mock_cli_command.assert_called_once()
        # Check args passed to cli_command
        call_args = mock_cli_command.call_args
        created_lazy_entry = call_args.args[2]  # This is the one created inside main from sys.argv
        assert call_args.args[0] == mock_app_instance  # parent app
        assert call_args.args[1] == "lazy_cmd"  # command name from sys.argv[1]
        # The LazyEntrypoint created *inside* main will be based on sys.argv
        assert isinstance(created_lazy_entry, LazyEntrypoint)
        # Adjust assertion to check the path within the internal _target_ object
        assert hasattr(created_lazy_entry, "_target_")
        assert hasattr(created_lazy_entry._target_, "import_path")
        assert created_lazy_entry._target_.import_path == "script.py lazy_cmd"
        # Check args are stored separately
        assert created_lazy_entry._args_ == [("arg1", "=", "val1")]

        assert call_args.kwargs.get("type") == "task"
        assert call_args.kwargs.get("default_factory") is not None
        assert call_args.kwargs.get("default_executor") is not None

        # Check that the app was run
        mock_app_instance.assert_called_once_with(standalone_mode=False)
        # The entrypoint's main method should NOT be called when setting up the typer app for a lazy entrypoint
        # We access the original mocked entrypoint's main via the fixture
        mock_entrypoint_main.assert_not_called()

    def test_main_invalid_plugins_type(self, decorated_func):
        """Test that providing non-Config plugins raises ValueError."""
        with pytest.raises(ValueError, match="must be a list of Config objects"):
            cli_main(decorated_func, default_plugins=[Mock()])  # List of non-Configs
        with pytest.raises(ValueError, match="must be a Config object"):
            cli_main(decorated_func, default_plugins=Mock())  # Single non-Config


class TestEntrypointCommandHelp:
    """Tests for the help formatting of EntrypointCommand."""

    @pytest.fixture
    def dummy_entrypoint_func(self):
        def _dummy_func(a: int):
            """Dummy help string."""
            pass

        return _dummy_func

    @pytest.fixture
    def mock_context_formatter(self):
        mock_ctx = Mock(spec=typer.Context)
        mock_formatter = Mock()
        return mock_ctx, mock_formatter

    @patch("nemo_run.cli.api.rich_utils.rich_format_help", return_value="Base Help")
    @patch("nemo_run.cli.api.rich_utils._get_rich_console")
    @patch("nemo_run.help.class_to_str")
    @patch("nemo_run.cli.api.Panel", autospec=True)
    @patch("nemo_run.cli.api.Table", autospec=True)
    def test_format_help_with_defaults(
        self,
        mock_table_cls,
        mock_panel_cls,
        mock_class_to_str,
        mock_get_console,
        mock_rich_format,
        dummy_entrypoint_func,
        mock_context_formatter,
    ):
        """Test help formatting when defaults are present."""
        mock_console = Mock(spec=Console)
        mock_get_console.return_value = mock_console
        mock_ctx, mock_formatter = mock_context_formatter
        mock_table_instance = mock_table_cls.return_value
        mock_table_instance.row_count = 3

        mock_factory = Mock()
        mock_executor = run.Config(run.LocalExecutor)
        # Replace WandbPlugin with a generic Mock
        mock_plugin_cls = Mock()
        mock_plugins = [run.Config(mock_plugin_cls)]
        mock_class_to_str.side_effect = ["FactoryStr", "ExecutorStr", "PluginsStr"]

        entrypoint = Entrypoint(
            dummy_entrypoint_func,
            namespace="test",
            default_factory=mock_factory,
            default_executor=mock_executor,
            default_plugins=mock_plugins,
        )
        entrypoint.help = Mock()

        cmd = EntrypointCommand(name="test_cmd", callback=dummy_entrypoint_func)
        cmd._entrypoint = entrypoint

        with patch("sys.argv", ["script.py", "test_cmd", "--help"]):
            result = cmd.format_help(mock_ctx, mock_formatter)

        assert result == "Base Help"
        mock_rich_format.assert_called_once()
        mock_get_console.assert_called_once()
        mock_table_cls.assert_called_once()
        mock_panel_cls.assert_called_once()

        assert mock_class_to_str.call_count == 3
        mock_class_to_str.assert_any_call(mock_factory)
        mock_class_to_str.assert_any_call(mock_executor)
        mock_class_to_str.assert_any_call(mock_plugins)

        assert mock_table_instance.add_row.call_count == 3
        mock_table_instance.add_row.assert_any_call("factory", "FactoryStr")
        mock_table_instance.add_row.assert_any_call("executor", "ExecutorStr")
        mock_table_instance.add_row.assert_any_call("plugins", "PluginsStr")

        mock_console.print.assert_called_once()
        mock_panel_cls.assert_called_once_with(
            mock_table_instance,
            title="Defaults",
            border_style=cli_api.rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title_align=cli_api.rich_utils.ALIGN_OPTIONS_PANEL,
        )
        mock_console.print.assert_called_once_with(mock_panel_cls.return_value)

        entrypoint.help.assert_called_once_with(mock_console, with_docs=False)

    @patch("nemo_run.cli.api.rich_utils.rich_format_help", return_value="Base Help")
    @patch("nemo_run.cli.api.rich_utils._get_rich_console")
    @patch("nemo_run.help.class_to_str")
    @patch("nemo_run.cli.api.Panel", autospec=True)
    @patch("nemo_run.cli.api.Table", autospec=True)
    def test_format_help_without_defaults(
        self,
        mock_table_cls,
        mock_panel_cls,
        mock_class_to_str,
        mock_get_console,
        mock_rich_format,
        dummy_entrypoint_func,
        mock_context_formatter,
    ):
        """Test help formatting when no defaults are present."""
        mock_console = Mock(spec=Console)
        mock_get_console.return_value = mock_console
        mock_ctx, mock_formatter = mock_context_formatter
        mock_table_instance = mock_table_cls.return_value
        # Configure row_count for the mock table instance
        mock_table_instance.row_count = 0

        entrypoint = Entrypoint(
            dummy_entrypoint_func,
            namespace="test",
        )
        entrypoint.help = Mock()

        cmd = EntrypointCommand(name="test_cmd", callback=dummy_entrypoint_func)
        cmd._entrypoint = entrypoint

        with patch("sys.argv", ["script.py", "test_cmd", "--help"]):
            result = cmd.format_help(mock_ctx, mock_formatter)

        assert result == "Base Help"
        mock_rich_format.assert_called_once()
        mock_get_console.assert_called_once()
        mock_table_cls.assert_called_once()
        mock_table_instance.add_row.assert_not_called()
        mock_panel_cls.assert_not_called()
        mock_class_to_str.assert_not_called()
        mock_console.print.assert_not_called()
        entrypoint.help.assert_called_once_with(mock_console, with_docs=False)

    @patch("nemo_run.cli.api.rich_utils.rich_format_help", return_value="Base Help")
    @patch("nemo_run.cli.api.rich_utils._get_rich_console")
    @patch("nemo_run.help.class_to_str")
    @patch("nemo_run.cli.api.Panel", autospec=True)
    @patch("nemo_run.cli.api.Table", autospec=True)
    def test_format_help_with_docs_flag(
        self,
        mock_table_cls,
        mock_panel_cls,
        mock_class_to_str,
        mock_get_console,
        mock_rich_format,
        dummy_entrypoint_func,
        mock_context_formatter,
    ):
        """Test help formatting checks sys.argv for --docs flag."""
        mock_console = Mock(spec=Console)
        mock_get_console.return_value = mock_console
        mock_ctx, mock_formatter = mock_context_formatter
        mock_table_instance = mock_table_cls.return_value
        # Configure row_count for the mock table instance
        mock_table_instance.row_count = 0

        entrypoint = Entrypoint(dummy_entrypoint_func, namespace="test")
        entrypoint.help = Mock()

        cmd = EntrypointCommand(name="test_cmd", callback=dummy_entrypoint_func)
        cmd._entrypoint = entrypoint

        # Mock sys.argv to include --docs
        with patch("sys.argv", ["script.py", "test_cmd", "--help", "--docs"]):
            cmd.format_help(mock_ctx, mock_formatter)

        entrypoint.help.assert_called_once_with(mock_console, with_docs=True)

        # Reset mock and test with -d flag
        entrypoint.help.reset_mock()
        with patch("sys.argv", ["script.py", "test_cmd", "--help", "-d"]):
            cmd.format_help(mock_ctx, mock_formatter)
        entrypoint.help.assert_called_once_with(mock_console, with_docs=True)


class TestExtractConstituentTypes:
    @pytest.mark.parametrize(
        "type_hint, expected_types",
        [
            (int, {int}),
            (str, {str}),
            (bool, {bool}),
            (float, {float}),
            (list[int], {list, int, list[int]}),
            (dict[str, float], {dict, str, float, dict[str, float]}),
            (Union[int, str], {int, str}),
            (Optional[int], {int}),  # Optional[T] is Union[T, NoneType]
            (list[Union[int, str]], {list, int, str, list[Union[int, str]]}),
            (dict[str, list[int]], {dict, str, list, int, dict[str, list[int]], list[int]}),
            (Optional[list[str]], {list, str, list[str]}),
            (Annotated[int, "meta"], {int}),
            (Annotated[list[str], "meta"], {list, str, list[str]}),
            (Annotated[Optional[dict[str, bool]], "meta"], {dict, str, bool, dict[str, bool]}),
            (Union[Annotated[int, "int_meta"], Annotated[str, "str_meta"]], {int, str}),
            (DummyModel, {DummyModel}),
            (Optional[DummyModel], {DummyModel}),
            (list[DummyModel], {list, DummyModel, list[DummyModel]}),
        ],
    )
    def test_various_type_hints(self, type_hint, expected_types):
        """Test get_underlying_types with various type hints."""
        assert extract_constituent_types(type_hint) == expected_types
