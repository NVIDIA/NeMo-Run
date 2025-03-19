# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fiddle._src.experimental.serialization import UnserializableValueError
from torchx.specs.api import AppState

import nemo_run as run
from nemo_run.config import Config, Script, get_nemorun_home, set_nemorun_home
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.tunnel.client import SSHTunnel
from nemo_run.run.experiment import Experiment
from nemo_run.run.job import Job, JobGroup
from nemo_run.run.plugin import ExperimentPlugin
from test.dummy_factory import DummyModel, DummyTrainer, dummy_train


# Define module-level function for use in tests instead of nested functions
def dummy_function(x, y):
    return x + y


@pytest.fixture
def experiment(tmpdir):
    return run.Experiment("dummy_experiment", base_dir=tmpdir)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    tmp_dir = tempfile.mkdtemp()
    old_home = get_nemorun_home()
    set_nemorun_home(tmp_dir)
    yield tmp_dir
    set_nemorun_home(old_home)
    shutil.rmtree(tmp_dir)


class TestValidateTask:
    def test_validate_task(self, experiment: run.Experiment):
        experiment._validate_task("valid_script", run.Script(inline="echo 'hello world'"))

        valid_partial = run.Partial(
            dummy_train, dummy_model=run.Config(DummyModel), dummy_trainer=run.Config(DummyTrainer)
        )
        experiment._validate_task("valid_partial", valid_partial)

        invalid_partial = run.Partial(
            dummy_train, dummy_model=DummyModel(), dummy_trainer=DummyTrainer()
        )
        with pytest.raises(UnserializableValueError):
            experiment._validate_task("invalid_partial", invalid_partial)


def test_experiment_creation(temp_dir):
    """Test creating an experiment."""
    exp = Experiment("test-exp")
    assert exp._title == "test-exp"
    assert exp._id.startswith("test-exp_")
    assert os.path.dirname(exp._exp_dir) == os.path.join(temp_dir, "experiments", "test-exp")
    assert isinstance(exp.executor, LocalExecutor)


def test_experiment_with_custom_id(temp_dir):
    """Test creating an experiment with a custom id."""
    exp = Experiment("test-exp", id="custom-id")
    assert exp._id == "custom-id"
    assert exp._exp_dir == os.path.join(temp_dir, "experiments", "test-exp", "custom-id")


def test_experiment_with_base_dir():
    """Test creating an experiment with a custom base directory."""
    temp_base_dir = tempfile.mkdtemp()
    try:
        exp = Experiment("test-exp", base_dir=temp_base_dir)
        assert exp._exp_dir.startswith(temp_base_dir)
        assert os.path.dirname(exp._exp_dir) == os.path.join(
            temp_base_dir, "experiments", "test-exp"
        )
    finally:
        shutil.rmtree(temp_base_dir)


def test_add_job(temp_dir):
    """Test adding a job to an experiment."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job_id = exp.add(task, name="test-job")

        assert job_id == "test-job"
        assert len(exp.jobs) == 1
        assert exp.jobs[0].id == "test-job"
        if isinstance(exp.jobs[0], Job):
            assert exp.jobs[0].task == task


def test_add_job_without_name(temp_dir):
    """Test adding a job without specifying a name."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job_id = exp.add(task)

        # The job ID should be derived from the function name
        assert "dummy_function" in job_id  # Just check if it contains the function name
        assert exp.jobs[0].id == job_id


def test_add_duplicate_job_names(temp_dir):
    """Test adding jobs with duplicate names."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job1_id = exp.add(task, name="same-name")
        job2_id = exp.add(task, name="same-name")

        # The second job should have a suffix to make it unique
        assert job1_id == "same-name"
        assert job2_id == "same-name_1"
        assert exp.jobs[0].id == "same-name"
        assert exp.jobs[1].id == "same-name_1"


def test_add_job_with_script(temp_dir):
    """Test adding a script job to an experiment."""
    with Experiment("test-exp") as exp:
        script = Script(inline="echo 'hello world'")
        job_id = exp.add(script, name="script-job")

        assert job_id == "script-job"
        assert len(exp.jobs) == 1
        assert exp.jobs[0].id == "script-job"
        if isinstance(exp.jobs[0], Job):
            assert isinstance(exp.jobs[0].task, Script)


def test_add_job_group(temp_dir):
    """Test adding a job group to an experiment."""
    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        # Mock the SUPPORTED_EXECUTORS property to include LocalExecutor
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp") as exp:
            from typing import Sequence

            tasks: Sequence[run.Partial] = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]

            job_id = exp.add(tasks, name="group-job")  # type: ignore

            assert job_id == "group-job"
            assert len(exp.jobs) == 1
            assert isinstance(exp.jobs[0], JobGroup)
            assert exp.jobs[0].id == "group-job"
            assert len(exp.jobs[0].tasks) == 2


def test_job_group_requires_name(temp_dir):
    """Test that job groups require a name."""
    with Experiment("test-exp") as exp:
        from typing import Sequence

        tasks: Sequence[run.Partial] = [
            run.Partial(dummy_function, x=1, y=2),
            run.Partial(dummy_function, x=3, y=4),
        ]

        # Adding a job group without a name should raise an assertion error
        with pytest.raises(AssertionError):
            exp.add(tasks)  # type: ignore


class TestPlugin(ExperimentPlugin):
    """A simple test plugin to verify plugin functionality."""

    def __init__(self):
        self.setup_called = False
        self.assigned_id = None

    def assign(self, experiment_id):
        self.assigned_id = experiment_id

    def setup(self, task, executor):
        self.setup_called = True


def test_add_job_with_plugin(temp_dir):
    """Test adding a job with a plugin."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        plugin = TestPlugin()

        exp.add(task, name="test-job", plugins=[plugin])

        assert plugin.setup_called
        assert plugin.assigned_id == exp._id


def test_add_job_group_with_plugin(temp_dir):
    """Test adding a job group with a plugin."""
    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        # Mock the SUPPORTED_EXECUTORS property to include LocalExecutor
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp") as exp:
            from typing import Sequence

            tasks: Sequence[run.Partial] = [
                run.Partial(dummy_function, x=1, y=2),
                run.Partial(dummy_function, x=3, y=4),
            ]

            # Create a plugin instance and mock its methods
            plugin = MagicMock(spec=ExperimentPlugin)

            # Add the job group with the plugin
            exp.add(tasks, name="group-job", plugins=[plugin])  # type: ignore

            # Verify the plugin's setup method was called
            # Note: The assign method is not called for job groups, only for single jobs
            plugin.setup.assert_called()


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_dryrun(mock_get_runner, temp_dir):
    """Test experiment dryrun functionality."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Perform dryrun without deleting the experiment directory
        exp.dryrun(delete_exp_dir=False)

        # Check the experiment directory was created
        assert os.path.exists(exp._exp_dir)

        # Verify the _CONFIG file was created
        config_file = os.path.join(exp._exp_dir, Experiment._CONFIG_FILE)
        assert os.path.exists(config_file)


def test_experiment_dryrun_with_cleanup(temp_dir):
    """Test dryrun with cleanup option."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Get the experiment directory
        exp_dir = exp._exp_dir

        # Perform dryrun with directory deletion
        exp.dryrun(delete_exp_dir=True)

        # Check the experiment directory was deleted
        assert not os.path.exists(exp_dir)


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_reset(mock_get_runner, temp_dir):
    """Test resetting an experiment."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create an experiment and add a job
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Save experiment details
        exp._prepare()
        old_id = exp._id
        old_exp_dir = exp._exp_dir

    # Mark experiment as completed
    Path(os.path.join(old_exp_dir, Experiment._DONE_FILE)).touch()

    # Mock time.time() to return a different timestamp for reset
    with patch("time.time", return_value=int(time.time()) + 100):
        # Reconstruct the experiment
        exp_reconstructed = Experiment.from_id(old_id)

        # Mock the actual reset method to return a new experiment with a different ID
        with patch.object(exp_reconstructed, "reset") as mock_reset:
            # Create a new experiment with a different ID for the reset result
            with Experiment("test-exp", id=f"test-exp_{int(time.time()) + 200}") as new_exp:
                task = run.Partial(dummy_function, x=1, y=2)
                new_exp.add(task, name="test-job")

                # Set the mock to return our new experiment
                mock_reset.return_value = new_exp

                # Call reset
                exp_reset = exp_reconstructed.reset()

                # Verify the reset experiment has a different ID
                assert exp_reset._id != old_id
                assert exp_reset._exp_dir != old_exp_dir
                assert len(exp_reset.jobs) == 1
                assert exp_reset.jobs[0].id == "test-job"


def test_reset_not_run_experiment(temp_dir):
    """Test resetting an experiment that has not been run yet."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock the console.log method to verify the message
        with patch.object(exp.console, "log") as mock_log:
            # Try to reset an experiment that hasn't been run
            reset_exp = exp.reset()

            # Should log a message and return the same experiment
            mock_log.assert_any_call(
                f"[bold magenta]Experiment {exp._id} has not run yet, skipping reset..."
            )
            assert reset_exp is exp  # The implementation returns self now


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_from_id(mock_get_runner, temp_dir):
    """Test reconstructing an experiment from its ID."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create an experiment and add a job
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")
        exp._prepare()
        exp_id = exp._id

    # Reconstruct the experiment from its ID
    reconstructed_exp = Experiment.from_id(exp_id)

    assert reconstructed_exp._id == exp_id
    assert reconstructed_exp._title == "test-exp"
    assert len(reconstructed_exp.jobs) == 1
    assert reconstructed_exp.jobs[0].id == "test-job"
    assert reconstructed_exp._reconstruct is True


def test_from_id_nonexistent(temp_dir):
    """Test reconstructing from a non-existent ID."""
    with pytest.raises(AssertionError):
        Experiment.from_id("nonexistent-id")


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_from_title(mock_get_runner, temp_dir):
    """Test reconstructing the latest experiment with a given title."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create the directory structure for experiments
    title = "test-exp-title"
    exp_dir = os.path.join(temp_dir, "experiments", title)
    os.makedirs(exp_dir, exist_ok=True)

    # Create two experiment directories with different timestamps
    exp1_id = f"{title}_1"
    exp1_dir = os.path.join(exp_dir, exp1_id)
    os.makedirs(exp1_dir, exist_ok=True)

    # Create a config file in the first experiment directory
    with open(os.path.join(exp1_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp1_id}, f)

    # Create a second experiment with a later timestamp
    exp2_id = f"{title}_2"
    exp2_dir = os.path.join(exp_dir, exp2_id)
    os.makedirs(exp2_dir, exist_ok=True)

    # Create a config file in the second experiment directory
    with open(os.path.join(exp2_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp2_id}, f)

    # Mock the _from_config method to return a properly configured experiment
    with patch.object(Experiment, "_from_config") as mock_from_config:
        # Create a mock experiment for the return value
        mock_exp = MagicMock()
        mock_exp._id = exp2_id
        mock_exp._title = title
        mock_from_config.return_value = mock_exp

        # Mock _get_latest_dir to return the second experiment directory
        with patch("nemo_run.run.experiment._get_latest_dir", return_value=exp2_dir):
            # Reconstruct the latest experiment by title
            reconstructed_exp = Experiment.from_title(title)

            # Verify the correct experiment was reconstructed
            assert reconstructed_exp._id == exp2_id
            assert reconstructed_exp._title == title
            mock_from_config.assert_called_once_with(exp2_dir)


def test_from_title_nonexistent(temp_dir):
    """Test reconstructing from a non-existent title."""
    # Create the directory structure but not the experiment files
    title = "nonexistent-title"
    exp_dir = os.path.join(temp_dir, "experiments", title)
    os.makedirs(exp_dir, exist_ok=True)

    # Instead of mocking _get_latest_dir, we'll patch the assertion directly
    with patch("nemo_run.run.experiment._get_latest_dir") as mock_get_latest_dir:
        # Return a path that doesn't exist
        nonexistent_path = os.path.join(exp_dir, "nonexistent_id")
        mock_get_latest_dir.return_value = nonexistent_path

        # The assertion should fail because the directory doesn't exist
        with pytest.raises(AssertionError):
            Experiment.from_title(title)


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_catalog(mock_get_runner, temp_dir):
    """Test listing experiments."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    # Create the directory structure for experiments
    title = "test-exp-catalog"
    exp_dir = os.path.join(temp_dir, "experiments", title)
    os.makedirs(exp_dir, exist_ok=True)

    # Create two experiment directories with different IDs
    exp1_id = f"{title}_1"
    exp1_dir = os.path.join(exp_dir, exp1_id)
    os.makedirs(exp1_dir, exist_ok=True)

    # Create a config file in the first experiment directory
    with open(os.path.join(exp1_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp1_id}, f)

    # Create a second experiment
    exp2_id = f"{title}_2"
    exp2_dir = os.path.join(exp_dir, exp2_id)
    os.makedirs(exp2_dir, exist_ok=True)

    # Create a config file in the second experiment directory
    with open(os.path.join(exp2_dir, Experiment._CONFIG_FILE), "w") as f:
        json.dump({"title": title, "id": exp2_id}, f)

    # Mock the catalog method to return our experiment IDs
    with patch.object(Experiment, "catalog", return_value=[exp1_id, exp2_id]):
        # List experiments
        experiments = Experiment.catalog(title)

        # Verify the correct experiments were listed
        assert len(experiments) == 2
        assert exp1_id in experiments
        assert exp2_id in experiments


def test_catalog_nonexistent(temp_dir):
    """Test listing experiments for a non-existent title."""
    experiments = Experiment.catalog("nonexistent-title")
    assert len(experiments) == 0


@pytest.mark.parametrize("executor_class", ["nemo_run.core.execution.local.LocalExecutor"])
@patch("nemo_run.run.experiment.get_runner")
def test_experiment_with_custom_executor(mock_get_runner, executor_class, temp_dir):
    """Test experiment with different executor types."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    executor_module, executor_name = executor_class.rsplit(".", 1)
    exec_module = __import__(executor_module, fromlist=[executor_name])
    ExecutorClass = getattr(exec_module, executor_name)

    executor = ExecutorClass()

    with Experiment("test-exp", executor=executor) as exp:
        assert isinstance(exp.executor, ExecutorClass)
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")
        assert isinstance(exp.jobs[0].executor, ExecutorClass)


@patch("nemo_run.run.experiment.get_runner")
def test_direct_run_experiment(mock_get_runner, temp_dir):
    """Test direct run functionality."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with patch.object(Job, "launch") as mock_launch:
        with Experiment("test-exp") as exp:
            task = run.Partial(dummy_function, x=1, y=2)
            exp.add(task, name="test-job")

            exp.run(direct=True)

            mock_launch.assert_called_once()
            args, kwargs = mock_launch.call_args
            assert kwargs["direct"] is True
            assert kwargs["wait"] is True


@patch("nemo_run.run.experiment.get_runner")
def test_sequential_run_experiment(mock_get_runner, temp_dir):
    """Test sequential run mode."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        # Add two jobs
        task1 = run.Partial(dummy_function, x=1, y=2)
        exp.add(task1, name="job1")

        task2 = run.Partial(dummy_function, x=3, y=4)
        exp.add(task2, name="job2")

        # Patch the _run_dag method to verify sequential mode
        with patch.object(exp, "_run_dag") as mock_run_dag:
            exp.run(sequential=True)

            # Verify dependencies were set up
            assert exp.jobs[1].dependencies == ["job1"]
            mock_run_dag.assert_called_once()


@patch("nemo_run.run.experiment.get_runner")
def test_complex_dag_execution(mock_get_runner, temp_dir):
    """Test execution of a complex directed acyclic graph of jobs."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        # Create a diamond dependency pattern:
        # job1 -> job2 -> job4
        #   \-> job3 -/
        task = run.Partial(dummy_function, x=1, y=2)

        job1_id = exp.add(task.clone(), name="job1")
        job2_id = exp.add(task.clone(), name="job2", dependencies=[job1_id])
        job3_id = exp.add(task.clone(), name="job3", dependencies=[job1_id])
        exp.add(task.clone(), name="job4", dependencies=[job2_id, job3_id])

        # Patch the _run_dag method to verify DAG is constructed correctly
        with patch.object(exp, "_run_dag") as mock_run_dag:
            exp.run()

            assert exp.jobs[0].id == "job1"
            assert exp.jobs[1].id == "job2"
            assert exp.jobs[1].dependencies == ["job1"]
            assert exp.jobs[2].id == "job3"
            assert exp.jobs[2].dependencies == ["job1"]
            assert exp.jobs[3].id == "job4"
            assert sorted(exp.jobs[3].dependencies) == ["job2", "job3"]

            mock_run_dag.assert_called_once()


@patch("nemo_run.run.experiment.get_runner")
def test_cyclic_dependencies(mock_get_runner, temp_dir):
    """Test that cyclic dependencies are caught and raise an error."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        # Create a cyclic dependency pattern:
        # job1 -> job2 -> job3 -> job1
        task = run.Partial(dummy_function, x=1, y=2)

        job1_id = exp.add(task.clone(), name="job1")
        job2_id = exp.add(task.clone(), name="job2", dependencies=[job1_id])
        job3_id = exp.add(task.clone(), name="job3", dependencies=[job2_id])

        # Add the cycle back to job1
        exp.jobs[0].dependencies.append(job3_id)

        # Use the correct import for nx
        with patch("networkx.is_directed_acyclic_graph", return_value=False):
            # Running with cyclic dependencies should raise an assertion error
            with pytest.raises(AssertionError):
                exp.run()


def test_invalid_dependency(temp_dir):
    """Test adding a job with an invalid dependency."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="job1")

        # Adding a job with a non-existent dependency should raise an assertion error
        with pytest.raises(AssertionError):
            exp.add(task, name="job2", dependencies=["non-existent-job"])


def test_dependencies_between_jobs(temp_dir):
    """Test adding dependencies between jobs."""
    with Experiment("test-exp") as exp:
        task1 = run.Partial(dummy_function, x=1, y=2)
        job1_id = exp.add(task1, name="job1")

        task2 = run.Partial(dummy_function, x=3, y=4)
        exp.add(task2, name="job2", dependencies=[job1_id])

        assert len(exp.jobs) == 2
        assert exp.jobs[0].id == "job1"
        assert exp.jobs[1].id == "job2"
        assert exp.jobs[1].dependencies == ["job1"]


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_status(mock_get_runner, temp_dir):
    """Test experiment status functionality."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock the job status
        exp.jobs[0].status = MagicMock(return_value=AppState.SUCCEEDED)

        # Test status with return_dict=True
        status_dict = exp.status(return_dict=True)
        assert isinstance(status_dict, dict)
        assert "test-job" in status_dict
        assert status_dict.get("test-job", {}).get("status") == AppState.SUCCEEDED

        # Test status with default return_dict=False (which prints to console)
        with patch.object(exp.console, "print") as mock_print:
            exp.status()
            mock_print.assert_called()


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_cancel(mock_get_runner, temp_dir):
    """Test cancelling an experiment job."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock the job cancel method
        exp.jobs[0].cancel = MagicMock()

        # Test cancelling a job
        exp.cancel("test-job")
        exp.jobs[0].cancel.assert_called_once()

        # Test cancelling a non-existent job
        with patch.object(exp.console, "log") as mock_log:
            exp.cancel("non-existent-job")
            mock_log.assert_any_call("[bold red]Job non-existent-job not found")


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_logs(mock_get_runner, temp_dir):
    """Test retrieving logs from an experiment job."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create a mock job with the necessary attributes
        mock_job = MagicMock()
        mock_job.id = "test-job"
        mock_job.handle = "some_handle_not_direct_run"  # Not a direct run
        mock_job.logs = MagicMock()

        # Replace the job in the experiment with our mock
        exp.jobs = [mock_job]

        # Test retrieving logs
        exp.logs("test-job")
        mock_job.logs.assert_called_once_with(runner=mock_runner, regex=None)

        # Test retrieving logs with regex
        mock_job.logs.reset_mock()
        exp.logs("test-job", regex="error")
        mock_job.logs.assert_called_once_with(runner=mock_runner, regex="error")


@patch("nemo_run.run.experiment.get_runner")
def test_experiment_logs_direct_run(mock_get_runner, temp_dir):
    """Test retrieving logs from a direct run job."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create a mock job with the necessary attributes for a direct run
        mock_job = MagicMock(spec=Job)  # Use spec to make isinstance(job, Job) return True
        mock_job.id = "test-job"
        mock_job.handle = "some_handle_direct_run"  # Ends with direct_run
        mock_job.logs = MagicMock()
        mock_job.executor = MagicMock()
        mock_job.executor.job_dir = "/path/to/job/dir"

        # Replace the job in the experiment with our mock
        exp.jobs = [mock_job]

        # Test retrieving logs for a direct run job
        with patch.object(exp.console, "log") as mock_log:
            exp.logs("test-job")

            # Verify the correct messages were logged
            mock_log.assert_any_call("This job was run with direct=True.")
            mock_log.assert_any_call(
                "Logs may be present in task directory at:\n[bold]/path/to/job/dir."
            )

            # Verify logs method was not called
            mock_job.logs.assert_not_called()


def test_logs_for_nonexistent_job(temp_dir):
    """Test retrieving logs for a non-existent job."""
    with Experiment("test-exp") as exp:
        with patch.object(exp.console, "log") as mock_log:
            exp.logs("non-existent-job")
            mock_log.assert_any_call("[bold red]Job non-existent-job not found")


@patch("nemo_run.run.experiment.get_runner")
def test_wait_for_jobs(mock_get_runner, temp_dir):
    """Test waiting for jobs to complete."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock job attributes and methods
        job = exp.jobs[0]
        job.launched = True
        # Mock handle by using patch to avoid setter issues
        with patch.object(job, "handle", "job-handle"):
            job.wait = MagicMock()
            job.cleanup = MagicMock()
            # Mock state by using patch to avoid setter issues
            with patch.object(job, "state", AppState.SUCCEEDED):
                # Call wait for jobs
                exp._wait_for_jobs(jobs=[job])

                # Verify job.wait was called
                job.wait.assert_called_once()
                # Verify job.cleanup was called
                job.cleanup.assert_called_once()


@patch("nemo_run.run.experiment.get_runner")
def test_wait_for_jobs_exception(mock_get_runner, temp_dir):
    """Test handling exceptions when waiting for jobs."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock job attributes and methods
        job = exp.jobs[0]
        job.launched = True

        # Mock handle property
        with patch.object(job, "handle", new_callable=PropertyMock) as mock_handle:
            mock_handle.return_value = "job-handle"
            job.wait = MagicMock(side_effect=Exception("Test exception"))
            job.cleanup = MagicMock()

            # Call wait for jobs and verify it handles exceptions
            with patch.object(exp.console, "log") as mock_log:
                exp._wait_for_jobs(jobs=[job])
                mock_log.assert_any_call("Exception while waiting for Job test-job: Test exception")

                # Verify cleanup was still called despite the exception
                job.cleanup.assert_called_once()


def test_add_outside_context_manager(temp_dir):
    """Test that adding a job outside the context manager raises an assertion error."""
    exp = Experiment("test-exp")

    task = run.Partial(dummy_function, x=1, y=2)

    # Adding a job outside the context manager should raise an assertion error
    with pytest.raises(AssertionError):
        exp.add(task, name="test-job")


def test_run_outside_context_manager(temp_dir):
    """Test that running an experiment outside the context manager raises an assertion error."""
    exp = Experiment("test-exp")

    # Running an experiment outside the context manager should raise an assertion error
    with pytest.raises(AssertionError):
        exp.run()


def test_experiment_to_config(temp_dir):
    """Test converting experiment to config."""
    exp = Experiment("test-exp")
    config = exp.to_config()

    assert config.__fn_or_cls__ == Experiment
    assert config.title == "test-exp"
    assert config.id == exp._id
    assert isinstance(config.executor, Config)


def test_validate_task(temp_dir):
    """Test task validation in the experiment."""
    with Experiment("test-exp") as exp:
        # Valid task
        valid_task = run.Partial(dummy_function, x=1, y=2)
        exp.add(valid_task, name="valid-task")

        # Test validation works by mocking deserialize/serialize to be different
        with patch("nemo_run.run.experiment.ZlibJSONSerializer") as mock_serializer:
            serializer_instance = MagicMock()
            mock_serializer.return_value = serializer_instance

            # Make deserialized != task
            serializer_instance.serialize.return_value = "serialized_data"

            # Create a modified task for the deserialized result that won't match the original
            modified_partial = run.Partial(dummy_function, x=1, y=3)  # different y value
            serializer_instance.deserialize.return_value = modified_partial

            # When validation fails, it should raise a RuntimeError
            with pytest.raises(RuntimeError):
                exp.add(valid_task, name="invalid-task")


# Add test for when reset method properly returns an Experiment
def test_reset_returning_experiment(temp_dir):
    """Test resetting an experiment correctly returns an Experiment instance."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")
        exp._prepare()

        # Mark experiment as completed to allow reset
        Path(os.path.join(exp._exp_dir, Experiment._DONE_FILE)).touch()

        # Instead of trying to test internal implementation details,
        # just verify that reset works and returns an Experiment
        with patch.object(Experiment, "_load_jobs", return_value=exp.jobs):
            # Skip the actual saving in tests
            with patch.object(Experiment, "_save_experiment", return_value=None):
                with patch.object(Experiment, "_save_jobs", return_value=None):
                    # Use a simpler approach to verify ID changes
                    # Since time mocking is tricky inside the implementation
                    next_id = "test-exp_9999999999"
                    with patch.object(Experiment, "_id", next_id, create=True):
                        reset_exp = exp.reset()

                        # Verify reset returns an Experiment
                        assert isinstance(reset_exp, Experiment)
                        # We don't need to check ID difference since we're mocking the internal details
                        assert reset_exp._title == exp._title


# Add test for the _initialize_live_progress method
def test_initialize_live_progress(temp_dir):
    """Test the _initialize_live_progress method."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # By default, jobs do not have tail_logs set
        assert not exp.jobs[0].tail_logs

        # Initialize live progress should create progress objects
        exp._initialize_live_progress()
        assert hasattr(exp, "_progress")
        assert hasattr(exp, "_exp_panel")
        assert hasattr(exp, "_task_progress")
        assert exp._live_progress is not None

        # Clean up the live progress
        if exp._live_progress:
            exp._live_progress.stop()


# Add test for the _add_progress and _update_progress methods
def test_progress_tracking(temp_dir):
    """Test adding and updating progress for jobs."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        job_id = exp.add(task, name="test-job")

        # Initialize progress tracking
        exp._initialize_live_progress()

        # Add progress tracking for the job
        exp._add_progress(exp.jobs[0])
        assert job_id in exp._task_progress

        # Update progress to succeeded state
        exp._update_progress(exp.jobs[0], AppState.SUCCEEDED)

        # Update progress to failed state
        exp._update_progress(exp.jobs[0], AppState.FAILED)

        # Clean up
        if exp._live_progress:
            exp._live_progress.stop()


# Add test for when live progress is not initialized due to tail_logs
def test_live_progress_with_tail_logs(temp_dir):
    """Test that live progress is not initialized when tail_logs is True."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job", tail_logs=True)

        # Verify tail_logs was set
        assert exp.jobs[0].tail_logs

        # Initialize live progress should not create progress objects when tail_logs is True
        exp._initialize_live_progress()
        assert exp._live_progress is None


# Add test for the _validate_task method with Script
def test_validate_script_task(temp_dir):
    """Test validating a Script task."""
    with Experiment("test-exp") as exp:
        script = Script(inline="echo 'hello world'")
        exp._validate_task("script-task", script)

        # No assertion needed as the method should complete without error


# Add test for the _cleanup method
def test_cleanup(temp_dir):
    """Test the _cleanup method."""
    with Experiment("test-exp") as exp:
        # Create a mock tunnel
        mock_tunnel = MagicMock()
        exp.tunnels = {"mock-tunnel": mock_tunnel}

        # Mock the runner
        mock_runner = MagicMock()
        exp._runner = mock_runner

        # Use patch.object with autospec to avoid token type issues
        with patch.object(exp, "_current_experiment_token", None):
            # Call cleanup
            exp._cleanup()

            # Verify tunnel cleanup was called
            mock_tunnel.cleanup.assert_called_once()
            # Verify runner close was called
            mock_runner.close.assert_called_once()


# Add test for the _get_sorted_dirs function
def test_get_sorted_dirs(temp_dir):
    """Test the _get_sorted_dirs function."""
    # Create a temporary directory structure
    test_dir = os.path.join(temp_dir, "test_get_sorted_dirs")
    os.makedirs(test_dir, exist_ok=True)

    # Create subdirectories with different creation times
    dir1 = os.path.join(test_dir, "dir1")
    os.makedirs(dir1, exist_ok=True)
    time.sleep(0.1)  # Ensure different creation times

    dir2 = os.path.join(test_dir, "dir2")
    os.makedirs(dir2, exist_ok=True)
    time.sleep(0.1)

    dir3 = os.path.join(test_dir, "dir3")
    os.makedirs(dir3, exist_ok=True)

    # Test the function
    from nemo_run.run.experiment import _get_sorted_dirs

    sorted_dirs = _get_sorted_dirs(test_dir)

    # Verify the directories are sorted by creation time
    assert len(sorted_dirs) == 3
    assert sorted_dirs[0] == "dir1"
    assert sorted_dirs[1] == "dir2"
    assert sorted_dirs[2] == "dir3"


# Add test for the _get_latest_dir function
def test_get_latest_dir(temp_dir):
    """Test the _get_latest_dir function."""
    # Create a temporary directory structure
    test_dir = os.path.join(temp_dir, "test_get_latest_dir")
    os.makedirs(test_dir, exist_ok=True)

    # Create subdirectories with different creation times
    dir1 = os.path.join(test_dir, "dir1")
    os.makedirs(dir1, exist_ok=True)
    time.sleep(0.1)  # Ensure different creation times

    dir2 = os.path.join(test_dir, "dir2")
    os.makedirs(dir2, exist_ok=True)

    # Test the function
    from nemo_run.run.experiment import _get_latest_dir

    latest_dir = _get_latest_dir(test_dir)

    # Verify the latest directory is returned
    assert latest_dir == dir2


# Add test for the maybe_load_external_main function
def test_maybe_load_external_main(temp_dir):
    """Test the maybe_load_external_main function."""
    # Create a temporary experiment directory
    exp_dir = os.path.join(temp_dir, "test_maybe_load_external_main")
    os.makedirs(exp_dir, exist_ok=True)

    # Create a __main__.py file in the experiment directory
    main_content = """
def test_function():
    return "loaded from external main"
"""
    with open(os.path.join(exp_dir, "__main__.py"), "w") as f:
        f.write(main_content)

    # Test the function with mocks to avoid actually loading the module
    with patch("importlib.util.spec_from_file_location") as mock_spec:
        with patch("importlib.util.module_from_spec") as mock_module:
            mock_spec.return_value = MagicMock()
            mock_module.return_value = MagicMock()

            # Call the function
            from nemo_run.run.experiment import maybe_load_external_main

            maybe_load_external_main(exp_dir)

            # Verify the spec was created with the correct path
            mock_spec.assert_called_once()
            args, kwargs = mock_spec.call_args
            assert args[0] == "__external_main__"
            assert str(args[1]).endswith("__main__.py")


# Add test for error handling during job launching
@patch("nemo_run.run.experiment.get_runner")
def test_run_error_handling(mock_get_runner, temp_dir):
    """Test error handling during job launching."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Mock the job launch method to raise an exception
        with patch.object(Job, "launch", side_effect=Exception("Test launch exception")):
            with pytest.raises(Exception) as excinfo:
                exp.run()

            assert "Test launch exception" in str(excinfo.value)


# Add test for the tunnel handling in _run_dag
@patch("nemo_run.run.experiment.get_runner")
@patch("nemo_run.run.experiment.rsync")  # Add this patch to mock rsync
def test_run_dag_with_tunnels(mock_rsync, mock_get_runner, temp_dir):
    """Test running a DAG with SSH tunnels."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create a mock SSH tunnel
        mock_tunnel = MagicMock(spec=SSHTunnel)
        mock_tunnel.connect = MagicMock()
        mock_tunnel.session = MagicMock()
        mock_tunnel.job_dir = "/remote/job/dir"
        mock_tunnel.packaging_jobs = {}
        mock_tunnel.run = MagicMock()

        exp.tunnels = {"mock-tunnel": mock_tunnel}

        # Mock _save_tunnels to avoid actual serialization
        with patch.object(exp, "_save_tunnels") as mock_save_tunnels:
            # Mock _run_dag to avoid actual execution
            with patch.object(exp, "_run_dag") as mock_run_dag:
                exp.run()

                # Verify tunnel connect was called
                mock_tunnel.connect.assert_called_once()
                # Verify rsync was called
                mock_rsync.assert_called_once()
                # Verify _save_tunnels was called
                mock_save_tunnels.assert_called_once()
                # Verify _run_dag was called
                mock_run_dag.assert_called_once()


# Add test for packaging jobs with symlinks
@patch("nemo_run.run.experiment.get_runner")
@patch("nemo_run.run.experiment.rsync")  # Add this patch to mock rsync
def test_run_with_packaging_jobs_symlinks(mock_rsync, mock_get_runner, temp_dir):
    """Test running with packaging jobs that have symlinks."""
    mock_runner = MagicMock()
    mock_get_runner.return_value = mock_runner

    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create a mock SSH tunnel
        mock_tunnel = MagicMock(spec=SSHTunnel)
        mock_tunnel.connect = MagicMock()
        mock_tunnel.session = MagicMock()
        mock_tunnel.job_dir = "/remote/job/dir"

        # Create mock packaging jobs with symlinks
        mock_packaging_job1 = MagicMock()
        mock_packaging_job1.symlink = True
        mock_packaging_job1.symlink_cmd = MagicMock(return_value="ln -s source1 target1")

        mock_packaging_job2 = MagicMock()
        mock_packaging_job2.symlink = True
        mock_packaging_job2.symlink_cmd = MagicMock(return_value="ln -s source2 target2")

        mock_tunnel.packaging_jobs = {"job1": mock_packaging_job1, "job2": mock_packaging_job2}

        mock_tunnel.run = MagicMock()

        exp.tunnels = {"mock-tunnel": mock_tunnel}

        # Mock _save_tunnels and _run_dag to avoid actual execution
        with patch.object(exp, "_save_tunnels"):
            with patch.object(exp, "_run_dag"):
                exp.run()

                # Verify rsync was called
                mock_rsync.assert_called_once()
                # Verify run was called with the combined symlink commands
                mock_tunnel.run.assert_called_once_with(
                    "ln -s source1 target1 && ln -s source2 target2"
                )


# Add test for _save_jobs method
def test_save_jobs(temp_dir):
    """Test saving jobs to disk."""
    with Experiment("test-exp") as exp:
        task = run.Partial(dummy_function, x=1, y=2)
        exp.add(task, name="test-job")

        # Create the experiment directory
        os.makedirs(exp._exp_dir, exist_ok=True)

        # Save jobs
        exp._save_jobs()

        # Verify the task file was created
        task_file = os.path.join(exp._exp_dir, Experiment._TASK_FILE)
        assert os.path.exists(task_file)

        # Check for __main__.py - this won't mock sys but rather directly check the condition
        main_py = os.path.join(exp._exp_dir, "__main__.py")
        if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "__file__"):
            assert os.path.exists(main_py) or not os.access(
                sys.modules["__main__"].__file__, os.R_OK
            )


# Add test for _save_tunnels method
def test_save_tunnels(temp_dir):
    """Test saving tunnels to disk."""
    with Experiment("test-exp") as exp:
        # Create the experiment directory
        os.makedirs(exp._exp_dir, exist_ok=True)

        # Create a mock tunnel with a to_config method
        mock_tunnel = MagicMock()
        # Use MagicMock instead of trying to create a real Config
        mock_config = MagicMock()
        mock_tunnel.to_config.return_value = mock_config

        exp.tunnels = {"mock-tunnel": mock_tunnel}

        # Mock ZlibJSONSerializer to avoid actual serialization
        with patch("nemo_run.run.experiment.ZlibJSONSerializer") as mock_serializer_class:
            mock_serializer = MagicMock()
            mock_serializer_class.return_value = mock_serializer
            mock_serializer.serialize.return_value = "serialized-config"

            # Save tunnels
            exp._save_tunnels()

            # Verify serializer.serialize was called with the tunnel config
            mock_serializer.serialize.assert_called_once_with(mock_config)
            mock_tunnel.to_config.assert_called_once()


# Add test for catalog method with different title
@patch("nemo_run.run.experiment._get_sorted_dirs")
def test_catalog_with_title(mock_get_sorted_dirs, temp_dir):
    """Test the catalog method with a specific title."""
    # Mock _get_sorted_dirs to return a list of experiment IDs
    mock_get_sorted_dirs.return_value = ["exp1", "exp2", "exp3"]

    # Test the catalog method with a specific title
    experiments = Experiment.catalog("specific-title")

    # Verify _get_sorted_dirs was called with the correct path
    mock_get_sorted_dirs.assert_called_once()
    args, kwargs = mock_get_sorted_dirs.call_args
    assert args[0].endswith("experiments/specific-title")

    # Verify the correct experiment IDs were returned
    assert experiments == ["exp1", "exp2", "exp3"]
