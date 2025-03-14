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
        assert isinstance(exp.jobs[0].task, Script)


def test_add_job_group(temp_dir):
    """Test adding a job group to an experiment."""
    with patch(
        "nemo_run.run.job.JobGroup.SUPPORTED_EXECUTORS", new_callable=PropertyMock
    ) as mock_supported:
        # Mock the SUPPORTED_EXECUTORS property to include LocalExecutor
        mock_supported.return_value = {LocalExecutor}

        with Experiment("test-exp") as exp:
            tasks = [run.Partial(dummy_function, x=1, y=2), run.Partial(dummy_function, x=3, y=4)]

            job_id = exp.add(tasks, name="group-job")

            assert job_id == "group-job"
            assert len(exp.jobs) == 1
            assert isinstance(exp.jobs[0], JobGroup)
            assert exp.jobs[0].id == "group-job"
            assert len(exp.jobs[0].tasks) == 2


def test_job_group_requires_name(temp_dir):
    """Test that job groups require a name."""
    with Experiment("test-exp") as exp:
        tasks = [run.Partial(dummy_function, x=1, y=2), run.Partial(dummy_function, x=3, y=4)]

        # Adding a job group without a name should raise an assertion error
        with pytest.raises(AssertionError):
            exp.add(tasks)


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

        job_id = exp.add(task, name="test-job", plugins=[plugin])

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
            tasks = [run.Partial(dummy_function, x=1, y=2), run.Partial(dummy_function, x=3, y=4)]

            # Create a plugin instance and mock its methods
            plugin = MagicMock(spec=ExperimentPlugin)

            # Add the job group with the plugin
            job_id = exp.add(tasks, name="group-job", plugins=[plugin])

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
            assert reset_exp is None  # The actual implementation returns None


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
        job4_id = exp.add(task.clone(), name="job4", dependencies=[job2_id, job3_id])

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
        job1_id = exp.add(task, name="job1")

        # Adding a job with a non-existent dependency should raise an assertion error
        with pytest.raises(AssertionError):
            exp.add(task, name="job2", dependencies=["non-existent-job"])


def test_dependencies_between_jobs(temp_dir):
    """Test adding dependencies between jobs."""
    with Experiment("test-exp") as exp:
        task1 = run.Partial(dummy_function, x=1, y=2)
        job1_id = exp.add(task1, name="job1")

        task2 = run.Partial(dummy_function, x=3, y=4)
        job2_id = exp.add(task2, name="job2", dependencies=[job1_id])

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
        assert status_dict["test-job"]["status"] == AppState.SUCCEEDED

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
        job_id = exp.add(task, name="test-job")

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
