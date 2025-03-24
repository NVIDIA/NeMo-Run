from unittest.mock import MagicMock, patch

import pytest
from torchx.specs.api import AppState

from nemo_run.config import Partial, Script
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.slurm import SlurmExecutor
from nemo_run.run.job import Job, JobGroup
from nemo_run.run.torchx_backend.runner import Runner


# Define a global function to avoid serialization issues with <locals>
def task_fn(x, y):
    return x + y


@pytest.fixture
def simple_task():
    return Partial(task_fn, 1, 2)


@pytest.fixture
def simple_script():
    return Script("echo hello")


@pytest.fixture
def docker_executor():
    return DockerExecutor(container_image="test:latest", job_dir="/tmp/test")


@pytest.fixture
def slurm_executor():
    return SlurmExecutor(
        account="test_account",
        job_name_prefix="test",
        partition="test",
        job_dir="/tmp/test",
    )


@pytest.fixture
def mock_runner():
    runner = MagicMock(spec=Runner)
    runner.status.return_value = MagicMock(state=AppState.SUCCEEDED)
    return runner


def test_job_serialize(simple_task, docker_executor):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
    )

    cfg_str, task_str = job.serialize()
    assert isinstance(cfg_str, str)
    assert isinstance(task_str, str)
    assert len(cfg_str) > 0
    assert len(task_str) > 0


def test_job_status_not_launched(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
    )

    assert job.status(mock_runner) == AppState.UNSUBMITTED
    assert not job.launched
    assert not job.handle


def test_job_status_launched(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
        state=AppState.RUNNING,
    )

    assert job.status(mock_runner) == AppState.SUCCEEDED
    mock_runner.status.assert_called_once_with("test-handle")


def test_job_status_exception(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
        state=AppState.RUNNING,
    )

    mock_runner.status.side_effect = Exception("Test exception")
    assert job.status(mock_runner) == AppState.RUNNING


def test_job_logs(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
    )

    with patch("nemo_run.run.job.get_logs") as mock_get_logs:
        job.logs(mock_runner)
        mock_get_logs.assert_called_once()
        args, kwargs = mock_get_logs.call_args
        assert kwargs["identifier"] == "test-handle"
        assert kwargs["runner"] == mock_runner


def test_job_prepare(simple_task, docker_executor):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
    )

    with patch.object(docker_executor, "create_job_dir") as mock_create_job_dir:
        with patch("nemo_run.run.job.package") as mock_package:
            mock_package.return_value = MagicMock()
            job.prepare()
            mock_create_job_dir.assert_called_once()
            mock_package.assert_called_once()
            assert hasattr(job, "_executable")


def test_job_launch_invalid_task(docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=5,  # Invalid task type
        executor=docker_executor,
    )

    with pytest.raises(TypeError):
        job.launch(wait=False, runner=mock_runner)


def test_job_launch_direct(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
    )

    with patch("nemo_run.run.job.direct_run_fn") as mock_direct_run_fn:
        job.prepare()
        job.launch(wait=False, runner=mock_runner, direct=True)
        mock_direct_run_fn.assert_called_once()
        assert job.launched
        assert job.handle
        assert job.state == AppState.SUCCEEDED


def test_job_launch_dryrun(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
    )

    with patch("nemo_run.run.job.launch") as mock_launch:
        job.prepare()
        job.launch(wait=False, runner=mock_runner, dryrun=True)
        mock_launch.assert_called_once()
        args, kwargs = mock_launch.call_args
        assert kwargs["dryrun"] is True


def test_job_launch(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
    )

    with patch("nemo_run.run.job.launch") as mock_launch:
        mock_launch.return_value = ("test-handle", MagicMock(state=AppState.RUNNING))
        job.prepare()
        job.launch(wait=False, runner=mock_runner)
        mock_launch.assert_called_once()
        assert job.launched
        assert job.handle == "test-handle"
        assert job.state == AppState.RUNNING


def test_job_wait(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
    )

    with patch("nemo_run.run.job.wait_and_exit") as mock_wait_and_exit:
        mock_wait_and_exit.return_value = MagicMock(state=AppState.SUCCEEDED)
        job.wait(mock_runner)
        mock_wait_and_exit.assert_called_once()
        assert job.state == AppState.SUCCEEDED


def test_job_wait_exception(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
    )

    with patch("nemo_run.run.job.wait_and_exit") as mock_wait_and_exit:
        from nemo_run.exceptions import UnknownStatusError

        mock_wait_and_exit.side_effect = UnknownStatusError()
        job.wait(mock_runner)
        assert job.state == AppState.UNKNOWN


def test_job_cancel(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
    )

    job.cancel(mock_runner)
    mock_runner.cancel.assert_called_once_with("test-handle")


def test_job_cancel_no_handle(simple_task, docker_executor, mock_runner):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
    )

    job.cancel(mock_runner)
    mock_runner.cancel.assert_not_called()


def test_job_cleanup(simple_task, docker_executor):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
        state=AppState.SUCCEEDED,
    )

    with patch.object(docker_executor, "cleanup") as mock_cleanup:
        job.cleanup()
        mock_cleanup.assert_called_once_with("test-handle")


def test_job_cleanup_not_terminal(simple_task, docker_executor):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
        state=AppState.RUNNING,
    )

    with patch.object(docker_executor, "cleanup") as mock_cleanup:
        job.cleanup()
        mock_cleanup.assert_not_called()


def test_job_cleanup_exception(simple_task, docker_executor):
    job = Job(
        id="test-job",
        task=simple_task,
        executor=docker_executor,
        launched=True,
        handle="test-handle",
        state=AppState.SUCCEEDED,
    )

    with patch.object(docker_executor, "cleanup") as mock_cleanup:
        mock_cleanup.side_effect = Exception("Test exception")
        with patch("nemo_run.run.job.CONSOLE") as mock_console:
            job.cleanup()
            mock_cleanup.assert_called_once_with("test-handle")
            mock_console.log.assert_called()


# JobGroup tests


def test_job_group_init_single_executor(simple_task, docker_executor):
    # Force DockerExecutor _merge to False for test purposes
    with patch.object(JobGroup, "__post_init__"):
        job_group = JobGroup(
            id="test-group",
            tasks=[simple_task, simple_task],
            executors=docker_executor,
        )
        job_group._merge = False

        assert job_group.executors == docker_executor
        assert not job_group._merge


def test_job_group_init_multiple_executors(simple_task):
    executors = [
        DockerExecutor(container_image="test1:latest", job_dir="/tmp/test1"),
        DockerExecutor(container_image="test2:latest", job_dir="/tmp/test2"),
    ]

    # Mock the merge process
    with patch.object(DockerExecutor, "merge") as mock_merge:
        mock_merge.return_value = DockerExecutor(
            container_image="merged:latest", job_dir="/tmp/merged"
        )
        job_group = JobGroup(
            id="test-group",
            tasks=[simple_task, simple_task],
            executors=executors,
        )

        mock_merge.assert_called_once()
        assert isinstance(job_group.executors, DockerExecutor)


def test_job_group_init_invalid_executor_count(simple_task):
    executors = [
        DockerExecutor(container_image="test1:latest", job_dir="/tmp/test1"),
        DockerExecutor(container_image="test2:latest", job_dir="/tmp/test2"),
        DockerExecutor(container_image="test3:latest", job_dir="/tmp/test3"),
    ]

    with pytest.raises(AssertionError):
        JobGroup(
            id="test-group",
            tasks=[simple_task, simple_task],  # 2 tasks
            executors=executors,  # 3 executors
        )


def test_job_group_init_mixed_executor_types(simple_task):
    executors = [
        DockerExecutor(container_image="test:latest", job_dir="/tmp/test1"),
        SlurmExecutor(account="test_account", partition="test", job_dir="/tmp/test2"),
    ]

    with pytest.raises(AssertionError):
        JobGroup(
            id="test-group",
            tasks=[simple_task, simple_task],
            executors=executors,
        )


def test_job_group_properties(simple_task, docker_executor):
    # Mock the property behavior directly
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    # Set properties explicitly for test
    job_group.handles = ["handle1"]
    job_group.states = [AppState.RUNNING]
    job_group.launched = True

    assert job_group.state == AppState.RUNNING
    assert job_group.handle == "handle1"
    assert job_group.executor == docker_executor


def test_job_group_serialize(simple_task, docker_executor):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    cfg_str, tasks_str = job_group.serialize()
    assert isinstance(cfg_str, str)
    assert isinstance(tasks_str, str)
    assert len(cfg_str) > 0
    assert len(tasks_str) > 0


def test_job_group_status_not_launched(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    assert job_group.status(mock_runner) == AppState.UNSUBMITTED


def test_job_group_status_launched(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["handle1"],
        states=[AppState.RUNNING],
    )

    assert job_group.status(mock_runner) == AppState.SUCCEEDED
    mock_runner.status.assert_called_once_with("handle1")


def test_job_group_status_exception(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["handle1"],
        states=[AppState.RUNNING],
    )

    mock_runner.status.side_effect = Exception("Test exception")
    status = job_group.status(mock_runner)
    assert status == AppState.UNKNOWN
    assert job_group.states == [AppState.UNKNOWN]


def test_job_group_logs(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["handle1"],
    )

    with patch("nemo_run.run.job.get_logs") as mock_get_logs:
        job_group.logs(mock_runner)
        mock_get_logs.assert_called_once()
        args, kwargs = mock_get_logs.call_args
        assert kwargs["identifier"] == "handle1"
        assert kwargs["runner"] == mock_runner


def test_job_group_prepare(simple_task, docker_executor):
    # Mock DockerExecutor merge behavior
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    # For non-merged case, we need to set executors to a list
    job_group._merge = False
    job_group.executors = [docker_executor] * len(job_group.tasks)

    with patch.object(docker_executor, "create_job_dir") as mock_create_job_dir:
        with patch("nemo_run.run.job.package") as mock_package:
            with patch("nemo_run.run.job.merge_executables") as mock_merge:
                mock_package.return_value = MagicMock()
                mock_merge.return_value = MagicMock()
                job_group.prepare()
                mock_create_job_dir.assert_called_once()
                assert mock_package.call_count == 2
                # Now we're explicitly not merging, so shouldn't be called
                mock_merge.assert_not_called()
                assert hasattr(job_group, "_executables")
                assert len(job_group._executables) == 2


def test_job_group_prepare_with_merge(simple_task, slurm_executor):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=slurm_executor,
    )

    # Make sure _merge is True
    job_group._merge = True

    with patch.object(slurm_executor, "create_job_dir") as mock_create_job_dir:
        with patch("nemo_run.run.job.package") as mock_package:
            with patch("nemo_run.run.job.merge_executables") as mock_merge:
                mock_package.return_value = MagicMock()
                mock_merge.return_value = MagicMock()
                job_group.prepare()
                mock_create_job_dir.assert_called_once()
                assert mock_package.call_count == 2
                mock_merge.assert_called_once()
                assert hasattr(job_group, "_executables")
                assert len(job_group._executables) == 1


def test_job_group_launch_invalid_task(docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[5, 10],  # Invalid task types
        executors=docker_executor,
    )

    with pytest.raises(TypeError):
        job_group.launch(wait=False, runner=mock_runner)


def test_job_group_launch_direct(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    with pytest.raises(NotImplementedError):
        job_group.launch(wait=False, runner=mock_runner, direct=True)


def test_job_group_launch_dryrun(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    # Set _merge to True for this test and patch _executables
    job_group._merge = True
    job_group._executables = [(MagicMock(), docker_executor)]

    with patch("nemo_run.run.job.launch") as mock_launch:
        job_group.launch(wait=False, runner=mock_runner, dryrun=True)
        # Now we have just one executable, which gets launch called once
        assert mock_launch.call_count == 1
        args, kwargs = mock_launch.call_args
        assert kwargs["dryrun"] is True


def test_job_group_launch(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    # Set _merge to True for this test and patch _executables
    job_group._merge = True
    job_group._executables = [(MagicMock(), docker_executor)]

    with patch("nemo_run.run.job.launch") as mock_launch:
        mock_launch.return_value = ("test-handle", MagicMock(state=AppState.RUNNING))
        job_group.launch(wait=False, runner=mock_runner)
        # Now we have just one executable, which gets launch called once
        assert mock_launch.call_count == 1
        assert job_group.launched
        assert len(job_group.handles) == 1
        assert job_group.handles[0] == "test-handle"
        assert job_group.states[0] == AppState.RUNNING


def test_job_group_wait(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["test-handle"],
        states=[AppState.RUNNING],
    )

    with patch("nemo_run.run.job.wait_and_exit") as mock_wait_and_exit:
        mock_wait_and_exit.return_value = MagicMock(state=AppState.SUCCEEDED)
        job_group.wait(mock_runner)
        mock_wait_and_exit.assert_called_once()
        assert job_group.states == [AppState.SUCCEEDED]


def test_job_group_wait_exception(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["test-handle"],
        states=[AppState.RUNNING],
    )

    with patch("nemo_run.run.job.wait_and_exit") as mock_wait_and_exit:
        from nemo_run.exceptions import UnknownStatusError

        mock_wait_and_exit.side_effect = UnknownStatusError()
        job_group.wait(mock_runner)
        assert job_group.states == [AppState.UNKNOWN]


def test_job_group_cancel(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["handle1", "handle2"],
    )

    job_group.cancel(mock_runner)
    assert mock_runner.cancel.call_count == 2
    mock_runner.cancel.assert_any_call("handle1")
    mock_runner.cancel.assert_any_call("handle2")


def test_job_group_cancel_no_handles(simple_task, docker_executor, mock_runner):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
    )

    job_group.cancel(mock_runner)
    mock_runner.cancel.assert_not_called()


def test_job_group_cleanup(simple_task, docker_executor):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["handle1", "handle2"],
        states=[AppState.SUCCEEDED],
    )

    with patch.object(docker_executor, "cleanup") as mock_cleanup:
        job_group.cleanup()
        assert mock_cleanup.call_count == 2
        mock_cleanup.assert_any_call("handle1")
        mock_cleanup.assert_any_call("handle2")


def test_job_group_cleanup_not_terminal(simple_task, docker_executor):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["handle1", "handle2"],
        states=[AppState.RUNNING],
    )

    with patch.object(docker_executor, "cleanup") as mock_cleanup:
        job_group.cleanup()
        mock_cleanup.assert_not_called()


def test_job_group_cleanup_exception(simple_task, docker_executor):
    job_group = JobGroup(
        id="test-group",
        tasks=[simple_task, simple_task],
        executors=docker_executor,
        launched=True,
        handles=["handle1", "handle2"],
        states=[AppState.SUCCEEDED],
    )

    with patch.object(docker_executor, "cleanup") as mock_cleanup:
        mock_cleanup.side_effect = Exception("Test exception")
        with patch("nemo_run.run.job.CONSOLE") as mock_console:
            job_group.cleanup()
            assert mock_cleanup.call_count == 2
            mock_console.log.assert_called()
