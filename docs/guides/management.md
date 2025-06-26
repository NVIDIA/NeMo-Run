# Manage NeMo Run Experiments

The central component for management of tasks in NeMo Run is the `Experiment` class. It allows you to define, launch, and manage complex workflows consisting of multiple tasks. This guide provides an overview of the `Experiment` class, its methods, and how to use it effectively.

## **Create an Experiment**

To create an experiment, you can instantiate the `Experiment` class by passing in a descriptive title:

```python
exp = Experiment("My Experiment")
```

When executed, it will automatically generate a unique experiment ID for you, which represents one unique run of the experiment.

> [!NOTE] > `Experiment` is a context manager and `Experiment.add` and `Experiment.run` methods can currently only be used after entering the context manager.

## **Add Tasks**

You can add tasks to an experiment using the `add` method. This method supports tasks of the following kind:

- A single task which is an instance of either `run.Partial` or `run.Script`, along with its executor.

  ```python
  with exp:
      exp.add(task_1, executor=run.LocalExecutor())
  ```

- A list of tasks, each of which is an instance of either `run.Partial` or `run.Script`, along with a single executor or a list of executors for each task in the group. Currently, all tasks in the group will be executed in parallel.

  ```python
  with exp:
      exp.add([task_2, task_3], executor=run.DockerExecutor(...))
  ```

You can specify a descriptive name for the task using the `name` keyword argument.

`add` also takes in a list of plugins, each an instance of `run.Plugin`. Plugins are used to make changes to the task and executor together, which is useful in some cases - for example, to enable a config option in the task and set an environment variable in the executor related to the config option.

`add` returns a unique id for the task/job. This unique id can be used to define complex dependencies between a group of tasks as follows:

```python
with run.Experiment("dag-experiment", log_level="INFO") as exp:
    id1 = exp.add([inline_script, inline_script_sleep], tail_logs=False, name="task-1")
    id2 = exp.add([inline_script, inline_script_sleep], tail_logs=False, name="task-2")
    exp.add(
       [inline_script, inline_script_sleep],
       tail_logs=False,
       name="task-3",
       dependencies=[id1, id2], # task-3 will only run after task-1 and task-2 have completed
   )
```

## **Launch an Experiment**

Once you have added all tasks to an experiment, you can launch it using the `run` method. This method takes several optional arguments, including `detach`, `sequential`, and `tail_logs` and `direct`:

- `detach`: If `True`, the experiment will detach from the process executing it. This is useful when launching an experiment on a remote cluster, where you may want to end the process after scheduling the tasks in that experiment.
- `sequential`: If `True`, all tasks will be executed sequentially. This is only applicable when the individual tasks do not have any dependencies on each other.
- `tail_logs`: If `True`, logs will be displayed in real-time.
- `direct`: If `True`, each task in the experiment will be executed directly in the same process on your local machine. This does not support task/job groups.

```python
with exp:
    # Add all tasks
    exp.run(detach=True, sequential=False, tail_logs=True, direct=False)
```

## **Experiment Status**

You can check the status of an experiment using the `status` method:

```python
exp.status()
```

This method will display information the status of each task in the experiment. The following is a sample output from the status of experiment in [hello_scripts.py](../../../examples/hello-world/hello_scripts.py):

```bash
Experiment Status for experiment_with_scripts_1730761155

Task 0: echo.sh
- Status: SUCCEEDED
- Executor: LocalExecutor
- Job id: echo.sh-zggz3tq0kpljs
- Local Directory: /home/your_user/.nemo_run/experiments/experiment_with_scripts/experiment_with_scripts_1730761155/echo.sh

Task 1: env_echo_
- Status: SUCCEEDED
- Executor: LocalExecutor
- Job id: env_echo_-f3fc3fbj1qjtc
- Local Directory: /home/your_user/.nemo_run/experiments/experiment_with_scripts/experiment_with_scripts_1730761155/env_echo_

Task 2: simple.add.add_object
- Status: RUNNING
- Executor: LocalExecutor
- Job id: simple.add.add_object-s1543tt3f7dcm
- Local Directory: /home/your_user/.nemo_run/experiments/experiment_with_scripts/experiment_with_scripts_1730761155/simple.add.add_object
```

## **Cancel a Task**

You can cancel a task using the `cancel` method:

```python
exp.cancel("task_id")
```

## **View Logs**

You can view the logs of a task using the `
