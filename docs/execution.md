---
description: "Learn how to execute NeMo Run experiments across different environments using various executors including local, Docker, Slurm, and cloud platforms."
tags: ["execution", "executors", "deployment", "distributed"]
categories: ["core"]
---

(execution-guide)=

# Execute NeMo Run

After configuring NeMo-Run, the next step is to execute it. Nemo-Run decouples configuration from execution, allowing you to configure a function or task once and then execute it across multiple environments. With Nemo-Run, you can choose to execute a single task or multiple tasks simultaneously on different remote clusters, managing them under an experiment. This brings us to the core building blocks for execution: `run.Executor` and `run.Experiment`.

Each execution of a single configured task requires an executor. Nemo-Run provides `run.Executor`, which are APIs to configure your remote executor and set up the packaging of your code. Currently we support:

- `run.LocalExecutor`
- `run.DockerExecutor`
- `run.SlurmExecutor` with an optional `SSHTunnel` for executing on Slurm clusters from your local machine
- `run.SkypilotExecutor` (available under the optional feature `skypilot` in the python package)
- `run.LeptonExecutor`

A tuple of task and executor form an execution unit. A key goal of NeMo-Run is to allow you to mix and match tasks and executors to arbitrarily define execution units.

Once an execution unit is created, the next step is to run it. The `run.run` function executes a single task, whereas `run.Experiment` offers more fine-grained control to define complex experiments. `run.run` wraps `run.Experiment` with a single task. `run.Experiment` is an API to launch and manage multiple tasks all using pure Python.

The `run.Experiment` takes care of storing the run metadata, launching it on the specified cluster, and syncing the logs, etc. Additionally, `run.Experiment` also provides management tools to easily inspect and reproduce past experiments. The `run.Experiment` is inspired from [xmanager](https://github.com/google-deepmind/xmanager/tree/main) and uses [TorchX](https://pytorch.org/torchx/latest/) under the hood to handle execution.

:::{warning}
NeMo-Run assumes familiarity with Docker and uses a docker image as the environment for remote execution. This means you must provide a Docker image that includes all necessary dependencies and configurations when using a remote executor.
:::

:::{note}
All the experiment metadata is stored under `NEMORUN_HOME` env var on the machine where you launch the experiments. By default, the value for `NEMORUN_HOME` value is `~/.run`. Be sure to change this according to your needs.
:::

## Executors

Executors are dataclasses that configure your remote executor and set up the packaging of your code. All supported executors inherit from the base class `run.Executor`, but have configuration parameters specific to their execution environment. There is an initial cost to understanding the specifics of your executor and setting it up, but this effort is easily amortized over time.

Each `run.Executor` has the two attributes: `packager` and `launcher`. The `packager` specifies how to package the code for execution, while the `launcher` determines which tool to use for launching the task.

### Launchers

We support the following `launchers`:

- `default` or `None`: This will directly launch your task without using any special launchers. Set `executor.launcher = None` (which is the default value) if you don't want to use a specific launcher.
- `torchrun` or `run.Torchrun`: This will launch the task using `torchrun`. See the `Torchrun` class for configuration options. You can use it using `executor.launcher = "torchrun"` or `executor.launcher = Torchrun(...)`.
- `ft` or `run.core.execution.FaultTolerance`: This will launch the task using NVIDIA's fault tolerant launcher. See the `FaultTolerance` class for configuration options. You can use it using `executor.launcher = "ft"` or `executor.launcher = FaultTolerance(...)`.

:::{warning}
Launcher may not work very well with `run.Script`. Please report any issues at <https://github.com/NVIDIA-NeMo/Run/issues>.
:::

### Packagers

The packager support matrix is described below:

| Executor | Packagers |
|----------|----------|
| LocalExecutor | run.Packager |
| DockerExecutor | run.Packager, run.GitArchivePackager, run.PatternPackager, run.HybridPackager |
| SlurmExecutor | run.Packager, run.GitArchivePackager, run.PatternPackager, run.HybridPackager |
| SkypilotExecutor | run.Packager, run.GitArchivePackager, run.PatternPackager, run.HybridPackager |
| DGXCloudExecutor | run.Packager, run.GitArchivePackager, run.PatternPackager, run.HybridPackager |
| LeptonExecutor   | run.Packager, run.GitArchivePackager, run.PatternPackager, run.HybridPackager |

`run.Packager` is a passthrough base packager.

`run.GitArchivePackager` uses `git archive` to package your code. Refer to the API reference for `run.GitArchivePackager` to see the exact mechanics of packaging using `git archive`.

At a high level, it works in the following way:

1. base_path = `git rev-parse --show-toplevel`
2. Optionally define a subpath as `base_path/GitArchivePackager.subpath` by setting `subpath` attribute on `GitArchivePackager`
3. `cd base_path && git archive --format=tar.gz --output={output_file} {GitArchivePackager.subpath}:{subpath}`

This extracted tar file becomes the working directory for your job. As an example, given the following directory structure with `subpath="src"`:

```
- docs
- src
  - your_library
- tests
```

Your working directory at the time of execution will look like:

```
- your_library
```

If you're executing a Python function, this working directory will automatically be included in your Python path.

:::{note}
git archive doesn't package uncommitted changes. In the future, we may add support for including uncommitted changes while honoring `.gitignore`.
:::

`run.PatternPackager` is a packager that uses a pattern to package your code. It is useful for packaging code that is not under version control. For example, if you have a directory structure like this:

```
- docs
- src
  - your_library
```

You can use `run.PatternPackager` to package your code by specifying `include_pattern` as `src/**` and `relative_path` as `os.getcwd()`. This will package the entire `src` directory. The command used to get the list of files to package is:

```bash
# relative_include_pattern = os.path.relpath(self.include_pattern, self.relative_path)
cd {relative_path} && find {relative_include_pattern} -type f
```

`run.HybridPackager` allows combining multiple packagers into a single archive. This is useful when you need to package different parts of your project using different strategies (e.g., a git archive for committed code and a pattern packager for generated artifacts).

Each sub-packager in the `sub_packagers` dictionary is assigned a key, which becomes the directory name under which its contents are placed in the final archive. If `extract_at_root` is set to `True`, all contents are placed directly in the root of the archive, potentially overwriting files if names conflict.

::::{dropdown} Hybrid Packager Example
:icon: code-square

```python
import nemo_run as run
import os

hybrid_packager = run.HybridPackager(
    sub_packagers={
        "code": run.GitArchivePackager(subpath="src"),
        "configs": run.PatternPackager(include_pattern="configs/*.yaml", relative_path=os.getcwd())
    }
)

# Usage with an executor:
# executor.packager = hybrid_packager
```

This would create an archive where the contents of `src` are under a `code/` directory and matched `configs/*.yaml` files are under a `configs/` directory.
::::

### Defining Executors

Next, We'll describe details on setting up each of the executors below.

#### LocalExecutor

The LocalExecutor is the simplest executor. It executes your task locally in a separate process or group from your current working directory.

The easiest way to define one is to call `run.LocalExecutor()`.

#### DockerExecutor

The DockerExecutor enables launching a task using `docker` on your local machine. It requires `docker` to be installed and running as a prerequisite.

The DockerExecutor uses the [docker python client](https://docker-py.readthedocs.io/en/stable/) and most of the options are passed directly to the client.

::::{dropdown} Docker Executor Example
:icon: code-square

```python
run.DockerExecutor(
    container_image="python:3.12",
    num_gpus=-1,
    runtime="nvidia",
    ipc_mode="host",
    shm_size="30g",
    volumes=["/local/path:/path/in/container"],
    env_vars={"PYTHONUNBUFFERED": "1"},
    packager=run.Packager(),
)
```

::::

#### SlurmExecutor

The SlurmExecutor enables launching the configured task on a Slurm Cluster with Pyxis. Additionally, you can configure a `run.SSHTunnel`, which enables you to execute tasks on the Slurm cluster from your local machine while NeMo-Run manages the SSH connection for you. This setup supports use cases such as launching the same task on multiple Slurm clusters.

::::{dropdown} Slurm Executor Example
:icon: code-square

```python
def your_slurm_executor(nodes: int = 1, container_image: str = DEFAULT_IMAGE):
    # SSH Tunnel
    ssh_tunnel = run.SSHTunnel(
        host="your-slurm-host",
        user="your-user",
        job_dir="directory-to-store-runs-on-the-slurm-cluster",
        identity="optional-path-to-your-key-for-auth",
    )
    # Local Tunnel to use if you're already on the cluster
    local_tunnel = run.LocalTunnel()

    packager = GitArchivePackager(
        # This will also be the working directory in your task.
        # If empty, the working directory will be toplevel of your git repo
        subpath="optional-subpath-from-toplevel-of-your-git-repo"
    )

    executor = run.SlurmExecutor(
        # Most of these parameters are specific to slurm
        account="your-account",
        partition="your-partition",
        ntasks_per_node=8,
        gpus_per_node=8,
        nodes=nodes,
        tunnel=ssh_tunnel,
        container_image=container_image,
        time="00:30:00",
        env_vars=common_envs(),
        container_mounts=mounts_for_your_hubs(),
    )
```

::::
