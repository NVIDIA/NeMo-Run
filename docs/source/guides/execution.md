# Execute NeMo Run

After configuring NeMo-Run, the next step is to execute it. Nemo-Run decouples configuration from execution, allowing you to configure a function or task once and then execute it across multiple environments. With Nemo-Run, you can choose to execute a single task or multiple tasks simultaneously on different remote clusters, managing them under an experiment. This brings us to the core building blocks for execution: `run.Executor` and `run.Experiment`.

Each execution of a single configured task requires an executor. Nemo-Run provides `run.Executor`, which are APIs to configure your remote executor and set up the packaging of your code. Currently we support:
- `run.LocalExecutor`
- `run.DockerExecutor`
- `run.SlurmExecutor` with an optional `SSHTunnel` for executing on Slurm clusters from your local machine
- `run.SkypilotExecutor` (available under the optional feature `skypilot` in the python package).
- `run.LeptonExecutor`

A tuple of task and executor form an execution unit. A key goal of NeMo-Run is to allow you to mix and match tasks and executors to arbitrarily define execution units.

Once an execution unit is created, the next step is to run it. The `run.run` function executes a single task, whereas `run.Experiment` offers more fine-grained control to define complex experiments. `run.run` wraps `run.Experiment` with a single task. `run.Experiment` is an API to launch and manage multiple tasks all using pure Python.
The `run.Experiment` takes care of storing the run metadata, launching it on the specified cluster, and syncing the logs, etc. Additionally, `run.Experiment` also provides management tools to easily inspect and reproduce past experiments. The `run.Experiment` is inspired from [xmanager](https://github.com/google-deepmind/xmanager/tree/main) and uses [TorchX](https://pytorch.org/torchx/latest/) under the hood to handle execution.

> **_NOTE:_** NeMo-Run assumes familiarity with Docker and uses a docker image as the environment for remote execution. This means you must provide a Docker image that includes all necessary dependencies and configurations when using a remote executor.

> **_NOTE:_** All the experiment metadata is stored under `NEMORUN_HOME` env var on the machine where you launch the experiments. By default, the value for `NEMORUN_HOME` value is `~/.run`. Be sure to change this according to your needs.

## Executors
Executors are dataclasses that configure your remote executor and set up the packaging of your code. All supported executors inherit from the base class `run.Executor`, but have configuration parameters specific to their execution environment. There is an initial cost to understanding the specifics of your executor and setting it up, but this effort is easily amortized over time.

Each `run.Executor` has the two attributes: `packager` and `launcher`. The `packager` specifies how to package the code for execution, while the `launcher` determines which tool to use for launching the task.

### Launchers
We support the following `launchers`:
- `default` or `None`: This will directly launch your task without using any special launchers. Set `executor.launcher = None` (which is the default value) if you don't want to use a specific launcher.
- `torchrun` or `run.Torchrun`: This will launch the task using `torchrun`. See the `Torchrun` class for configuration options. You can use it using `executor.launcher = "torchrun"` or `executor.launcher = Torchrun(...)`.
- `ft` or `run.core.execution.FaultTolerance`: This will launch the task using NVIDIA's fault tolerant launcher. See the `FaultTolerance` class for configuration options. You can use it using `executor.launcher = "ft"` or `executor.launcher = FaultTolerance(...)`.

> **_NOTE:_** Launcher may not work very well with `run.Script`. Please report any issues at https://github.com/NVIDIA/NeMo-Run/issues.

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
1. base_path = `git rev-parse --show-toplevel`.
2. Optionally define a subpath as `base_path/GitArchivePackager.subpath` by setting `subpath` attribute on `GitArchivePackager`.
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

> **_NOTE:_** git archive doesn't package uncommitted changes. In the future, we may add support for including uncommitted changes while honoring `.gitignore`.

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

Example:
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

### Defining Executors
Next, We'll describe details on setting up each of the executors below.

#### LocalExecutor

The LocalExecutor is the simplest executor. It executes your task locally in a separate process or group from your current working directory.

The easiest way to define one is to call `run.LocalExecutor()`.

#### DockerExecutor

The DockerExecutor enables launching a task using `docker` on your local machine. It requires `docker` to be installed and running as a prerequisite.

The DockerExecutor uses the [docker python client](https://docker-py.readthedocs.io/en/stable/) and most of the options are passed directly to the client.

Below is an example of configuring a Docker Executor

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

#### SlurmExecutor

The SlurmExecutor enables launching the configured task on a Slurm Cluster with Pyxis.  Additionally, you can configure a `run.SSHTunnel`, which enables you to execute tasks on the Slurm cluster from your local machine while NeMo-Run manages the SSH connection for you. This setup supports use cases such as launching the same task on multiple Slurm clusters.

Below is an example of configuring a Slurm Executor
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
        packager=packager,
    )

# You can then call the executor in your script like
executor = your_slurm_cluster(nodes=8, container_image="your-nemo-image")
```

Use the SSH Tunnel when launching from your local machine, or the Local Tunnel if you're already on the Slurm cluster.

##### Job Dependencies

`SlurmExecutor` supports defining dependencies between [jobs](management.md#adding-tasks), allowing you to create workflows where jobs run in a specific order. Additionally, you can specify the `dependency_type` parameter:

```python
executor = run.SlurmExecutor(
    # ... other parameters ...
    dependency_type="afterok",
)
```

The `dependency_type` parameter specifies the type of dependency relationship:

- `afterok` (default): Job will start only after the specified jobs have completed successfully
- `afterany`: Job will start after the specified jobs have terminated (regardless of exit code)
- `afternotok`: Job will start after the specified jobs have failed
- Other options are available as defined in the [Slurm documentation](https://slurm.schedmd.com/sbatch.html#OPT_dependency)

This functionality enables you to create complex workflows with proper orchestration between different tasks, such as starting a training job only after data preparation is complete, or running an evaluation only after training finishes successfully.

#### SkypilotExecutor
This executor is used to configure [Skypilot](https://skypilot.readthedocs.io/en/latest/docs/index.html). Make sure Skypilot is installed using `pip install "nemo_run[skypilot]"` and atleast one cloud is configured using `sky check`.

Here's an example of the `SkypilotExecutor` for Kubernetes:
```python
def your_skypilot_executor(nodes: int, devices: int, container_image: str):
    return SkypilotExecutor(
        gpus="RTX5880-ADA-GENERATION",
        gpus_per_node=devices,
        nodes = nodes
        env_vars=common_envs()
        container_image=container_image,
        cloud="kubernetes",
        # Optional to reuse Skypilot cluster
        cluster_name="tester",
        setup="""
    conda deactivate
    nvidia-smi
    ls -al ./
    """,
    )

# You can then call the executor in your script like
executor = your_skypilot_cluster(nodes=8, devices=8, container_image="your-nemo-image")
```

As demonstrated in the examples, defining executors in Python offers great flexibility. You can easily mix and match things like common environment variables, and the separation of tasks from executors enables you to run the same configured task on any supported executor.

#### DGXCloudExecutor

The `DGXCloudExecutor` integrates with a DGX Cloud cluster's Run:ai API to launch distributed jobs. It uses REST API calls to authenticate, identify the target project and cluster, and submit the job specification.

> **_WARNING:_** Currently, the `DGXCloudExecutor` is only supported when launching experiments *from* a pod running on the DGX Cloud cluster itself. Furthermore, this launching pod must have access to a Persistent Volume Claim (PVC) where the experiment/job directories will be created, and this same PVC must also be configured to be mounted by the job being launched.

Here's an example configuration:

```python
def your_dgx_executor(nodes: int, gpus_per_node: int, container_image: str):
    # Ensure these are set correctly for your DGX Cloud environment
    # You might fetch these from environment variables or a config file
    base_url = "YOUR_DGX_CLOUD_API_ENDPOINT" # e.g., https://<cluster-name>.<domain>/api/v1
    app_id = "YOUR_RUNAI_APP_ID"
    app_secret = "YOUR_RUNAI_APP_SECRET"
    project_name = "YOUR_RUNAI_PROJECT_NAME"
    # Define the PVC that will be mounted in the job pods
    # Ensure the path specified here contains your NEMORUN_HOME
    pvc_name = "your-pvc-k8s-name" # The Kubernetes name of the PVC
    pvc_mount_path = "/your_custom_path" # The path where the PVC will be mounted inside the container

    executor = run.DGXCloudExecutor(
        base_url=base_url,
        app_id=app_id,
        app_secret=app_secret,
        project_name=project_name,
        container_image=container_image,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        pvcs=[{"name": pvc_name, "path": pvc_mount_path}],
        # Optional: Add custom environment variables or Slurm specs if needed
        env_vars=common_envs(),
        # packager=run.GitArchivePackager() # Choose appropriate packager
    )
    return executor

# Example usage:
# executor = your_dgx_executor(nodes=4, gpus_per_node=8, container_image="your-nemo-image")

```

For a complete end-to-end example using DGX Cloud with NeMo, refer to the [NVIDIA DGX Cloud NeMo End-to-End Workflow Example](https://docs.nvidia.com/dgx-cloud/run-ai/latest/nemo-e2e-example.html).

#### LeptonExecutor

The `LeptonExecutor` integrates with an NVIDIA DGX Cloud Lepton cluster's Python SDK to launch distributed jobs. It uses API calls behind the Lepton SDK to authenticate, identify the target node group and resource shapes, and submit the job specification which will be launched as a batch job on the cluster.

Here's an example configuration:

```python
def your_lepton_executor(nodes: int, gpus_per_node: int, container_image: str):
    # Ensure these are set correctly for your DGX Cloud environment
    # You might fetch these from environment variables or a config file
    resource_shape = "gpu.8xh100-80gb" # Replace with your desired resource shape representing the number of GPUs in a pod
    node_group = "my-node-group"  # The node group to run the job in
    nemo_run_dir = "/nemo-workspace/nemo-run"  # The NeMo-Run directory where experiments are saved
    # Define the remote storage directory that will be mounted in the job pods
    # Ensure the path specified here contains your NEMORUN_HOME
    storage_path = "/nemo-workspace" # The remote storage directory to mount in jobs
    mount_path = "/nemo-workspace" # The path where the remote storage directory will be mounted inside the container

    executor = run.LeptonExecutor(
        resource_shape=resource_shape,
        node_group=node_group,
        container_image=container_image,
        nodes=nodes,
        nemo_run_dir=nemo_run_dir,
        gpus_per_node=gpus_per_node,
        mounts=[{"path": storage_path, "mount_path": mount_path}],
        # Optional: Add custom environment variables or PyTorch specs if needed
        env_vars=common_envs(),
        # packager=run.GitArchivePackager() # Choose appropriate packager
    )
    return executor

# Example usage:
executor = your_lepton_executor(nodes=4, gpus_per_node=8, container_image="your-nemo-image")

```
