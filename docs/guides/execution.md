---
description: "Execute NeMo Run experiments across different environments using various executors and launchers."
tags: ["execution", "executors", "Docker", "Slurm", "Kubernetes", "cloud", "distributed-computing"]
categories: ["guides"]
---

(execution)=

# Execute NeMo Run

This guide covers how to execute NeMo Run experiments across different computing environments. NeMo Run separates configuration from execution, allowing you to define your task once and run it on various platforms without code changes.

## Execution Overview

NeMo Run provides a unified execution framework that abstracts away the complexity of different computing environments. The execution process involves:

1. **Configuration**: Define your task using `run.Config` or `run.Partial`
2. **Packaging**: Bundle your code and dependencies for remote execution
3. **Launching**: Execute the task using appropriate launchers (torchrun, fault tolerance, etc.)
4. **Management**: Monitor and retrieve results through the experiment interface

### Key Components

- **`run.Executor`**: Configures the execution environment and packaging strategy
- **`run.Experiment`**: Manages multiple tasks and provides experiment lifecycle management
- **`run.run()`**: Simple function for single task execution

> **Important**: NeMo Run requires Docker for remote execution. All remote executors use containerized environments to ensure reproducibility and dependency isolation.

> **Note**: Experiment metadata is stored in `NEMORUN_HOME` (default: `~/.nemorun`). Configure this environment variable to control where experiment data is stored.

## Core Concepts

### Execution Units

An execution unit consists of a task configuration paired with an executor. This separation allows you to:

- Run the same task on different platforms
- Mix and match tasks and executors
- Scale experiments across multiple environments

```python
import nemo_run as run

# Define your task
task_config = run.Config(MyTrainingFunction, learning_rate=0.001, batch_size=32)

# Choose your executor
executor = run.SlurmExecutor(partition="gpu", nodes=2, gpus_per_node=4)

# Create execution unit
experiment = run.submit(task_config, executor)
```

### Experiment Management

NeMo Run provides comprehensive experiment management through the `run.Experiment` class:

```python
# Create experiment with multiple tasks
experiment = run.Experiment()

# Add tasks with different configurations
experiment.add_task(
    run.Config(MyModel, model_size="small"),
    run.LocalExecutor()
)

experiment.add_task(
    run.Config(MyModel, model_size="large"),
    run.SlurmExecutor(partition="gpu", nodes=4)
)

# Launch all tasks
experiment.launch()

# Monitor progress
for task in experiment.tasks:
    print(f"Task {task.id}: {task.status}")
    if task.completed:
        logs = run.get_logs(task)
        print(f"Exit code: {logs.exit_code}")
```

## Code Packaging

NeMo Run uses packagers to bundle your code and dependencies for remote execution. Each executor supports different packaging strategies.

### Packager Support Matrix

| Executor | Supported Packagers |
|----------|-------------------|
| `LocalExecutor` | `run.Packager` |
| `DockerExecutor` | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| `SlurmExecutor` | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| `SkypilotExecutor` | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| `DGXCloudExecutor` | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |
| `LeptonExecutor` | `run.Packager`, `run.GitArchivePackager`, `run.PatternPackager`, `run.HybridPackager` |

### Packager Types

#### `run.Packager` (Base Packager)

A pass-through packager that doesn't modify your code:

```python
executor = run.LocalExecutor(packager=run.Packager())
```

#### `run.GitArchivePackager`

Packages your Git repository using `git archive`:

```python
packager = run.GitArchivePackager(
    subpath="src"  # Optional: package only a subdirectory
)

executor = run.SlurmExecutor(
    packager=packager,
    # ... other parameters
)
```

**How it works:**
1. Determines the Git repository root using `git rev-parse --show-toplevel`
2. Creates a tar.gz archive using `git archive --format=tar.gz`
3. Extracts the archive as the working directory for your job

**Directory structure example:**
```
Repository structure:
├── docs/
├── src/
│   └── my_library/
└── tests/

With subpath="src", working directory becomes:
└── my_library/
```

> **Important**: `git archive` only includes committed changes. Uncommitted modifications are not packaged.

#### `run.PatternPackager`

Packages files based on pattern matching, useful for non-Git repositories:

```python
import os

packager = run.PatternPackager(
    include_pattern="src/**",  # Include all files under src/
    relative_path=os.getcwd()  # Base directory for pattern matching
)

executor = run.DockerExecutor(packager=packager)
```

**Pattern matching command:**
```bash
cd {relative_path} && find {include_pattern} -type f
```

#### `run.HybridPackager`

Combines multiple packagers into a single archive:

```python
import os

hybrid_packager = run.HybridPackager(
    sub_packagers={
        "code": run.GitArchivePackager(subpath="src"),
        "configs": run.PatternPackager(
            include_pattern="configs/*.yaml",
            relative_path=os.getcwd()
        ),
        "data": run.PatternPackager(
            include_pattern="data/processed/**",
            relative_path=os.getcwd()
        )
    }
)

executor = run.SlurmExecutor(packager=hybrid_packager)
```

**Resulting archive structure:**
```
archive/
├── code/
│   └── my_library/
├── configs/
│   ├── model_config.yaml
│   └── data_config.yaml
└── data/
    └── processed/
        └── dataset.parquet
```

## Task Launchers

Launchers determine how your task is executed within the container. They handle distributed training, fault tolerance, and other execution requirements.

### Available Launchers

#### Default Launcher (`None`)

Direct execution without special launchers:

```python
executor = run.SlurmExecutor(launcher=None)  # Default behavior
```

#### `torchrun` Launcher

Launches distributed PyTorch training using `torchrun`:

```python
from nemo_run import Torchrun

executor = run.SlurmExecutor(
    launcher=Torchrun(
        nnodes=2,
        nproc_per_node=4,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:29400"
    )
)
```

**Configuration options:**
- `nnodes`: Number of nodes
- `nproc_per_node`: Processes per node
- `rdzv_backend`: Rendezvous backend (c10d, static, etc.)
- `rdzv_endpoint`: Rendezvous endpoint
- `rdzv_id`: Unique rendezvous ID

#### Fault Tolerance Launcher

Uses NVIDIA's fault-tolerant launcher for resilient training:

```python
from nemo_run.core.execution import FaultTolerance

executor = run.SlurmExecutor(
    launcher=FaultTolerance(
        max_restarts=3,
        restart_delay=60,
        checkpoint_interval=1000
    )
)
```

**Configuration options:**
- `max_restarts`: Maximum number of restart attempts
- `restart_delay`: Delay between restarts (seconds)
- `checkpoint_interval`: Checkpoint frequency (steps)

> **Note**: Launchers may not work optimally with `run.Script`. Report issues at the [NeMo Run GitHub repository](https://github.com/NVIDIA-NeMo/Run/issues).

## Executor Types

### Local Execution

#### `run.LocalExecutor`

Executes tasks locally in a separate process:

```python
executor = run.LocalExecutor(
    packager=run.Packager(),
    env_vars={"CUDA_VISIBLE_DEVICES": "0"}
)
```

**Use cases:**
- Development and debugging
- Quick testing of configurations
- Local experimentation

### Containerized Execution

#### `run.DockerExecutor`

Executes tasks in Docker containers on your local machine:

```python
executor = run.DockerExecutor(
    container_image="nvidia/cuda:11.8-devel-ubuntu20.04",
    num_gpus=4,
    runtime="nvidia",
    ipc_mode="host",
    shm_size="32g",
    volumes=[
        "/local/data:/container/data",
        "/local/models:/container/models"
    ],
    env_vars={
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    },
    packager=run.GitArchivePackager()
)
```

**Key parameters:**
- `container_image`: Docker image with required dependencies
- `num_gpus`: Number of GPUs to allocate (-1 for all available)
- `runtime`: Docker runtime (nvidia for GPU support)
- `volumes`: Host-to-container volume mappings
- `env_vars`: Environment variables for the container

### High-Performance Computing

#### `run.SlurmExecutor`

Executes tasks on Slurm clusters with container support via Pyxis:

```python
def create_slurm_executor(
    nodes: int = 1,
    gpus_per_node: int = 8,
    container_image: str = "nvidia/cuda:11.8-devel-ubuntu20.04"
):
    # SSH tunnel for remote execution
    ssh_tunnel = run.SSHTunnel(
        host="cluster.login.node",
        user="username",
        job_dir="/home/username/nemo-run-experiments",
        identity="~/.ssh/id_rsa"
    )

    # Local tunnel for execution from login node
    local_tunnel = run.LocalTunnel()

    packager = run.GitArchivePackager(
        subpath="src"  # Package only the src directory
    )

    return run.SlurmExecutor(
        # Slurm-specific parameters
        account="ml_research",
        partition="gpu",
        nodes=nodes,
        ntasks_per_node=8,
        gpus_per_node=gpus_per_node,
        cpus_per_node=32,
        memory_per_node="128G",
        time="24:00:00",

        # Container configuration
        container_image=container_image,
        container_mounts=[
            "/shared/data:/data",
            "/shared/models:/models"
        ],

        # Execution configuration
        tunnel=ssh_tunnel,  # Use local_tunnel if on login node
        packager=packager,
        launcher=Torchrun(nnodes=nodes, nproc_per_node=gpus_per_node),

        # Environment variables
        env_vars={
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "0",
            "PYTHONUNBUFFERED": "1"
        }
    )

# Usage
executor = create_slurm_executor(nodes=4, gpus_per_node=8)
```

#### Job Dependencies

Create workflow dependencies between Slurm jobs:

```python
# Data preparation job
data_job = run.submit(
    run.Config(PrepareData, dataset="wikitext-103"),
    run.SlurmExecutor(partition="cpu", time="02:00:00")
)

# Training job that depends on data preparation
training_job = run.submit(
    run.Config(TrainModel, dataset="wikitext-103"),
    run.SlurmExecutor(
        partition="gpu",
        nodes=4,
        gpus_per_node=8,
        dependency_type="afterok",  # Start after data job succeeds
        dependencies=[data_job.id]
    )
)
```

**Dependency types:**
- `afterok` (default): Start after dependency jobs complete successfully
- `afterany`: Start after dependency jobs terminate (any exit code)
- `afternotok`: Start after dependency jobs fail
- `aftercorr`: Start after dependency jobs are cancelled

### Cloud Execution

#### `run.SkypilotExecutor`

Executes tasks on cloud platforms using SkyPilot:

```python
def create_skypilot_executor(
    nodes: int = 1,
    gpus_per_node: int = 8,
    container_image: str = "nvidia/cuda:11.8-devel-ubuntu20.04"
):
    return run.SkypilotExecutor(
        # Resource specification
        gpus="A100-80GB",  # GPU type
        gpus_per_node=gpus_per_node,
        nodes=nodes,

        # Container configuration
        container_image=container_image,

        # Cloud configuration
        cloud="aws",  # or "gcp", "azure", "kubernetes"
        region="us-west-2",

        # Optional cluster reuse
        cluster_name="nemo-training-cluster",

        # Setup commands
        setup="""
        # Install additional dependencies
        pip install transformers datasets

        # Verify GPU availability
        nvidia-smi

        # Check working directory
        ls -la ./
        """,

        # Environment variables
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"
        },

        # Packaging
        packager=run.GitArchivePackager()
    )

# Usage
executor = create_skypilot_executor(nodes=2, gpus_per_node=8)
```

**Prerequisites:**
```bash
# Install SkyPilot support
pip install "nemo_run[skypilot]"

# Configure cloud credentials
sky check
```

#### `run.DGXCloudExecutor`

Executes tasks on NVIDIA DGX Cloud using Run:ai API:

```python
def create_dgx_executor(
    nodes: int = 1,
    gpus_per_node: int = 8,
    container_image: str = "nvidia/cuda:11.8-devel-ubuntu20.04"
):
    return run.DGXCloudExecutor(
        # API configuration
        base_url="https://your-cluster.domain.com/api/v1",
        app_id="your-runai-app-id",
        app_secret="your-runai-app-secret",
        project_name="your-project",

        # Resource configuration
        nodes=nodes,
        gpus_per_node=gpus_per_node,

        # Container configuration
        container_image=container_image,

        # Storage configuration
        pvcs=[
            {
                "name": "nemo-data-pvc",
                "path": "/workspace/data"
            }
        ],

        # Environment variables
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "NEMORUN_HOME": "/workspace/nemo-run"
        },

        # Packaging
        packager=run.GitArchivePackager()
    )
```

> **Warning**: DGXCloudExecutor currently only supports launching from pods running on the DGX Cloud cluster itself. The launching pod must have access to a Persistent Volume Claim (PVC) for experiment storage.

#### `run.LeptonExecutor`

Executes tasks on NVIDIA DGX Cloud Lepton clusters:

```python
def create_lepton_executor(
    nodes: int = 1,
    gpus_per_node: int = 8,
    container_image: str = "nvidia/cuda:11.8-devel-ubuntu20.04"
):
    return run.LeptonExecutor(
        # Resource configuration
        resource_shape="gpu.8xh100-80gb",  # Resource shape per node
        node_group="training-nodes",
        nodes=nodes,
        gpus_per_node=gpus_per_node,

        # Container configuration
        container_image=container_image,

        # Storage configuration
        nemo_run_dir="/workspace/nemo-run",
        mounts=[
            {
                "path": "/workspace/data",
                "mount_path": "/workspace/data"
            }
        ],

        # Environment variables
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "NEMORUN_HOME": "/workspace/nemo-run"
        },

        # Packaging
        packager=run.GitArchivePackager()
    )
```

## Advanced Execution Features

### Multi-Environment Execution

Run the same task across different environments:

```python
# Define task once
task_config = run.Config(
    TrainModel,
    model_size="1.3B",
    dataset="wikitext-103",
    learning_rate=0.0001
)

# Create executors for different environments
executors = {
    "local": run.LocalExecutor(),
    "docker": run.DockerExecutor(
        container_image="nvidia/cuda:11.8-devel-ubuntu20.04",
        num_gpus=2
    ),
    "slurm": run.SlurmExecutor(
        partition="gpu",
        nodes=2,
        gpus_per_node=4
    ),
    "cloud": run.SkypilotExecutor(
        gpus="A100-80GB",
        nodes=2,
        gpus_per_node=4
    )
}

# Launch experiments
experiments = {}
for name, executor in executors.items():
    experiments[name] = run.submit(
        task_config,
        executor,
        metadata={"environment": name}
    )
```

### Resource Optimization

Dynamically configure resources based on task requirements:

```python
def create_adaptive_executor(task_config):
    """Create executor with resources optimized for the task."""

    # Analyze task requirements
    model_size = task_config.model_size
    batch_size = task_config.batch_size

    # Calculate optimal resources
    if model_size == "small" and batch_size <= 32:
        return run.SlurmExecutor(
            partition="gpu",
            nodes=1,
            gpus_per_node=2,
            memory_per_node="64G"
        )
    elif model_size == "medium" and batch_size <= 128:
        return run.SlurmExecutor(
            partition="gpu",
            nodes=2,
            gpus_per_node=4,
            memory_per_node="128G"
        )
    else:
        return run.SlurmExecutor(
            partition="gpu",
            nodes=4,
            gpus_per_node=8,
            memory_per_node="256G"
        )

# Usage
executor = create_adaptive_executor(task_config)
experiment = run.submit(task_config, executor)
```

### Fault Tolerance and Recovery

Implement robust execution with automatic recovery:

```python
# Configure fault-tolerant launcher
fault_tolerant_launcher = FaultTolerance(
    max_restarts=5,
    restart_delay=120,
    checkpoint_interval=500,
    checkpoint_dir="/workspace/checkpoints"
)

# Use with any executor
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=4,
    gpus_per_node=8,
    launcher=fault_tolerant_launcher,
    time="48:00:00"  # Long time for fault tolerance
)

# Submit with automatic recovery
experiment = run.submit(task_config, executor)
```

## Best Practices

### Configuration Management

1. **Use environment-specific configurations:**
```python
def get_executor_for_environment(env: str):
    if env == "development":
        return run.LocalExecutor()
    elif env == "staging":
        return run.DockerExecutor(container_image="staging-image")
    elif env == "production":
        return run.SlurmExecutor(partition="production-gpu")
    else:
        raise ValueError(f"Unknown environment: {env}")
```

2. **Parameterize executor creation:**
```python
def create_executor_factory(
    base_image: str = "nvidia/cuda:11.8-devel-ubuntu20.04",
    default_gpus: int = 4
):
    def create_executor(nodes: int, gpus_per_node: int = None):
        if gpus_per_node is None:
            gpus_per_node = default_gpus

        return run.SlurmExecutor(
            container_image=base_image,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            partition="gpu"
        )

    return create_executor

# Usage
factory = create_executor_factory()
executor = factory(nodes=2)
```

### Resource Management

1. **Monitor resource usage:**
```python
# Check cluster status before submission
executor = run.SlurmExecutor(partition="gpu")
status = executor.get_cluster_status()
print(f"Available nodes: {status.available_nodes}")
print(f"Queue length: {status.queue_length}")

# Only submit if resources are available
if status.available_nodes >= 2:
    experiment = run.submit(task_config, executor)
else:
    print("Insufficient resources, waiting...")
```

2. **Use resource quotas:**
```python
executor = run.SlurmExecutor(
    partition="gpu",
    qos="high_priority",  # Quality of service
    account="ml_research",  # Account/charge code
    exclusive=True  # Exclusive node access
)
```

### Error Handling

1. **Implement comprehensive error handling:**
```python
try:
    experiment = run.submit(task_config, executor)

    # Wait for completion with timeout
    experiment.wait(timeout=3600)  # 1 hour timeout

    if experiment.failed:
        logs = run.get_logs(experiment)
        print(f"Experiment failed: {logs.stderr}")

        # Implement retry logic
        if experiment.retry_count < 3:
            experiment.retry()

except Exception as e:
    print(f"Execution failed: {e}")
    # Implement fallback strategy
```

2. **Validate configurations before execution:**
```python
def validate_executor_config(executor):
    """Validate executor configuration before submission."""

    if isinstance(executor, run.SlurmExecutor):
        # Check if partition exists
        partitions = executor.list_partitions()
        if executor.partition not in partitions:
            raise ValueError(f"Partition {executor.partition} not found")

        # Check resource availability
        if executor.nodes > executor.get_max_nodes():
            raise ValueError(f"Requested {executor.nodes} nodes, max available: {executor.get_max_nodes()}")

    return True

# Usage
validate_executor_config(executor)
experiment = run.submit(task_config, executor)
```

### Performance Optimization

1. **Optimize container images:**
```dockerfile
# Use multi-stage builds for smaller images
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Install only necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Final stage
FROM base as runtime
WORKDIR /workspace
CMD ["python3"]
```

2. **Use efficient packaging strategies:**
```python
# For development with frequent changes
dev_packager = run.PatternPackager(
    include_pattern="src/**",
    relative_path=os.getcwd()
)

# For production with version control
prod_packager = run.GitArchivePackager(
    subpath="src"
)

# Choose based on environment
packager = dev_packager if is_development else prod_packager
executor = run.SlurmExecutor(packager=packager)
```

This comprehensive guide covers all aspects of NeMo Run execution, from basic usage to advanced features and best practices. Use these patterns to build robust, scalable machine learning workflows across different computing environments.
