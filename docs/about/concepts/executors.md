---
description: "Learn about NeMo Run's executor abstraction and how it enables multi-environment execution for ML experiments."
tags: ["executors", "concepts", "execution", "environments", "distributed"]
categories: ["about"]
---

(about-concepts-executors)=
# Executor Pattern

NeMo Run's executor system provides a unified abstraction for executing ML experiments across different environments, from local development to distributed clusters. The executor pattern decouples your experiment logic from the execution environment, making your code portable and scalable.

## Core Concepts

### Executor Abstraction

An executor is responsible for:
- **Environment Setup**: Preparing the execution environment
- **Code Packaging**: Packaging your code for the target environment
- **Task Execution**: Running your experiment with proper isolation
- **Resource Management**: Managing compute resources and cleanup
- **Result Collection**: Gathering outputs and artifacts

```python
import nemo_run as run

# Configure an executor
executor = run.Config(
    run.LocalExecutor,
    working_dir="/tmp/experiments",
    timeout=3600
)

# Use the executor to run an experiment
experiment = run.Experiment(
    task=my_training_function,
    executor=executor
)
```

## Built-in Executors

### Local Executor

The `run.LocalExecutor` runs experiments in isolated local processes:

```python
local_executor = run.Config(
    run.LocalExecutor,
    working_dir="./experiments",
    timeout=1800,  # 30 minutes
    env_vars={"CUDA_VISIBLE_DEVICES": "0,1"},
    python_path="/path/to/venv/bin/python"
)
```

**Use Cases:**
- Development and testing
- Small-scale experiments
- Debugging and profiling
- Local prototyping

### Docker Executor

The `run.DockerExecutor` runs experiments in isolated containers:

```python
docker_executor = run.Config(
    run.DockerExecutor,
    image="nvidia/cuda:11.8-devel-ubuntu20.04",
    working_dir="/workspace",
    volumes={
        "/host/data": "/container/data",
        "/host/models": "/container/models"
    },
    environment={
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "NCCL_DEBUG": "INFO"
    }
)
```

**Use Cases:**
- Reproducible environments
- Dependency isolation
- Multi-GPU training
- Production deployments

### Slurm Executor

The `run.SlurmExecutor` runs experiments on HPC clusters:

```python
slurm_executor = run.Config(
    run.SlurmExecutor,
    partition="gpu",
    nodes=2,
    ntasks_per_node=1,
    cpus_per_task=8,
    gpus_per_node=4,
    time="24:00:00",
    account="my_account",
    working_dir="/scratch/experiments"
)
```

**Use Cases:**
- Large-scale distributed training
- HPC cluster utilization
- Multi-node experiments
- Resource-intensive workloads

### SkyPilot Executor

The `run.SkypilotExecutor` runs experiments on cloud platforms:

```python
skypilot_executor = run.Config(
    run.SkypilotExecutor,
    cluster_name="nemo-run-cluster",
    region="us-west-2",
    instance_type="g4dn.xlarge",
    num_nodes=4,
    disk_size=100,
    working_dir="/workspace"
)
```

**Use Cases:**
- Cloud-based training
- Cost-effective scaling
- Multi-cloud deployments
- On-demand resources

## Executor Configuration

### Common Parameters

All executors share common configuration parameters:

```python
executor_config = run.Config(
    run.LocalExecutor,  # or any other executor
    working_dir="/path/to/workspace",
    timeout=3600,
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHONPATH": "/path/to/code"
    },
    python_path="/path/to/python",
    log_level="INFO"
)
```

### Environment Variables

Control the execution environment:

```python
executor = run.Config(
    run.LocalExecutor,
    env_vars={
        # GPU configuration
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "NCCL_DEBUG": "INFO",

        # Python configuration
        "PYTHONPATH": "/path/to/code:/path/to/libs",
        "PYTHONUNBUFFERED": "1",

        # Custom variables
        "EXPERIMENT_NAME": "transformer_training",
        "DATA_PATH": "/path/to/data"
    }
)
```

### Timeout and Resource Limits

Set execution limits:

```python
executor = run.Config(
    run.LocalExecutor,
    timeout=7200,  # 2 hours
    memory_limit="32G",
    cpu_limit=8
)
```

## Execution Patterns

### Basic Execution

```python
# Configure executor
executor = run.Config(run.LocalExecutor, working_dir="./experiments")

# Create experiment
experiment = run.Experiment(
    task=my_training_function,
    executor=executor,
    name="transformer_training"
)

# Run experiment
result = experiment.run()
```

### Distributed Execution

```python
# Multi-node Slurm execution
slurm_executor = run.Config(
    run.SlurmExecutor,
    nodes=4,
    ntasks_per_node=1,
    gpus_per_node=8,
    partition="gpu"
)

# The executor handles distributed setup automatically
experiment = run.Experiment(
    task=distributed_training_function,
    executor=slurm_executor
)
```

### Conditional Execution

```python
def get_executor(environment="local"):
    if environment == "local":
        return run.Config(run.LocalExecutor)
    elif environment == "docker":
        return run.Config(run.DockerExecutor, image="nvidia/cuda:11.8")
    elif environment == "slurm":
        return run.Config(run.SlurmExecutor, partition="gpu")
    else:
        raise ValueError(f"Unknown environment: {environment}")

# Use different executors based on environment
executor = get_executor(environment="slurm")
experiment = run.Experiment(task=my_function, executor=executor)
```

## Custom Executors

You can create custom executors by subclassing the base executor:

```python
class CustomExecutor(run.Executor):
    def __init__(self, custom_param=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def execute(self, task, config):
        """Custom execution logic."""
        # Implement your custom execution logic
        print(f"Executing with custom parameter: {self.custom_param}")

        # Call the parent execution method
        return super().execute(task, config)

# Use custom executor
custom_executor = run.Config(
    CustomExecutor,
    custom_param="special_value",
    working_dir="./experiments"
)
```

## Executor Lifecycle

### 1. Initialization

```python
executor = run.Config(run.LocalExecutor, working_dir="./experiments")
```

### 2. Environment Setup

The executor prepares the execution environment:
- Creates working directories
- Sets up environment variables
- Installs dependencies (if needed)
- Configures resource limits

### 3. Code Packaging

The executor packages your code for the target environment:
- Collects source files
- Resolves dependencies
- Creates deployment packages
- Handles file transfers

### 4. Task Execution

The executor runs your experiment:
- Launches the execution process
- Manages process lifecycle
- Handles signals and cleanup
- Monitors resource usage

### 5. Result Collection

The executor gathers results:
- Collects stdout/stderr
- Gathers artifacts
- Handles error conditions
- Cleans up resources

## Best Practices

### 1. Choose the Right Executor

```python
# Development and debugging
dev_executor = run.Config(run.LocalExecutor, timeout=300)

# Production training
prod_executor = run.Config(
    run.SlurmExecutor,
    nodes=8,
    gpus_per_node=8,
    time="48:00:00"
)

# Reproducible environments
docker_executor = run.Config(
    run.DockerExecutor,
    image="nvidia/cuda:11.8-devel-ubuntu20.04"
)
```

### 2. Configure Resource Limits

```python
executor = run.Config(
    run.LocalExecutor,
    timeout=3600,  # Prevent runaway experiments
    memory_limit="64G",  # Prevent OOM
    cpu_limit=16  # Control CPU usage
)
```

### 3. Use Environment Variables

```python
executor = run.Config(
    run.LocalExecutor,
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1",  # Control GPU usage
        "PYTHONPATH": "/path/to/code",  # Set Python path
        "EXPERIMENT_NAME": "my_experiment"  # Custom variables
    }
)
```

### 4. Handle Errors Gracefully

```python
try:
    experiment = run.Experiment(task=my_function, executor=executor)
    result = experiment.run()
except run.ExecutionError as e:
    print(f"Execution failed: {e}")
    # Handle error appropriately
```

### 5. Monitor Resource Usage

```python
# Enable resource monitoring
executor = run.Config(
    run.LocalExecutor,
    monitor_resources=True,
    log_level="DEBUG"
)
```

## Integration with Experiments

Executors work seamlessly with NeMo Run's experiment management:

```python
# Create experiment with executor
experiment = run.Experiment(
    task=training_function,
    executor=run.Config(run.SlurmExecutor, nodes=4),
    name="distributed_training"
)

# Run and monitor
experiment.run()

# Check status
status = experiment.status()
print(f"Experiment status: {status}")

# Get results
results = experiment.results()
print(f"Training completed: {results}")
```

The executor pattern is central to NeMo Run's design, providing the flexibility to run experiments anywhere while maintaining a consistent interface and ensuring reproducibility across environments.
