# Manage NeMo Run Experiments

NeMo Run provides a comprehensive experiment management system centered around the `Experiment` class. This system enables you to define, launch, monitor, and manage complex machine learning workflows with multiple interdependent tasks. This guide covers all aspects of experiment lifecycle management.

## Experiment Management Overview

The `Experiment` class serves as the central orchestrator for managing multi-task workflows in NeMo Run. It provides:

- **Task Orchestration**: Define and manage multiple tasks with dependencies
- **Execution Control**: Launch, monitor, and control experiment execution
- **Metadata Tracking**: Automatic tracking of experiment metadata and artifacts
- **Logging and Monitoring**: Real-time log access and status monitoring
- **Reproducibility**: Complete experiment state capture for reproducibility

## Creating and Configuring Experiments

### Basic Experiment Creation

Create an experiment with a descriptive title:

```python
import nemo_run as run

# Create a simple experiment
experiment = run.Experiment("transformer-finetuning")

# Create with additional metadata
experiment = run.Experiment(
    "llama3-pretraining",
    description="Pre-training Llama3 8B model on custom dataset",
    tags=["llama3", "pretraining", "8b"],
    metadata={
        "dataset": "wikitext-103",
        "model_size": "8B",
        "framework": "NeMo"
    }
)
```

### Experiment Configuration

Configure experiment behavior and logging:

```python
experiment = run.Experiment(
    "distributed-training",
    log_level="INFO",  # DEBUG, INFO, WARNING, ERROR
    max_retries=3,     # Number of retry attempts for failed tasks
    timeout=3600,      # Global timeout in seconds
    checkpoint_interval=300,  # Checkpoint frequency in seconds
    metadata={
        "project": "ml-research",
        "team": "ai-team",
        "priority": "high"
    }
)
```

## Task Management

### Adding Individual Tasks

Add single tasks to your experiment:

```python
# Define your task configurations
model_config = run.Config(
    TrainModel,
    model_size="8b",
    learning_rate=0.001,
    batch_size=32
)

data_config = run.Config(
    PreprocessData,
    input_path="/data/raw",
    output_path="/data/processed"
)

# Create experiment and add tasks
with run.Experiment("ml-pipeline") as exp:
    # Add training task
    training_id = exp.add(
        model_config,
        executor=run.SlurmExecutor(
            partition="gpu",
            nodes=2,
            gpus_per_node=4
        ),
        name="model-training"
    )

    # Add data preprocessing task
    data_id = exp.add(
        data_config,
        executor=run.LocalExecutor(),
        name="data-preprocessing"
    )
```

### Adding Task Groups

Add multiple tasks that execute in parallel:

```python
# Define multiple model configurations
model_configs = [
    run.Config(TrainModel, model_size="8b", learning_rate=0.001),
    run.Config(TrainModel, model_size="8b", learning_rate=0.0001),
    run.Config(TrainModel, model_size="70b", learning_rate=0.001)
]

# Create experiment with parallel tasks
with run.Experiment("hyperparameter-sweep") as exp:
    # Add all model configurations to run in parallel
    task_ids = exp.add(
        model_configs,
        executor=run.SlurmExecutor(
            partition="gpu",
            nodes=1,
            gpus_per_node=8
        ),
        name="model-variants"
    )

    # Add evaluation task that depends on all training tasks
    exp.add(
        run.Config(EvaluateModels, model_paths="/checkpoints/*"),
        executor=run.LocalExecutor(),
        name="model-evaluation",
        dependencies=task_ids  # Wait for all training tasks to complete
    )
```

### Task Dependencies and Workflows

Create complex workflows with task dependencies:

```python
def create_ml_pipeline():
    """Create a complete ML pipeline with dependencies."""

    with run.Experiment("complete-ml-pipeline") as exp:
        # Stage 1: Data preparation
        data_prep_id = exp.add(
            run.Config(PrepareData, dataset="wikitext-103"),
            executor=run.LocalExecutor(),
            name="data-preparation"
        )

        # Stage 2: Model training (depends on data preparation)
        training_id = exp.add(
            run.Config(TrainModel, data_path="/data/processed"),
            executor=run.SlurmExecutor(partition="gpu", nodes=4),
            name="model-training",
            dependencies=[data_prep_id]
        )

        # Stage 3: Model evaluation (depends on training)
        eval_id = exp.add(
            run.Config(EvaluateModel, model_path="/checkpoints/best"),
            executor=run.LocalExecutor(),
            name="model-evaluation",
            dependencies=[training_id]
        )

        # Stage 4: Model deployment (depends on evaluation)
        deploy_id = exp.add(
            run.Config(DeployModel, model_path="/checkpoints/best"),
            executor=run.DockerExecutor(),
            name="model-deployment",
            dependencies=[eval_id]
        )

        return exp

# Usage
pipeline = create_ml_pipeline()
```

### Using Plugins

Plugins allow you to modify tasks and executors together:

```python
# Define a custom plugin
class MixedPrecisionPlugin(run.Plugin):
    """Plugin to enable mixed precision training."""

    def modify_task(self, task):
        """Modify task to use mixed precision."""
        if hasattr(task, 'precision'):
            task.precision = "bf16-mixed"
        return task

    def modify_executor(self, executor):
        """Modify executor with mixed precision environment variables."""
        if not hasattr(executor, 'env_vars'):
            executor.env_vars = {}
        executor.env_vars.update({
            "NCCL_P2P_DISABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
        })
        return executor

# Use plugin with tasks
with run.Experiment("mixed-precision-training") as exp:
    exp.add(
        run.Config(TrainModel, model_size="8b"),
        executor=run.SlurmExecutor(partition="gpu"),
        plugins=[MixedPrecisionPlugin()],
        name="training-with-mixed-precision"
    )
```

## Experiment Execution

### Launching Experiments

Launch experiments with different execution modes:

```python
with run.Experiment("my-experiment") as exp:
    # Add tasks...

    # Launch with different options
    exp.run(
        detach=False,      # Stay attached to monitor progress
        sequential=False,  # Execute tasks in parallel where possible
        tail_logs=True,    # Show real-time logs
        direct=False       # Use remote execution
    )
```

### Execution Modes

#### Attached Execution (Default)

Monitor experiment progress in real-time:

```python
with run.Experiment("monitored-experiment") as exp:
    exp.add(task_config, executor=executor)

    # Launch and monitor
    exp.run(
        detach=False,      # Stay attached
        tail_logs=True     # Show logs in real-time
    )

    # Check final status
    print(f"Experiment completed with status: {exp.status()}")
```

#### Detached Execution

Launch experiments and detach for long-running tasks:

```python
with run.Experiment("long-running-experiment") as exp:
    exp.add(task_config, executor=executor)

    # Launch and detach
    exp.run(detach=True)

    # Save experiment ID for later monitoring
    experiment_id = exp.experiment_id
    print(f"Experiment launched with ID: {experiment_id}")

    # Later, retrieve and monitor
    retrieved_exp = run.get_experiment(experiment_id)
    print(retrieved_exp.status())
```

#### Sequential Execution

Execute tasks one after another:

```python
with run.Experiment("sequential-experiment") as exp:
    exp.add([task1, task2, task3], executor=executor)

    # Execute sequentially
    exp.run(sequential=True)
```

#### Direct Execution

Execute tasks directly in the current process:

```python
with run.Experiment("direct-execution") as exp:
    exp.add(task_config, executor=executor)

    # Execute directly (no remote execution)
    exp.run(direct=True)
```

## Monitoring and Status

### Checking Experiment Status

Monitor experiment and task status:

```python
# Check overall experiment status
status = experiment.status()
print(f"Experiment status: {status}")

# Get detailed task information
for task in experiment.tasks:
    print(f"Task {task.name}: {task.status}")
    print(f"  - Job ID: {task.job_id}")
    print(f"  - Executor: {task.executor}")
    print(f"  - Directory: {task.local_directory}")
    print(f"  - Start time: {task.start_time}")
    print(f"  - End time: {task.end_time}")
```

### Real-Time Monitoring

Monitor experiments in real-time:

```python
import time

def monitor_experiment(experiment, check_interval=30):
    """Monitor experiment progress in real-time."""

    print(f"Monitoring experiment: {experiment.experiment_id}")

    while True:
        status = experiment.status()
        print(f"\nStatus at {time.strftime('%H:%M:%S')}:")

        for task in experiment.tasks:
            print(f"  {task.name}: {task.status}")

        # Check if all tasks are complete
        if all(task.status in ['SUCCEEDED', 'FAILED', 'CANCELLED']
               for task in experiment.tasks):
            print("\nExperiment completed!")
            break

        time.sleep(check_interval)

# Usage
with run.Experiment("monitored-experiment") as exp:
    exp.add(task_config, executor=executor)
    exp.run(detach=False)
    monitor_experiment(exp)
```

### Task Control

Control individual tasks:

```python
# Cancel a specific task
experiment.cancel("task_id")

# Cancel all tasks
experiment.cancel_all()

# Retry a failed task
experiment.retry("task_id")

# Pause/resume tasks
experiment.pause("task_id")
experiment.resume("task_id")
```

## Logging and Debugging

### Accessing Task Logs

Retrieve and analyze task logs:

```python
# Get logs for a specific task
task_logs = experiment.get_logs("task_id")
print(f"Exit code: {task_logs.exit_code}")
print(f"Stdout: {task_logs.stdout}")
print(f"Stderr: {task_logs.stderr}")

# Stream logs in real-time
for log_entry in experiment.stream_logs("task_id"):
    print(f"{log_entry.timestamp}: {log_entry.message}")

# Get logs for all tasks
all_logs = experiment.get_all_logs()
for task_name, logs in all_logs.items():
    print(f"\n=== {task_name} ===")
    print(f"Exit code: {logs.exit_code}")
    if logs.stderr:
        print(f"Errors: {logs.stderr}")
```

### Log Analysis

Analyze logs for debugging and monitoring:

```python
import re
from typing import Dict, List

def analyze_experiment_logs(experiment) -> Dict[str, Dict]:
    """Analyze logs from all tasks in an experiment."""

    analysis = {}

    for task in experiment.tasks:
        logs = experiment.get_logs(task.id)

        # Extract metrics
        metrics = {}
        if logs.stdout:
            # Extract loss values
            loss_pattern = r"loss: (\d+\.\d+)"
            losses = re.findall(loss_pattern, logs.stdout)
            if losses:
                metrics['losses'] = [float(l) for l in losses]
                metrics['final_loss'] = float(losses[-1])

            # Extract accuracy values
            acc_pattern = r"accuracy: (\d+\.\d+)"
            accuracies = re.findall(acc_pattern, logs.stdout)
            if accuracies:
                metrics['accuracies'] = [float(a) for a in accuracies]
                metrics['final_accuracy'] = float(accuracies[-1])

        # Extract errors
        errors = []
        if logs.stderr:
            error_lines = logs.stderr.split('\n')
            errors = [line.strip() for line in error_lines if line.strip()]

        analysis[task.name] = {
            'status': task.status,
            'exit_code': logs.exit_code,
            'metrics': metrics,
            'errors': errors,
            'duration': task.end_time - task.start_time if task.end_time else None
        }

    return analysis

# Usage
analysis = analyze_experiment_logs(experiment)
for task_name, data in analysis.items():
    print(f"\n=== {task_name} Analysis ===")
    print(f"Status: {data['status']}")
    print(f"Exit code: {data['exit_code']}")
    if data['metrics']:
        print(f"Final loss: {data['metrics'].get('final_loss', 'N/A')}")
        print(f"Final accuracy: {data['metrics'].get('final_accuracy', 'N/A')}")
    if data['errors']:
        print(f"Errors: {len(data['errors'])} found")
```

## Experiment Metadata and Artifacts

### Metadata Management

Track and retrieve experiment metadata:

```python
# Add metadata during creation
experiment = run.Experiment(
    "hyperparameter-sweep",
    metadata={
        "dataset": "wikitext-103",
        "model_family": "llama",
        "sweep_type": "learning_rate",
        "num_trials": 10
    }
)

# Add metadata after creation
experiment.add_metadata({
    "completed_at": time.time(),
    "best_accuracy": 0.95,
    "best_model_path": "/checkpoints/best"
})

# Retrieve metadata
metadata = experiment.get_metadata()
print(f"Dataset: {metadata.get('dataset')}")
print(f"Best accuracy: {metadata.get('best_accuracy')}")
```

### Artifact Management

Track and retrieve experiment artifacts:

```python
# Add artifacts
experiment.add_artifact(
    "best_model",
    "/checkpoints/best_model.pt",
    description="Best performing model checkpoint"
)

experiment.add_artifact(
    "training_logs",
    "/logs/training.log",
    description="Complete training logs"
)

# Retrieve artifacts
artifacts = experiment.get_artifacts()
for name, artifact in artifacts.items():
    print(f"{name}: {artifact.path}")
    print(f"  Description: {artifact.description}")
    print(f"  Size: {artifact.size} bytes")
```

## Experiment Reproducibility

### Experiment Snapshots

Create reproducible experiment snapshots:

```python
# Create a snapshot of the current experiment state
snapshot = experiment.create_snapshot()

# Save snapshot to file
snapshot.save("/path/to/snapshot.json")

# Load and reproduce experiment
loaded_snapshot = run.ExperimentSnapshot.load("/path/to/snapshot.json")
reproduced_experiment = loaded_snapshot.reproduce()
```

### Configuration Tracking

Track configuration changes:

```python
# Track configuration versions
experiment.track_config_version(
    "model_config",
    model_config,
    description="Initial model configuration"
)

# Update configuration
model_config.learning_rate = 0.0005
experiment.track_config_version(
    "model_config",
    model_config,
    description="Updated learning rate"
)

# Retrieve configuration history
config_history = experiment.get_config_history("model_config")
for version, config in config_history.items():
    print(f"Version {version}: {config.description}")
```

## Best Practices

### Experiment Organization

```python
def create_organized_experiment(
    project_name: str,
    experiment_type: str,
    model_size: str
) -> run.Experiment:
    """Create a well-organized experiment with consistent naming."""

    # Create descriptive name
    experiment_name = f"{project_name}-{experiment_type}-{model_size}"

    # Add comprehensive metadata
    metadata = {
        "project": project_name,
        "experiment_type": experiment_type,
        "model_size": model_size,
        "created_by": os.getenv("USER", "unknown"),
        "created_at": time.time(),
        "git_commit": get_git_commit_hash(),
        "environment": os.getenv("NEMO_ENV", "development")
    }

    return run.Experiment(experiment_name, metadata=metadata)

# Usage
exp = create_organized_experiment(
    project_name="llama-finetuning",
    experiment_type="supervised",
    model_size="8b"
)
```

### Error Handling and Recovery

```python
def run_experiment_with_recovery(experiment_config, max_retries=3):
    """Run experiment with automatic error recovery."""

    for attempt in range(max_retries):
        try:
            with run.Experiment("recovery-experiment") as exp:
                exp.add(experiment_config, executor=executor)
                exp.run()

                # Check for failures
                failed_tasks = [task for task in exp.tasks if task.status == 'FAILED']
                if failed_tasks:
                    print(f"Attempt {attempt + 1} failed with {len(failed_tasks)} failed tasks")
                    if attempt < max_retries - 1:
                        print("Retrying...")
                        continue
                    else:
                        raise Exception("Max retries exceeded")

                print("Experiment completed successfully!")
                return exp

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
```

### Resource Management

```python
def create_resource_aware_experiment(resource_constraints):
    """Create experiment that respects resource constraints."""

    with run.Experiment("resource-aware-experiment") as exp:
        # Check available resources
        available_gpus = get_available_gpus()
        available_memory = get_available_memory()

        # Adjust configuration based on resources
        if available_gpus < resource_constraints['min_gpus']:
            raise ValueError("Insufficient GPU resources")

        # Create adaptive executor
        executor = run.SlurmExecutor(
            partition="gpu",
            nodes=min(resource_constraints['max_nodes'], available_gpus // 8),
            gpus_per_node=min(8, available_gpus)
        )

        exp.add(task_config, executor=executor)
        return exp
```

This comprehensive guide covers all aspects of NeMo Run experiment management, from basic usage to advanced monitoring and reproducibility features. Use these patterns to build robust, maintainable, and reproducible machine learning workflows.
