---
description: "Learn about NeMo Run's experiment lifecycle including creation, execution, monitoring, and reconstruction."
tags: ["experiments", "concepts", "lifecycle", "monitoring", "reconstruction"]
categories: ["about"]
---

(about-concepts-experiments)=
# Experiment Lifecycle

NeMo Run's experiment management system provides a complete lifecycle for ML experiments, from creation and execution to monitoring and reconstruction. Understanding the experiment lifecycle is key to effective experiment management.

## Core Concepts

### Experiment Abstraction

An experiment in NeMo Run represents:
- **Task Definition**: The function or script to execute
- **Configuration**: Parameters and settings for the experiment
- **Execution Environment**: Where and how the experiment runs
- **Metadata**: Information about the experiment's lifecycle
- **Results**: Outputs, artifacts, and metrics

```python
import nemo_run as run

# Create an experiment
experiment = run.Experiment(
    task=training_function,
    executor=run.Config(run.LocalExecutor),
    name="transformer_training",
    description="Training a transformer model on text data"
)
```

## Experiment Lifecycle Stages

### 1. Creation

Experiments are created with a task, executor, and metadata:

```python
experiment = run.Experiment(
    task=run.Config(
        training_function,
        model=run.Config(TransformerModel, hidden_size=768),
        data=run.Config(DataLoader, batch_size=32),
        optimizer=run.Config(Adam, learning_rate=1e-4)
    ),
    executor=run.Config(
        run.SlurmExecutor,
        nodes=4,
        gpus_per_node=8,
        time="24:00:00"
    ),
    name="large_transformer_training",
    description="Training a large transformer model on distributed cluster",
    tags=["transformer", "distributed", "large-scale"]
)
```

### 2. Execution

Experiments are executed with monitoring and control:

```python
# Start execution
experiment.run()

# Or run asynchronously
future = experiment.run_async()

# Wait for completion
result = future.result()
```

### 3. Monitoring

Monitor experiment progress in real-time:

```python
# Get current status
status = experiment.status()
print(f"Status: {status}")

# Stream logs
for log_line in experiment.logs():
    print(log_line)

# Monitor resources
resources = experiment.resources()
print(f"CPU: {resources.cpu_percent}%, Memory: {resources.memory_percent}%")
```

### 4. Reconstruction

Reconstruct experiments from metadata:

```python
# Reconstruct from experiment ID
reconstructed = run.Experiment.from_id("exp_12345")

# Reconstruct from configuration
reconstructed = run.Experiment.from_config(saved_config)

# Verify reconstruction
assert reconstructed.task == experiment.task
assert reconstructed.executor == experiment.executor
```

## Experiment Configuration

### Task Configuration

Configure the experiment task:

```python
task_config = run.Config(
    training_function,
    model=run.Config(
        TransformerModel,
        hidden_size=768,
        num_layers=12,
        num_heads=12
    ),
    data=run.Config(
        DataLoader,
        batch_size=32,
        num_workers=4,
        shuffle=True
    ),
    training=run.Config(
        TrainingConfig,
        epochs=100,
        learning_rate=1e-4,
        weight_decay=1e-5
    )
)

experiment = run.Experiment(task=task_config, executor=executor)
```

### Executor Configuration

Configure the execution environment:

```python
executor_config = run.Config(
    run.SlurmExecutor,
    partition="gpu",
    nodes=4,
    ntasks_per_node=1,
    cpus_per_task=8,
    gpus_per_node=8,
    time="24:00:00",
    account="my_account",
    working_dir="/scratch/experiments"
)

experiment = run.Experiment(task=task, executor=executor_config)
```

### Metadata Configuration

Add rich metadata to experiments:

```python
experiment = run.Experiment(
    task=task,
    executor=executor,
    name="transformer_finetuning",
    description="Fine-tuning a pre-trained transformer model on domain-specific data",
    tags=["transformer", "finetuning", "nlp"],
    metadata={
        "dataset": "domain_specific_corpus",
        "base_model": "bert-base-uncased",
        "expected_accuracy": 0.85,
        "team": "nlp_research"
    }
)
```

## Experiment Management

### Listing Experiments

```python
# List all experiments
experiments = run.Experiment.list()

# List with filters
experiments = run.Experiment.list(
    status="completed",
    tags=["transformer"],
    limit=10
)

# Search experiments
experiments = run.Experiment.search("transformer training")
```

### Experiment Status

```python
# Check status
status = experiment.status()
print(f"Status: {status}")

# Possible statuses:
# - "pending": Waiting to start
# - "running": Currently executing
# - "completed": Successfully finished
# - "failed": Execution failed
# - "cancelled": Manually cancelled
# - "timeout": Execution timed out
```

### Experiment Results

```python
# Get results
results = experiment.results()

# Access specific outputs
model_path = results.artifacts["model"]
metrics = results.metrics
logs = results.logs

# Download artifacts
experiment.download_artifacts("/local/path")
```

## Advanced Patterns

### Experiment Dependencies

Create experiments that depend on others:

```python
# Create dependent experiments
data_prep = run.Experiment(
    task=data_preparation_function,
    executor=local_executor,
    name="data_preparation"
)

training = run.Experiment(
    task=training_function,
    executor=slurm_executor,
    name="model_training",
    dependencies=[data_prep]  # Wait for data_prep to complete
)

# Run with dependencies
training.run()  # Automatically waits for data_prep
```

### Experiment Templates

Create reusable experiment templates:

```python
def create_training_experiment(model_config, data_config, name):
    return run.Experiment(
        task=run.Config(
            training_function,
            model=model_config,
            data=data_config
        ),
        executor=run.Config(run.SlurmExecutor, nodes=4),
        name=name,
        tags=["training", "template"]
    )

# Use template
experiment = create_training_experiment(
    model_config=run.Config(TransformerModel, hidden_size=768),
    data_config=run.Config(DataLoader, batch_size=32),
    name="transformer_training_v1"
)
```

### Experiment Orchestration

Orchestrate complex experiment workflows:

```python
# Create experiment pipeline
experiments = []

# Data preparation
data_prep = run.Experiment(task=data_prep_function, name="data_prep")
experiments.append(data_prep)

# Model training
training = run.Experiment(
    task=training_function,
    dependencies=[data_prep],
    name="training"
)
experiments.append(training)

# Evaluation
evaluation = run.Experiment(
    task=evaluation_function,
    dependencies=[training],
    name="evaluation"
)
experiments.append(evaluation)

# Run pipeline
for exp in experiments:
    exp.run()
```

## Monitoring and Debugging

### Real-time Monitoring

```python
# Monitor experiment progress
experiment.run()

# Stream logs in real-time
for log_line in experiment.logs(stream=True):
    print(log_line, end="")

# Monitor resources
import time
while experiment.status() == "running":
    resources = experiment.resources()
    print(f"CPU: {resources.cpu_percent}%, GPU: {resources.gpu_percent}%")
    time.sleep(30)
```

### Debugging Failed Experiments

```python
# Check experiment status
if experiment.status() == "failed":
    # Get error information
    error_info = experiment.error()
    print(f"Error: {error_info.message}")
    print(f"Traceback: {error_info.traceback}")

    # Get logs for debugging
    logs = experiment.logs()
    for log in logs[-100:]:  # Last 100 log lines
        print(log)

    # Reconstruct for debugging
    debug_experiment = experiment.reconstruct()
    debug_experiment.run(debug=True)
```

### Experiment Comparison

```python
# Compare experiments
exp1 = run.Experiment.from_id("exp_12345")
exp2 = run.Experiment.from_id("exp_12346")

# Compare configurations
diff = exp1.compare_config(exp2)
print(f"Configuration differences: {diff}")

# Compare results
results_diff = exp1.compare_results(exp2)
print(f"Results differences: {results_diff}")
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good: Descriptive and specific
experiment = run.Experiment(
    task=task,
    name="transformer_finetuning_bert_base_uncased_domain_data_v1"
)

# Avoid: Generic names
bad_experiment = run.Experiment(task=task, name="experiment_1")
```

### 2. Add Rich Metadata

```python
experiment = run.Experiment(
    task=task,
    executor=executor,
    name="transformer_training",
    description="Training a transformer model for text classification",
    tags=["transformer", "nlp", "classification"],
    metadata={
        "dataset_size": "1M examples",
        "expected_duration": "4 hours",
        "team": "nlp_research",
        "priority": "high"
    }
)
```

### 3. Handle Dependencies

```python
# Create experiments with clear dependencies
data_prep = run.Experiment(task=data_prep_function, name="data_prep")
training = run.Experiment(
    task=training_function,
    dependencies=[data_prep],
    name="training"
)
evaluation = run.Experiment(
    task=evaluation_function,
    dependencies=[training],
    name="evaluation"
)
```

### 4. Monitor Resources

```python
# Set appropriate resource limits
executor = run.Config(
    run.SlurmExecutor,
    time="24:00:00",  # Set reasonable timeout
    memory="64G",     # Set memory limit
    gpus_per_node=8   # Set GPU limit
)

experiment = run.Experiment(task=task, executor=executor)
```

### 5. Archive Important Experiments

```python
# Archive completed experiments
if experiment.status() == "completed":
    experiment.archive()

# Restore archived experiments
archived = run.Experiment.from_archive("archive_path")
```

## Integration with Other Systems

### Version Control Integration

```python
# Include Git information
experiment = run.Experiment(
    task=task,
    executor=executor,
    metadata={
        "git_commit": "abc123",
        "git_branch": "main",
        "git_repo": "https://github.com/org/repo"
    }
)
```

### Artifact Storage

```python
# Configure artifact storage
experiment = run.Experiment(
    task=task,
    executor=executor,
    artifact_storage=run.Config(
        run.S3Storage,
        bucket="my-experiments",
        prefix="transformer_training"
    )
)
```

The experiment lifecycle is central to NeMo Run's experiment management capabilities, providing a complete framework for creating, executing, monitoring, and reconstructing ML experiments with full reproducibility and traceability.
