---
description: "Learn about NeMo Run's task dependency system and how to create complex workflows with orchestration."
tags: ["dependencies", "concepts", "workflows", "orchestration", "pipelines"]
categories: ["about"]
---

(about-concepts-dependencies)=
# Task Dependencies

NeMo Run's dependency system enables you to create complex workflows by defining relationships between experiments. Dependencies ensure that experiments run in the correct order and handle failures gracefully.

## Core Concepts

### Dependency Types

NeMo Run supports several types of dependencies:

- **Execution Dependencies**: Experiments that must complete before others start
- **Data Dependencies**: Experiments that produce data consumed by others
- **Resource Dependencies**: Experiments that require specific resources
- **Conditional Dependencies**: Dependencies that depend on experiment outcomes

```python
import nemo_run as run

# Create experiments with dependencies
data_prep = run.Experiment(task=data_preparation, name="data_prep")
training = run.Experiment(
    task=training_function,
    dependencies=[data_prep],  # Wait for data_prep to complete
    name="training"
)
```

## Basic Dependencies

### Sequential Dependencies

Create simple sequential workflows:

```python
# Data preparation experiment
data_prep = run.Experiment(
    task=run.Config(
        data_preparation_function,
        input_path="/path/to/raw/data",
        output_path="/path/to/processed/data"
    ),
    executor=run.Config(run.LocalExecutor),
    name="data_preparation"
)

# Training experiment (depends on data_prep)
training = run.Experiment(
    task=run.Config(
        training_function,
        data_path="/path/to/processed/data",
        model_config=run.Config(TransformerModel, hidden_size=768)
    ),
    executor=run.Config(run.SlurmExecutor, nodes=4),
    dependencies=[data_prep],  # Must wait for data_prep
    name="model_training"
)

# Evaluation experiment (depends on training)
evaluation = run.Experiment(
    task=run.Config(
        evaluation_function,
        model_path="/path/to/trained/model",
        test_data="/path/to/test/data"
    ),
    executor=run.Config(run.LocalExecutor),
    dependencies=[training],  # Must wait for training
    name="model_evaluation"
)
```

### Parallel Dependencies

Run independent experiments in parallel:

```python
# Independent data processing tasks
text_prep = run.Experiment(
    task=text_preprocessing_function,
    name="text_preprocessing"
)

image_prep = run.Experiment(
    task=image_preprocessing_function,
    name="image_preprocessing"
)

# Training that depends on both
multimodal_training = run.Experiment(
    task=multimodal_training_function,
    dependencies=[text_prep, image_prep],  # Wait for both
    name="multimodal_training"
)
```

## Advanced Dependency Patterns

### Conditional Dependencies

Create dependencies based on experiment outcomes:

```python
# Initial experiment
baseline = run.Experiment(
    task=baseline_training_function,
    name="baseline_training"
)

# Conditional experiment based on baseline results
def should_run_advanced(baseline_results):
    return baseline_results.accuracy > 0.8

advanced_training = run.Experiment(
    task=advanced_training_function,
    dependencies=[baseline],
    condition=should_run_advanced,  # Only run if condition is met
    name="advanced_training"
)
```

### Data Dependencies

Create dependencies based on data availability:

```python
# Data generation experiment
data_gen = run.Experiment(
    task=data_generation_function,
    name="data_generation"
)

# Training with data dependency
training = run.Experiment(
    task=training_function,
    dependencies=[data_gen],
    data_dependencies={
        "training_data": "/path/to/generated/data",
        "validation_data": "/path/to/generated/validation"
    },
    name="training"
)
```

### Resource Dependencies

Create dependencies based on resource availability:

```python
# Resource-intensive preprocessing
heavy_prep = run.Experiment(
    task=heavy_preprocessing_function,
    executor=run.Config(run.SlurmExecutor, nodes=8),
    name="heavy_preprocessing"
)

# Training that requires the same resources
training = run.Experiment(
    task=training_function,
    dependencies=[heavy_prep],
    resource_dependencies={
        "gpu_cluster": "gpu_partition",
        "memory": "64G"
    },
    name="training"
)
```

## Workflow Orchestration

### Pipeline Creation

Create complex pipelines with multiple stages:

```python
def create_ml_pipeline():
    experiments = []

    # Stage 1: Data preparation
    data_prep = run.Experiment(
        task=data_preparation_function,
        name="data_preparation"
    )
    experiments.append(data_prep)

    # Stage 2: Feature engineering
    feature_eng = run.Experiment(
        task=feature_engineering_function,
        dependencies=[data_prep],
        name="feature_engineering"
    )
    experiments.append(feature_eng)

    # Stage 3: Model training (parallel)
    model_a = run.Experiment(
        task=model_a_training_function,
        dependencies=[feature_eng],
        name="model_a_training"
    )
    experiments.append(model_a)

    model_b = run.Experiment(
        task=model_b_training_function,
        dependencies=[feature_eng],
        name="model_b_training"
    )
    experiments.append(model_b)

    # Stage 4: Ensemble training
    ensemble = run.Experiment(
        task=ensemble_training_function,
        dependencies=[model_a, model_b],
        name="ensemble_training"
    )
    experiments.append(ensemble)

    # Stage 5: Evaluation
    evaluation = run.Experiment(
        task=evaluation_function,
        dependencies=[ensemble],
        name="evaluation"
    )
    experiments.append(evaluation)

    return experiments

# Create and run pipeline
pipeline = create_ml_pipeline()
for exp in pipeline:
    exp.run()
```

### Dependency Graphs

Visualize and manage dependency relationships:

```python
# Create dependency graph
graph = run.DependencyGraph()

# Add experiments
graph.add_experiment(data_prep)
graph.add_experiment(training, dependencies=[data_prep])
graph.add_experiment(evaluation, dependencies=[training])

# Visualize graph
graph.visualize("pipeline_graph.png")

# Check for cycles
if graph.has_cycles():
    print("Warning: Circular dependencies detected!")

# Get execution order
execution_order = graph.topological_sort()
print(f"Execution order: {execution_order}")
```

## Error Handling and Recovery

### Dependency Failure Handling

Handle failures in dependent experiments:

```python
# Configure failure handling
training = run.Experiment(
    task=training_function,
    dependencies=[data_prep],
    on_dependency_failure="retry",  # or "skip", "fail"
    max_retries=3,
    retry_delay=300,  # 5 minutes
    name="training"
)
```

### Fallback Dependencies

Provide fallback experiments:

```python
# Primary data source
primary_data = run.Experiment(
    task=primary_data_preparation,
    name="primary_data_prep"
)

# Fallback data source
fallback_data = run.Experiment(
    task=fallback_data_preparation,
    name="fallback_data_prep"
)

# Training with fallback
training = run.Experiment(
    task=training_function,
    dependencies=[primary_data],
    fallback_dependencies=[fallback_data],  # Use if primary fails
    name="training"
)
```

### Dependency Monitoring

Monitor dependency status:

```python
# Check dependency status
for dep in training.dependencies:
    status = dep.status()
    print(f"Dependency {dep.name}: {status}")

    if status == "failed":
        print(f"Error: {dep.error()}")
        print(f"Logs: {dep.logs()[-10:]}")  # Last 10 log lines
```

## Best Practices

### 1. Clear Dependency Naming

```python
# Good: Clear and descriptive names
data_prep = run.Experiment(task=data_prep_function, name="text_data_preparation")
training = run.Experiment(
    task=training_function,
    dependencies=[data_prep],
    name="transformer_training"
)

# Avoid: Unclear names
exp1 = run.Experiment(task=func1, name="experiment_1")
exp2 = run.Experiment(task=func2, dependencies=[exp1], name="experiment_2")
```

### 2. Minimize Dependencies

```python
# Good: Minimal, clear dependencies
data_prep = run.Experiment(task=data_prep_function, name="data_prep")
training = run.Experiment(
    task=training_function,
    dependencies=[data_prep],  # Only essential dependency
    name="training"
)

# Avoid: Unnecessary dependencies
training = run.Experiment(
    task=training_function,
    dependencies=[data_prep, unrelated_exp],  # Unnecessary dependency
    name="training"
)
```

### 3. Handle Failures Gracefully

```python
# Configure proper error handling
experiment = run.Experiment(
    task=task,
    dependencies=dependencies,
    on_dependency_failure="retry",
    max_retries=3,
    retry_delay=300,
    timeout=3600
)
```

### 4. Use Conditional Dependencies

```python
# Only run expensive experiments when needed
def should_run_expensive(baseline_results):
    return baseline_results.accuracy > 0.7

expensive_training = run.Experiment(
    task=expensive_training_function,
    dependencies=[baseline],
    condition=should_run_expensive,
    name="expensive_training"
)
```

### 5. Monitor Dependency Health

```python
# Regular dependency health checks
def check_dependencies(experiment):
    for dep in experiment.dependencies:
        if dep.status() == "failed":
            print(f"Warning: Dependency {dep.name} failed")
            # Send notification or take corrective action

# Check before running
check_dependencies(training)
training.run()
```

## Integration with External Systems

### CI/CD Integration

```python
# Trigger experiments based on CI/CD events
def on_code_change():
    # Run tests first
    tests = run.Experiment(task=run_tests, name="tests")

    # Only run training if tests pass
    training = run.Experiment(
        task=training_function,
        dependencies=[tests],
        condition=lambda results: results.tests_passed,
        name="training"
    )

    return training
```

### Data Pipeline Integration

```python
# Integrate with data pipelines
def on_data_update():
    # Wait for data pipeline completion
    data_pipeline = run.Experiment(
        task=external_data_pipeline,
        name="data_pipeline"
    )

    # Run ML experiments after data is ready
    ml_experiments = []
    for model in models:
        exp = run.Experiment(
            task=training_function,
            dependencies=[data_pipeline],
            name=f"training_{model}"
        )
        ml_experiments.append(exp)

    return ml_experiments
```

The dependency system is essential for creating complex, reliable ML workflows in NeMo Run, enabling you to build sophisticated pipelines while maintaining clarity and manageability.
