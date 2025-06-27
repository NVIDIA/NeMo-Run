---
description: "Comprehensive tutorials and learning resources for NeMo Run - Learn by example with hands-on guides, notebooks, and practical exercises."
tags: ["tutorials", "examples", "hello-world", "notebooks", "learning", "hands-on"]
categories: ["get-started"]
---

(tutorials)=

# Tutorials and Learning Resources

Welcome to the NeMo Run tutorial collection! This comprehensive guide provides hands-on learning experiences to help you master NeMo Run's capabilities for machine learning experiment management and distributed computing.

## Learning Path

Our tutorials are designed to guide you from basic concepts to advanced workflows:

1. **Getting Started**: Basic configuration and execution
2. **Experiment Management**: Creating and tracking experiments
3. **Advanced Workflows**: Script-based execution and automation
4. **Distributed Computing**: Ray clusters and cloud execution
5. **Production Deployment**: Best practices and optimization

## Tutorial Series

### Hello World Series

The `hello_world` tutorial series provides a comprehensive introduction to NeMo Run, demonstrating its core capabilities through practical examples.

#### What You'll Learn

- **Configuration Management**: Using `Partial` and `Config` classes for flexible parameter management
- **Execution Backends**: Running functions locally and on remote clusters
- **Visualization**: Creating configuration diagrams with `graphviz`
- **Experiment Tracking**: Managing experiments with `run.Experiment`
- **Automation**: Script-based execution and workflow automation

#### Tutorial Structure

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Part 1: Hello World
:link: <https://github.com/NVIDIA-NeMo/Run/tree/main/examples/hello-world/hello_world.ipynb>
:link-type: url
:link-alt: Hello World tutorial part 1

**Basic Configuration and Execution**

Learn the fundamentals of NeMo Run configuration and execution:

- Creating and configuring Python functions
- Using `Partial` for parameter management
- Basic execution on local and remote backends
- Understanding the execution model
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Part 2: Hello Experiments
:link: <https://github.com/NVIDIA-NeMo/Run/tree/main/examples/hello-world/hello_experiments.ipynb>
:link-type: url
:link-alt: Hello World tutorial part 2

**Experiment Management and Tracking**

Master experiment lifecycle management:

- Creating and managing experiments
- Parameter tracking and versioning
- Result collection and analysis
- Experiment comparison and visualization
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Part 3: Hello Scripts
:link: <https://github.com/NVIDIA-NeMo/Run/tree/main/examples/hello-world/hello_scripts.py>
:link-type: url
:link-alt: Hello World tutorial part 3

**Script-Based Execution and Automation**

Build automated workflows:

- Script-based experiment execution
- Batch processing and automation
- Integration with CI/CD pipelines
- Production deployment patterns
:::

::::

## Advanced Tutorials

### Ray Distributed Computing

Learn to leverage Ray for distributed computing across Kubernetes and Slurm environments.

#### Ray Cluster Management

```python
# Example: Deploying a Ray cluster on Kubernetes
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup
from nemo_run.run.ray.cluster import RayCluster

# Configure KubeRay executor
executor = KubeRayExecutor(
    namespace="ml-team",
    worker_groups=[KubeRayWorkerGroup(
        group_name="worker",
        replicas=2,
        gpus_per_worker=8
    )]
)

# Deploy cluster
cluster = RayCluster(name="training-cluster", executor=executor)
cluster.start()
```

#### Ray Job Submission

```python
# Example: Submitting jobs to Ray cluster
from nemo_run.run.ray.job import RayJob

# Submit training job
job = RayJob(name="training-job", executor=executor)
job.start(
    command="python train.py --config config.yaml",
    workdir="/workspace/project/"
)

# Monitor execution
job.logs(follow=True)
```

### Cloud Execution with SkyPilot

Master cloud-native execution with automatic resource provisioning and cost optimization.

#### Multi-Cloud Deployment

```python
# Example: SkyPilot multi-cloud execution
import nemo_run as run
from nemo_run.core.execution.skypilot import SkyPilotExecutor

# Configure SkyPilot executor
executor = SkyPilotExecutor(
    cloud="aws",  # or "gcp", "azure", "lambda"
    instance_type="g4dn.xlarge",
    region="us-west-2"
)

# Execute function on cloud
@run.partial(executor=executor)
def train_model(config):
    # Training logic here
    return model

result = train_model(config)
```

### Experiment Management

Learn advanced experiment tracking and management techniques.

#### Experiment Lifecycle

```python
# Example: Comprehensive experiment management
import nemo_run as run

# Create experiment
experiment = run.Experiment(
    name="hyperparameter-sweep",
    description="Sweep learning rate and batch size"
)

# Define parameter space
configs = [
    {"lr": 0.001, "batch_size": 32},
    {"lr": 0.01, "batch_size": 64},
    {"lr": 0.1, "batch_size": 128}
]

# Run experiments
for config in configs:
    with experiment.run(config) as run_context:
        result = train_model(config)
        run_context.log_metrics({
            "accuracy": result.accuracy,
            "loss": result.loss
        })
```

## Interactive Examples

### Jupyter Notebooks

Explore interactive examples in Jupyter notebooks:

- **Configuration Examples**: Learn configuration patterns and best practices
- **Execution Backends**: Compare different execution environments
- **Experiment Tracking**: Visualize experiment results and metrics
- **Distributed Computing**: Hands-on Ray cluster management

### Code Examples

Run complete code examples to understand NeMo Run workflows:

```python
# Complete example: ML training pipeline
import nemo_run as run
from nemo_run.core.execution.docker import DockerExecutor

# Define training function
@run.partial
def train_model(
    model_name: str = "gpt2",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10
):
    """Train a machine learning model."""
    # Training implementation
    return {"accuracy": 0.95, "loss": 0.05}

# Configure executor
executor = DockerExecutor(
    image="nvidia/pytorch:24.05-py3",
    gpus=1
)

# Execute training
result = train_model.with_executor(executor)(
    model_name="gpt2-large",
    learning_rate=0.0001,
    batch_size=64
)

print(f"Training completed: {result}")
```

## Learning Resources

### Documentation

- **Configuration Guide**: Deep dive into configuration management
- **Execution Guide**: Understanding execution backends and environments
- **CLI Guide**: Command-line interface usage and automation
- **Ray Guide**: Distributed computing with Ray clusters

### Community Resources

- **GitHub Repository**: Source code and issue tracking
- **Discussions**: Community forums and Q&A
- **Examples**: Additional code examples and use cases
- **Contributing**: Guidelines for contributing to NeMo Run

### Best Practices

#### Configuration Management

```python
# Best practice: Structured configuration
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    hidden_size: int = 512
    num_layers: int = 12
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"

# Use structured configs
model_config = ModelConfig(name="gpt2", hidden_size=768)
training_config = TrainingConfig(learning_rate=0.0001, batch_size=64)
```

#### Error Handling

```python
# Best practice: Robust error handling
import nemo_run as run
from nemo_run.core.execution.docker import DockerExecutor

@run.partial
def robust_training(config):
    try:
        # Training logic
        result = train_model(config)
        return result
    except Exception as e:
        # Log error and return fallback
        run.logger.error(f"Training failed: {e}")
        return {"error": str(e), "status": "failed"}

# Execute with error handling
executor = DockerExecutor(image="nvidia/pytorch:24.05-py3")
result = robust_training.with_executor(executor)(config)
```

#### Resource Management

```python
# Best practice: Resource-aware execution
from nemo_run.core.execution.kuberay import KubeRayExecutor

# Configure resource limits
executor = KubeRayExecutor(
    worker_groups=[KubeRayWorkerGroup(
        group_name="worker",
        replicas=2,
        gpus_per_worker=4,
        cpu_per_worker="8",
        memory_per_worker="32Gi"
    )],
    # Set resource limits
    resource_limits={
        "cpu": "16",
        "memory": "64Gi",
        "nvidia.com/gpu": "8"
    }
)
```

## Getting Help

### Troubleshooting

Common issues and solutions:

1. **Installation Problems**: Check prerequisites and dependencies
2. **Configuration Errors**: Validate configuration syntax and structure
3. **Execution Failures**: Review logs and resource requirements
4. **Performance Issues**: Optimize resource allocation and execution settings

### Support Channels

- **Documentation**: Comprehensive guides and API reference
- **GitHub Issues**: Bug reports and feature requests
- **Community Forums**: Discussion and Q&A
- **Examples Repository**: Working code examples

## Next Steps

After completing the tutorials:

1. **Explore Advanced Features**: Dive into distributed computing and cloud execution
2. **Build Your Workflows**: Create custom experiment pipelines
3. **Optimize Performance**: Learn best practices for production deployment
4. **Contribute**: Share your experiences and contribute to the community

For additional learning resources and community support, visit the [NeMo Run GitHub repository](https://github.com/NVIDIA-NeMo/Run) and [documentation](https://docs.nemo.run).

---

:::{note}
**Note**: The tutorial files referenced in this guide are available in the [NeMo Run examples repository](https://github.com/NVIDIA-NeMo/Run/tree/main/examples). Clone the repository to access the complete tutorial notebooks and scripts.
:::
