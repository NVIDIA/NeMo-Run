---
description: "Explore NeMo Run's technical capabilities and implementation features for ML experiment management, including configuration systems, execution backends, and experiment tracking."
tags: ["features", "capabilities", "technical", "implementation", "ml", "experiment-management"]
categories: ["about"]
---

(about-key-features)=
# Technical Capabilities

NeMo Run provides a comprehensive set of technical capabilities designed for modern ML experiment management. This document outlines the specific features and implementation details that power NeMo Run's functionality.

## Configuration System

### Core Configuration Classes

**`run.Config`** - Direct configuration objects that build to instances
- Type-safe configuration with automatic validation
- Nested configuration support with dot notation access
- Integration with Python dataclasses and type hints
- Configuration broadcasting and transformation capabilities

**`run.Partial`** - Partial configurations that build to callable objects
- Lazy evaluation and configuration
- CLI parameter exposure with automatic argument parsing
- Factory function support for complex object creation
- Configuration composition and inheritance

**`run.Script`** - Script-based execution configurations
- External script execution with parameter passing
- Environment variable management
- Working directory and path configuration
- Script validation and preprocessing

### Configuration Features

- **Type Validation**: Runtime type checking using Python's type system
- **Configuration Walking**: Transform configurations with custom functions
- **Configuration Diffing**: Compare and visualize configuration changes
- **Configuration Export**: Export to YAML, TOML, JSON, or Python code
- **Configuration Broadcasting**: Apply changes across nested structures

::::{dropdown} Configuration System Example
:icon: code-square

```python
from dataclasses import dataclass
import nemo_run as run

@dataclass
class ModelConfig:
    hidden_size: int = 512
    num_layers: int = 12
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100

# Direct configuration
config = run.Config(ModelConfig, hidden_size=1024, num_layers=24)
model = config.build()  # Returns ModelConfig instance

# Partial configuration
train_fn = run.Partial(train_model, model=ModelConfig(), training=TrainingConfig())

# CLI usage: python train.py model.hidden_size=1024 training.learning_rate=2e-4
```
::::

## Execution Backends

### Local Execution

**`run.LocalExecutor`** - Local process execution
- Process isolation and resource management
- Environment variable configuration
- Working directory and path management
- Log capture and redirection

### Container Execution

**`run.DockerExecutor`** - Docker container execution
- Custom container images with GPU support
- Volume mounting and file sharing
- Network configuration and port forwarding
- Resource limits and constraints

### Cluster Execution

**`run.SlurmExecutor`** - HPC cluster execution
- Slurm job submission and management
- Multi-node and multi-GPU support
- Resource allocation and scheduling
- SSH tunnel support for remote access

**`run.SkypilotExecutor`** - Cloud platform execution
- Multi-cloud support (AWS, GCP, Azure)
- Automatic resource provisioning
- Cost optimization and spot instances
- Cloud-specific optimizations

### Cloud Execution

**`run.DGXCloudExecutor`** - NVIDIA DGX Cloud execution
- DGX Cloud cluster management
- Lepton integration and authentication
- GPU resource allocation
- Cloud-native optimizations

**`run.LeptonExecutor`** - Lepton cloud execution
- Lepton cluster deployment
- Automatic scaling and resource management
- Cost tracking and optimization
- Integration with Lepton services

::::{dropdown} Execution Backend Example
:icon: code-square

```python
import nemo_run as run

# Local execution
local_exec = run.LocalExecutor(
    env_vars={"CUDA_VISIBLE_DEVICES": "0,1"},
    working_dir="/path/to/project"
)

# Docker execution
docker_exec = run.DockerExecutor(
    container_image="pytorch/pytorch:2.0.0",
    num_gpus=4,
    volumes=["/data:/data", "/models:/models"],
    ports=[8080:8080]
)

# Slurm execution
slurm_exec = run.SlurmExecutor(
    nodes=2,
    gpus_per_node=8,
    time="04:00:00",
    account="gpu-dept",
    partition="a100"
)

# Use with any configuration
config = run.Partial(train_model, model_size=1024)
run.run(config, executor=local_exec)  # or docker_exec, slurm_exec
```
::::

## Experiment Management System

### Experiment Lifecycle

**`run.Experiment`** - Main experiment management class
- Experiment creation and initialization
- Task addition and dependency management
- Execution orchestration and monitoring
- Metadata capture and storage

### Task Management

- **Task Addition**: Add individual tasks or task groups
- **Dependency Management**: Define complex task dependencies
- **Parallel Execution**: Concurrent execution of independent tasks
- **Status Tracking**: Real-time task status monitoring

### Metadata Management

- **Configuration Capture**: Automatic configuration serialization
- **Log Management**: Centralized log collection and storage
- **Artifact Tracking**: Automatic artifact collection and synchronization
- **Experiment Reconstruction**: Full experiment reproduction from metadata

::::{dropdown} Experiment Management Example
:icon: code-square

```python
import nemo_run as run

# Create experiment
with run.Experiment("hyperparameter-sweep") as exp:
    # Add tasks with different configurations
    task1 = exp.add(
        train_model,
        model_size=512,
        executor=slurm_exec,
        name="small-model"
    )

    task2 = exp.add(
        train_model,
        model_size=1024,
        executor=slurm_exec,
        name="large-model"
    )

    # Add dependent task
    exp.add(
        evaluate_models,
        dependencies=[task1, task2],
        name="evaluation"
    )

    # Launch experiment
    exp.run(tail_logs=True, sequential=False)

# Later reconstruction
exp = run.Experiment.from_id("hyperparameter-sweep_20241201_123456")
exp.status()
exp.logs("large-model")
```
::::

## CLI System

### Command-Line Interface

**`run.cli.entrypoint`** - CLI entry point decorator
- Automatic parameter exposure to CLI
- Type-safe argument parsing and validation
- Nested configuration overrides with dot notation
- Error correction and intelligent suggestions

### CLI Features

- **Factory Functions**: Create complex objects via CLI
- **Configuration Files**: Load configurations with `@` syntax
- **Dry Run Mode**: Preview execution without running
- **Configuration Export**: Export to various formats
- **Rich Output**: Formatted tables and syntax highlighting

::::{dropdown} CLI System Example
:icon: code-square

```bash
# Basic parameter overrides
python train.py model.hidden_size=1024 training.learning_rate=2e-4

# Configuration file loading
python train.py --factory @configs/base.yaml model.layers=24

# Nested configuration with operations
python train.py model.size*=2 training.batch_size+=16

# Factory function usage
python train.py --factory executor=@executors/slurm.yaml

# Dry run to preview
python train.py --dryrun model.size=512

# Export configuration
python train.py --to-yaml config.yaml model.size=512
```
::::

## Packaging System

### Packaging Strategies

**`run.GitArchivePackager`** - Git-based packaging
- Package committed code using git archive
- Version control integration
- Automatic dependency resolution
- Clean, reproducible packages

**`run.PatternPackager`** - Pattern-based packaging
- Selective file inclusion with glob patterns
- Custom inclusion/exclusion rules
- File filtering and transformation
- Flexible packaging strategies

**`run.HybridPackager`** - Combined packaging
- Multiple packaging strategy combination
- Custom packaging logic
- Conditional packaging rules
- Advanced packaging workflows

::::{dropdown} Packaging System Example
:icon: code-square

```python
import nemo_run as run

# Git archive packaging
git_packager = run.GitArchivePackager()

# Pattern-based packaging
pattern_packager = run.PatternPackager(
    include=["*.py", "*.yaml", "*.json"],
    exclude=["__pycache__", "*.pyc", "tests/"]
)

# Hybrid packaging
hybrid_packager = run.HybridPackager([
    git_packager,
    pattern_packager
])

# Use with executor
executor = run.SlurmExecutor(packager=hybrid_packager)
```
::::

## Ray Integration

### Ray Cluster Management

**`run.ray.cluster.RayCluster`** - Ray cluster lifecycle management
- Cluster creation and initialization
- Resource allocation and configuration
- Port forwarding and dashboard access
- Cluster cleanup and resource management

### Ray Job Management

**`run.ray.job.RayJob`** - Ray job submission and monitoring
- Job submission to Ray clusters
- Runtime environment configuration
- Log streaming and monitoring
- Job status tracking and management

### Backend Support

- **KubeRay**: Kubernetes-based Ray cluster management
- **Slurm Ray**: Ray clusters on HPC systems
- **Local Ray**: Local Ray cluster for development

::::{dropdown} Ray Integration Example
:icon: code-square

```python
from nemo_run.run.ray import RayCluster, RayJob

# Create Ray cluster
cluster = RayCluster(
    name="ml-cluster",
    executor=kuberay_executor
)
cluster.start(timeout=900)
cluster.port_forward(port=8265)  # Ray dashboard

# Submit job to cluster
job = RayJob(name="training-job", executor=kuberay_executor)
job.start(
    command="python train.py --config config.yaml",
    workdir="/path/to/project",
    runtime_env_yaml="/path/to/runtime_env.yaml"
)
job.logs(follow=True)

# Clean up
cluster.stop()
```
::::

## Plugin System

### Plugin Architecture

**`run.Plugin`** - Base plugin class for extensibility
- Task and executor modification
- Configuration injection and transformation
- Environment setup and cleanup
- Custom functionality integration

### Plugin Features

- **Setup Hooks**: Modify tasks and executors before execution
- **Configuration Injection**: Add configuration parameters
- **Environment Management**: Set up execution environments
- **Custom Logic**: Implement custom experiment logic

::::{dropdown} Plugin System Example
:icon: code-square

```python
import nemo_run as run

class LoggingPlugin(run.Plugin):
    def setup(self, task, executor):
        # Add logging configuration
        if hasattr(executor, 'env_vars'):
            executor.env_vars['LOG_LEVEL'] = 'DEBUG'

        # Modify task configuration
        if hasattr(task, 'config'):
            task.config.logging = True

# Use plugin
plugin = LoggingPlugin()
run.run(config, executor=executor, plugins=[plugin])
```
::::

## Tunneling System

### SSH Tunneling

**`run.SSHTunnel`** - SSH tunnel management
- Secure remote access to clusters
- Port forwarding and connection management
- Authentication and key management
- Connection monitoring and health checks

### Local Tunneling

**`run.LocalTunnel`** - Local tunnel management
- Local port forwarding
- Service discovery and connection
- Network configuration
- Tunnel lifecycle management

::::{dropdown} Tunneling System Example
:icon: code-square

```python
import nemo_run as run

# SSH tunnel to remote cluster
ssh_tunnel = run.SSHTunnel(
    host="login.cluster.com",
    user="username",
    identity="~/.ssh/id_rsa",
    job_dir="/scratch/username/runs"
)

# Use with executor
executor = run.SlurmExecutor(tunnel=ssh_tunnel)
run.run(config, executor=executor)
```
::::

These technical capabilities provide the foundation for NeMo Run's comprehensive ML experiment management system, enabling users to build sophisticated, scalable, and reproducible ML workflows across diverse computing environments.
