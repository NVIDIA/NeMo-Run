---
description: "Explore NeMo Run's key features and capabilities for ML experiment management, including configuration, execution, and experiment tracking."
tags: ["features", "capabilities", "ml", "experiment-management", "distributed-computing"]
categories: ["about"]
---

(about-key-features)=
# Key Features

NeMo Run provides a comprehensive suite of features designed to streamline ML experiment management across diverse computing environments. Here are the key capabilities that make NeMo Run an essential tool for modern ML workflows.

## 🔧 Configuration Management

### Type-Safe Python Configurations
- **Automatic Type Validation**: Leverage Python's type annotations for runtime validation
- **Nested Configuration Support**: Intuitive dot notation for complex parameter hierarchies
- **Fiddle Integration**: Built on Google's Fiddle framework for robust configuration management
- **YAML Interoperability**: Seamless integration with external YAML, TOML, and JSON files

### Advanced Configuration Features
- **Configuration Broadcasting**: Apply changes across nested configuration trees
- **Configuration Walking**: Transform configurations with custom functions
- **Configuration Visualization**: Built-in graph visualization for complex configurations
- **Configuration Diffing**: Compare and visualize configuration changes

::::{dropdown} Configuration Example
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

@run.cli.entrypoint
def train(model: ModelConfig, training: TrainingConfig):
    # All parameters automatically validated and accessible via CLI
    print(f"Training model with {model.hidden_size} hidden units")
    print(f"Learning rate: {training.learning_rate}")

# CLI usage: python train.py model.hidden_size=1024 training.learning_rate=2e-4
```
::::

## 🚀 Multi-Environment Execution

### Executor Abstraction
- **Local Execution**: Run experiments locally with process isolation
- **Docker Execution**: Containerized execution with GPU support
- **Slurm Integration**: High-performance computing cluster support
- **Kubernetes Support**: Cloud-native execution with KubeRay
- **Cloud Platforms**: Integration with Skypilot, Lepton, and DGX Cloud

### Intelligent Code Packaging
- **Git Archive Packaging**: Package committed code using git archive
- **Pattern-Based Packaging**: Selective file inclusion with glob patterns
- **Hybrid Packaging**: Combine multiple packaging strategies
- **Dependency Management**: Automatic Python path configuration

::::{dropdown} Multi-Environment Example
:icon: code-square

```python
import nemo_run as run

# Same configuration, different environments
config = run.Partial(train_model, model_size=1024, epochs=100)

# Local execution
run.run(config, executor=run.LocalExecutor())

# Docker execution
docker_exec = run.DockerExecutor(
    container_image="pytorch/pytorch:2.0.0",
    num_gpus=4,
    volumes=["/data:/data"]
)
run.run(config, executor=docker_exec)

# Slurm execution
slurm_exec = run.SlurmExecutor(
    nodes=2,
    gpus_per_node=8,
    time="04:00:00"
)
run.run(config, executor=slurm_exec)
```
::::

## 📊 Experiment Management

### Comprehensive Experiment Tracking
- **Metadata Preservation**: Automatic capture of configurations, logs, and artifacts
- **Experiment Reconstruction**: One-command experiment reproduction from metadata
- **Status Monitoring**: Real-time experiment status and log access
- **Dependency Management**: Complex workflow orchestration with task dependencies

### Advanced Management Features
- **Experiment Cataloging**: Browse and search past experiments
- **Log Streaming**: Real-time log access with filtering and search
- **Artifact Management**: Automatic artifact collection and synchronization
- **Experiment Comparison**: Side-by-side comparison of experiment configurations

::::{dropdown} Experiment Management Example
:icon: code-square

```python
import nemo_run as run

# Create and manage experiments
with run.Experiment("hyperparameter-sweep") as exp:
    # Add multiple tasks with different configurations
    exp.add(train_model, model_size=512, executor=slurm_exec, name="small-model")
    exp.add(train_model, model_size=1024, executor=slurm_exec, name="large-model")

    # Run with dependency management
    exp.add(evaluate_model, dependencies=["small-model", "large-model"])

    # Launch and monitor
    exp.run(tail_logs=True)

# Later, reconstruct and inspect
exp = run.Experiment.from_id("hyperparameter-sweep_20241201_123456")
exp.status()
exp.logs("large-model")
```
::::

## 🎯 Rich CLI Interface

### Type-Safe Command Line
- **Automatic Parameter Exposure**: All function parameters automatically available via CLI
- **Nested Configuration Overrides**: Intuitive dot notation for complex configurations
- **Error Correction**: Intelligent suggestions for typos and invalid parameters
- **Configuration Export**: Export configurations to YAML, TOML, or JSON

### Advanced CLI Features
- **Factory Functions**: Create complex objects via CLI with factory functions
- **Configuration Files**: Load configurations from external files with `@` syntax
- **Dry Run Mode**: Preview execution without actually running
- **Rich Output**: Beautiful, formatted output with tables and syntax highlighting

::::{dropdown} CLI Features Example
:icon: code-square

```bash
# Basic parameter overrides
python train.py model.hidden_size=1024 training.learning_rate=2e-4

# Configuration file loading
python train.py --factory @configs/base.yaml model.layers=24

# Nested configuration with operations
python train.py model.size*=2 training.batch_size+=16

# Dry run to preview
python train.py --dryrun model.size=512

# Export configuration
python train.py --to-yaml config.yaml model.size=512
```
::::

## 🔌 Integration Ecosystem

### Ray Integration
- **RayCluster Management**: Long-lived Ray clusters for interactive development
- **RayJob Submission**: Ephemeral Ray jobs for batch processing
- **KubeRay Support**: Kubernetes-based Ray cluster management
- **Slurm Ray Integration**: Ray clusters on HPC systems

### Framework Integrations
- **PyTorch Integration**: Native support for PyTorch Lightning and torchrun
- **NeMo 2.0 Integration**: Seamless integration with NVIDIA NeMo framework
- **Custom Framework Support**: Extensible architecture for any ML framework

::::{dropdown} Ray Integration Example
:icon: code-square

```python
from nemo_run.run.ray import RayCluster, RayJob

# Create a Ray cluster
cluster = RayCluster(name="ml-cluster", executor=kuberay_executor)
cluster.start(timeout=900)
cluster.port_forward(port=8265)  # Ray dashboard

# Submit jobs to the cluster
job = RayJob(name="training-job", executor=kuberay_executor)
job.start(
    command="python train.py --config config.yaml",
    workdir="/path/to/project"
)
job.logs(follow=True)

# Clean up
cluster.stop()
```
::::

## 🛡️ Production Features

### Reliability and Fault Tolerance
- **Fault Tolerant Launchers**: NVIDIA's fault tolerance framework integration
- **Automatic Retries**: Configurable retry policies for failed experiments
- **Resource Management**: Intelligent resource allocation and cleanup
- **Security**: SSH tunnel support for secure remote execution

### Scalability
- **Distributed Execution**: Support for multi-node, multi-GPU experiments
- **Parallel Processing**: Concurrent execution of independent experiments
- **Resource Optimization**: Efficient resource utilization across platforms
- **Load Balancing**: Intelligent workload distribution

## 🎨 Developer Experience

### Rich Development Tools
- **Configuration Visualization**: Interactive configuration graphs
- **Experiment Dashboard**: Web-based experiment monitoring (future)
- **Debugging Support**: Comprehensive logging and error reporting
- **IDE Integration**: Full IDE support with autocomplete and type hints

### Documentation and Learning
- **Comprehensive Documentation**: Detailed guides and API references
- **Example Gallery**: Rich collection of working examples
- **Community Support**: Active community and support channels
- **Best Practices**: Proven patterns and recommendations

These features work together to provide a complete solution for ML experiment management, from initial prototyping to production deployment, across any computing environment.
