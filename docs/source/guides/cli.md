# NeMo Run CLI Guide

The NeMo Run Command Line Interface (CLI) is a powerful, Pythonic tool designed to streamline the configuration, execution, and management of machine learning experiments. This guide provides a comprehensive overview of the NeMo Run CLI, from installation to advanced features, with a special focus on an improved "Configuration Files" section as requested. Whether you're running simple tasks locally or managing distributed experiments, this guide will equip you with the knowledge to use NeMo Run effectively.

## Table of Contents
- [NeMo Run CLI Guide](#nemo-run-cli-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Key Features](#key-features)
  - [Installation](#installation)
  - [Core Concepts](#core-concepts)
    - [Entrypoints](#entrypoints)
    - [Factories](#factories)
    - [Executors](#executors)
    - [RunContext](#runcontext)
  - [Basic Usage](#basic-usage)
  - [Working with Factories](#working-with-factories)
    - [Overview of Factories](#overview-of-factories)
    - [Basic Factory Registration](#basic-factory-registration)
    - [Advanced Factory Registration with `target` and `target_arg`](#advanced-factory-registration-with-target-and-target_arg)
    - [Setting Default Factories with `is_target_default`](#setting-default-factories-with-is_target_default)
    - [Customizing with `name` and `namespace`](#customizing-with-name-and-namespace)
    - [Using Factories in Entrypoints](#using-factories-in-entrypoints)
    - [Advanced Factory Usage](#advanced-factory-usage)
  - [Configuration Files](#configuration-files)
    - [Supported Formats](#supported-formats)
    - [Factory Configuration File (`--factory @file.yaml`)](#factory-configuration-file---factory-fileyaml)
    - [Argument-Specific Configuration File (`argument=@file.yaml`)](#argument-specific-configuration-file-argumentfileyaml)
    - [Selecting a Section from a Config File (`argument=@file:section`)](#selecting-a-section-from-a-config-file-argumentfilesection)
  - [Advanced Features](#advanced-features)
    - [Partial Configurations](#partial-configurations)
    - [Nested Configurations](#nested-configurations)
    - [Experiment Management](#experiment-management)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)
  - [Conclusion](#conclusion)

## Introduction

The NeMo Run CLI is designed to simplify machine learning workflows by integrating Python's intuitive syntax with a flexible command-line interface. It emphasizes type-driven configuration, portability, and reproducibility, making it suitable for both small experiments and large-scale deployments.

### Key Features
- **Pythonic Design**: Execute Python code via an intuitive CLI.
- **Type-Driven**: Use Python type hints for structured, safe configurations.
- **Portable Execution**: Run tasks locally or on clusters/cloud by switching executors.
- **Rich Configuration**: Support for nested structures and file-based overrides.
- **Reproducibility**: Save and share experiment configurations easily.

This guide covers everything you need to get started and master the CLI.

## Installation

To use the NeMo Run CLI, install the NeMo toolkit:

```bash
pip install nemo-toolkit
```

Ensure your Python version is 3.8 or higher. If NeMo Run is a standalone package, refer to its official documentation for specific instructions.

## Core Concepts

### Entrypoints
An entrypoint is a Python function decorated with `@run.cli.entrypoint`, serving as the main CLI command. It defines the parameters your script accepts.

### Factories
A factory is a function decorated with `@run.cli.factory`, creating preconfigured objects (e.g., models, optimizers) for reuse in entrypoints.

### Executors
Executors control where and how your code runs:
- **LocalExecutor**: Runs locally.
- **SlurmExecutor**: Submits to SLURM clusters.
- **SkypilotExecutor**: Deploys to the cloud.

### RunContext
The RunContext manages execution settings, such as the executor or configuration exports.

## Basic Usage

Here's a simple example:

```python
import nemo_run as run

@run.cli.entrypoint
def train_model(model: str, epochs: int = 10):
    print(f"Training {model} for {epochs} epochs")

if __name__ == "__main__":
    run.cli.main(train_model)
```

Run it:
```bash
python script.py model=resnet50 epochs=20
```

Output: `Training resnet50 for 20 epochs`

This shows how NeMo Run maps CLI arguments to Python function calls using type hints.

## Working with Factories

Factories allow you to define reusable configurations for complex objects. Below, we explore their usage in detail.

### Overview of Factories
Factories are decorated with `@run.cli.factory` and instantiate objects for entrypoints. Parameters include:
- `target`: Type or function to register under.
- `target_arg`: Specific argument to target.
- `is_target_default`: Sets as default for its type.
- `name`: Custom factory name.
- `namespace`: Groups factories.

### Basic Factory Registration

Register a factory for a type:

```python
from dataclasses import dataclass
import nemo_run as run

@dataclass
class Model:
    hidden_size: int
    num_layers: int

@run.cli.factory
def my_model() -> run.Config[Model]:
    return run.Config(Model, hidden_size=256, num_layers=3)
```

Run with an entrypoint:

```python
@run.cli.entrypoint
def train(model: Model):
    print(f"Training with {model.hidden_size} hidden units and {model.num_layers} layers")

if __name__ == "__main__":
    run.cli.main(train)
```

```bash
python script.py model=my_model
```

### Advanced Factory Registration with `target` and `target_arg`

```python
@run.cli.factory
def my_model(
    hidden_size: int = 256,
    num_layers: int = 3,
    activation: str = 'relu'
) -> run.Config[Model]:
    return run.Config(Model, hidden_size=hidden_size, num_layers=num_layers, activation=activation)

@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

Now you can use this factory in the CLI:

```bash
# Use the factory with default settings
python train.py

# Override specific parameters
python train.py model.hidden_size=512 learning_rate=0.01

# Use a different factory altogether
python train.py model=large_model
```

### Setting Default Factories with `is_target_default`

```python
@run.cli.factory
def my_model(
    hidden_size: int = 256,
    num_layers: int = 3,
    activation: str = 'relu'
) -> run.Config[Model]:
    return run.Config(Model, hidden_size=hidden_size, num_layers=num_layers, activation=activation)

@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

Now you can use this factory in the CLI:

```bash
# Use the factory with default settings
python train.py

# Override specific parameters
python train.py model.hidden_size=512 learning_rate=0.01

# Use a different factory altogether
python train.py model=large_model
```

### Customizing with `name` and `namespace`

```python
@run.cli.factory
def my_model(
    hidden_size: int = 256,
    num_layers: int = 3,
    activation: str = 'relu'
) -> run.Config[Model]:
    return run.Config(Model, hidden_size=hidden_size, num_layers=num_layers, activation=activation)

@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

Now you can use this factory in the CLI:

```bash
# Use the factory with default settings
python train.py

# Override specific parameters
python train.py model.hidden_size=512 learning_rate=0.01

# Use a different factory altogether
python train.py model=large_model
```

### Using Factories in Entrypoints

```python
@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

```bash
# Run locally
python train.py

# Run in Docker
python train.py executor=docker_executor

# Run on a Slurm cluster
python train.py executor=slurm_executor executor.nodes=2 executor.gpus_per_node=8
```

### Advanced Factory Usage

```python
@run.cli.factory
def my_model(
    hidden_size: int = 256,
    num_layers: int = 3,
    activation: str = 'relu'
) -> run.Config[Model]:
    return run.Config(Model, hidden_size=hidden_size, num_layers=num_layers, activation=activation)

@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

```bash
# Use the factory with default settings
python train.py

# Override specific parameters
python train.py model.hidden_size=512 learning_rate=0.01

# Use a different factory altogether
python train.py model=large_model
```

## Configuration Files

NeMo Run provides powerful tools for managing your configurations, allowing you to load, modify, and export them with ease.

### Supported Formats

Load configurations from YAML, TOML, or JSON files:

```bash
python train.py --yaml config.yaml
```

Override specific values from the loaded configuration:

```bash
python train.py --yaml config.yaml learning_rate=0.01
```

### Factory Configuration File (`--factory @file.yaml`)

```bash
python train.py --factory @file.yaml
```

### Argument-Specific Configuration File (`argument=@file.yaml`)

```bash
python train.py argument=@file.yaml
```

### Selecting a Section from a Config File (`argument=@file:section`)

```bash
python train.py argument=@file:section
```

## Advanced Features

### Partial Configurations

```python
@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

```bash
# Run locally
python train.py

# Run in Docker
python train.py executor=docker_executor

# Run on a Slurm cluster
python train.py executor=slurm_executor executor.nodes=2 executor.gpus_per_node=8
```

### Nested Configurations

```python
@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

```bash
# Run locally
python train.py

# Run in Docker
python train.py executor=docker_executor

# Run on a Slurm cluster
python train.py executor=slurm_executor executor.nodes=2 executor.gpus_per_node=8
```

### Experiment Management

```python
@run.cli.entrypoint
def train_model(
    model: Model = my_model(),
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Implementation...
```

```bash
# Run locally
python train.py

# Run in Docker
python train.py executor=docker_executor

# Run on a Slurm cluster
python train.py executor=slurm_executor executor.nodes=2 executor.gpus_per_node=8
```

## Best Practices

1. **Use Type-Driven Configuration**: Leverage Python type hints for safe and structured configurations.
2. **Portability**: Run tasks locally or on clusters/cloud by switching executors.
3. **Reproducibility**: Save and share experiment configurations easily.
4. **Configuration Management**: Use configuration files and command-line arguments effectively.

## Troubleshooting

1. **Installation Issues**: Ensure NeMo toolkit is installed correctly.
2. **Configuration Errors**: Check your configuration files and command-line arguments.
3. **Execution Issues**: Verify your environment and executor settings.

## Conclusion

The NeMo Run CLI is a powerful, Pythonic tool designed to streamline the configuration, execution, and management of machine learning experiments. Whether you're running simple tasks locally or managing distributed experiments, this guide has equipped you with the knowledge to use NeMo Run effectively.
