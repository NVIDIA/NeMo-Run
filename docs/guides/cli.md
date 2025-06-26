---
description: "Complete guide to NeMo Run's command-line interface, including entrypoint creation, factory functions, and CLI argument parsing."
tags: ["cli", "command-line", "entrypoints", "factories", "arguments"]
categories: ["guides"]
---

(cli)=

# Command-Line Interface

NeMo Run provides a powerful command-line interface that allows you to execute Python functions and experiments directly from the terminal with rich argument parsing and configuration capabilities.

## Overview

The CLI system transforms Python functions into command-line tools with:

- **Rich Argument Parsing**: Support for complex Python types and nested configurations
- **Factory Functions**: Reusable configuration components
- **Executor Integration**: Seamless integration with execution backends
- **Interactive Mode**: REPL-style interaction for exploration
- **Configuration Export**: Export configurations to YAML, TOML, or JSON

## Basic CLI Usage

### Create Entrypoints

Use the `@run.cli.entrypoint` decorator to expose Python functions as CLI commands:

```python
import nemo_run as run

@run.cli.entrypoint
def train_model(
    model_name: str = "gpt2",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10
):
    """Train a machine learning model with specified parameters."""
    print(f"Training {model_name} with lr={learning_rate}, batch_size={batch_size}")
    # Your training logic here
    return {"accuracy": 0.95, "loss": 0.1}

@run.cli.entrypoint
def evaluate_model(
    model_path: str,
    test_data: str,
    metrics: list[str] = ["accuracy", "precision", "recall"]
):
    """Evaluate a trained model on test data."""
    print(f"Evaluating {model_path} on {test_data}")
    print(f"Metrics: {metrics}")
    # Your evaluation logic here
```

### CLI Argument Syntax

NeMo Run supports rich Python-like argument syntax:

```bash
# Basic arguments
python script.py model_name=gpt2 learning_rate=0.001

# Nested attribute setting
python script.py model.hidden_size=512 data.batch_size=64

# List and dictionary arguments
python script.py layers=[128,256,512] config={'dropout': 0.1}

# Operations on arguments
python script.py counter+=1 rate*=2 flags|=0x1

# Type casting
python script.py int_arg=42 float_arg=3.14 bool_arg=true

# None values
python script.py optional_arg=None
```

## Factory Functions

Factory functions allow you to create reusable configuration components:

```python
import nemo_run as run

@run.cli.factory
def create_optimizer(optimizer_type: str = "adam", lr: float = 0.001):
    """Create an optimizer configuration."""
    if optimizer_type == "adam":
        return {"type": "adam", "lr": lr, "beta1": 0.9, "beta2": 0.999}
    elif optimizer_type == "sgd":
        return {"type": "sgd", "lr": lr, "momentum": 0.9}
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

@run.cli.entrypoint
def train_with_optimizer(
    model: str,
    optimizer=create_optimizer(optimizer_type="adam", lr=0.001)
):
    """Train a model with a configurable optimizer."""
    print(f"Training {model} with {optimizer}")
```

### Use Factories in CLI

```bash
# Use default factory
python script.py train_with_optimizer model=resnet50

# Override factory parameters
python script.py train_with_optimizer model=resnet50 optimizer=create_optimizer(optimizer_type=sgd,lr=0.01)

# Nested factory usage
python script.py train_with_optimizer model=resnet50 optimizer.lr=0.005
```

## Executor Integration

### Default Executors

Set default executors for your entrypoints:

```python
import nemo_run as run

@run.cli.entrypoint(
    default_executor=run.DockerExecutor(
        container_image="pytorch/pytorch:latest",
        num_gpus=1
    )
)
def train_model(model: str, epochs: int = 10):
    """Train a model using Docker executor by default."""
    print(f"Training {model} for {epochs} epochs")
```

### CLI Executor Override

```bash
# Use default executor
python script.py train_model model=resnet50

# Override with different executor
python script.py train_model model=resnet50 executor=run.LocalExecutor()

# Configure executor parameters
python script.py train_model model=resnet50 executor=run.SlurmExecutor(partition=gpu,time=02:00:00)
```

## Advanced CLI Features

### Interactive Mode (REPL)

Start an interactive session to explore configurations:

```bash
python script.py train_model --repl
```

This opens an interactive Python shell where you can:

```python
>>> model_config = create_optimizer(optimizer_type="adam", lr=0.001)
>>> print(model_config)
{'type': 'adam', 'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999}

>>> # Modify and test configurations
>>> model_config['lr'] = 0.0001
>>> print(model_config)
{'type': 'adam', 'lr': 0.0001, 'beta1': 0.9, 'beta2': 0.999}
```

### Configuration Export

Export your configurations to various formats:

```bash
# Export to YAML
python script.py train_model model=resnet50 --to-yaml config.yaml

# Export to TOML
python script.py train_model model=resnet50 --to-toml config.toml

# Export to JSON
python script.py train_model model=resnet50 --to-json config.json
```

### Dry Run Mode

Preview what would be executed without actually running:

```bash
python script.py train_model model=resnet50 --dryrun
```

### Detached Execution

Run tasks in the background:

```bash
python script.py train_model model=resnet50 --detach
```

### Tail Logs

Follow logs in real-time:

```bash
python script.py train_model model=resnet50 --tail-logs
```

## CLI Options Reference

### Global Options

| Option | Description | Example |
|--------|-------------|---------|
| `--name, -n` | Name of the run | `--name my_experiment` |
| `--direct` | Execute directly (no executor) | `--direct` |
| `--dryrun` | Preview without execution | `--dryrun` |
| `--factory, -f` | Use predefined factory | `--factory my_factory` |
| `--load, -l` | Load factory from directory | `--load ./configs/` |
| `--yaml, -y` | Load from YAML file | `--yaml config.yaml` |
| `--repl, -r` | Enter interactive mode | `--repl` |
| `--detach` | Run in background | `--detach` |
| `--yes, -y` | Skip confirmation | `--yes` |
| `--tail-logs` | Follow logs | `--tail-logs` |
| `--verbose, -v` | Enable verbose logging | `--verbose` |

### Output Options

| Option | Description | Example |
|--------|-------------|---------|
| `--to-yaml` | Export to YAML | `--to-yaml output.yaml` |
| `--to-toml` | Export to TOML | `--to-toml output.toml` |
| `--to-json` | Export to JSON | `--to-json output.json` |

### Rich Output Options

| Option | Description | Example |
|--------|-------------|---------|
| `--rich-exceptions` | Enable rich exception formatting | `--rich-exceptions` |
| `--rich-traceback-short` | Short traceback format | `--rich-traceback-short` |
| `--rich-traceback-full` | Full traceback format | `--rich-traceback-full` |
| `--rich-show-locals` | Show local variables in exceptions | `--rich-show-locals` |
| `--rich-hide-locals` | Hide local variables in exceptions | `--rich-hide-locals` |
| `--rich-theme` | Color theme (dark/light/monochrome) | `--rich-theme dark` |

## Best Practices

### 1. Use Descriptive Help Text

```python
@run.cli.entrypoint(
    help="Train a machine learning model with configurable hyperparameters"
)
def train_model(model: str, epochs: int = 10):
    """Train a machine learning model."""
    pass
```

### 2. Provide Sensible Defaults

```python
@run.cli.entrypoint
def train_model(
    model: str,
    learning_rate: float = 0.001,  # Sensible default
    batch_size: int = 32,          # Sensible default
    epochs: int = 10               # Sensible default
):
    pass
```

### 3. Use Type Hints

```python
@run.cli.entrypoint
def process_data(
    input_path: str,
    output_path: str,
    batch_size: int = 32,
    num_workers: int = 4
):
    pass
```

### 4. Create Reusable Factories

```python
@run.cli.factory
def create_model_config(
    model_type: str = "transformer",
    hidden_size: int = 512,
    num_layers: int = 6
):
    """Create a reusable model configuration."""
    return {
        "type": model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers
    }
```

### 5. Handle Complex Configurations

```python
@run.cli.entrypoint
def complex_training(
    model_config: dict = create_model_config(),
    optimizer_config: dict = create_optimizer(),
    data_config: dict = create_data_config()
):
    """Handle complex nested configurations."""
    print(f"Model: {model_config}")
    print(f"Optimizer: {optimizer_config}")
    print(f"Data: {data_config}")
```

## Troubleshoot

### Common Issues

1. **Type Conversion Errors**
   ```bash
   # Error: Cannot convert string to int
   python script.py batch_size=32.5  # Should be int

   # Fix: Use explicit type
   python script.py batch_size=32
   ```

2. **Nested Configuration Issues**
   ```bash
   # Error: Cannot set nested attribute
   python script.py model.config.hidden_size=512

   # Fix: Use factory or direct assignment
   python script.py model=create_model(hidden_size=512)
   ```

3. **Factory Resolution Issues**
   ```bash
   # Error: Factory not found
   python script.py optimizer=unknown_factory()

   # Fix: Use registered factory
   python script.py optimizer=create_optimizer()
   ```

### Debug

1. **Use `--verbose` for detailed output**
2. **Use `--dryrun` to preview execution**
3. **Use `--repl` for interactive debugging**
4. **Export configurations to inspect them**

## Examples

### Complete Example: Training Pipeline

```python
import nemo_run as run

@run.cli.factory
def create_model(name: str, hidden_size: int = 512):
    return {"name": name, "hidden_size": hidden_size}

@run.cli.factory
def create_optimizer(optimizer: str = "adam", lr: float = 0.001):
    return {"optimizer": optimizer, "lr": lr}

@run.cli.factory
def create_data(data_path: str, batch_size: int = 32):
    return {"path": data_path, "batch_size": batch_size}

@run.cli.entrypoint(
    help="Complete training pipeline with configurable components",
    default_executor=run.DockerExecutor(container_image="pytorch/pytorch:latest")
)
def train_pipeline(
    model=create_model(name="transformer"),
    optimizer=create_optimizer(optimizer="adam", lr=0.001),
    data=create_data(data_path="./data", batch_size=32),
    epochs: int = 10,
    save_path: str = "./models"
):
    """Complete training pipeline."""
    print(f"Training {model['name']} model")
    print(f"Using {optimizer['optimizer']} optimizer with lr={optimizer['lr']}")
    print(f"Data from {data['path']} with batch_size={data['batch_size']}")
    print(f"Training for {epochs} epochs")
    print(f"Saving to {save_path}")

    # Your training logic here
    return {"status": "completed", "accuracy": 0.95}
```

### CLI Usage

```bash
# Use defaults
python script.py train_pipeline

# Customize components
python script.py train_pipeline \
    model=create_model(name=resnet50,hidden_size=1024) \
    optimizer=create_optimizer(optimizer=sgd,lr=0.01) \
    data=create_data(data_path=/path/to/data,batch_size=64) \
    epochs=20 \
    save_path=/path/to/save

# Export configuration
python script.py train_pipeline --to-yaml config.yaml

# Dry run
python script.py train_pipeline --dryrun
```

This CLI system provides a powerful and flexible way to interact with NeMo Run, making it easy to create command-line tools for your ML workflows while maintaining the full power of Python's type system and configuration capabilities.
