---
description: "Complete guide to NeMo Run's command-line interface, including entry point creation, factory functions, and CLI argument parsing."
tags: ["cli", "command-line", "entry points", "factories", "arguments"]
categories: ["guides"]
---

(cli)=

# Command-Line Interface

NeMo Run provides a powerful command-line interface that transforms Python functions into sophisticated CLI tools with rich argument parsing, type safety, and seamless integration with execution backends. This system enables AI researchers to create reproducible, configurable experiments that can be executed across diverse computing environments.

## Overview

The CLI system transforms Python functions into command-line tools with:

- **Rich Argument Parsing**: Support for complex Python types, nested configurations, and operations
- **Factory Functions**: Reusable configuration components for complex objects
- **Executor Integration**: Seamless integration with execution backends (Docker, Slurm, Kubernetes, etc.)
- **Interactive Mode**: REPL-style interaction for configuration exploration
- **Configuration Export**: Export configurations to YAML, TOML, or JSON formats
- **Type Safety**: Full type checking and validation with intelligent error correction
- **Error Correction**: Intelligent suggestions for typos and parameter names

## Core Concepts

### Entry Points

Entry points are Python functions decorated with `@run.cli.entrypoint` that become accessible as CLI commands. They support:

- **Parameter Discovery**: Automatic exposure of function parameters as CLI arguments
- **Type Safety**: Type hints are used for validation and parsing
- **Default Values**: Sensible defaults for rapid prototyping
- **Help Text**: Rich documentation and usage information

### Factory Functions

Factory functions (decorated with `@run.cli.factory`) create reusable configuration components:

- **Object Creation**: Instantiate complex objects from CLI arguments
- **Type Registration**: Register factories for specific types or parameters
- **Default Factories**: Provide sensible defaults for complex configurations
- **Composition**: Chain and nest factories for complex workflows

### Run Context

The `RunContext` manages execution settings and provides:

- **Executor Configuration**: Specify execution environments
- **Plugin Management**: Configure and manage execution plugins
- **Execution Control**: Dry run, detached execution, and log management
- **Configuration Export**: Export configurations in various formats

## Basic CLI Usage

### Create Entry Points

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

# Factory function usage
python script.py model=create_model(hidden_size=256)
```

## Factory Functions

Factory functions allow you to create reusable configuration components:

```python
import nemo_run as run
from dataclasses import dataclass
from typing import List

@dataclass
class OptimizerConfig:
    type: str
    lr: float
    betas: List[float]
    weight_decay: float

@run.cli.factory
def create_optimizer(optimizer_type: str = "adam", lr: float = 0.001) -> OptimizerConfig:
    """Create an optimizer configuration."""
    if optimizer_type == "adam":
        return OptimizerConfig(
            type="adam",
            lr=lr,
            betas=[0.9, 0.999],
            weight_decay=1e-5
        )
    elif optimizer_type == "sgd":
        return OptimizerConfig(
            type="sgd",
            lr=lr,
            betas=[0.0, 0.0],
            weight_decay=1e-4
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

@run.cli.entrypoint
def train_with_optimizer(
    model: str,
    optimizer: OptimizerConfig = create_optimizer(optimizer_type="adam", lr=0.001)
):
    """Train a model with a configurable optimizer."""
    print(f"Training {model} with {optimizer.type} optimizer")
    print(f"Learning rate: {optimizer.lr}")
```

### Factory Registration Patterns

#### Type-Based Registration

Register factories for specific types:

```python
@run.cli.factory
def create_transformer_model() -> run.Config[TransformerModel]:
    """Create a default transformer model configuration."""
    return run.Config(
        TransformerModel,
        hidden_size=512,
        num_layers=6,
        num_attention_heads=8
    )

@run.cli.factory
def create_cnn_model() -> run.Config[CNNModel]:
    """Create a default CNN model configuration."""
    return run.Config(
        CNNModel,
        channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3]
    )
```

#### Parameter-Specific Registration

Register factories for specific parameters:

```python
@run.cli.factory(target=train_model, target_arg="model")
def create_default_model() -> run.Config[BaseModel]:
    """Default model factory for train_model function."""
    return create_transformer_model()

@run.cli.factory(target=train_model, target_arg="optimizer")
def create_default_optimizer() -> OptimizerConfig:
    """Default optimizer factory for train_model function."""
    return create_optimizer(optimizer_type="adam", lr=0.001)
```

### Use Factories in CLI

```bash
# Use default factory
python script.py train_with_optimizer model=resnet50

# Override factory parameters
python script.py train_with_optimizer model=resnet50 optimizer=create_optimizer(optimizer_type=sgd,lr=0.01)

# Nested factory usage
python script.py train_with_optimizer model=resnet50 optimizer.lr=0.005

# Use type-based factories
python script.py train_model model=create_transformer_model optimizer=create_optimizer
```

## Executor Integration

### Default Executors

Set default executors for your entry points:

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

# Override executor settings
python script.py train_model model=resnet50 executor.num_gpus=4 executor.memory=32g
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
OptimizerConfig(type='adam', lr=0.001, betas=[0.9, 0.999], weight_decay=1e-05)

>>> # Modify and test configurations
>>> model_config.lr = 0.0001
>>> print(model_config)
OptimizerConfig(type='adam', lr=0.0001, betas=[0.9, 0.999], weight_decay=1e-05)

>>> # Test complex configurations
>>> training_config = run.Config(
...     TrainingJob,
...     model=create_transformer_model(),
...     optimizer=model_config,
...     epochs=100
... )
>>> print(training_config)
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

# Export specific sections
python script.py train_model model=resnet50 --to-yaml config.yaml --section model
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

## Advanced Patterns

### Complex Configuration Management

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import nemo_run as run

@dataclass
class ModelConfig:
    architecture: str
    hidden_size: int
    num_layers: int
    dropout: float = 0.1

@dataclass
class DataConfig:
    batch_size: int
    num_workers: int
    data_path: str
    augmentation: Dict[str, Any] = None

@dataclass
class TrainingConfig:
    learning_rate: float
    epochs: int
    optimizer: str
    scheduler: Optional[str] = None

@run.cli.factory
def create_model_config(
    architecture: str = "transformer",
    hidden_size: int = 512,
    num_layers: int = 6
) -> ModelConfig:
    """Create a standardized model configuration."""
    return ModelConfig(
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

@run.cli.factory
def create_data_config(
    batch_size: int = 32,
    data_path: str = "./data"
) -> DataConfig:
    """Create a standardized data configuration."""
    return DataConfig(
        batch_size=batch_size,
        num_workers=4,
        data_path=data_path
    )

@run.cli.factory
def create_training_config(
    learning_rate: float = 0.001,
    epochs: int = 100
) -> TrainingConfig:
    """Create a standardized training configuration."""
    return TrainingConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        optimizer="adam"
    )

@run.cli.entrypoint(
    help="Complete training pipeline with comprehensive configuration",
    default_executor=run.DockerExecutor(container_image="pytorch/pytorch:latest")
)
def train_pipeline(
    model: ModelConfig = create_model_config(),
    data: DataConfig = create_data_config(),
    training: TrainingConfig = create_training_config(),
    experiment_name: str = "default_experiment",
    seed: int = 42
):
    """Complete training pipeline with comprehensive configuration."""
    print(f"Training {model.architecture} model")
    print(f"Hidden size: {model.hidden_size}, Layers: {model.num_layers}")
    print(f"Batch size: {data.batch_size}, Data path: {data.data_path}")
    print(f"Learning rate: {training.learning_rate}, Epochs: {training.epochs}")
    print(f"Experiment: {experiment_name}, Seed: {seed}")

    # Your training logic here
    return {"status": "completed", "accuracy": 0.95}
```

### Experiment Entry Points

Create entry points for multi-task experiments:

```python
@run.cli.entrypoint(type="experiment")
def multi_stage_training(
    ctx: run.cli.RunContext,
    pretrain: run.Partial[train_pipeline] = run.Partial(
        train_pipeline,
        model=create_model_config(architecture="transformer", hidden_size=768),
        training=create_training_config(epochs=50)
    ),
    finetune: run.Partial[train_pipeline] = run.Partial(
        train_pipeline,
        model=create_model_config(architecture="transformer", hidden_size=768),
        training=create_training_config(epochs=10, learning_rate=1e-5)
    )
):
    """Multi-stage training experiment."""
    # Pretrain stage
    pretrain_result = ctx.run(pretrain)

    # Finetune stage
    finetune_result = ctx.run(finetune)

    return {
        "pretrain": pretrain_result,
        "finetune": finetune_result
    }
```

## Best Practices

### 1. Use Descriptive Help Text

```python
@run.cli.entrypoint(
    help="Train a machine learning model with configurable hyperparameters and advanced features"
)
def train_model(model: str, epochs: int = 10):
    """Train a machine learning model with comprehensive logging and validation."""
    pass
```

### 2. Provide Sensible Defaults

```python
@run.cli.entrypoint
def train_model(
    model: str,
    learning_rate: float = 0.001,  # Sensible default for most models
    batch_size: int = 32,          # Good balance of memory and speed
    epochs: int = 10,              # Reasonable training duration
    seed: int = 42                 # Reproducible default
):
    pass
```

### 3. Use Type Hints Consistently

```python
from typing import Optional, List, Dict, Any

@run.cli.entrypoint
def process_data(
    input_path: str,
    output_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    config: Optional[Dict[str, Any]] = None
):
    pass
```

### 4. Create Reusable Factories

```python
@run.cli.factory
def create_model_config(
    model_type: str = "transformer",
    hidden_size: int = 512,
    num_layers: int = 6,
    dropout: float = 0.1
) -> ModelConfig:
    """Create a reusable model configuration with validation."""
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must be between 0 and 1")

    return ModelConfig(
        model_type=model_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
```

### 5. Handle Complex Configurations

```python
@run.cli.entrypoint
def complex_training(
    model_config: ModelConfig = create_model_config(),
    optimizer_config: OptimizerConfig = create_optimizer(),
    data_config: DataConfig = create_data_config(),
    training_config: TrainingConfig = create_training_config()
):
    """Handle complex nested configurations with validation."""
    # Validate configuration compatibility
    if data_config.batch_size % model_config.hidden_size != 0:
        raise ValueError("batch_size must be divisible by hidden_size")

    print(f"Model: {model_config}")
    print(f"Optimizer: {optimizer_config}")
    print(f"Data: {data_config}")
    print(f"Training: {training_config}")
```

### 6. Use Configuration Export for Reproducibility

```python
# Export configuration for reproducibility
python script.py train_pipeline --to-yaml experiment_config.yaml

# Load and modify configuration
python script.py train_pipeline --yaml experiment_config.yaml model.hidden_size=1024
```

## Troubleshooting

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

4. **Executor Configuration Issues**

   ```bash
   # Error: Invalid executor parameter
   python script.py executor=run.SlurmExecutor(invalid_param=value)

   # Fix: Check executor documentation for valid parameters
   python script.py executor=run.SlurmExecutor(partition=gpu,time=02:00:00)
   ```

### Debug Strategies

1. **Use `--verbose` for detailed output**

   ```bash
   python script.py train_model --verbose
   ```

2. **Use `--dryrun` to preview execution**

   ```bash
   python script.py train_model --dryrun
   ```

3. **Use `--repl` for interactive debugging**

   ```bash
   python script.py train_model --repl
   ```

4. **Export configurations to inspect them**

   ```bash
   python script.py train_model --to-yaml debug_config.yaml
   ```

5. **Check available factories**

   ```python
   import nemo_run as run
   factories = run.cli.list_factories()
   print(factories)
   ```

## Examples

### Complete Example: Advanced Training Pipeline

```python
import nemo_run as run
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_layers: int
    dropout: float = 0.1

@dataclass
class OptimizerConfig:
    type: str
    lr: float
    weight_decay: float = 1e-5
    betas: List[float] = None

@dataclass
class DataConfig:
    path: str
    batch_size: int
    num_workers: int = 4

@run.cli.factory
def create_model(name: str, hidden_size: int = 512) -> ModelConfig:
    return ModelConfig(name=name, hidden_size=hidden_size, num_layers=6)

@run.cli.factory
def create_optimizer(optimizer: str = "adam", lr: float = 0.001) -> OptimizerConfig:
    betas = [0.9, 0.999] if optimizer == "adam" else [0.0, 0.0]
    return OptimizerConfig(type=optimizer, lr=lr, betas=betas)

@run.cli.factory
def create_data(data_path: str, batch_size: int = 32) -> DataConfig:
    return DataConfig(path=data_path, batch_size=batch_size)

@run.cli.entrypoint(
    help="Advanced training pipeline with comprehensive configuration and validation",
    default_executor=run.DockerExecutor(container_image="pytorch/pytorch:latest")
)
def advanced_training_pipeline(
    model: ModelConfig = create_model(name="transformer"),
    optimizer: OptimizerConfig = create_optimizer(optimizer="adam", lr=0.001),
    data: DataConfig = create_data(data_path="./data", batch_size=32),
    epochs: int = 10,
    save_path: str = "./models",
    experiment_name: str = "default_experiment",
    seed: int = 42,
    debug: bool = False
):
    """Advanced training pipeline with comprehensive configuration."""
    print(f"=== Training Configuration ===")
    print(f"Model: {model.name} (hidden_size={model.hidden_size}, layers={model.num_layers})")
    print(f"Optimizer: {optimizer.type} (lr={optimizer.lr}, weight_decay={optimizer.weight_decay})")
    print(f"Data: {data.path} (batch_size={data.batch_size}, workers={data.num_workers})")
    print(f"Training: {epochs} epochs, save_path={save_path}")
    print(f"Experiment: {experiment_name}, Seed: {seed}, Debug: {debug}")

    # Validation
    if model.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if optimizer.lr <= 0:
        raise ValueError("learning_rate must be positive")
    if data.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Your training logic here
    return {
        "status": "completed",
        "accuracy": 0.95,
        "loss": 0.1,
        "config": {
            "model": model,
            "optimizer": optimizer,
            "data": data,
            "training": {
                "epochs": epochs,
                "save_path": save_path,
                "experiment_name": experiment_name,
                "seed": seed
            }
        }
    }
```

### CLI Usage Examples

```bash
# Use defaults
python script.py advanced_training_pipeline

# Customize components
python script.py advanced_training_pipeline \
    model=create_model(name=resnet50,hidden_size=1024) \
    optimizer=create_optimizer(optimizer=sgd,lr=0.01) \
    data=create_data(data_path=/path/to/data,batch_size=64) \
    epochs=20 \
    save_path=/path/to/save \
    experiment_name=resnet_experiment

# Export configuration
python script.py advanced_training_pipeline --to-yaml config.yaml

# Dry run
python script.py advanced_training_pipeline --dryrun

# Interactive mode
python script.py advanced_training_pipeline --repl

# Detached execution
python script.py advanced_training_pipeline --detach

# Follow logs
python script.py advanced_training_pipeline --tail-logs
```

This CLI system provides a powerful and flexible way to interact with NeMo Run, making it easy to create command-line tools for your ML workflows while maintaining the full power of Python's type system and configuration capabilities. The system is designed to be intuitive for AI researchers while providing the robustness and reproducibility needed for serious research workflows.
