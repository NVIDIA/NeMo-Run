---
description: "Learn about NeMo Run's CLI architecture and how type-safe command-line interfaces work."
tags: ["cli", "concepts", "command-line", "type-safety", "interfaces"]
categories: ["about"]
---

(about-concepts-cli)=
# CLI Architecture

NeMo Run's CLI system provides type-safe, intuitive command-line interfaces that automatically convert string arguments to the appropriate Python types. The CLI architecture leverages NeMo Run's configuration system to create powerful, user-friendly interfaces.

## Core Concepts

### CLI Abstraction

NeMo Run's CLI system provides:
- **Type Safety**: Automatic conversion from strings to Python types
- **Configuration Integration**: Seamless integration with `run.Config` objects
- **Help Generation**: Automatic help text and documentation
- **Argument Validation**: Runtime validation of input parameters
- **Subcommand Support**: Hierarchical command structure

```python
import nemo_run as run

@run.script
def train_model(model, optimizer, batch_size=32, epochs=100):
    """Train a model with the given configuration."""
    # Arguments are automatically converted to appropriate types
    print(f"Training {model} with {optimizer}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")

# CLI usage:
# nemo-run train-model --model="TransformerModel(hidden_size=768)" --optimizer="Adam(learning_rate=1e-4)"
```

## Basic CLI Patterns

### Simple Functions

Convert any Python function to a CLI command:

```python
@run.script
def hello_world(name: str, count: int = 1, verbose: bool = False):
    """Say hello to the world."""
    for i in range(count):
        message = f"Hello, {name}!"
        if verbose:
            message += f" (iteration {i+1})"
        print(message)

# CLI usage:
# nemo-run hello-world --name="Alice" --count=3 --verbose
```

### Configuration Objects

Use `run.Config` objects as CLI arguments:

```python
@run.script
def train_transformer(
    model: run.Config,
    data: run.Config,
    training: run.Config,
    output_dir: str = "./output"
):
    """Train a transformer model."""
    # model, data, and training are automatically converted to run.Config objects
    model_instance = run.build(model)
    data_instance = run.build(data)
    training_instance = run.build(training)

    print(f"Training {model_instance} with {data_instance}")
    print(f"Output directory: {output_dir}")

# CLI usage:
# nemo-run train-transformer \
#   --model="TransformerModel(hidden_size=768,num_layers=12)" \
#   --data="DataLoader(batch_size=32,num_workers=4)" \
#   --training="TrainingConfig(epochs=100,learning_rate=1e-4)" \
#   --output-dir="./models"
```

### Nested Configurations

Handle complex nested configurations:

```python
@run.script
def complex_training(
    model_config: run.Config,
    optimizer_config: run.Config,
    scheduler_config: run.Config = None
):
    """Train with complex configuration."""
    # All arguments are automatically converted to run.Config objects
    model = run.build(model_config)
    optimizer = run.build(optimizer_config)

    if scheduler_config:
        scheduler = run.build(scheduler_config)
        print(f"Using scheduler: {scheduler}")

    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")

# CLI usage:
# nemo-run complex-training \
#   --model-config="TransformerModel(encoder=TransformerEncoder(layers=12),decoder=TransformerDecoder(layers=6))" \
#   --optimizer-config="Adam(learning_rate=1e-4,weight_decay=1e-5)" \
#   --scheduler-config="CosineAnnealingLR(T_max=100)"
```

## Advanced CLI Features

### Type Annotations

Leverage Python type annotations for better CLI behavior:

```python
from typing import Optional, List, Dict, Union
from pathlib import Path

@run.script
def advanced_function(
    text: str,
    numbers: List[int],
    config: Dict[str, Union[str, int, float]],
    file_path: Optional[Path] = None,
    verbose: bool = False
):
    """Advanced function with complex type annotations."""
    print(f"Text: {text}")
    print(f"Numbers: {numbers}")
    print(f"Config: {config}")

    if file_path:
        print(f"File path: {file_path}")

    if verbose:
        print("Verbose mode enabled")

# CLI usage:
# nemo-run advanced-function \
#   --text="Hello World" \
#   --numbers="[1,2,3,4,5]" \
#   --config='{"key1":"value1","key2":42,"key3":3.14}' \
#   --file-path="./data.txt" \
#   --verbose
```

### Custom Type Converters

Create custom type converters for complex objects:

```python
import json
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_layers: int

def parse_model_config(config_str: str) -> ModelConfig:
    """Parse model configuration from string."""
    data = json.loads(config_str)
    return ModelConfig(**data)

@run.script
def custom_training(
    model: ModelConfig = run.arg(converter=parse_model_config),
    batch_size: int = 32
):
    """Train with custom model configuration."""
    print(f"Training {model.name} with hidden_size={model.hidden_size}")
    print(f"Batch size: {batch_size}")

# CLI usage:
# nemo-run custom-training \
#   --model='{"name":"transformer","hidden_size":768,"num_layers":12}' \
#   --batch-size=64
```

### Subcommands

Create hierarchical command structures:

```python
@run.script
def main():
    """Main CLI application."""
    pass

@main.command
def train(
    model: str,
    data: str,
    epochs: int = 100
):
    """Train a model."""
    print(f"Training {model} on {data} for {epochs} epochs")

@main.command
def evaluate(
    model: str,
    test_data: str
):
    """Evaluate a model."""
    print(f"Evaluating {model} on {test_data}")

@main.command
def serve(
    model: str,
    port: int = 8000
):
    """Serve a model."""
    print(f"Serving {model} on port {port}")

# CLI usage:
# nemo-run main train --model="transformer" --data="dataset" --epochs=200
# nemo-run main evaluate --model="transformer" --test-data="test_dataset"
# nemo-run main serve --model="transformer" --port=9000
```

## CLI Configuration

### Argument Validation

Add validation rules to CLI arguments:

```python
@run.script
def validated_training(
    learning_rate: float = run.arg(gt=0.0, le=1.0),
    batch_size: int = run.arg(gt=0, le=1024),
    epochs: int = run.arg(gt=0, le=10000)
):
    """Train with validated parameters."""
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

# CLI usage with validation:
# nemo-run validated-training --learning-rate=0.001 --batch-size=64 --epochs=100
# Error if values are outside valid ranges
```

### Default Values

Set intelligent default values:

```python
@run.script
def smart_defaults(
    model: str = "transformer",
    optimizer: str = "adam",
    batch_size: int = 32,
    epochs: int = 100,
    verbose: bool = False
):
    """Function with smart defaults."""
    print(f"Model: {model}")
    print(f"Optimizer: {optimizer}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Verbose: {verbose}")

# CLI usage:
# nemo-run smart-defaults  # Uses all defaults
# nemo-run smart-defaults --model="bert" --epochs=200  # Override specific defaults
```

### Help and Documentation

Automatic help generation:

```python
@run.script
def documented_function(
    model: str = run.arg(help="Model architecture to use"),
    data_path: str = run.arg(help="Path to training data"),
    epochs: int = run.arg(default=100, help="Number of training epochs"),
    learning_rate: float = run.arg(default=1e-4, help="Learning rate for training")
):
    """Train a machine learning model.

    This function trains a model using the specified configuration.
    It supports various model architectures and training parameters.
    """
    print(f"Training {model} for {epochs} epochs")
    print(f"Data path: {data_path}")
    print(f"Learning rate: {learning_rate}")

# CLI usage:
# nemo-run documented-function --help
# Shows detailed help with argument descriptions
```

## Integration Patterns

### Experiment Integration

Integrate CLI with NeMo Run experiments:

```python
@run.script
def run_experiment(
    task: run.Config,
    executor: run.Config,
    name: str,
    description: str = ""
):
    """Run a NeMo Run experiment from CLI."""
    experiment = run.Experiment(
        task=task,
        executor=executor,
        name=name,
        description=description
    )

    result = experiment.run()
    print(f"Experiment {name} completed with result: {result}")

# CLI usage:
# nemo-run run-experiment \
#   --task="TrainingFunction(model=TransformerModel(hidden_size=768))" \
#   --executor="SlurmExecutor(nodes=4,gpus_per_node=8)" \
#   --name="transformer_training" \
#   --description="Training large transformer model"
```

### Configuration Management

Manage configurations through CLI:

```python
@run.script
def config_manager(
    action: str = run.arg(choices=["create", "load", "save", "validate"]),
    config: run.Config = None,
    file_path: str = None
):
    """Manage NeMo Run configurations."""
    if action == "create":
        print(f"Created configuration: {config}")
    elif action == "load":
        loaded_config = run.Config.from_yaml(file_path)
        print(f"Loaded configuration: {loaded_config}")
    elif action == "save":
        config.to_yaml(file_path)
        print(f"Saved configuration to {file_path}")
    elif action == "validate":
        if config.is_valid():
            print("Configuration is valid")
        else:
            print("Configuration is invalid")

# CLI usage:
# nemo-run config-manager create --config="TransformerModel(hidden_size=768)"
# nemo-run config-manager load --file-path="./config.yaml"
# nemo-run config-manager save --config="ModelConfig()" --file-path="./output.yaml"
# nemo-run config-manager validate --config="ModelConfig()"
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good: Clear and descriptive
@run.script
def train_transformer_model(
    model_config: run.Config,
    training_config: run.Config
):
    pass

# Avoid: Unclear names
@run.script
def train(model, config):
    pass
```

### 2. Provide Good Documentation

```python
@run.script
def well_documented_function(
    model: str = run.arg(help="Model architecture (transformer, cnn, rnn)"),
    epochs: int = run.arg(default=100, help="Number of training epochs")
):
    """Train a machine learning model.

    This function provides a simple interface for training various
    machine learning models with configurable parameters.

    Examples:
        nemo-run well-documented-function --model="transformer" --epochs=200
        nemo-run well-documented-function --model="cnn" --epochs=50
    """
    pass
```

### 3. Use Type Annotations

```python
# Good: Clear type annotations
@run.script
def typed_function(
    model: run.Config,
    data_path: str,
    batch_size: int = 32,
    verbose: bool = False
):
    pass

# Avoid: No type annotations
@run.script
def untyped_function(model, data_path, batch_size=32, verbose=False):
    pass
```

### 4. Validate Inputs

```python
@run.script
def validated_function(
    learning_rate: float = run.arg(gt=0.0, le=1.0),
    batch_size: int = run.arg(gt=0, le=1024),
    epochs: int = run.arg(gt=0, le=10000)
):
    pass
```

### 5. Provide Sensible Defaults

```python
@run.script
def sensible_defaults(
    model: str = "transformer",
    optimizer: str = "adam",
    batch_size: int = 32,
    epochs: int = 100
):
    pass
```

## Error Handling

### Graceful Error Handling

```python
@run.script
def robust_function(
    config: run.Config,
    fallback_config: run.Config = None
):
    """Function with robust error handling."""
    try:
        # Try to use primary config
        instance = run.build(config)
        print(f"Using primary config: {instance}")
    except Exception as e:
        if fallback_config:
            print(f"Primary config failed: {e}")
            print("Using fallback config")
            instance = run.build(fallback_config)
        else:
            raise

# CLI usage:
# nemo-run robust-function \
#   --config="ComplexModel(complex_param=invalid)" \
#   --fallback-config="SimpleModel()"
```

The CLI architecture makes NeMo Run accessible and user-friendly while maintaining the power and flexibility of the underlying configuration system.
