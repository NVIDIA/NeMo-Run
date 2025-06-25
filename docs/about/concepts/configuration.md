---
description: "Learn about NeMo Run's configuration objects including run.Config, run.Partial, and configuration management patterns."
tags: ["configuration", "concepts", "fiddle", "type-safety"]
categories: ["about"]
---

(about-concepts-configuration)=
# Configuration Objects

NeMo Run's configuration system is built on Google's Fiddle framework, providing type-safe, composable configurations for ML experiments.

## Core Configuration Classes

### `run.Config`

`run.Config` is the primary configuration class that represents a direct configuration of an object or function. When built, it creates an instance of the configured object.

```python
import nemo_run as run

# Configure a model
model_config = run.Config(
    MyModel,
    hidden_size=512,
    num_layers=12,
    dropout=0.1
)

# Build to get an instance
model = run.build(model_config)
```

### `run.Partial`

`run.Partial` creates a partially configured function that can be called later with additional arguments. It's useful for creating reusable, configurable functions.

```python
# Create a partially configured training function
train_fn = run.Partial(
    train_model,
    model=model_config,
    optimizer=run.Config(Adam, lr=1e-4),
    batch_size=32
)

# Call with additional arguments
result = train_fn(epochs=100, data_path="/path/to/data")
```

### `run.Script`

`run.Script` configures script-based execution, either from files or inline content.

```python
# File-based script
script = run.Script(path="./scripts/train.sh")

# Inline script
script = run.Script(
    inline="""
    python train.py --config config.yaml
    echo "Training complete"
    """
)
```

## Configuration Patterns

### Nested Configurations

NeMo Run supports deeply nested configurations with intuitive access patterns:

```python
config = run.Config(
    TrainingPipeline,
    model=run.Config(
        TransformerModel,
        encoder=run.Config(
            TransformerEncoder,
            layers=12,
            hidden_size=768
        ),
        decoder=run.Config(
            TransformerDecoder,
            layers=6,
            hidden_size=768
        )
    ),
    data=run.Config(
        DataLoader,
        batch_size=32,
        num_workers=4
    )
)

# Access nested values
config.model.encoder.layers = 24
config.data.batch_size = 64
```

### Configuration Broadcasting

Use `.broadcast()` to apply changes across nested configuration trees:

```python
# Apply the same value to all matching attributes
config.broadcast(hidden_size=1024)
```

### Configuration Walking

Use `.walk()` to transform configurations with custom functions:

```python
# Double all hidden_size values
config.walk(hidden_size=lambda cfg: cfg.hidden_size * 2)
```

## Type Safety

NeMo Run leverages Python's type annotations for runtime validation:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float = 0.1
    activation: Optional[str] = "relu"

# Type validation happens automatically
config = run.Config(ModelConfig, hidden_size=512, num_layers=12)
```

## Configuration Visualization

NeMo Run provides built-in visualization for complex configurations:

```python
# Generate a visual representation
config.visualize()

# Compare configurations
config.diff(other_config)

# Save configuration graph
config.save_config_img("config_graph.png")
```

## Integration with External Files

NeMo Run supports loading configurations from external files:

```python
# Load from YAML
config = run.Config.from_yaml("config.yaml")

# Load from JSON
config = run.Config.from_json("config.json")

# Export configurations
config.to_yaml("exported_config.yaml")
```

## Best Practices

### 1. Use Type Annotations
Always use type annotations for better validation and documentation:

```python
@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"
```

### 2. Leverage Composition
Build complex configurations by composing simpler ones:

```python
# Reusable base configurations
base_model = run.Config(TransformerModel, hidden_size=768)
large_model = base_model.clone().set(hidden_size=1024, layers=24)
```

### 3. Use Meaningful Names
Give configurations descriptive names for better debugging:

```python
config = run.Config(
    TrainingPipeline,
    name="transformer_finetuning",
    model=large_model,
    data=data_config
)
```

### 4. Validate Early
Use configuration validation to catch errors early:

```python
# Validate before building
if not config.is_valid():
    raise ValueError("Invalid configuration")

# Build with validation
instance = run.build(config)
```

Configuration objects are the foundation of NeMo Run's type-safe, composable approach to ML experiment management. Understanding these patterns will help you create robust, maintainable experiment configurations.
