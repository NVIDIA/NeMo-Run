---
description: "Learn about NeMo Run's @run.autoconvert decorator and automatic configuration conversion patterns."
tags: ["autoconvert", "concepts", "decorators", "type-conversion"]
categories: ["about"]
---

(about-concepts-autoconvert)=
# Autoconvert Pattern

The `@run.autoconvert` decorator is a powerful feature in NeMo Run that automatically converts function arguments to configuration objects, enabling seamless integration between regular Python functions and NeMo Run's configuration system.

## Basic Usage

The `@run.autoconvert` decorator automatically converts function arguments to `run.Config` objects when they're not already configuration objects:

```python
import nemo_run as run

@run.autoconvert
def train_model(model, optimizer, batch_size=32):
    """Train a model with the given configuration."""
    # model and optimizer are automatically converted to run.Config objects
    model_instance = run.build(model)
    optimizer_instance = run.build(optimizer)

    print(f"Training with batch size: {batch_size}")
    return model_instance, optimizer_instance

# Call with regular Python objects
result = train_model(
    model=MyModel(hidden_size=512),
    optimizer=Adam(learning_rate=1e-4),
    batch_size=64
)
```

## How Autoconvert Works

### Automatic Conversion

When you call a function decorated with `@run.autoconvert`:

1. **Regular objects** are automatically wrapped in `run.Config`
2. **Already configured objects** (like `run.Config`, `run.Partial`) are left unchanged
3. **Primitive values** (strings, numbers, booleans) are passed through unchanged
4. **Lists and dictionaries** are recursively converted

```python
@run.autoconvert
def process_data(data_loader, preprocessor, augmentations=None):
    # data_loader becomes run.Config(DataLoader, ...)
    # preprocessor becomes run.Config(Preprocessor, ...)
    # augmentations remains as a list of run.Config objects
    pass

# These are equivalent:
process_data(DataLoader(batch_size=32), Preprocessor())
process_data(run.Config(DataLoader, batch_size=32), run.Config(Preprocessor))
```

### Type Preservation

Autoconvert preserves the original types and structure:

```python
@run.autoconvert
def create_pipeline(model, data_config, training_config):
    # All arguments are converted to run.Config while preserving their structure
    model_config = model  # run.Config(MyModel, ...)
    data_config = data_config  # run.Config(DataConfig, ...)
    training_config = training_config  # run.Config(TrainingConfig, ...)

    return run.Config(Pipeline, model=model_config, data=data_config, training=training_config)
```

## Advanced Patterns

### Nested Object Conversion

Autoconvert handles nested objects and complex structures:

```python
@run.autoconvert
def setup_experiment(model_config, data_config, optimizer_config):
    # All nested objects are converted to run.Config
    return run.Config(
        Experiment,
        model=model_config,  # run.Config(TransformerModel, ...)
        data=data_config,    # run.Config(DataLoader, ...)
        optimizer=optimizer_config  # run.Config(Adam, ...)
    )

# Complex nested configuration
experiment = setup_experiment(
    model_config={
        "encoder": TransformerEncoder(layers=12),
        "decoder": TransformerDecoder(layers=6)
    },
    data_config=DataLoader(batch_size=32),
    optimizer_config=Adam(learning_rate=1e-4)
)
```

### Partial Configuration

Autoconvert works seamlessly with `run.Partial`:

```python
@run.autoconvert
def create_training_fn(model, optimizer, loss_fn):
    return run.Partial(
        train_model,
        model=model,  # Automatically converted to run.Config
        optimizer=optimizer,  # Automatically converted to run.Config
        loss_fn=loss_fn  # Automatically converted to run.Config
    )

# Create a partially configured training function
train_fn = create_training_fn(
    model=MyModel(hidden_size=768),
    optimizer=Adam(learning_rate=1e-4),
    loss_fn=CrossEntropyLoss()
)

# Call with additional arguments
result = train_fn(epochs=100, data_path="/path/to/data")
```

### Conditional Conversion

You can control which arguments are converted:

```python
@run.autoconvert(convert=["model", "optimizer"])  # Only convert specific arguments
def train_with_config(model, optimizer, epochs, data_path):
    # model and optimizer are converted to run.Config
    # epochs and data_path remain as primitive values
    pass

@run.autoconvert(skip=["data_path"])  # Skip conversion for specific arguments
def process_experiment(model, data_loader, data_path):
    # model and data_loader are converted to run.Config
    # data_path remains as a string
    pass
```

## Integration with CLI

Autoconvert is particularly powerful when combined with NeMo Run's CLI system:

```python
@run.autoconvert
@run.script
def main(model, optimizer, training_config):
    """Main training script with automatic configuration conversion."""
    # All arguments are automatically converted from CLI strings to run.Config objects
    model_instance = run.build(model)
    optimizer_instance = run.build(optimizer)
    training_instance = run.build(training_config)

    # Execute training
    train(model_instance, optimizer_instance, training_instance)

# CLI usage:
# nemo-run main --model="MyModel(hidden_size=512)" --optimizer="Adam(learning_rate=1e-4)"
```

## Best Practices

### 1. Use Type Annotations

Always use type annotations for better autoconvert behavior:

```python
from typing import Optional, List

@run.autoconvert
def create_model(
    encoder: TransformerEncoder,
    decoder: Optional[TransformerDecoder] = None,
    layers: List[int] = [12, 12, 6]
):
    # Type annotations help autoconvert understand the expected types
    pass
```

### 2. Document Expected Types

Clearly document what types of objects your function expects:

```python
@run.autoconvert
def train_pipeline(
    model,  # Should be a model class or instance
    data_loader,  # Should be a DataLoader class or instance
    optimizer,  # Should be an optimizer class or instance
    **kwargs
):
    """Train a model pipeline with automatic configuration conversion.

    Args:
        model: Model class or instance (will be converted to run.Config)
        data_loader: DataLoader class or instance (will be converted to run.Config)
        optimizer: Optimizer class or instance (will be converted to run.Config)
        **kwargs: Additional arguments passed through unchanged
    """
    pass
```

### 3. Handle Edge Cases

Consider how autoconvert affects your function's behavior:

```python
@run.autoconvert
def flexible_function(obj, use_config=True):
    if use_config:
        # obj is a run.Config object
        instance = run.build(obj)
    else:
        # obj might be a regular object or run.Config
        instance = obj if not isinstance(obj, run.Config) else run.build(obj)

    return instance
```

### 4. Test Both Patterns

Test your functions with both regular objects and configuration objects:

```python
# Test with regular objects
result1 = train_model(MyModel(), Adam())

# Test with configuration objects
result2 = train_model(
    run.Config(MyModel, hidden_size=512),
    run.Config(Adam, learning_rate=1e-4)
)

# Both should work identically
assert result1 == result2
```

## Common Use Cases

### 1. Configuration Factories

Create functions that generate configurations:

```python
@run.autoconvert
def create_model_config(model_class, hidden_size, num_layers):
    return run.Config(
        model_class,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

# Usage
config = create_model_config(TransformerModel, hidden_size=768, num_layers=12)
```

### 2. Pipeline Assembly

Assemble complex pipelines from components:

```python
@run.autoconvert
def create_training_pipeline(model, data, optimizer, scheduler=None):
    pipeline_config = run.Config(
        TrainingPipeline,
        model=model,
        data=data,
        optimizer=optimizer
    )

    if scheduler:
        pipeline_config.scheduler = scheduler

    return pipeline_config
```

### 3. Experiment Templates

Create reusable experiment templates:

```python
@run.autoconvert
def create_experiment_template(model, dataset, training_config):
    return run.Config(
        Experiment,
        name=f"{model.__name__}_{dataset.__name__}",
        model=model,
        dataset=dataset,
        training=training_config
    )
```

The `@run.autoconvert` decorator makes NeMo Run's configuration system feel natural and intuitive, allowing you to work with regular Python objects while still getting all the benefits of type-safe, composable configurations.
