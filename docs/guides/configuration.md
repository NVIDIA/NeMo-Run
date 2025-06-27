---
description: "Configure NeMo Run experiments using Python-based configuration with Fiddle or raw scripts and commands."
tags: ["configuration", "fiddle", "python", "YAML", "scripts", "machine-learning"]
categories: ["guides"]
---

(configuration)=

# Configure NeMo Run

NeMo Run provides a flexible configuration system that allows you to define machine learning experiments in a type-safe, reproducible manner. This guide covers the two main configuration approaches supported by NeMo Run.

## Configuration Overview

NeMo Run supports two primary configuration systems:

1. **Python-based configuration**: Type-safe, structured configuration using Fiddle
2. **Raw scripts and commands**: Direct script execution for custom workflows

The Python-based system is the recommended approach for most use cases, offering better type safety, validation, and reproducibility. Raw scripts provide flexibility for legacy workflows or custom execution requirements.

## Python-Based Configuration

NeMo Run's Python configuration system is built on top of Fiddle, providing a powerful and flexible way to define experiments. The system uses two main primitives: `run.Config` and `run.Partial`.

### Core Configuration Primitives

#### `run.Config`

`run.Config` creates a complete configuration for a class or function:

```python
import nemo_run as run
from nemo.collections.llm import LlamaModel, Llama3Config8B

# Create a configuration for a model
model_config = run.Config(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384,
        hidden_size=4096,
        num_attention_heads=32
    )
)

# Build the configuration to instantiate the object
model = fdl.build(model_config)
```

#### `run.Partial`

`run.Partial` creates a partially applied function with some arguments fixed:

```python
# Create a partial function for training
train_fn = run.Partial(
    train_model,
    optimizer="adam",
    learning_rate=0.001,
    batch_size=32
)

# Later, you can call it with additional arguments
result = fdl.build(train_fn)(data_path="/path/to/data")
```

### Configuration Patterns

#### Basic Model Configuration

```python
def create_model_config(
    model_size: str = "8b",
    seq_length: int = 16384,
    hidden_size: int = 4096
) -> run.Config:
    """Create a standardized model configuration."""

    if model_size == "8b":
        return run.Config(
            LlamaModel,
            config=run.Config(
                Llama3Config8B,
                seq_length=seq_length,
                hidden_size=hidden_size
            )
        )
    elif model_size == "70b":
        return run.Config(
            LlamaModel,
            config=run.Config(
                Llama3Config70B,
                seq_length=seq_length,
                hidden_size=8192
            )
        )
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

# Usage
model_config = create_model_config(model_size="8b", seq_length=8192)
```

#### Training Configuration

```python
def create_training_config(
    model_config: run.Config,
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    batch_size: int = 512
) -> run.Config:
    """Create a complete training configuration."""

    return run.Config(
        TrainingJob,
        model=model_config,
        trainer=run.Config(
            Trainer,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            precision="bf16-mixed",
            max_epochs=100
        ),
        data=run.Config(
            DataModule,
            batch_size=batch_size,
            num_workers=4
        ),
        optimizer=run.Config(
            AdamW,
            lr=3e-4,
            weight_decay=0.01
        )
    )

# Usage
training_config = create_training_config(
    model_config=model_config,
    num_nodes=4,
    gpus_per_node=8
)
```

### Advanced Configuration Features

#### Configuration Composition

Combine multiple configurations into complex workflows:

```python
# Data preprocessing configuration
preprocess_config = run.Config(
    PreprocessData,
    input_path="/data/raw",
    output_path="/data/processed",
    tokenizer="llama-tokenizer"
)

# Training configuration
training_config = run.Config(
    TrainModel,
    model=model_config,
    data_path="/data/processed"
)

# Evaluation configuration
eval_config = run.Config(
    EvaluateModel,
    model_path="/checkpoints/best",
    test_data="/data/test"
)

# Complete pipeline
pipeline_config = run.Config(
    TrainingPipeline,
    preprocess=preprocess_config,
    training=training_config,
    evaluation=eval_config
)
```

#### Configuration Validation

Add validation to your configurations:

```python
def validate_training_config(config: run.Config) -> bool:
    """Validate training configuration parameters."""

    trainer = config.trainer
    data = config.data

    # Check resource requirements
    if trainer.num_nodes * trainer.gpus_per_node < 1:
        raise ValueError("At least one GPU is required")

    # Check batch size compatibility
    if data.batch_size % trainer.gpus_per_node != 0:
        raise ValueError("Batch size must be divisible by number of GPUs")

    # Check memory requirements
    estimated_memory = data.batch_size * config.model.config.seq_length * 4  # bytes
    if estimated_memory > 32 * 1024**3:  # 32GB
        print("Warning: High memory usage detected")

    return True

# Usage
if validate_training_config(training_config):
    experiment = run.submit(training_config, executor)
```

### Using `run.autoconvert`

The `@run.autoconvert` decorator automatically converts regular Python functions to NeMo Run configurations:

```python
import nemo_run as run
from nemo.collections.llm import LlamaModel, Llama3Config8B

@run.autoconvert
def create_llama_model(seq_length: int = 16384) -> LlamaModel:
    """Create a Llama model with specified sequence length."""
    return LlamaModel(
        config=Llama3Config8B(
            seq_length=seq_length,
            hidden_size=4096,
            num_attention_heads=32
        )
    )

# This automatically becomes a run.Config
model_config = create_llama_model(seq_length=8192)
```

**Limitations of `@run.autoconvert`:**

- No support for control flow (if/else, loops, comprehensions)
- No support for complex expressions
- Limited to simple function definitions

**Workaround for complex logic:**

```python
def create_adaptive_model_config(
    model_size: str,
    seq_length: int,
    use_flash_attention: bool = True
) -> run.Config:
    """Create model configuration with complex logic."""

    # Complex logic that can't be in @run.autoconvert
    if model_size == "8b":
        base_config = Llama3Config8B
        hidden_size = 4096
    elif model_size == "70b":
        base_config = Llama3Config70B
        hidden_size = 8192
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

    # Dynamic parameter calculation
    attention_heads = hidden_size // 128
    if use_flash_attention:
        attention_implementation = "flash_attention_2"
    else:
        attention_implementation = "eager"

    return run.Config(
        LlamaModel,
        config=run.Config(
            base_config,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_attention_heads=attention_heads,
            attention_implementation=attention_implementation
        )
    )
```

### Configuration Utilities

#### Broadcasting Values

Apply values across nested configurations:

```python
# Create base configuration
config = run.Config(
    TrainingJob,
    model=run.Config(LlamaModel, config=run.Config(Llama3Config8B)),
    data=run.Config(DataModule, batch_size=32),
    optimizer=run.Config(AdamW, lr=0.001)
)

# Broadcast learning rate to all optimizers
config.broadcast(lr=0.0001)

# Broadcast batch size to all data modules
config.broadcast(batch_size=64)
```

#### Walking Configurations

Apply transformations to nested configurations:

```python
# Double all learning rates
config.walk(lr=lambda cfg: cfg.lr * 2)

# Set all sequence lengths to a specific value
config.walk(seq_length=lambda cfg: 8192)

# Apply custom transformation
def scale_batch_size(cfg):
    if hasattr(cfg, 'batch_size'):
        cfg.batch_size = min(cfg.batch_size * 2, 1024)
    return cfg

config.walk(scale_batch_size)
```

## YAML Equivalence

NeMo Run configurations can be understood in terms of YAML/Hydra syntax, making it easier to transition from YAML-based systems.

### Basic Configuration Mapping

**Python configuration:**
```python
config = run.Config(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384,
        hidden_size=4096
    )
)
```

**Equivalent YAML:**
```yaml
_target_: nemo.collections.llm.gpt.model.llama.LlamaModel
config:
    _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
    seq_length: 16384
    hidden_size: 4096
```

### Partial Function Mapping

**Python partial:**
```python
partial = run.Partial(
    train_model,
    optimizer="adam",
    learning_rate=0.001
)
```

**Equivalent YAML:**
```yaml
_target_: train_model
_partial_: true
optimizer: adam
learning_rate: 0.001
```

### Configuration Operations

**Python operations:**
```python
# Modify configuration
config.config.seq_length *= 2
config.config.hidden_size = 8192

# Broadcast values
config.broadcast(learning_rate=0.0001)
```

**Equivalent YAML transformations:**
```yaml
# After modification
_target_: nemo.collections.llm.gpt.model.llama.LlamaModel
config:
    _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
    seq_length: 32768  # Doubled
    hidden_size: 8192  # Changed
```

## Raw Script Configuration

For legacy workflows or custom execution requirements, NeMo Run supports direct script execution.

### File-Based Scripts

Execute scripts from files:

```python
# Execute a shell script
script = run.Script("./scripts/train_model.sh")

# Execute with environment variables
script = run.Script(
    "./scripts/train_model.sh",
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONPATH": "/path/to/code",
        "DATA_PATH": "/path/to/data"
    }
)

# Execute with arguments
script = run.Script(
    "./scripts/train_model.sh",
    args=["--model-size", "8b", "--batch-size", "512"]
)
```

### Inline Scripts

Execute scripts defined inline:

```python
# Simple inline script
inline_script = run.Script(
    inline="""
#!/bin/bash
set -e

echo "Starting training..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/path/to/code

python train.py \
    --model-size 8b \
    --batch-size 512 \
    --learning-rate 0.001 \
    --max-epochs 100
"""
)

# Complex inline script with multiple commands
complex_script = run.Script(
    inline="""
#!/bin/bash
set -e

# Setup environment
source /opt/conda/etc/profile.d/conda.sh
conda activate nemo

# Download data if not exists
if [ ! -d "/data/dataset" ]; then
    echo "Downloading dataset..."
    python download_data.py --output /data/dataset
fi

# Preprocess data
echo "Preprocessing data..."
python preprocess.py \
    --input /data/dataset \
    --output /data/processed \
    --tokenizer llama-tokenizer

# Train model
echo "Starting training..."
python train.py \
    --model-size 8b \
    --data-path /data/processed \
    --batch-size 512 \
    --learning-rate 0.001 \
    --max-epochs 100 \
    --checkpoint-dir /checkpoints

# Evaluate model
echo "Evaluating model..."
python evaluate.py \
    --model-path /checkpoints/best \
    --test-data /data/test
"""
)
```

### Script Configuration Patterns

#### Parameterized Scripts

Create reusable script templates:

```python
def create_training_script(
    model_size: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int
) -> run.Script:
    """Create a parameterized training script."""

    script_content = f"""
#!/bin/bash
set -e

# Training parameters
MODEL_SIZE={model_size}
BATCH_SIZE={batch_size}
LEARNING_RATE={learning_rate}
MAX_EPOCHS={max_epochs}

echo "Training configuration:"
echo "  Model size: $MODEL_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS"

# Execute training
python train.py \\
    --model-size $MODEL_SIZE \\
    --batch-size $BATCH_SIZE \\
    --learning-rate $LEARNING_RATE \\
    --max-epochs $MAX_EPOCHS \\
    --checkpoint-dir /checkpoints
"""

    return run.Script(inline=script_content)

# Usage
script = create_training_script(
    model_size="8b",
    batch_size=512,
    learning_rate=0.001,
    max_epochs=100
)
```

#### Multi-Stage Scripts

Create complex workflows with multiple stages:

```python
def create_pipeline_script() -> run.Script:
    """Create a complete ML pipeline script."""

    return run.Script(
        inline="""
#!/bin/bash
set -e

# Stage 1: Data preparation
echo "=== Stage 1: Data Preparation ==="
python prepare_data.py \
    --input /data/raw \
    --output /data/processed \
    --tokenizer llama-tokenizer

# Stage 2: Model training
echo "=== Stage 2: Model Training ==="
python train.py \
    --model-size 8b \
    --data-path /data/processed \
    --batch-size 512 \
    --learning-rate 0.001 \
    --max-epochs 100 \
    --checkpoint-dir /checkpoints

# Stage 3: Model evaluation
echo "=== Stage 3: Model Evaluation ==="
python evaluate.py \
    --model-path /checkpoints/best \
    --test-data /data/test \
    --output /results/evaluation.json

# Stage 4: Model deployment preparation
echo "=== Stage 4: Deployment Preparation ==="
python export_model.py \
    --model-path /checkpoints/best \
    --output /deployment/model.pt \
    --format torchscript

echo "Pipeline completed successfully!"
"""
    )
```

## Configuration Best Practices

### Type Safety and Validation

```python
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Type-safe training configuration."""
    model_size: str
    batch_size: int
    learning_rate: float
    max_epochs: int
    use_mixed_precision: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_size not in ["8b", "70b"]:
            raise ValueError(f"Unsupported model size: {self.model_size}")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.max_epochs <= 0:
            raise ValueError("Max epochs must be positive")

def create_validated_config(config: TrainingConfig) -> run.Config:
    """Create NeMo Run configuration from validated config."""
    return run.Config(
        TrainingJob,
        model=create_model_config(config.model_size),
        trainer=run.Config(
            Trainer,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            max_epochs=config.max_epochs,
            precision="bf16-mixed" if config.use_mixed_precision else "32"
        )
    )
```

### Environment-Specific Configurations

```python
import os

def get_environment_config() -> run.Config:
    """Get configuration based on environment."""

    env = os.getenv("NEMO_ENV", "development")

    if env == "development":
        return run.Config(
            TrainingJob,
            model=create_model_config("8b"),
            trainer=run.Config(
                Trainer,
                num_nodes=1,
                gpus_per_node=1,
                batch_size=32,
                max_epochs=5
            )
        )
    elif env == "staging":
        return run.Config(
            TrainingJob,
            model=create_model_config("8b"),
            trainer=run.Config(
                Trainer,
                num_nodes=2,
                gpus_per_node=4,
                batch_size=256,
                max_epochs=50
            )
        )
    elif env == "production":
        return run.Config(
            TrainingJob,
            model=create_model_config("70b"),
            trainer=run.Config(
                Trainer,
                num_nodes=8,
                gpus_per_node=8,
                batch_size=512,
                max_epochs=100
            )
        )
    else:
        raise ValueError(f"Unknown environment: {env}")
```

### Configuration Composition and Reuse

```python
# Base configurations for reuse
BASE_MODEL_CONFIG = run.Config(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        hidden_size=4096,
        num_attention_heads=32
    )
)

BASE_TRAINER_CONFIG = run.Config(
    Trainer,
    precision="bf16-mixed",
    max_epochs=100,
    gradient_clip_val=1.0
)

# Compose configurations
def create_experiment_config(
    model_size: str,
    seq_length: int,
    batch_size: int
) -> run.Config:
    """Create experiment configuration by composing base configs."""

    # Start with base configurations
    model_config = BASE_MODEL_CONFIG.copy()
    trainer_config = BASE_TRAINER_CONFIG.copy()

    # Customize model configuration
    model_config.config.seq_length = seq_length
    if model_size == "70b":
        model_config.config = run.Config(Llama3Config70B)
        model_config.config.hidden_size = 8192
        model_config.config.num_attention_heads = 64

    # Customize trainer configuration
    trainer_config.batch_size = batch_size

    return run.Config(
        TrainingJob,
        model=model_config,
        trainer=trainer_config
    )
```

This comprehensive guide covers all aspects of NeMo Run configuration, from basic usage to advanced patterns and best practices. Use these patterns to create robust, maintainable, and type-safe machine learning experiment configurations.
