---
description: "Comprehensive CLI reference for NeMo Run, including command-line interface usage, configuration options, and advanced features."
tags: ["cli", "reference", "command-line", "interface"]
categories: ["reference"]
---

(cli-guide)=

# NeMo Run CLI Guide

NeMo Run CLI is a Python-based command-line tool designed to efficiently configure and execute machine learning experiments. It provides a type-safe, Python-centric alternative to argparse and Hydra, streamlining workflows from prototyping to scaling across diverse environments.

## Introduction

NeMo Run CLI simplifies experiment management by leveraging Python's capabilities:

- **Type-Safe Configurations**: Automatically validates arguments using Python's type annotations
- **Intuitive Overrides**: Enables modification of nested settings via command line (e.g., `model_config.size=256`)
- **Flexible Configuration Inputs**: Supports configuration through Python defaults and external files (YAML, TOML, JSON)
- **Executors**: Effortlessly transitions execution contexts (local, Docker, Slurm)

### Comparison to Other CLI Solutions

#### vs. argparse

**argparse** requires manually exposing each parameter and lacks support for nested configurations, resulting in verbose boilerplate code as complexity grows. **NeMo Run CLI** automatically exposes all nested parameters through intuitive dot notation (e.g., `model_config.layer_size=256`), allowing direct access to any level of configuration without additional code.

#### vs. Hydra

**Hydra** offers powerful YAML-based configuration but requires context-switching between YAML files and Python code, creating friction in the development workflow. **NeMo Run CLI** achieves similar nested override capabilities while maintaining a seamless Python-first approach through fiddle Configs, using the same OmegaConf library for config file integration to ensure compatibility with existing YAML configurations while also supporting TOML and JSON formats for additional flexibility.

#### vs. Typer

**Typer** provides excellent type-based CLI interfaces but has limited support for nested configurations common in ML workflows. **NeMo Run CLI** builds on Typer's foundations, adding support for nested configuration overwrites (like `model.hidden_size=256`) and fiddle Config integration for a complete ML experiment framework.

#### Code Example Comparison

::::{dropdown} CLI Comparison Examples
:icon: code-square

```python
# NeMo Run CLI - Simple, automatic parameter exposure
@run.cli.entrypoint
def train(model_config: ModelConfig, optimizer: OptimizerConfig):
    # All nested parameters automatically available via CLI:
    # python train.py model_config.hidden_size=1024 optimizer.lr=0.01
    ...

# argparse - Requires manual exposure of every parameter
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-hidden-size", type=int, default=512)
    parser.add_argument("--optimizer-lr", type=float, default=0.001)
    # Must manually add every nested parameter you want to expose
    ...
```
::::

#### Feature Comparison

| Feature | NeMo Run CLI | Hydra | Typer | argparse |
|---------|-------------|-------|-------|----------|
| Type checking | ✅ (Python types) | ⚠️ (via OmegaConf) | ✅ | ❌ (manual) |
| Nested configs | ✅ (native) | ✅ | ❌ | ❌ |
| Config files | ✅ (YAML/JSON/TOML) | ✅ (YAML) | ❌ (manual) | ❌ (manual) |
| Python-native | ✅ | ⚠️ | ✅ | ✅ |
| ML experiment focus | ✅ | ⚠️ | ❌ | ❌ |
| Executor abstraction | ✅ | ❌ | ❌ | ❌ |
| Default suggestions | ✅ | ❌ | ✅ | ❌ |
| Factory system | ✅ | ❌ | ❌ | ❌ |
| Learning curve | Medium | High | Low | Low |

#### When to Choose Each Solution

- **NeMo Run CLI**: ML experiments with complex nested configurations, particularly when moving between development and production environments
- **Hydra**: Applications with extensive YAML configuration needs where Python integration is secondary
- **Typer**: General-purpose CLIs with good documentation that don't require nested configuration
- **argparse**: Simple scripts with minimal configuration needs and standard library requirements

## Core Concepts

- **Entrypoints**: Python functions decorated with `@run.cli.entrypoint` serving as primary CLI commands
- **Factories**: Functions decorated with `@run.cli.factory` that configure complex objects (e.g., models, optimizers)
- **Partials**: Reusable, partially configured functions enabling flexible experiment definitions
- **Experiments**: Groups of tasks executed sequentially or concurrently
- **RunContext**: Manages execution settings, including executor configurations

## Getting Started

### Example 1: Basic Entrypoint

Create `script.py`:

::::{dropdown} Basic Entrypoint Example
:icon: code-square

```python
import nemo_run as run

@run.cli.entrypoint
def train(model: str, epochs: int = 10):
    print(f"Training {model} for {epochs} epochs")

if __name__ == "__main__":
    run.cli.main(train)
```
::::

Execute:

```bash
python script.py model=alexnet epochs=5
```

Output:

```
Training alexnet for 5 epochs
```

### Example 2: Error Correction

NeMo Run CLI helps prevent silent failures by catching typos:

```bash
python script.py model=alexnet epocks=5
```

Output:

```
Unknown argument 'epocks'. Did you mean 'epochs'?
```

## Advanced Configuration

### Nested Configurations with Dataclasses

Create structured configurations:

::::{dropdown} Nested Configuration Example
:icon: code-square

```python
from dataclasses import dataclass
import nemo_run as run

@dataclass
class ModelConfig:
    size: int = 128
    layers: int = 2

@run.cli.entrypoint
def train(model_config: ModelConfig):
    print(f"Model size: {model_config.size}, layers: {model_config.layers}")

if __name__ == "__main__":
    run.cli.main(train)
```
::::

Execute with overrides:

```bash
python script.py model_config.size=256
```

Output:

```
Configuring global options
Dry run for task __main__:train
Resolved Arguments
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Argument Name        ┃ Resolved Value                                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ model_config         │ ModelConfig(size=256)                                        │
└──────────────────────┴──────────────────────────────────────────────────────────────┘
Continue? [y/N]: y
Launching train...
Model size: 256, layers: 2
```

### Configuration Files and the @ Syntax

NeMo Run CLI supports external configuration files (YAML, TOML, JSON) using the `@` syntax:

- Use `--factory` combined with `@` to load entire configurations:

```bash
python script.py --factory @path/to/config.yaml
```

- Load specific nested configurations using the `@` syntax:

```bash
python script.py model_config=@configs/model.yaml optimizer=@configs/optimizer.json
```

Overrides can still be applied directly alongside file-based inputs:

```bash
python script.py --factory @path/to/config.yaml model_config.layers=4
```

### Exporting Configurations
