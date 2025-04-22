# NeMo Run CLI Guide

NeMo Run CLI is a Python-based command-line tool designed to efficiently configure and execute machine learning experiments. It provides a type-safe, Python-centric alternative to argparse and Hydra, streamlining workflows from prototyping to scaling across diverse environments.

## 1. Introduction

NeMo Run CLI simplifies experiment management by leveraging Python's capabilities:

- **Type-Safe Configurations**: Automatically validates arguments using Python's type annotations.
- **Intuitive Overrides**: Enables modification of nested settings via command line (e.g., `model_config.size=256`).
- **Flexible Configuration Inputs**: Supports configuration through Python defaults and external files (YAML, TOML, JSON).
- **Executors**: Effortlessly transitions execution contexts (local, Docker, Slurm).

### Comparison to Other CLI Solutions

#### vs. argparse

**argparse** requires manually exposing each parameter and lacks support for nested configurations, resulting in verbose boilerplate code as complexity grows. **NeMo Run CLI** automatically exposes all nested parameters through intuitive dot notation (e.g., `model_config.layer_size=256`), allowing direct access to any level of configuration without additional code.

#### vs. Hydra

**Hydra** offers powerful YAML-based configuration but requires context-switching between YAML files and Python code, creating friction in the development workflow. **NeMo Run CLI** achieves similar nested override capabilities while maintaining a seamless Python-first approach through fiddle Configs, using the same OmegaConf library for config file integration to ensure compatibility with existing YAML configurations while also supporting TOML and JSON formats for additional flexibility.

#### vs. Typer

**Typer** provides excellent type-based CLI interfaces but has limited support for nested configurations common in ML workflows. **NeMo Run CLI** builds on Typer's foundations, adding support for nested configuration overwrites (like `model.hidden_size=256`) and fiddle Config integration for a complete ML experiment framework.

#### Code Example Comparison

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

## 2. Core Concepts

- **Entrypoints**: Python functions decorated with `@run.cli.entrypoint` serving as primary CLI commands.
- **Factories**: Functions decorated with `@run.cli.factory` that configure complex objects (e.g., models, optimizers).
- **Partials**: Reusable, partially configured functions enabling flexible experiment definitions.
- **Experiments**: Groups of tasks executed sequentially or concurrently.
- **RunContext**: Manages execution settings, including executor configurations.

## 3. Getting Started

### Example 1: Basic Entrypoint

Create `script.py`:

```python
import nemo_run as run

@run.cli.entrypoint
def train(model: str, epochs: int = 10):
    print(f"Training {model} for {epochs} epochs")

if __name__ == "__main__":
    run.cli.main(train)
```

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

## 4. Advanced Configuration

### Nested Configurations with Dataclasses

Create structured configurations:

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

NeMo Run CLI allows you to export your configuration (including all applied overrides) to YAML, TOML, or JSON files using the `--to-yaml`, `--to-toml`, and `--to-json` flags:

```bash
python script.py --factory @config.yaml model.num_layers=5 --to-yaml exported.yaml
```

This feature is particularly useful for:

- Capturing the final configuration after applying command-line overrides
- Creating a reproducible snapshot of your experiment's configuration
- Converting between different configuration formats (e.g., YAML to TOML)
- Sharing configurations with teammates

When you use any export flag, NeMo Run will output the configuration file and skip execution:

```bash
python task.py --factory @task.yaml model.num_layers=5 --to-toml test.toml
```

Output:

```
File contents:
╭───────────────────────────────────────────────────────── test.toml ──────────────────────────────────────────────────────────╮
│ 1 partial = true
│ 2 target = "main.train_model"
│ 3 batch_size = 32
│ 4 epochs = 10
│ 5 
│ 6 [model]
│ 7 target = "main.Model"
│ 8 activation = "relu"
│ 9 hidden_size = 256
│ 10 num_layers = 5
│ 11 
│ 12 [optimizer]
│ 13 target = "main.Optimizer"
│ 14 betas = [ 0.9, 0.999,]
│ 15 learning_rate = 0.001
│ 16 weight_decay = 1e-5
│ 17 
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Export complete. Skipping execution.

```

## 5. Executors

Executors determine where your code runs, such as local environments, Docker containers, or Slurm clusters.

### Docker Executor

Define and register a Docker executor:

```python
import nemo_run as run

@run.cli.factory
@run.autoconvert
def docker() -> run.Executor:
    return run.DockerExecutor(
        container_image="nvcr.io/nvidia/nemo:dev",
        volumes=[
            "/home/marc/models:/workspaces/models",
            "/home/marc/data:/workspaces/data",
            "/var/run/docker.sock:/var/run/docker.sock",
            f"{BASE_DIR}/opt/NeMo-Run:/opt/NeMo-Run",
            f"{BASE_DIR}/opt/NeMo:/opt/NeMo",
            f"{BASE_DIR}/opt/megatron-lm:/opt/Megatron-LM",
        ],
        env_vars={
            "HF_HOME": "/workspaces/models/hf",
            "NEMO_HOME": "/workspaces/models/nemo",
        }
    )
```

Execute:

```bash
python script.py model=alexnet epochs=5 run.executor=docker
```

### Slurm Executor

Define and register a Slurm executor:

```python
@run.cli.factory
@run.autoconvert
def slurm_cluster() -> run.Executor:
    return run.SlurmExecutor(
        account=ACCOUNT,
        partition=SLURM_PARTITION,
        job_name_prefix=f"{ACCOUNT}-nemo-ux:",
        job_dir=BASE_DIR,
        container_image="nvcr.io/nvidia/nemo:dev",
        container_mounts=[
            f"/home/{USER}:/home/{USER}", 
            "/lustre:/lustre",
        ],
        time="4:00:00",
        gpus_per_node=8,
        tunnel=run.SSHTunnel(host=SLURM_LOGIN_NODE, user=USER, job_dir=BASE_DIR)
    )
```

Execute lazily:

```bash
python script.py --lazy model=alexnet epochs=5 run.executor=slurm_cluster run.executor.nodes=2
```

## 6. Advanced CLI Features

### Dry Runs and Help Messages

Use `--dryrun` to preview execution:

```bash
python task.py --yaml task.yaml --dryrun
```

Output:

```
Configuring global options
Dry run for task __main__:train_model
Resolved Arguments
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Argument Name        ┃ Resolved Value                                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ batch_size           │ 32                                                           │
│ epochs               │ 10                                                           │
│ model                │ Model(hidden_size=256, num_layers=3, activation='relu')      │
│ optimizer            │ Optimizer(learning_rate=0.001, weight_decay=1e-05,           │
│                      │ betas=[0.9, 0.999])                                          │
└──────────────────────┴──────────────────────────────────────────────────────────────┘
Dry run for train_model:
```

Use `--help` to see detailed CLI usage information, including registered factory functions for each argument:

```bash
python task.py --help
```

Output:

```
Usage: task.py [OPTIONS] [ARGUMENTS]

[Entrypoint] train_model
Train a model using the specified configuration.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --name                  -n                           TEXT  Name of the run [default: None]
│ --direct                    --no-direct                    Execute the run directly [default: no-direct]
│ --dryrun                                                   Print the scheduler request without submitting
│ --factory               -f                           TEXT  Predefined factory to use [default: None]
│ --load                  -l                           TEXT  Load a factory from a directory [default: None]
│ --yaml                  -y                           TEXT  Path to a YAML file to load [default: None]
│ --repl                  -r                                 Enter interactive mode
│ --detach                                                   Detach from the run
│ --yes,--no-confirm      -y                                 Skip confirmation before execution
│ --tail-logs                 --no-tail-logs                 Tail logs after execution [default: no-tail-logs]
│ --verbose               -v                                 Enable verbose logging
│ --rich-exceptions           --no-rich-exceptions           Enable rich exception formatting [default: no-rich-exceptions]
│ --rich-traceback-short      --rich-traceback-full          Control traceback verbosity [default: rich-traceback-full]
│ --rich-show-locals          --rich-hide-locals             Toggle local variables in exceptions [default: rich-show-locals]
│ --rich-theme                                         TEXT  Color theme (dark/light/monochrome) [default: None]
│ --to-yaml                                            TEXT  Export config to YAML file [default: None]
│ --to-toml                                            TEXT  Export config to TOML file [default: None]
│ --to-json                                            TEXT  Export config to JSON file [default: None]
│ --install-completion                                       Install completion for the current shell.
│ --show-completion                                          Show completion for the current shell, to copy it or customize
│                                                            the installation.
│ --help                                                     Show this message and exit.
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Pre-loaded entrypoint factories, run with --factory ────────────────────────────────────────────────────────────────────────╮
│ train_recipe                               task.train_recipe                        line 72
│
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ model                                      task.Model
│ optimizer                                  task.Optimizer
│ epochs                                     int                                      10
│ batch_size                                 int                                      32
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Factory for model: task.Model ──────────────────────────────────────────────────────────────────────────────────────────────╮
│ my_model                                   task.my_model                            line 25
│
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Factory for optimizer: task.Optimizer ──────────────────────────────────────────────────────────────────────────────────────╮
│ my_optimizer                               task.my_optimizer                        line 34
│
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The help output clearly shows:

1. The entrypoint function name and description
2. Available command-line options with their descriptions and default values
3. Pre-loaded entrypoint factories that can be used with the `--factory` option
4. Arguments expected by the function with their types and default values
5. Registered factory functions for each complex argument type

This makes it easy for users to discover what factory functions they can use to configure complex arguments like `model` and `optimizer`, along with information about where these factories are defined (module name and line number).

