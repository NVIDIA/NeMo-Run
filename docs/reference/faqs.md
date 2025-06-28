---
description: "Frequently asked questions about NeMo Run"
tags: ["FAQs", "troubleshooting", "help", "configuration", "execution", "management"]
categories: ["help"]
---

(faqs)=

# Frequently Asked Questions

This section provides comprehensive answers to common questions about NeMo Run, organized by functionality and complexity.

## Getting Started

### **Q:** What is NeMo Run and when should I use it?

**A:** NeMo Run is a Python framework designed for distributed machine learning experimentation and execution. It provides:

- **Unified Configuration Management**: Use `run.Config` and `run.Partial` for type-safe, serializable configurations
- **Multi-Platform Execution**: Support for local, Slurm, Kubernetes, Docker, and cloud platforms
- **Automatic Code Packaging**: Git-based packaging for reproducible experiments
- **Built-in Logging and Monitoring**: Centralized experiment tracking and log retrieval

Use NeMo Run when you need to:

- Run ML experiments across different compute environments
- Ensure reproducibility through configuration management
- Scale experiments from local development to production clusters
- Maintain consistent logging and monitoring across platforms

### **Q:** How do I install and set up NeMo Run?

**A:** Install NeMo Run using pip from the GitHub repository:

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

Basic setup involves:

1. **Configure your environment**:

   ```bash
   export NEMORUN_HOME=~/.nemo_run  # Optional: customize home directory
   ```

2. **Initialize a project**:

   ```python
   import nemo_run as run

   # Basic configuration
   config = run.Config(YourModel, learning_rate=0.001, batch_size=32)
   ```

3. **Choose an executor**:

   ```python
   # Local execution
   executor = run.LocalExecutor()

   # Remote execution
   executor = run.SlurmExecutor(
       partition="gpu",
       nodes=1,
       gpus_per_node=4
   )
   ```

## Configuration and Serialization

### **Q:** What's the difference between `run.Config` and `run.Partial`?

**A:** Both are configuration primitives, but they serve different purposes:

- **`run.Config`**: Creates a complete configuration for a class or function

  ```python
  model_config = run.Config(
      MyModel,
      hidden_size=512,
      num_layers=6,
      dropout=0.1
  )
  ```

- **`run.Partial`**: Creates a partially applied function with some arguments fixed

  ```python
  train_fn = run.Partial(
      train_model,
      optimizer="adam",
      learning_rate=0.001
  )
  # Can be called later with additional arguments
  ```

### **Q:** How do I handle serialization errors with complex objects?

**A:** NeMo Run uses Fiddle's serialization system, which requires all configuration values to be JSON-serializable. Common issues and solutions:

**Problem**: Non-serializable objects like `pathlib.Path`:

```python
# ❌ This will fail
config = run.Config(MyClass, data_path=Path("/tmp/data"))
```

**Solution**: Wrap non-serializable objects in `run.Config`:

```python
# ✅ This works
config = run.Config(MyClass, data_path=run.Config(Path, "/tmp/data"))
```

**Problem**: Custom classes or complex objects:

```python
# ❌ This will fail
config = run.Config(MyClass, custom_obj=MyCustomObject())
```

**Solution**: Create factory functions or use `run.Partial`:

```python
# ✅ Using a factory function
def create_custom_obj(param1, param2):
    return MyCustomObject(param1, param2)

config = run.Config(MyClass, custom_obj=run.Config(create_custom_obj, "value1", "value2"))
```

### **Q:** How do I validate my configuration before execution?

**A:** Use the serialization round-trip test to validate configurations:

```python
from nemo_run.config import ZlibJSONSerializer

def validate_config(config):
    """Validate that a configuration can be serialized and deserialized."""
    serializer = ZlibJSONSerializer()

    try:
        # Serialize and deserialize
        serialized = serializer.serialize(config)
        deserialized = serializer.deserialize(serialized)

        # Verify equality
        assert config == deserialized, "Configuration changed during serialization"
        print("✅ Configuration is valid")
        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Usage
config = run.Config(MyModel, param1="value1", param2=run.Config(Path, "/tmp"))
validate_config(config)
```

### **Q:** How do I handle control flow in configurations?

**A:** NeMo Run's `@run.autoconvert` decorator doesn't support control flow constructs like list comprehensions or loops. Here are the recommended approaches:

**Problem**: Control flow in `@run.autoconvert`:

```python
# ❌ This will fail
@run.autoconvert
def create_dataset():
    return Dataset(
        paths=[Path(f"data_{i}.txt") for i in range(10)],  # List comprehension
        weights=[1.0 for _ in range(10)]
    )
```

**Solution 1**: Use `run.Config` directly:

```python
# ✅ Direct configuration
def create_dataset_config():
    return run.Config(
        Dataset,
        paths=[run.Config(Path, f"data_{i}.txt") for i in range(10)],
        weights=[1.0 for _ in range(10)]
    )
```

**Solution 2**: Use factory functions:

```python
# ✅ Factory function approach
def create_paths(num_files):
    return [run.Config(Path, f"data_{i}.txt") for i in range(num_files)]

def create_dataset_config():
    return run.Config(
        Dataset,
        paths=create_paths(10),
        weights=[1.0 for _ in range(10)]
    )
```

## Execution and Backends

### **Q:** How does NeMo Run package my code for remote execution?

**A:** NeMo Run uses packagers to bundle your code and dependencies for remote execution:

- **`run.Packager`**: Pass-through packager (no modification)
- **`run.GitArchivePackager`**: Packages Git repository using `git archive`
- **`run.PatternPackager`**: Packages files based on pattern matching
- **`run.HybridPackager`**: Combines multiple packagers

Example:

```python
# Git-based packaging
packager = run.GitArchivePackager(subpath="src")
executor = run.SlurmExecutor(packager=packager)

# Pattern-based packaging
packager = run.PatternPackager(
    include_pattern="src/**",
    relative_path=os.getcwd()
)
executor = run.DockerExecutor(packager=packager)
```

### **Q:** What execution backends does NeMo Run support?

**A:** NeMo Run supports multiple execution backends:

- **`run.LocalExecutor`**: Local process execution
- **`run.DockerExecutor`**: Docker container execution
- **`run.SlurmExecutor`**: HPC cluster execution via Slurm
- **`run.SkypilotExecutor`**: Multi-cloud execution with cost optimization
- **`run.DGXCloudExecutor`**: NVIDIA DGX Cloud execution
- **`run.LeptonExecutor`**: Lepton cloud execution

Each executor supports different packaging strategies and resource configurations.

### **Q:** How do I configure Slurm execution?

**A:** Configure Slurm execution with the `run.SlurmExecutor`:

```python
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=2,
    gpus_per_node=4,
    time="02:00:00",
    job_name="my_experiment",
    account="my_account"
)
```

For SSH tunnel access from your local machine:

```python
from nemo_run.core.execution.slurm import SSHTunnel

tunnel = SSHTunnel(
    host="cluster.example.com",
    username="your_username",
    port=22
)

executor = run.SlurmExecutor(
    partition="gpu",
    tunnel=tunnel
)
```

### **Q:** How does NeMo Run handle logging and experiment tracking?

**A:** NeMo Run provides centralized logging and experiment management:

- **Automatic Log Capture**: All stdout/stderr is captured and stored
- **Experiment Metadata**: Configuration, timestamps, and status are tracked
- **Log Retrieval**: Access logs through the experiment interface
- **Centralized Storage**: All data stored in `NEMORUN_HOME`

Example:

```python
# Launch experiment
experiment = run.submit(task_config, executor)

# Monitor progress
print(f"Status: {experiment.status}")
print(f"Logs: {experiment.logs}")

# Retrieve logs
logs = run.get_logs(experiment)
print(f"Exit code: {logs.exit_code}")
print(f"Output: {logs.stdout}")
```

## Troubleshooting

### **Q:** How do I debug configuration issues?

**A:** Use these debugging techniques:

1. **Validate Configuration**:
   ```python
   from nemo_run.config import ZlibJSONSerializer
   serializer = ZlibJSONSerializer()

   try:
       serialized = serializer.serialize(config)
       print("✅ Configuration is serializable")
   except Exception as e:
       print(f"❌ Serialization failed: {e}")
   ```

2. **Check Type Hints**:
   ```python
   import inspect
   sig = inspect.signature(MyFunction)
   print(sig.parameters)
   ```

3. **Test CLI Parsing**:
   ```bash
   python script.py --help
   python script.py --dryrun param1=value1
   ```

### **Q:** How do I resolve dependency conflicts?

**A:** Resolve dependency conflicts with these approaches:

1. **Use Virtual Environment**:
   ```bash
   python -m venv nemo-run-env
   source nemo-run-env/bin/activate
   pip install git+https://github.com/NVIDIA-NeMo/Run.git
   ```

2. **Install with --no-deps**:
   ```bash
   pip install git+https://github.com/NVIDIA-NeMo/Run.git --no-deps
   pip install inquirerpy catalogue fabric fiddle torchx typer rich jinja2 cryptography networkx omegaconf leptonai packaging toml
   ```

3. **Use Compatible Versions**:
   ```bash
   pip install "torchx>=0.7.0" "fiddle>=0.3.0" "omegaconf>=2.3.0"
   ```

### **Q:** How do I recover from experiment failures?

**A:** NeMo Run provides several recovery mechanisms:

1. **Check Experiment Status**:
   ```python
   experiment = run.get_experiment(experiment_id)
   print(f"Status: {experiment.status}")
   print(f"Error: {experiment.error}")
   ```

2. **Retrieve Logs**:
   ```python
   logs = run.get_logs(experiment)
   print(f"Exit code: {logs.exit_code}")
   print(f"Error output: {logs.stderr}")
   ```

3. **Restart with Modified Config**:
   ```python
   # Modify configuration based on error
   new_config = config.clone()
   new_config.learning_rate = 0.0001

   # Restart experiment
   new_experiment = run.submit(new_config, executor)
   ```

### **Q:** How do I manage NeMo Run home directory issues?

**A:** NeMo Run home directory issues can be resolved by:

1. **Check Current Home**:
   ```bash
   echo $NEMORUN_HOME
   ls ~/.nemo_run/experiments/
   ```

2. **Reset Home Directory**:
   ```bash
   export NEMORUN_HOME=~/.nemo_run
   mkdir -p ~/.nemo_run
   ```

3. **Recover from Backup**:
   ```bash
   export NEMORUN_HOME=/path/to/original/home
   cp -r ~/.nemo_run.backup ~/.nemo_run
   ```

## Advanced Topics

### **Q:** How do I create custom executors?

**A:** Create custom executors by inheriting from `run.Executor`:

```python
from nemo_run.core.execution.base import Executor

class CustomExecutor(Executor):
    def __init__(self, custom_param: str):
        self.custom_param = custom_param

    def submit(self, task_config, **kwargs):
        # Custom submission logic
        pass

    def get_status(self, job_id):
        # Custom status checking
        pass
```

### **Q:** How do I integrate with external experiment tracking?

**A:** Integrate with external tracking systems using plugins:

```python
from nemo_run.run.plugin import ExperimentPlugin

class WandBPlugin(ExperimentPlugin):
    def on_experiment_start(self, experiment):
        import wandb
        wandb.init(project="my_project")

    def on_experiment_end(self, experiment):
        import wandb
        wandb.finish()

# Use plugin
executor = run.LocalExecutor(plugins=[WandBPlugin()])
```

### **Q:** How do I optimize performance for large-scale experiments?

**A:** Optimize performance with these strategies:

1. **Use Efficient Packagers**:
   ```python
   # Use Git packager for large codebases
   packager = run.GitArchivePackager(subpath="src")
   ```

2. **Configure Resource Limits**:
   ```python
   executor = run.SlurmExecutor(
       partition="gpu",
       nodes=4,
       gpus_per_node=8,
       memory="64GB"
   )
   ```

3. **Use Parallel Execution**:
   ```python
   experiment = run.Experiment()
   for config in configs:
       experiment.add_task(config, executor)
   experiment.launch(sequential=False)
   ```

This FAQ covers the most common questions about NeMo Run. For more detailed information, refer to the specific guides for [Configuration](../guides/configuration), [CLI Reference](cli.md), [Execution](../guides/execution), and [Management](../guides/management).
