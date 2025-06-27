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

**A:** Install NeMo Run using pip:

```bash
pip install nemo-run
```

Basic setup involves:

1. **Configure your environment**:

   ```bash
   export NEMORUN_HOME=~/.nemorun  # Optional: customize home directory
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
        paths=run.Config(create_paths, 10),
        weights=[1.0] * 10
    )
```

## Code Packaging and Deployment

### **Q:** How does NeMo Run package my code for remote execution?

**A:** NeMo Run uses packagers to bundle your code and dependencies for remote execution:

**GitArchivePackager** (default):

- Creates a Git archive of your current repository
- Includes only committed changes
- Excludes files in `.gitignore`
- Provides reproducible code snapshots

**DockerPackager**:

- Builds a Docker image with your code
- Includes all dependencies
- Provides isolated execution environment

**HybridPackager**:

- Combines Git archive with Docker image
- Offers flexibility for different deployment scenarios

### **Q:** Why aren't my local changes reflected in remote jobs?

**A:** This is typically due to packaging behavior. Here are the common scenarios:

**Scenario 1**: Uncommitted changes in Git repository

```bash
# ❌ Changes not committed
git add my_changes.py
# Missing: git commit -m "Add changes"

# ✅ Commit changes first
git add my_changes.py
git commit -m "Add changes"
```

**Scenario 2**: Changes outside Git repository

```python
# ❌ Files outside repo aren't packaged
external_file = "/path/to/external/data.csv"

# ✅ Solutions:
# 1. Move files into your repository
# 2. Use DockerPackager with volume mounts
# 3. Copy files to remote cluster manually
```

**Scenario 3**: Using wrong packager

```python
# For local development with frequent changes
packager = run.DockerPackager(
    dockerfile="Dockerfile.dev",
    build_context="."
)

# For production with Git-based versioning
packager = run.GitArchivePackager()
```

### **Q:** How do I include external dependencies or data files?

**A:** Several approaches depending on your needs:

**For Python dependencies**:

```python
# Use requirements.txt or pyproject.toml
packager = run.GitArchivePackager(
    requirements_file="requirements.txt"
)
```

**For data files**:

```python
# Option 1: Include in repository
# Add data/ directory to your Git repository

# Option 2: Use Docker with volume mounts
packager = run.DockerPackager(
    volumes={
        "/host/data": "/container/data"
    }
)

# Option 3: Copy files during execution
def setup_data():
    import shutil
    shutil.copy("/host/data", "/container/data")

config = run.Config(MyJob, setup_fn=run.Config(setup_data))
```

## Execution and Scheduling

### **Q:** How do I choose the right executor for my use case?

**A:** Choose based on your compute environment and requirements:

**LocalExecutor**: Development and testing

```python
executor = run.LocalExecutor()
# Pros: Fast iteration, no setup required
# Cons: Limited resources, no isolation
```

**SlurmExecutor**: HPC clusters

```python
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=2,
    gpus_per_node=4,
    time_limit="24:00:00"
)
# Pros: Resource management, job queuing
# Cons: Requires Slurm cluster access
```

**DockerExecutor**: Containerized execution

```python
executor = run.DockerExecutor(
    image="nvidia/cuda:11.8-devel-ubuntu20.04",
    gpus="all"
)
# Pros: Environment isolation, reproducibility
# Cons: Build time, image size
```

**KubernetesExecutor**: Cloud-native deployment

```python
executor = run.KubernetesExecutor(
    namespace="ml-experiments",
    resources={"gpu": 4}
)
# Pros: Scalability, resource efficiency
# Cons: Kubernetes complexity
```

### **Q:** How do I execute Slurm jobs from different environments?

**A:** NeMo Run supports multiple execution modes for Slurm:

**From local machine via SSH**:

```python
ssh_tunnel = run.SSHTunnel(
    host="cluster.login.node",
    user="username",
    job_dir="/home/username/nemo-run-experiments",
    ssh_key_path="~/.ssh/id_rsa"
)

executor = run.SlurmExecutor(
    partition="gpu",
    nodes=1,
    gpus_per_node=4,
    tunnel=ssh_tunnel
)
```

**From login node directly**:

```python
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=1,
    gpus_per_node=4,
    tunnel=run.LocalTunnel()  # Direct execution
)
```

**From within a Slurm job**:

```python
# When running inside a Slurm allocation
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=1,
    gpus_per_node=4,
    tunnel=run.LocalTunnel(),
    submit_from_allocated_job=True  # Submit from within allocation
)
```

### **Q:** How do I handle resource requirements and constraints?

**A:** Configure resources based on your workload:

**Basic resource specification**:

```python
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=2,
    gpus_per_node=4,
    cpus_per_node=32,
    memory_per_node="128G",
    time_limit="24:00:00"
)
```

**Advanced resource management**:

```python
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=2,
    gpus_per_node=4,
    exclusive=True,  # Exclusive node access
    constraint="gpu_type:rtx6000",  # Specific GPU type
    qos="high_priority",  # Quality of service
    account="ml_research"  # Account/charge code
)
```

**Dynamic resource allocation**:

```python
def adaptive_resources(batch_size):
    """Calculate resources based on batch size."""
    if batch_size <= 32:
        return {"nodes": 1, "gpus_per_node": 2}
    elif batch_size <= 128:
        return {"nodes": 2, "gpus_per_node": 4}
    else:
        return {"nodes": 4, "gpus_per_node": 8}

config = run.Config(MyJob, batch_size=64)
resources = adaptive_resources(config.batch_size)
executor = run.SlurmExecutor(**resources)
```

## Logging and Monitoring

### **Q:** How does NeMo Run handle logging and experiment tracking?

**A:** NeMo Run provides centralized logging and experiment management:

**Automatic log collection**:

```python
# Logs are automatically captured and stored
experiment = run.submit(config, executor)

# Retrieve logs
logs = run.get_logs(experiment)
print(logs.stdout)
print(logs.stderr)
```

**Custom logging configuration**:

```python
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )

config = run.Config(MyJob, setup_fn=run.Config(setup_logging))
```

**Experiment metadata**:

```python
# Add metadata to experiments
experiment = run.submit(
    config,
    executor,
    metadata={
        "experiment_name": "transformer_finetuning",
        "dataset": "wikitext-103",
        "model_size": "1.3B"
    }
)
```

### **Q:** How do I retrieve and analyze experiment logs?

**A:** Multiple ways to access and analyze logs:

**Basic log retrieval**:

```python
# Get experiment by ID
experiment = run.get_experiment("exp_123")

# Get logs
logs = run.get_logs(experiment)
print(f"Exit code: {logs.exit_code}")
print(f"Stdout: {logs.stdout}")
print(f"Stderr: {logs.stderr}")
```

**Streaming logs during execution**:

```python
# Stream logs in real-time
for log_entry in run.stream_logs(experiment):
    print(log_entry.timestamp, log_entry.message)
```

**Log analysis and filtering**:

```python
import re

def analyze_logs(experiment):
    logs = run.get_logs(experiment)

    # Extract metrics
    loss_pattern = r"loss: (\d+\.\d+)"
    losses = re.findall(loss_pattern, logs.stdout)

    # Extract errors
    error_pattern = r"ERROR: (.+)"
    errors = re.findall(error_pattern, logs.stderr)

    return {
        "losses": [float(l) for l in losses],
        "errors": errors,
        "exit_code": logs.exit_code
    }
```

## Troubleshooting

### **Q:** Why can't I retrieve logs for my experiment?

**A:** Common causes and solutions:

**NeMo Run home directory issues**:

```bash
# Check current home directory
echo $NEMORUN_HOME

# Verify experiment exists
ls ~/.nemorun/experiments/

# If home changed, set it correctly
export NEMORUN_HOME=/path/to/original/home
```

**Remote cluster issues**:

```python
# Check if cluster is still running
executor = run.SlurmExecutor(...)
status = executor.get_cluster_status()

# For Kubernetes, check pod status
executor = run.KubernetesExecutor(...)
pods = executor.list_pods()
```

**Network connectivity issues**:

```python
# Test SSH connection
ssh_tunnel = run.SSHTunnel(host="cluster", user="user")
try:
    ssh_tunnel.test_connection()
    print("✅ SSH connection working")
except Exception as e:
    print(f"❌ SSH connection failed: {e}")
```

### **Q:** How do I debug configuration and serialization issues?

**A:** Systematic debugging approach:

**Step 1: Validate configuration structure**:

```python
def debug_config(config):
    """Recursively inspect configuration for issues."""
    def inspect(obj, path=""):
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                current_path = f"{path}.{key}" if path else key
                print(f"Checking {current_path}: {type(value)}")

                if not isinstance(value, (run.Config, run.Partial, str, int, float, bool, list, dict)):
                    print(f"⚠️  Non-serializable object at {current_path}: {type(value)}")

                if isinstance(value, (run.Config, run.Partial)):
                    inspect(value, current_path)

    inspect(config)
    print("✅ Configuration inspection complete")

# Usage
debug_config(my_config)
```

**Step 2: Test serialization incrementally**:

```python
def test_serialization_incremental(config):
    """Test serialization by building config incrementally."""
    serializer = ZlibJSONSerializer()

    # Test each component separately
    for key, value in config.__dict__.items():
        try:
            test_config = run.Config(type(config), **{key: value})
            serializer.serialize(test_config)
            print(f"✅ {key} serializes successfully")
        except Exception as e:
            print(f"❌ {key} fails: {e}")
```

**Step 3: Use minimal reproduction**:

```python
# Create minimal test case
minimal_config = run.Config(MyClass, param1="test")
try:
    serializer.serialize(minimal_config)
    print("✅ Minimal config works")

    # Add parameters one by one
    for param, value in original_config.__dict__.items():
        if param not in minimal_config.__dict__:
            test_config = run.Config(MyClass, **{**minimal_config.__dict__, param: value})
            serializer.serialize(test_config)
            print(f"✅ Adding {param} works")

except Exception as e:
    print(f"❌ Issue found: {e}")
```

### **Q:** How do I handle executor-specific issues?

**A:** Common executor problems and solutions:

**SlurmExecutor issues**:

```python
# Check Slurm configuration
executor = run.SlurmExecutor(...)

# Verify partition exists
partitions = executor.list_partitions()
print(f"Available partitions: {partitions}")

# Check queue status
queue = executor.get_queue_status()
print(f"Queue status: {queue}")

# Test job submission
test_job = executor.submit_test_job()
print(f"Test job ID: {test_job}")
```

**DockerExecutor issues**:

```python
# Check Docker daemon
import docker
client = docker.from_env()
try:
    client.ping()
    print("✅ Docker daemon accessible")
except Exception as e:
    print(f"❌ Docker daemon issue: {e}")

# Check image availability
try:
    client.images.get("nvidia/cuda:11.8-devel-ubuntu20.04")
    print("✅ CUDA image available")
except docker.errors.ImageNotFound:
    print("❌ CUDA image not found, pulling...")
    client.images.pull("nvidia/cuda:11.8-devel-ubuntu20.04")
```

**KubernetesExecutor issues**:

```python
# Check cluster connectivity
executor = run.KubernetesExecutor(...)

# Verify namespace exists
namespaces = executor.list_namespaces()
print(f"Available namespaces: {namespaces}")

# Check resource quotas
quotas = executor.get_resource_quotas()
print(f"Resource quotas: {quotas}")

# Test pod creation
test_pod = executor.create_test_pod()
print(f"Test pod created: {test_pod}")
```
