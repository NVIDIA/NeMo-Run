---
description: "Comprehensive troubleshooting guide for NeMo Run covering common issues, error messages, debugging techniques, and solutions."
tags: ["troubleshooting", "debugging", "errors", "solutions", "help", "support"]
categories: ["help"]
---

(troubleshooting)=

# Troubleshooting NeMo Run

This guide helps you diagnose and resolve common issues when using NeMo Run. It covers error messages, debugging techniques, and solutions for various scenarios.

## Quick Diagnostic Commands

### Check NeMo Run Status

Run these commands to quickly assess your NeMo Run installation:

```bash
# Check NeMo Run installation
python -c "import nemo_run; print(nemo_run.__version__)"

# Check environment variables
echo $NEMORUN_HOME

# Check Python environment
python -c "import nemo_run as run; print(dir(run))"
```

## Common Issues and Solutions

### Installation Issues

#### Package Installation Problems

**Problem**: Unable to install NeMo Run from GitHub

**Solution**: Use the correct installation method:

```bash
# ✅ Correct installation
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# ❌ Incorrect (this package doesn't exist)
pip install nemo-run
```

**Problem**: Git installation fails

**Solution**: Ensure Git is available and use HTTPS:

```bash
# Check Git installation
git --version

# Use HTTPS instead of SSH
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Or install manually
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run
pip install .
```

#### Dependency Conflicts

**Problem**: Version conflicts with dependencies

**Solution**: Install with compatible versions:

```bash
# Install with --no-deps and resolve manually
pip install git+https://github.com/NVIDIA-NeMo/Run.git --no-deps

# Install core dependencies
pip install inquirerpy catalogue fabric fiddle torchx typer rich jinja2 cryptography networkx omegaconf leptonai packaging toml

# Install optional dependencies
pip install "skypilot[kubernetes]>=0.9.2"
pip install "ray[kubernetes]"
```

### Configuration Issues

#### Serialization Errors

**Problem**: Configuration serialization fails

**Solution**: Wrap non-serializable objects in `run.Config`:

```python
# ❌ This will fail
partial = run.Partial(some_function, something=Path("/tmp"))

# ✅ Correct: Wrap in run.Config
partial = run.Partial(some_function, something=run.Config(Path, "/tmp"))
```

**Problem**: Complex object serialization

**Solution**: Use factory functions or `run.Partial`:

```python
from nemo_run.config import ZlibJSONSerializer

# Test serialization
serializer = ZlibJSONSerializer()
partial = run.Partial(some_function, something=run.Config(Path, "/tmp"))

try:
    serialized = serializer.serialize(partial)
    print("✅ Configuration serializes successfully")
except Exception as e:
    print(f"❌ Serialization failed: {e}")
```

#### Control Flow Issues

**Problem**: Control flow constructs in `@run.autoconvert`

**Solution**: Use `run.Config` directly or factory functions:

```python
# ❌ This will fail
@run.autoconvert
def control_flow_config() -> run.Config[llm.PreTrainingDataModule]:
    return run.Config(
        llm.PreTrainingDataModule,
        paths=[Path(f"some_doc_{i}") for i in range(10)],  # List comprehension
        weights=[1.0 for _ in range(10)]
    )

# ✅ Correct: Use run.Config directly
def control_flow_config() -> run.Config[llm.PreTrainingDataModule]:
    return run.Config(
        llm.PreTrainingDataModule,
        paths=[run.Config(Path, f"some_doc_{i}") for i in range(10)],
        weights=[1.0 for _ in range(10)]
    )
```

### Execution Issues

#### Packager Problems

**Problem**: Code not packaged correctly

**Solution**: Check packager configuration:

```python
# Test packager
packager = run.GitArchivePackager(subpath="src")
executor = run.LocalExecutor(packager=packager)

# Verify Git repository
git status
git add .
git commit -m "Test commit"
```

**Problem**: Files missing from package

**Solution**: Use appropriate packager:

```python
# For non-Git repositories
packager = run.PatternPackager(
    include_pattern="src/**",
    relative_path=os.getcwd()
)
executor = run.DockerExecutor(packager=packager)
```

#### Executor Configuration Issues

**Problem**: Slurm executor fails

**Solution**: Check Slurm configuration:

```python
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=1,
    gpus_per_node=4,
    time="02:00:00"
)

# Test with dry run
experiment = run.submit(config, executor)
experiment.dryrun = True
```

**Problem**: Docker executor fails

**Solution**: Check Docker configuration:

```python
executor = run.DockerExecutor(
    container_image="nvidia/pytorch:24.05-py3",
    gpus="all"
)

# Test Docker daemon
import docker
client = docker.from_env()
client.ping()
```

**Problem**: SkyPilot executor fails

**Solution**: Check SkyPilot configuration:

```python
executor = run.SkypilotExecutor(
    cluster_name="my-cluster",
    region="us-west1"
)

# Verify SkyPilot installation
pip list | grep skypilot
```

### Logging and Monitoring Issues

#### Log Retrieval Problems

**Problem**: Cannot retrieve experiment logs

**Solution**: Check experiment status and home directory:

```python
# Check experiment status
experiment = run.get_experiment(experiment_id)
print(f"Status: {experiment.status}")

# Check logs
logs = run.get_logs(experiment)
print(f"Exit code: {logs.exit_code}")
print(f"Output: {logs.stdout}")
```

**Problem**: NeMo Run home directory issues

**Solution**: Check and fix home directory:

```bash
# Check current home
echo $NEMORUN_HOME

# Set correct home
export NEMORUN_HOME=~/.nemo_run

# Create directory if missing
mkdir -p ~/.nemo_run
```

### Network and Connectivity Issues

#### SSH Tunnel Problems

**Problem**: SSH tunnel connection fails

**Solution**: Check SSH configuration:

```python
from nemo_run.core.execution.slurm import SSHTunnel

tunnel = SSHTunnel(
    host="cluster.example.com",
    username="your_username",
    port=22
)

# Test SSH connection
ssh -T your_username@cluster.example.com
```

**Problem**: Network timeout issues

**Solution**: Configure network timeouts:

```bash
# Set network timeouts
export NEMORUN_NETWORK_TIMEOUT=60
export NEMORUN_MAX_CONNECTIONS=50
```

## Debugging Techniques

### Enable Debug Mode

Enable comprehensive debugging:

```bash
# Enable debug logging
export NEMORUN_DEBUG=true
export NEMORUN_LOG_LEVEL=DEBUG

# Run with verbose output
python -c "import nemo_run; print('Debug mode enabled')"
```

### Configuration Validation

Validate configurations before execution:

```python
from nemo_run.config import ZlibJSONSerializer

def validate_config(config):
    """Validate configuration serialization."""
    serializer = ZlibJSONSerializer()

    try:
        serialized = serializer.serialize(config)
        deserialized = serializer.deserialize(serialized)

        if config == deserialized:
            print("✅ Configuration is valid")
            return True
        else:
            print("❌ Configuration changed during serialization")
            return False

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Usage
validate_config(my_config)
```

### CLI Debugging

Debug CLI issues:

```bash
# Test CLI help
python script.py --help

# Test with dry run
python script.py --dryrun param1=value1

# Test with verbose output
python script.py --verbose param1=value1
```

### Executor Testing

Test executor configurations:

```python
# Test local executor
executor = run.LocalExecutor()
print("✅ Local executor created")

# Test Docker executor
executor = run.DockerExecutor(container_image="python:3.9")
print("✅ Docker executor created")

# Test Slurm executor
executor = run.SlurmExecutor(partition="cpu", time="00:10:00")
print("✅ Slurm executor created")
```

## Performance Issues

### Resource Optimization

**Problem**: High memory usage

**Solution**: Configure memory limits:

```bash
# Set memory limits
export NEMORUN_MAX_MEMORY=8GB
export NEMORUN_MEMORY_POOL_SIZE=2GB

# Monitor memory usage
free -h
```

**Problem**: Slow execution

**Solution**: Optimize configuration:

```python
# Use efficient packager
packager = run.GitArchivePackager(subpath="src")

# Configure resource limits
executor = run.SlurmExecutor(
    partition="gpu",
    nodes=2,
    gpus_per_node=4,
    memory="64GB"
)
```

### Network Optimization

**Problem**: Slow network transfers

**Solution**: Configure network settings:

```bash
# Enable compression
export NEMORUN_COMPRESSION=true
export NEMORUN_CHUNK_SIZE=1MB

# Configure timeouts
export NEMORUN_NETWORK_TIMEOUT=30
export NEMORUN_KEEPALIVE=true
```

## Error Message Reference

### Common Error Messages

#### Import Errors

```
ModuleNotFoundError: No module named 'nemo_run'
```
**Solution**: Install NeMo Run correctly:
```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

#### Serialization Errors

```
TypeError: Object of type Path is not JSON serializable
```
**Solution**: Wrap in `run.Config`:
```python
config = run.Config(MyClass, path=run.Config(Path, "/tmp"))
```

#### Executor Errors

```
ExecutorError: Failed to submit job
```
**Solution**: Check executor configuration and connectivity.

#### Configuration Errors

```
ConfigurationError: Invalid configuration
```
**Solution**: Validate configuration before execution.

## Getting Help

### Diagnostic Information

When reporting issues, include this diagnostic information:

```bash
# System information
python --version
pip --version
echo $NEMORUN_HOME

# NeMo Run information
python -c "import nemo_run; print(f'Version: {nemo_run.__version__}')"

# Environment information
env | grep NEMORUN
```

### Reporting Issues

When reporting issues to the NeMo Run team:

1. **Include diagnostic information** (see above)
2. **Provide error messages** and stack traces
3. **Describe the steps** to reproduce the issue
4. **Include configuration files** (if applicable)
5. **Specify your environment** (OS, Python version, etc.)

### Example Issue Report

```
NeMo Run Version: 1.0.0
Python Version: 3.9.7
OS: Ubuntu 20.04
NEMORUN_HOME: ~/.nemo_run

Error: Configuration serialization fails
Steps to reproduce:
1. Create configuration with Path object
2. Attempt to serialize
3. Get TypeError

Error message:
TypeError: Object of type Path is not JSON serializable
```

This troubleshooting guide should help you resolve most common issues with NeMo Run. If you continue to experience problems, please report them with the information requested above.
