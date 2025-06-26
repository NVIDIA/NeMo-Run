---
description: "Comprehensive troubleshooting guide for NeMo Run covering common issues, error messages, debugging techniques, and solutions."
tags: ["troubleshooting", "debugging", "errors", "help", "support"]
categories: ["help"]
---

(troubleshooting-overview)=

# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using NeMo Run. It covers error messages, debugging techniques, and solutions for various scenarios.

## Quick Diagnosis

### Check NeMo Run Status

```bash
# Check NeMo Run installation
python -c "import nemo_run; print(nemo_run.__version__)"

# Check environment
echo $NEMORUN_HOME
echo $PYTHONPATH

# Check available executors
python -c "import nemo_run as run; print(dir(run))"
```

### Common Error Patterns

| Error Pattern | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| `ModuleNotFoundError` | Missing dependencies | Install required packages |
| `PermissionError` | File/directory permissions | Check file permissions |
| `ConnectionError` | Network/SSH issues | Verify connectivity |
| `ConfigError` | Configuration issues | Validate configuration |
| `ExecutorError` | Execution environment | Check executor setup |

## Configuration Issues

### UnserializableValueError

**Error:**

```
fiddle._src.experimental.serialization.UnserializableValueError:
Unserializable value .tmp of type <class 'pathlib.PosixPath'>.
Error occurred at path '<root>.something'
```

**Cause:** Non-serializable objects in configuration

**Solution:**

```python
# ❌ Wrong: Direct path object
partial = run.Partial(some_function, something=Path("/tmp"))

# ✅ Correct: Wrap in run.Config
partial = run.Partial(some_function, something=run.Config(Path, "/tmp"))
```

### Deserialization Error

**Error:**

```
ValueError: Using the Buildable constructor to convert a buildable to a new type
or to override arguments is forbidden; please use either `fdl.cast(new_type, buildable)`
(for casting) or `fdl.copy_with(buildable, **kwargs)` (for overriding arguments).
```

**Cause:** Mixed configuration types

**Solution:**

```python
# Ensure only Config or Partial objects in nested configuration
from nemo_run.config import ZlibJSONSerializer

serializer = ZlibJSONSerializer()
partial = run.Partial(some_function, something=run.Config(Path, "/tmp"))

# Test serialization
serializer.deserialize(serializer.serialize(partial)) == partial
```

### Control Flow in autoconvert

**Error:**

```
UnsupportedLanguageConstructError: Control flow (ListComp) is unsupported by auto_config.
```

**Cause:** Complex control flow in autoconvert decorator

**Solution:**

```python
# ❌ Wrong: Control flow in autoconvert
@run.autoconvert
def control_flow() -> llm.PreTrainingDataModule:
    return llm.PreTrainingDataModule(
        paths=[Path(f"some_doc_{i}") for i in range(10)],
        weights=[1 for i in range(10)]
    )

# ✅ Correct: Direct config return
def control_flow_config() -> run.Config[llm.PreTrainingDataModule]:
    return run.Config(
        llm.PreTrainingDataModule,
        paths=[run.Config(Path, f"some_doc_{i}") for i in range(10)],
        weights=[1 for i in range(10)]
    )
```

## Execution Issues

### Local Changes Not Reflected

**Problem:** Local changes not appearing in remote jobs

**Cause:** Uncommitted changes or wrong packaging strategy

**Solutions:**

1. **Commit Git Changes:**

   ```bash
   git add .
   git commit -m "Update configuration"
   ```

2. **Check Packaging Strategy:**

   ```python
   # Use GitArchivePackager for version-controlled code
   packager = run.GitArchivePackager(subpath="src")

   # Use PatternPackager for non-git code
   packager = run.PatternPackager(include_pattern="src/**")
   ```

3. **Verify Working Directory:**

   ```python
   # Check what gets packaged
   executor = run.LocalExecutor(packager=packager)
   ```

### Slurm Execution Issues

**Problem:** Jobs not starting or failing on Slurm

**Solutions:**

1. **Check Slurm Configuration:**

   ```python
   executor = run.SlurmExecutor(
       account="your_account",      # Verify account
       partition="your_partition",  # Verify partition
       time="02:00:00",            # Set appropriate time limit
       nodes=1,                    # Check node requirements
       ntasks_per_node=8,          # Verify task count
       gpus_per_node=8             # Verify GPU count
   )
   ```

2. **Verify SSH Tunnel:**

   ```python
   # For remote execution
   ssh_tunnel = run.SSHTunnel(
       host="your-slurm-host",
       user="your-user",
       job_dir="/scratch/your-user/runs",
       identity="~/.ssh/id_ed25519"  # Verify SSH key
   )

   # For local execution on cluster
   local_tunnel = run.LocalTunnel()
   ```

3. **Check Resource Availability:**

   ```bash
   # Check Slurm status
   sinfo
   squeue
   scontrol show partition your_partition
   ```

### Docker Execution Issues

**Problem:** Docker containers failing to start

**Solutions:**

1. **Verify Docker Installation:**

   ```bash
   docker --version
   docker ps
   ```

2. **Check Image Availability:**

   ```python
   executor = run.DockerExecutor(
       container_image="pytorch/pytorch:latest",  # Verify image exists
       num_gpus=1,                               # Check GPU requirements
       runtime="nvidia"                          # For GPU support
   )
   ```

3. **Verify Volume Mounts:**

   ```python
   executor = run.DockerExecutor(
       container_image="pytorch/pytorch:latest",
       volumes=["/local/path:/path/in/container"],  # Check paths
       env_vars={"PYTHONUNBUFFERED": "1"}
   )
   ```

### Skypilot Execution Issues

**Problem:** Skypilot jobs failing

**Solutions:**

1. **Install Skypilot:**

   ```bash
   pip install "nemo_run[skypilot]"
   ```

2. **Configure Cloud:**

   ```bash
   sky check
   sky status
   ```

3. **Check Cloud Configuration:**

   ```python
   executor = run.SkypilotExecutor(
       cluster_name="my-cluster",
       region="us-west-2",           # Verify region
       zone="us-west-2a",            # Verify zone
       instance_type="g4dn.xlarge"   # Verify instance type
   )
   ```

## Management Issues

### Cannot Retrieve Logs

**Problem:** Unable to access experiment logs

**Causes and Solutions:**

1. **NeMo Run Home Changed:**

   ```bash
   # Check current home
   echo $NEMORUN_HOME

   # Set consistent home
   export NEMORUN_HOME=~/.nemorun
   ```

2. **Home Directory Deleted:**

   ```bash
   # Recreate home directory
   mkdir -p ~/.nemorun
   ```

3. **Remote Logs Unavailable:**

   ```python
   # For Kubernetes/Skypilot, logs may be lost if cluster is terminated
   # Use persistent storage for important logs
   executor = run.SkypilotExecutor(
       # ... other config ...
       file_mounts={
           "/logs": "s3://my-bucket/logs"  # Persistent storage
       }
   )
   ```

### Experiment Status Issues

**Problem:** Experiments stuck or showing incorrect status

**Solutions:**

1. **Check Experiment Status:**

   ```python
   experiment = run.Experiment.from_id("experiment_id")
   experiment.status()
   ```

2. **Cancel Stuck Jobs:**

   ```python
   experiment.cancel("task_name")
   ```

3. **Check Job Dependencies:**

   ```python
   # Verify dependency configuration
   exp.add(task1, name="task1")
   exp.add(task2, name="task2", dependencies=["task1"])
   ```

## CLI Issues

### Command Not Found

**Problem:** CLI commands not recognized

**Solutions:**

1. **Check Installation:**

   ```bash
   pip list | grep nemo-run
   ```

2. **Verify Entrypoint Registration:**

   ```python
   @run.cli.entrypoint
   def my_command():
       pass

   if __name__ == "__main__":
       my_command.main()
   ```

3. **Check Python Path:**

   ```bash
   python -c "import sys; print(sys.path)"
   ```

### Argument Parsing Errors

**Problem:** CLI arguments not parsed correctly

**Solutions:**

1. **Use Correct Syntax:**

   ```bash
   # ✅ Correct
   python script.py model=resnet50 learning_rate=0.001

   # ❌ Wrong
   python script.py --model resnet50 --learning-rate 0.001
   ```

2. **Check Type Hints:**

   ```python
   @run.cli.entrypoint
   def train_model(
       model: str,           # String type
       learning_rate: float, # Float type
       epochs: int          # Integer type
   ):
       pass
   ```

3. **Use Factory Functions:**

   ```python
   @run.cli.factory
   def create_model(name: str, hidden_size: int = 512):
       return {"name": name, "hidden_size": hidden_size}

   @run.cli.entrypoint
   def train(model=create_model(name="transformer")):
       pass
   ```

## Performance Issues

### Slow Job Submission

**Problem:** Jobs taking too long to submit

**Solutions:**

1. **Optimize Package Size:**

   ```python
   # Use specific subpaths
   packager = run.GitArchivePackager(subpath="src/models")

   # Exclude unnecessary files
   packager = run.PatternPackager(
       include_pattern="src/**/*.py",
       exclude_pattern="**/*_test.py **/__pycache__/**"
   )
   ```

2. **Use Local Testing:**

   ```python
   # Test locally first
   executor = run.LocalExecutor()
   ```

3. **Check Network Connectivity:**

   ```bash
   # Test SSH connection
   ssh user@remote-host "echo 'Connection successful'"
   ```

### Memory Issues

**Problem:** Jobs running out of memory

**Solutions:**

1. **Adjust Resource Requests:**

   ```python
   executor = run.SlurmExecutor(
       mem="64G",           # Request more memory
       mem_per_cpu="8G"     # Or per CPU
   )
   ```

2. **Optimize Code:**

   ```python
   # Use generators for large datasets
   def data_generator():
       for item in large_dataset:
           yield process_item(item)
   ```

3. **Monitor Resource Usage:**

   ```bash
   # Check memory usage
   htop
   nvidia-smi  # For GPU memory
   ```

## Network and Connectivity Issues

### SSH Connection Problems

**Problem:** Cannot connect to remote clusters

**Solutions:**

1. **Verify SSH Configuration:**

   ```bash
   # Test SSH connection
   ssh -i ~/.ssh/id_ed25519 user@host

   # Check SSH key permissions
   chmod 600 ~/.ssh/id_ed25519
   ```

2. **Configure SSH Tunnel:**

   ```python
   ssh_tunnel = run.SSHTunnel(
       host="your-host",
       user="your-user",
       job_dir="/scratch/user/runs",
       identity="~/.ssh/id_ed25519"
   )
   ```

3. **Check Firewall Settings:**

   ```bash
   # Test port connectivity
   telnet host 22
   ```

### Network Timeouts

**Problem:** Network operations timing out

**Solutions:**

1. **Increase Timeouts:**

   ```python
   # Configure longer timeouts
   executor = run.SlurmExecutor(
       # ... other config ...
       timeout=3600  # 1 hour timeout
   )
   ```

2. **Use Retry Logic:**

   ```python
   # Implement retry logic in your code
   import time
   from functools import wraps

   def retry_on_timeout(max_retries=3, delay=5):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except TimeoutError:
                       if attempt == max_retries - 1:
                           raise
                       time.sleep(delay)
           return wrapper
       return decorator
   ```

## Debug Techniques

### Enable Verbose Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or use CLI verbose flag
# python script.py --verbose
```

### Use Dry Run Mode

```bash
# Preview execution without running
python script.py train_model --dryrun
```

### Debug Interactively

```bash
# Enter interactive mode
python script.py train_model --repl
```

### Export Configurations

```bash
# Export to inspect configuration
python script.py train_model --to-yaml config.yaml
python script.py train_model --to-json config.json
```

### Check Experiment Metadata

```python
# Inspect experiment details
experiment = run.Experiment.from_id("experiment_id")
print(experiment.metadata)
```

## Get Help

### Self-Diagnosis Checklist

- [ ] Check NeMo Run version and installation
- [ ] Verify environment variables
- [ ] Test with minimal example
- [ ] Check executor configuration
- [ ] Verify network connectivity
- [ ] Review error logs
- [ ] Test with different executor

### Useful Commands

```bash
# Check system information
python -c "import nemo_run; print(nemo_run.__version__)"
echo $NEMORUN_HOME
echo $PYTHONPATH

# Check available resources
nvidia-smi  # GPU status
sinfo       # Slurm status
docker ps   # Docker status

# Test connectivity
ssh user@host "echo 'SSH working'"
ping host
telnet host port
```

### Report Issues

When reporting issues, include:

1. **NeMo Run version**
2. **Python version**
3. **Operating system**
4. **Error message and traceback**
5. **Minimal reproduction example**
6. **Environment configuration**
7. **Steps to reproduce**

### Example Issue Report

```
NeMo Run Version: 1.0.0
Python Version: 3.9.0
OS: Ubuntu 20.04

Error:
[Full error message and traceback]

Reproduction Steps:
1. Create configuration with run.Partial
2. Use SlurmExecutor with SSH tunnel
3. Submit job

Configuration:
[Minimal configuration that reproduces the issue]

Environment:
NEMORUN_HOME=~/.nemorun
[Other relevant environment variables]
```

This troubleshooting guide should help you resolve most common issues with NeMo Run. If you continue to experience problems, please report them with the information requested above.
